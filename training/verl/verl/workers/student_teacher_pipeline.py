# Copyright 2026 Open-MOPD contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""S(old) -> T0 -> ... -> Tn -> S(new), at stable actor/vLLM addresses."""

import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
from torch.distributed.fsdp._runtime_utils import _lazy_init

from verl.workers.teacher_layer_pipeline import TeacherLayerPipeline


class StudentTeacherPipeline(TeacherLayerPipeline):
    def __init__(self, student, optimizer, models, names, max_mb, overlap=True):
        self.optimizer = optimizer
        self.student = student
        self.overlap = overlap
        self.condition = threading.Condition()
        self.scoring = False
        self.student_released = set()
        self.restore_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="student-restore")
        self.restores = []
        _lazy_init(models[0], models[0])
        teacher_modules = {
            n: m for n, m in models[0].named_modules()
            if isinstance(m, type(student)) and m._handle is not None
        }
        if set(teacher_modules) != set(optimizer.modules):
            raise ValueError("student/teacher FSDP wrapping must match")
        slots = {}
        for name, module in teacher_modules.items():
            source = module._handle.flat_param
            target = optimizer.modules[name]._handle.flat_param
            if (source.dtype != target.dtype or source.numel() != target.numel()
                    or tuple(source._fqns) != tuple(target._fqns)
                    or tuple(source._shapes) != tuple(target._shapes)
                    or tuple(source._numels_with_padding) != tuple(target._numels_with_padding)):
                raise ValueError(f"student/teacher flat parameter layouts differ: {name or 'root'}")
            # Tied embeddings/head stay live across the entire student forward.
            # A dedicated teacher root slot breaks this otherwise cyclic dependency.
            slots[name] = (
                torch.empty_like(source, device=student.compute_device) if not name
                else optimizer.gpu_weights[name]
            )
        super().__init__(models, names, max_mb, slot_tensors=slots)
        self.extra_gpu_bytes = self.slots[""].tensor.nbytes
        for name, module in optimizer.modules.items():
            self.hooks.append(module.register_forward_hook(partial(self._student_release, name)))

    def begin_student(self):
        self._check_live()
        if self.scoring or self.expected != 0 or self.active is not None:
            raise RuntimeError("previous student/teacher cycle is unfinished")
        self.optimizer.start()  # Also support fixed-input scoring without a rollout.
        self.scoring = True
        self.student_released.clear()
        self.restores = []
        for slot in self.slots.values():
            slot.owner = -1
            slot.ready = None
        self._stage(0, "")

    def prefetch_first(self):
        # Decoder copies are submitted by student post-hooks, never before last use.
        if not self.scoring:
            raise RuntimeError("begin student scoring before launching teachers")

    def _prime(self):
        if not self.scoring:
            raise RuntimeError("teacher called outside the student scoring cycle")

    def _student_release(self, name, module, args, output):
        if not self.scoring:
            return
        if torch.is_grad_enabled() or name in self.student_released:
            raise RuntimeError("student scoring must execute each layer once under no_grad")
        self.student_released.add(name)
        if name:
            last_use = torch.cuda.Event()
            last_use.record(torch.cuda.current_stream(self.device))
            self._stage(0, name, last_use)

    def _stage(self, index, name, last_use=None):
        # Last teacher returns decoder slots to the new student, instead of T0.
        if index == 0 and self.active is not None and self.active[0] == len(self.names) - 1:
            with self.condition:
                self.slots[name].owner = -1
                self.slots[name].ready = None
            future = self.restore_executor.submit(self._restore, name, last_use)
            self.restores.append(future)
            if not self.overlap:
                future.result()
            return
        with self.condition:
            super()._stage(index, name, last_use)
            if not self.overlap:
                self.stream.synchronize()
            self.condition.notify_all()

    def _acquire(self, index, name):
        with self.condition:
            # A CUDA wait on an unrecorded event is a no-op, so first wait until
            # the producer has submitted the copy and recorded its ready event.
            self.condition.wait_for(lambda: self.broken or self.slots[name].owner == index)
            self._check_live()
            return super()._acquire(index, name)

    @torch.no_grad()
    def _restore(self, name, last_use):
        self.optimizer.wait()
        with torch.cuda.device(self.device), self.condition, torch.cuda.stream(self.stream):
            with torch.cuda.nvtx.range(f"openmopd::io::h2d::student_layer_restore::{name or 'root'}"):
                self.stream.wait_event(last_use)
                self.optimizer.gpu_weights[name].copy_(self.optimizer.host_weights[name], non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(self.stream)
                self.slots[name].ready = ready
            if not self.overlap:
                self.stream.synchronize()

    def finish_student(self):
        if self.student_released != set(self.slots):
            self.abort()
            raise RuntimeError("student scoring skipped a layer")

    def finish_cycle(self):
        if self.expected != 0 or len(self.restores) != len(self.slots):
            raise RuntimeError("all teachers must run before restoring the student")
        for future in self.restores:
            future.result()
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        self.optimizer.publish()
        self.scoring = False

    def abort(self):
        with self.condition:
            self.broken = True
            self.condition.notify_all()

    def close(self):
        self.abort()
        self.restore_executor.shutdown(wait=True)
        super().close()
