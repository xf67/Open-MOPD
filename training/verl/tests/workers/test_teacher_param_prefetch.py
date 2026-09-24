# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker, RewardModelWorker
from verl.workers.teacher_forward_overlap import TeacherForwardOverlap
from verl.workers.teacher_param_prefetch import TeacherHandlePrefetch


class TestTeacherParamPrefetch(unittest.TestCase):
    def test_lora_reference_scoring_skips_teacher_prefetch(self):
        teacher = SimpleNamespace(start_param_prefetch=Mock())
        disable_adapter = Mock(side_effect=nullcontext)

        def score(data, calculate_entropy, prefetch_callback, prefetch_point):
            if prefetch_callback is not None:
                prefetch_callback()
            return torch.zeros(1, 2), torch.ones(1, 2), None, None

        worker = SimpleNamespace(
            _is_actor=True,
            _is_lora=True,
            _is_offload_param=False,
            world_size=1,
            actor=SimpleNamespace(
                actor_module=SimpleNamespace(disable_adapter=disable_adapter), compute_log_prob=score
            ),
            ulysses_sharding_manager=nullcontext(),
            get_fused_worker_by_name=lambda name: teacher,
            config=OmegaConf.create(
                {
                    "rollout": {
                        "teacher_param_prefetch": True,
                        "teacher_param_prefetch_max_mb": 1,
                        "log_prob_micro_batch_size_per_gpu": 1,
                        "log_prob_max_token_len_per_gpu": 4,
                        "log_prob_use_dynamic_bsz": False,
                        "temperature": 1.0,
                    }
                }
            ),
        )
        for name in ("_compute_log_prob", "compute_log_prob", "compute_ref_log_prob"):
            setattr(worker, name, MethodType(getattr(ActorRolloutRefWorker, name), worker))

        def batch():
            return DataProto.from_dict({"input_ids": torch.ones(1, 4, dtype=torch.long)})

        worker.compute_log_prob(batch())
        teacher.start_param_prefetch.assert_called_once_with(1)
        reference = worker.compute_ref_log_prob(batch())
        teacher.start_param_prefetch.assert_called_once_with(1)
        disable_adapter.assert_called_once_with()
        torch.testing.assert_close(reference.batch["ref_log_prob"], torch.zeros(1, 2))
        worker.compute_log_prob(batch())
        self.assertEqual(teacher.start_param_prefetch.call_count, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_teacher_forward_reuses_prefetched_shard(self):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        with tempfile.TemporaryDirectory() if world_size == 1 else nullcontext() as directory:
            if world_size == 1:
                dist.init_process_group("nccl", init_method=Path(directory, "init").as_uri(), rank=0, world_size=1)
            else:
                torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
                dist.init_process_group("nccl", init_method="env://")
            try:
                torch.manual_seed(42)
                model = FSDP(
                    torch.nn.Linear(16, 16, bias=False),
                    cpu_offload=CPUOffload(offload_params=True),
                    device_id=torch.cuda.current_device(),
                ).eval()
                x = torch.randn(2, 16, device="cuda")
                with torch.no_grad():
                    baseline = model(x)

                prefetch = TeacherHandlePrefetch(model, max_mb=1)
                original_calls = 0
                original = prefetch.original_pre_unshard

                def counted_original():
                    nonlocal original_calls
                    original_calls += 1
                    return original()

                prefetch.original_pre_unshard = counted_original
                staged_ptr = prefetch.staged.data_ptr()
                prefetch.start()
                self.assertIsNotNone(prefetch.pending)
                self.assertEqual(prefetch.pending[0].data_ptr(), staged_ptr)
                with torch.no_grad():
                    result = model(x)
                prefetch.assert_consumed(previous_reuse_count=0)
                torch.testing.assert_close(result, baseline)
                self.assertEqual(original_calls, 0)  # no second HtoD through FSDP pre_unshard

                prefetch.start()
                self.assertEqual(prefetch.pending[0].data_ptr(), staged_ptr)
                with torch.no_grad():
                    result = model(x)
                prefetch.assert_consumed(previous_reuse_count=1)
                torch.testing.assert_close(result, baseline)
                self.assertEqual(original_calls, 0)

                with torch.no_grad():
                    model(x)
                self.assertEqual(original_calls, 1)  # normal path resumes without a pending shard
            finally:
                dist.destroy_process_group()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_serial_and_overlapped_logits_release_offloaded_views(self):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            self.skipTest("forward overlap requires one GPU")

        class Teacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(32, 16)
                self.layer = torch.nn.Linear(16, 16)
                self.head = torch.nn.Linear(16, 32, bias=False)
                self.head.weight = self.embed.weight

            def forward(self, input_ids, **kwargs):
                return (self.head(self.layer(self.embed(input_ids))),)

        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group("nccl", init_method=Path(directory, "init").as_uri(), rank=0, world_size=1)
            runner = None
            try:
                torch.manual_seed(42)
                raw = Teacher().eval()
                kwargs = {"cpu_offload": CPUOffload(offload_params=True), "device_id": torch.cuda.current_device()}
                raw.layer = FSDP(raw.layer, **kwargs)
                model = FSDP(raw, **kwargs)
                worker = SimpleNamespace(reward_module=model, use_fused_kernels=False)
                inputs = {"input_ids": torch.arange(8, device="cuda").unsqueeze(0)}
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
                inputs["position_ids"] = inputs["input_ids"].clone()

                def forward():
                    return RewardModelWorker._forward_model_logits(worker, inputs)

                reference = forward().cpu()
                prefetch = TeacherHandlePrefetch(model, max_mb=1)
                runner = TeacherForwardOverlap(model.compute_device)
                ready = torch.cuda.Event()
                ready.record()
                for tensor in inputs.values():
                    tensor.record_stream(runner.stream)

                for index, overlap in enumerate((False, True, True, False)):
                    if runner.last_done is not None:
                        prefetch.stream.wait_event(runner.last_done)
                    prefetch.start()
                    if overlap:
                        runner.start(forward, ready)
                        output = runner.finish()
                    else:
                        output = forward()
                    prefetch.assert_consumed(index)
                    torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)
                    self.assertIs(raw.embed.weight, raw.head.weight)
                    for module in FSDP.fsdp_modules(model):
                        for name, owner, _ in module._handle.flat_param._param_infos:
                            self.assertEqual(getattr(owner, name).device.type, "cpu")
            finally:
                if runner is not None:
                    runner.close()
                dist.destroy_process_group()
