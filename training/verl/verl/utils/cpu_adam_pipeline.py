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
"""One pending CPU AdamW update, with explicit publication to stable GPU storage."""

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch


def _cpu_adam_worker(connection, groups, states, gradients, threads):
    """CPUAdam's ordinary C++ binding holds the GIL: use a spawned process."""
    try:
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        torch.set_num_threads(threads)
        optimizer = DeepSpeedCPUAdam(groups, adamw_mode=True, fp32_optimizer_states=True)
        params = [p for group in optimizer.param_groups for p in group["params"]]
        for param, state in zip(params, states, strict=True):
            optimizer.state[param] = state
        connection.send((True, None))
        while True:
            message = connection.recv()
            if message is None:
                break
            live, options, steps = message
            for group, opt in zip(optimizer.param_groups, options, strict=True):
                group.update(opt)
            for i, param in enumerate(params):
                param.grad = gradients[i] if i in live else None
                optimizer.state[param]["step"] = steps[i]
            start = time.perf_counter()
            with torch.cuda.nvtx.range("openmopd::cpu::adam_native"):
                optimizer.step()
            connection.send((True, time.perf_counter() - start))
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


class CPUAdamPipeline(torch.optim.Optimizer):
    """GPU-facing optimizer with FP32 CPU masters and FP32 CPU moments.

    FSDP hooks offload completed layer gradients during backward. step() seals
    that gradient batch; start() submits CPU work just before rollout. With early
    offload disabled, start() also submits gradient DMA. Neither writes GPU
    weights. The scoring pipeline publishes the result after old-policy scoring.
    The public groups/state remain keyed by actor parameters for FSDP checkpoints.
    """

    def __init__(self, optimizer, model, overlap=True, threads=8, gradient_offload=True):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp._runtime_utils import _lazy_init

        if optimizer.state:
            raise ValueError("CPUAdamPipeline must be constructed before any optimizer updates")
        if not isinstance(model, FSDP) or threads < 1:
            raise ValueError("CPUAdamPipeline requires FSDP1 and a positive CPU thread count")
        # Configure once at initialization: parent-side BF16/FP32 staging is also
        # substantial. Do not resize the intra-op pool during concurrent work.
        torch.set_num_threads(threads)
        _lazy_init(model, model)
        self.modules = {
            n: m for n, m in model.named_modules() if isinstance(m, FSDP) and m._handle is not None
        }
        self.device = model.compute_device
        self.host_weights = {}
        self.gpu_weights = {}
        storage = {}
        for name, module in self.modules.items():
            handle = module._handle
            flat = handle.flat_param._local_shard
            if (handle.world_size != 1 or handle.uses_sharded_strategy or handle._offload_params
                    or not handle._use_orig_params or handle._uses_param_mixed_precision
                    or flat.dtype != torch.bfloat16):
                raise ValueError("CPU pipeline requires BF16 NO_SHARD original parameters resident on one GPU")
            host = torch.empty_like(flat, device="cpu", pin_memory=True)
            host.copy_(flat)
            self.host_weights[name] = host
            self.gpu_weights[name] = flat.detach()
            storage[flat.untyped_storage().data_ptr()] = (flat, host)
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self.pairs = []
        self.shared_gradients = []
        cpu_groups = []
        for group in self.param_groups:
            if group.get("maximize", False) or group.get("amsgrad", False):
                raise ValueError("CPU pipeline supports ordinary AdamW only")
            masters = []
            for param in group["params"]:
                flat, host = storage[param.untyped_storage().data_ptr()]
                offset = param.storage_offset() - flat.storage_offset()
                output = host.narrow(0, offset, param.numel()).view(param.shape)
                master = torch.nn.Parameter(output.float().share_memory_(), requires_grad=False)
                gradient = torch.empty_like(output, pin_memory=True)
                self.shared_gradients.append(torch.empty_like(master).share_memory_())
                self.state[param] = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(master).share_memory_(),
                    "exp_avg_sq": torch.zeros_like(master).share_memory_(),
                    "cpu_master": master.detach(),
                }
                masters.append(master)
                self.pairs.append((param, master, gradient, output))
            cpu_groups.append(dict(params=masters, **self._options(group)))
        self.cpu_groups = cpu_groups
        context = torch.multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        states = [{k: v for k, v in self.state[p].items() if k != "cpu_master"} for p, *_ in self.pairs]
        self.process = context.Process(
            target=_cpu_adam_worker,
            args=(child, cpu_groups, states, self.shared_gradients, threads),
            name="openmopd-cpu-adam",
            daemon=True,
        )
        self.process.start()
        child.close()
        self._receive()
        self.overlap = overlap
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpu-adam")
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.future = None
        self.pending = None
        self.submit_lock = threading.Lock()
        self.version = 0
        self.last_metrics = {}
        self.gradient_offload = gradient_offload
        self.gradient_norms = {}
        self.gradient_scale = None
        self.gradient_layers = set()
        self.backward_hooks = {}
        self.forward_hooks = []
        indices = {id(p): index for index, (p, *_) in enumerate(self.pairs)}
        self.layer_indices = {
            name: [indices[id(p)] for p in module._handle.flat_param._params if id(p) in indices]
            for name, module in self.modules.items()
        }
        if gradient_offload:
            for name, module in self.modules.items():
                self.forward_hooks.append(module.register_forward_hook(partial(self._arm_gradient_offload, name)))

    def _arm_gradient_offload(self, name, module, args, output):
        if not torch.is_grad_enabled() or name in self.backward_hooks:
            return
        flat = module._handle.flat_param
        if not flat.requires_grad:
            return
        state = getattr(flat, "_post_backward_hook_state", None)
        if state is None or len(state) != 2:
            raise RuntimeError("gradient offload requires eager FSDP1 AccumulateGrad hooks")
        # FSDP registered its reduction hook during pre-forward. Register AFTER
        # it, so .grad views contain the final reduced gradient of this layer.
        self.backward_hooks[name] = state[0].register_hook(partial(self._offload_layer_gradient, name, module))

    @torch.no_grad()
    def _offload_layer_gradient(self, name, module, *unused):
        self.backward_hooks.pop(name).remove()
        if name in self.gradient_layers:
            raise RuntimeError("early gradient offload requires one backward per optimizer step")
        if self.pending is not None or self.future is not None or not module._sync_gradients:
            raise RuntimeError("publish the previous CPU update before a synchronized backward")
        self.gradient_layers.add(name)
        handle = module._handle
        live = [(i, self.pairs[i][0].grad) for i in self.layer_indices[name] if self.pairs[i][0].grad is not None]
        # Keep only scalar norms on GPU. Using the same foreach norm and BF16
        # coefficient as ordinary clipping preserves the existing numerics.
        with torch.cuda.stream(module._post_backward_stream):
            if live:
                norms = torch._foreach_norm([g for _, g in live], 2.0)
                self.gradient_norms.update((i, norm) for (i, _), norm in zip(live, norms, strict=True))
        self.copy_stream.wait_stream(module._post_backward_stream)
        with torch.cuda.stream(self.copy_stream), torch.cuda.nvtx.range(
            f"openmopd::io::d2h::actor_gradients::{name or 'root'}"
        ):
            for i, gradient in live:
                self.pairs[i][2].copy_(gradient, non_blocking=True)
                gradient.record_stream(self.copy_stream)
        if not self.overlap:
            self.copy_stream.synchronize()
        # record_stream keeps storage alive until DMA completes without keeping
        # all layers' .grad references through the remainder of backward.
        handle.flat_param.grad = None
        handle._use_unsharded_grad_views()

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm):
        if not self.gradient_offload:
            return torch.nn.utils.clip_grad_norm_([p for p, *_ in self.pairs], max_norm)
        for module in self.modules.values():
            torch.cuda.current_stream(self.device).wait_stream(module._post_backward_stream)
        norms = [self.gradient_norms[i] for i in sorted(self.gradient_norms)]
        total = torch.linalg.vector_norm(torch.stack(norms), 2.0) if norms else torch.zeros((), device=self.device)
        self.gradient_scale = (max_norm / (total + 1e-6)).clamp(max=1.0).cpu()
        return total

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        # In the non-finite-gradient path no step() consumed these buffers.
        if self.pending is None and self.future is None:
            self.gradient_layers.clear()
            self.gradient_norms.clear()
            self.gradient_scale = None

    @staticmethod
    def _options(group):
        return {k: group[k] for k in ("lr", "betas", "eps", "weight_decay")}

    def _receive(self):
        ok, result = self.connection.recv()
        if not ok:
            raise RuntimeError(f"CPU Adam worker failed:\n{result}")
        return result

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("CPU pipeline does not support optimizer closures")
        if self.future is not None or self.pending is not None:
            raise RuntimeError("previous CPU update has not been published")
        produced = torch.cuda.Event()
        if self.gradient_offload:
            if self.gradient_scale is None:
                raise RuntimeError("call CPUAdamPipeline.clip_grad_norm_ before step")
            produced.record(self.copy_stream)
            gradients = None
            live = sorted(self.gradient_norms)
            scale = self.gradient_scale
        else:
            produced.record(torch.cuda.current_stream(self.device))
            gradients = [(i, p.grad.detach()) for i, (p, *_) in enumerate(self.pairs) if p.grad is not None]
            live, scale = None, None
        # Early mode retains host gradients; the comparison mode retains GPU
        # gradients across zero_grad and defers DMA past metrics / vLLM wake.
        self.pending = (gradients, produced, live, scale, [self._options(group) for group in self.param_groups])
        self.gradient_layers.clear()
        self.gradient_norms.clear()
        self.gradient_scale = None
        if not self.overlap:
            self.wait()

    def start(self):
        """Start CPU work (and deferred DMA, if any) immediately before generation."""
        with self.submit_lock:
            if self.pending is not None:
                self._start()

    @torch.no_grad()
    def _start(self):
        gradients, ready, live, scale, options = self.pending
        if gradients is not None:
            live = []
            self.copy_stream.wait_event(ready)
            with torch.cuda.stream(self.copy_stream), torch.cuda.nvtx.range("openmopd::io::d2h::actor_gradients"):
                for index, gradient in gradients:
                    self.pairs[index][2].copy_(gradient, non_blocking=True)
                    gradient.record_stream(self.copy_stream)
                    live.append(index)
                ready = torch.cuda.Event()
                ready.record(self.copy_stream)
        self.future = self.executor.submit(self._update, ready, live, options, scale)
        self.pending = None

    @torch.no_grad()
    def _update(self, ready, live, options, scale):
        with torch.cuda.device(self.device):
            with torch.cuda.nvtx.range("openmopd::cpu::optimizer_wait_gradients"):
                ready.synchronize()
            start = time.perf_counter()
            with torch.cuda.nvtx.range("openmopd::cpu::optimizer_update"):
                with torch.cuda.nvtx.range("openmopd::cpu::gradient_cast"):
                    for index in live:
                        if scale is not None:
                            self.pairs[index][2].mul_(scale)
                        self.shared_gradients[index].copy_(self.pairs[index][2])
                cast_seconds = time.perf_counter() - start
                steps = [int(self.state[p]["step"]) for p, *_ in self.pairs]
                self.connection.send((set(live), options, steps))
                native_seconds = self._receive()
                publish_start = time.perf_counter()
                with torch.cuda.nvtx.range("openmopd::cpu::weight_cast"):
                    for index in live:
                        param, master, _, output = self.pairs[index]
                        output.copy_(master)
                        self.state[param]["step"] += 1
            self.last_metrics = {
                "pipeline/cpu_optimizer_seconds": time.perf_counter() - start,
                "pipeline/cpu_adam_native_seconds": native_seconds,
                "pipeline/gradient_cast_seconds": cast_seconds,
                "pipeline/weight_cast_seconds": time.perf_counter() - publish_start,
            }

    def wait(self):
        self.start()
        if self.future is not None:
            self.future.result()

    def publish(self):
        """Called only after the slot manager has ordered all weight copies."""
        if self.future is not None or self.pending is not None:
            self.wait()
            self.future = None
            self.version += 1

    @torch.no_grad()
    def flush(self):
        """Drain pending work and restore GPU weights at validation/save/final boundaries."""
        if self.future is None and self.pending is None:
            return
        self.wait()
        with torch.cuda.nvtx.range("openmopd::io::h2d::actor_weights_flush"):
            for name, destination in self.gpu_weights.items():
                destination.copy_(self.host_weights[name], non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        self.publish()

    def state_dict(self):
        if self.future is not None or self.pending is not None:
            raise RuntimeError("flush CPU pipeline before saving optimizer state")
        result = super().state_dict()
        result["pipeline_version"] = self.version
        return result

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        if self.future is not None or self.pending is not None:
            raise RuntimeError("cannot load an optimizer with a pending CPU update")
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups) or any(
            len(a["params"]) != len(b["params"]) for a, b in zip(saved_groups, self.param_groups, strict=True)
        ):
            raise ValueError("CPU optimizer checkpoint parameter groups do not match")
        for group, saved in zip(self.param_groups, saved_groups, strict=True):
            group.update({key: value for key, value in saved.items() if key != "params"})
        indices = [index for group in saved_groups for index in group["params"]]
        for index, (param, master, _, output) in zip(indices, self.pairs, strict=True):
            saved = state_dict["state"][index]
            if "cpu_master" not in saved:
                raise ValueError("CPU pipeline resume requires a checkpoint with FP32 master weights")
            state = self.state[param]
            master.copy_(saved["cpu_master"])
            for key in ("exp_avg", "exp_avg_sq"):
                state[key].copy_(saved[key])
            state["step"] = int(saved["step"])
            output.copy_(master)
        self.version = state_dict.get("pipeline_version", 0)

    def close(self):
        try:
            self.flush()
        finally:
            for hook in self.forward_hooks + list(self.backward_hooks.values()):
                hook.remove()
            self.executor.shutdown(wait=True)
            if self.process.is_alive():
                self.connection.send(None)
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
            self.connection.close()
