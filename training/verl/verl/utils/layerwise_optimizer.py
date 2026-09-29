# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Keep Adam states on CPU and stage one model layer at a time for its update."""

from collections import defaultdict
from contextlib import nullcontext
from itertools import groupby

import torch

from verl.utils.bf16_optimizer import BF16StochasticAdamW


def _parameter_layers(model):
    # HF decoder blocks define the layer boundary. Outside those blocks (e.g.
    # embeddings and the output head), use the module owning the parameter.
    # FSDP1 flat parameters naturally remain indivisible local shards.
    block_classes = {name for module in model.modules() for name in (getattr(module, "_no_split_modules", None) or ())}
    owners = {}

    def visit(module, name, block=None):
        if type(module).__name__ in block_classes:
            block = name
        for param in module.parameters(recurse=False):
            owners.setdefault(id(param), block if block is not None else name)
        for child_name, child in module.named_children():
            visit(child, f"{name}.{child_name}" if name else child_name, block)

    visit(model, "root")
    return owners


class LayerwiseOffloadOptimizer(torch.optim.Optimizer):
    """Stage Adam/AdamW moments after backward and global gradient clipping.

    Only one layer's states reside on the accelerator. D2H completes before
    the next layer is loaded, so queued transfers cannot retain the full model's
    states. Pinned CPU buffers are reused. Parameter groups, update order, and
    the checkpoint format are preserved; parameters are always updated in place.

    This is intended for FSDP1 (flat or original parameters) and ordinary CUDA
    modules. FSDP2/DTensor and graph-captured/differentiable optimizers are not
    supported. It deliberately does not update parameters inside backward:
    gradient accumulation and global clipping must finish first.
    """

    def __init__(self, optimizer, model):
        if type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW, BF16StochasticAdamW):
            raise ValueError("Layerwise optimizer offload supports torch Adam/AdamW and BF16StochasticAdamW only")
        self._validate_groups(optimizer.param_groups)
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self.defaults = optimizer.defaults
        self._owners = _parameter_layers(model)
        for group in self.param_groups:
            for param in group["params"]:
                if id(param) not in self._owners:
                    raise ValueError("Optimizer parameter is not owned by the model")
        self.offload_state()

    @staticmethod
    def _validate_groups(groups):
        for group in groups:
            for option in ("foreach", "fused", "capturable", "differentiable"):
                if group.get(option, False):
                    raise ValueError(f"Layerwise optimizer offload requires {option}=False")
            # PyTorch's default None can otherwise select foreach on CUDA.
            group["foreach"] = False

    @staticmethod
    def _range(name, params):
        if any(param.is_cuda for param in params):
            return torch.cuda.nvtx.range(f"openmopd::{name}")
        return nullcontext()

    def _load_layer(self, params):
        host_states = {}
        for param in params:
            state = self.state.get(param, {})
            host_states[param] = state.copy()
            for key, value in state.items():
                # Non-capturable Adam keeps its scalar step on CPU.
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(param.device, non_blocking=True)
        return host_states

    def _offload_layer(self, params, host_states):
        devices = set()
        for param in params:
            state = self.state.get(param, {})
            for key, value in state.items():
                if not isinstance(value, torch.Tensor):
                    continue
                if value.device.type == "cpu":
                    if param.is_cuda and key != "step" and not value.is_pinned():
                        state[key] = value.pin_memory()
                    continue
                host = host_states.get(param, {}).get(key)
                if host is None or host.shape != value.shape or host.dtype != value.dtype:
                    host = torch.empty_like(value, device="cpu", pin_memory=param.is_cuda)
                host.copy_(value, non_blocking=True)
                state[key] = host
                devices.add(value.device)
        # Besides making CPU checkpoints immediately safe, this bounds live
        # CUDA allocations even when the CPU can enqueue layers faster than DMA.
        for device in devices:
            torch.cuda.current_stream(device).synchronize()

    @torch.no_grad()
    def offload_state(self):
        self._offload_layer([p for group in self.param_groups for p in group["params"]], {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Process contiguous runs instead of regrouping parameters: this also
        # preserves stochastic BF16 rounding's RNG order across all layers.
        try:
            for group in self.param_groups:
                for layer, run in groupby(group["params"], key=lambda p: self._owners[id(p)]):
                    params = [p for p in run if p.grad is not None]
                    if not params:
                        continue
                    host_states = {}
                    try:
                        with self._range(f"io::h2d::optimizer_layer::{layer}", params):
                            host_states = self._load_layer(params)
                        self.optimizer.param_groups = [dict(group, params=params)]
                        with self._range(f"compute::optimizer_layer::{layer}", params):
                            self.optimizer.step()
                    finally:
                        with self._range(f"io::d2h::optimizer_layer::{layer}", params):
                            self._offload_layer(params, host_states)
        finally:
            self.optimizer.param_groups = self.param_groups
        return loss

    def load_state_dict(self, state_dict):
        # Use the native loader's validation/backward compatibility, but bind it
        # to zero-sized CPU placeholders. Loading against real CUDA parameters
        # would materialize *all* moments on GPU before we could offload them.
        # FP32 placeholders also avoid rounding BF16StochasticAdamW's moments to
        # BF16 in torch.optim.Optimizer.load_state_dict.
        state_dict = state_dict.copy()
        for hook in self._optimizer_load_state_dict_pre_hooks.values():
            result = hook(self, state_dict)
            if result is not None:
                state_dict = result
        state_dict["param_groups"] = [dict(group) for group in state_dict["param_groups"]]
        self._validate_groups(state_dict["param_groups"])
        original_groups, original_state = self.param_groups, self.state
        cpu_groups = []
        param_map = {}
        for group in original_groups:
            placeholders = []
            for param in group["params"]:
                dtype = torch.float32 if isinstance(self.optimizer, BF16StochasticAdamW) else param.dtype
                placeholder = torch.empty(0, dtype=dtype, device="cpu")
                placeholders.append(placeholder)
                param_map[placeholder] = param
            cpu_groups.append(dict(group, params=placeholders))
        self.optimizer.param_groups = cpu_groups
        try:
            self.optimizer.load_state_dict(state_dict)
            restored_state = defaultdict(dict)
            for param, state in self.optimizer.state.items():
                # Clone so subsequent steps cannot mutate the caller's checkpoint.
                restored_state[param_map.get(param, param)] = {
                    key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                    for key, value in state.items()
                }
            for group, original in zip(self.optimizer.param_groups, original_groups, strict=True):
                group["params"] = original["params"]
            self.param_groups = self.optimizer.param_groups
            self.state = self.optimizer.state = restored_state
            self.offload_state()
        except Exception:
            self.param_groups = self.optimizer.param_groups = original_groups
            self.state = self.optimizer.state = original_state
            raise
        for hook in self._optimizer_load_state_dict_post_hooks.values():
            hook(self)
