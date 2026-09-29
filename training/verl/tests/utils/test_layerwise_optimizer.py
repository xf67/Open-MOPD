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

import copy
import gc
from functools import partial

import pytest
import torch
from torch import nn

from verl.utils.bf16_optimizer import BF16StochasticAdamW
from verl.utils.layerwise_optimizer import LayerwiseOffloadOptimizer

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


class Block(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)

    def forward(self, x):
        return x + self.second(self.first(x).tanh())


class Student(nn.Module):
    _no_split_modules = ["Block"]

    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(8, 8)
        self.layers = nn.Sequential(Block(), Block())
        self.head = nn.Linear(8, 8)
        self.head.weight = self.embedding.weight
        self.unused = nn.Parameter(torch.ones(3))

    def forward(self, x):
        return self.head(self.layers(self.embedding(x)))


def make_optimizer(model, bf16, layerwise):
    params = list(model.parameters())
    groups = [{"params": params[::2], "lr": 2e-3}, {"params": params[1::2], "lr": 1e-3}]
    cls = BF16StochasticAdamW if bf16 else torch.optim.AdamW
    kwargs = {} if bf16 else {"amsgrad": True}
    optimizer = cls(groups, weight_decay=0.03, foreach=False, **kwargs)
    return LayerwiseOffloadOptimizer(optimizer, model) if layerwise else optimizer


def assert_cpu_states(optimizer):
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"


def assert_equal(left, right):
    for left_group, right_group in zip(left.param_groups, right.param_groups, strict=True):
        assert left_group["lr"] == right_group["lr"]
        for a, b in zip(left_group["params"], right_group["params"], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert left.state.get(a, {}).keys() == right.state.get(b, {}).keys()
            for key, value in left.state.get(a, {}).items():
                other = right.state[b][key]
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value.cpu(), other.cpu(), rtol=0, atol=0)
                else:
                    assert value == other


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bf16", [False, True])
def test_accumulation_clipping_scheduler_and_checkpoint(device, bf16):
    torch.manual_seed(123)
    dtype = torch.bfloat16 if bf16 else torch.float32
    reference = Student().to(device=device, dtype=dtype)
    candidate = copy.deepcopy(reference)
    base = make_optimizer(reference, bf16, False)
    wrapped = make_optimizer(candidate, bf16, True)
    schedulers = [torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9) for opt in (base, wrapped)]
    pointers = [param.data_ptr() for param in candidate.parameters()]

    for step in range(4):
        base.zero_grad()
        wrapped.zero_grad()
        # Two microbatches with a single globally clipped update.
        for _ in range(2):
            x = torch.randn(3, 8, device=device, dtype=dtype)
            for model in (reference, candidate):
                model(x).float().square().mean().backward()
        assert_cpu_states(wrapped)
        norms = [torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) for model in (reference, candidate)]
        torch.testing.assert_close(*norms, rtol=0, atol=0)
        for opt in (base, wrapped):
            torch.manual_seed(100 + step)
            opt.step()
        for scheduler in schedulers:
            scheduler.step()
        assert_equal(base, wrapped)
        assert_cpu_states(wrapped)
        assert candidate.unused not in wrapped.state
        assert [param.data_ptr() for param in candidate.parameters()] == pointers

        if step == 1:
            saved = copy.deepcopy(wrapped.state_dict())
            # Resume directly on CPU, without rounding FP32 moments to BF16.
            wrapped.load_state_dict(saved)
            assert_equal(base, wrapped)
            assert_cpu_states(wrapped)
            if bf16:
                assert all(state["exp_avg"].dtype == torch.float32 for state in wrapped.state.values())


@pytest.mark.parametrize("device", DEVICES)
def test_only_current_layer_resident_and_host_buffers_reused(device):
    model = nn.Sequential(Block(), Block()).to(device)
    base = torch.optim.AdamW(model.parameters(), foreach=False)
    wrapped = LayerwiseOffloadOptimizer(base, model)
    calls = []

    def before_step(opt, args, kwargs):
        active = {id(p) for g in opt.param_groups for p in g["params"]}
        for param, state in wrapped.state.items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    assert value.device.type == (device if id(param) in active else "cpu")
        calls.append(active)

    base.register_step_pre_hook(before_step)
    previous_pointers = None
    for step in range(3):
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        wrapped.step()
        assert_cpu_states(wrapped)
        pointers = [state["exp_avg"].data_ptr() for state in wrapped.state.values()]
        if step:
            assert pointers == previous_pointers
        previous_pointers = pointers
    assert len(calls) == 12  # Four leaf Linear modules per update.
    if device == "cuda":
        assert all(state["exp_avg"].is_pinned() for state in wrapped.state.values())


def test_closure_runs_once_and_failure_restores_groups(monkeypatch):
    model = Student()
    base = torch.optim.AdamW(model.parameters(), foreach=False)
    wrapped = LayerwiseOffloadOptimizer(base, model)
    calls = []

    def closure():
        calls.append(1)
        loss = model(torch.ones(1, 8)).sum()
        loss.backward()
        return loss

    assert wrapped.step(closure).ndim == 0
    assert calls == [1]
    groups = wrapped.param_groups

    def fail():
        raise RuntimeError("injected step failure")

    monkeypatch.setattr(base, "step", fail)
    with pytest.raises(RuntimeError, match="injected step failure"):
        wrapped.step()
    assert base.param_groups is groups
    assert_cpu_states(wrapped)


@pytest.mark.parametrize("flag", ["foreach", "fused", "capturable", "differentiable"])
def test_reject_incompatible_execution_modes(flag):
    model = nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), **{flag: True})
    with pytest.raises(ValueError, match=flag):
        LayerwiseOffloadOptimizer(opt, model)


def test_native_checkpoint_and_legacy_offload_helpers():
    from verl.utils.fsdp_utils import load_fsdp_optimizer, offload_fsdp_optimizer

    model = Student()
    native = make_optimizer(model, False, False)
    model(torch.ones(1, 8)).sum().backward()
    native.step()
    wrapped = make_optimizer(model, False, True)
    wrapped.load_state_dict(native.state_dict())
    assert_equal(native, wrapped)
    # The old worker's eager load must be a no-op even with a CUDA target.
    load_fsdp_optimizer(wrapped, "cuda")
    offload_fsdp_optimizer(wrapped)
    assert_cpu_states(wrapped)
    native.load_state_dict(wrapped.state_dict())
    assert_equal(native, wrapped)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("use_orig_params", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_fsdp_flat_and_original_parameters(tmp_path, use_orig_params, bf16):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/init", rank=0, world_size=1)
    try:
        dtype = torch.bfloat16 if bf16 else torch.float32
        model = Student().cuda().to(dtype)
        other = copy.deepcopy(model)
        kwargs = {
            "auto_wrap_policy": partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}),
            "device_id": torch.cuda.current_device(),
            "use_orig_params": use_orig_params,
        }
        reference, candidate = FSDP(model, **kwargs), FSDP(other, **kwargs)
        base = make_optimizer(reference, bf16, False)
        wrapped = make_optimizer(candidate, bf16, True)
        for step in range(3):
            base.zero_grad()
            wrapped.zero_grad()
            for _ in range(2):
                x = torch.randn(3, 8, device="cuda", dtype=dtype)
                reference(x).float().square().mean().backward()
                candidate(x).float().square().mean().backward()
            torch.testing.assert_close(reference.clip_grad_norm_(0.1), candidate.clip_grad_norm_(0.1))
            for opt in (base, wrapped):
                torch.manual_seed(100 + step)
                opt.step()
            assert_equal(base, wrapped)
            assert_cpu_states(wrapped)
            if step == 0:
                wrapped.load_state_dict(copy.deepcopy(wrapped.state_dict()))
                assert_equal(base, wrapped)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bf16", [False, True])
def test_cuda_peak_memory(bf16):
    def measure(layerwise):
        gc.collect()
        torch.cuda.empty_cache()
        dtype = torch.bfloat16 if bf16 else torch.float32
        model = nn.Sequential(*(nn.Linear(1024, 1024, bias=False) for _ in range(8))).cuda().to(dtype)
        cls = BF16StochasticAdamW if bf16 else torch.optim.AdamW
        opt = cls(model.parameters(), foreach=False)
        if layerwise:
            opt = LayerwiseOffloadOptimizer(opt, model)
        peaks = []
        backward_peaks = []
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            # Baseline eagerly reloads every state before student backward.
            if not layerwise:
                for param, state in opt.state.items():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor) and key != "step":
                            state[key] = value.to(param.device)
            model(torch.ones(2, 1024, device="cuda", dtype=dtype)).float().square().mean().backward()
            backward_peaks.append(torch.cuda.max_memory_allocated())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated())
            if not layerwise:
                for state in opt.state.values():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.cpu()
                del state, value
        return peaks, backward_peaks

    baseline, baseline_backward = measure(False)
    layerwise, layerwise_backward = measure(True)
    print(f"\nbf16={bf16}: total peaks {baseline} -> {layerwise}; backward {baseline_backward} -> {layerwise_backward}")
    assert all(new < old * 0.8 for new, old in zip(layerwise, baseline, strict=True))
    assert layerwise_backward[1] < baseline_backward[1] * 0.7
