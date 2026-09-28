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

import gc
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils.fsdp_utils import release_fsdp_cpu_offloaded_param_views
from verl.workers.fsdp_workers import ActorRolloutRefWorker, RewardModelWorker
from verl.workers.teacher_forward_overlap import TeacherForwardOverlap
from verl.workers.teacher_layer_pipeline import TeacherLayerPipeline


@pytest.fixture
def gpu(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    dist.init_process_group("nccl", init_method=(tmp_path / "init").as_uri(), rank=0, world_size=1)
    try:
        yield
    finally:
        # FSDP hook cycles from failed-forward cases must not be collected in a
        # subsequent test's before/after live-allocation measurement.
        gc.collect()
        torch.cuda.synchronize()
        dist.destroy_process_group()


class _Layer(torch.nn.Linear):
    def forward(self, x):
        # Keep GPU reads pending after Python returns, exposing an early-overwrite bug.
        torch.cuda._sleep(200_000)
        self.last_pointer = self.weight.untyped_storage().data_ptr()
        return super().forward(x).tanh()


class _Teacher(torch.nn.Module):
    def __init__(self, width, **fsdp_kwargs):
        super().__init__()
        self.embed = torch.nn.Embedding(32, width)
        self.layers = torch.nn.ModuleList([FSDP(_Layer(width, width), **fsdp_kwargs) for _ in range(3)])
        self.head = torch.nn.Linear(width, 32, bias=False)
        self.head.weight = self.embed.weight
        self.fail = False
        self.skip = False

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.layers[:-1] if self.skip else self.layers:
            x = layer(x)
            if self.fail:
                raise RuntimeError("intentional forward failure")
        return self.head(x)


def _model(width=16, seed=0):
    torch.manual_seed(seed)
    kwargs = dict(cpu_offload=CPUOffload(offload_params=True), device_id=torch.cuda.current_device())
    return FSDP(_Teacher(width, **kwargs), **kwargs).eval()


@pytest.mark.parametrize("count", [2, 3])
def test_shared_slots_match_serial_and_handoff_before_root_exit(gpu, count):
    models = [_model(seed=index) for index in range(count)]
    ids = torch.arange(16, device="cuda").reshape(2, 8)
    with torch.no_grad():
        reference = [model(ids).cpu() for model in models]
        for model in models:
            release_fsdp_cpu_offloaded_param_views(model)
    assert not torch.equal(reference[0], reference[1])
    pool = TeacherLayerPipeline(models, [f"teacher{i}" for i in range(count)], max_mb=1)
    runner = TeacherForwardOverlap(torch.cuda.current_device())
    pointers = {name: slot.tensor.data_ptr() for name, slot in pool.slots.items()}
    assert pool.bytes == sum(m._handle.flat_param._local_shard.nbytes for m in pool.modules[0].values())
    observed = []

    def before_head(module, args):
        # Every decoder slot already handed off, while root/tied head is still in use.
        index = pool.active[0]
        assert pool.slots[""].owner == index and pool.slots[""].in_use
        for name, slot in pool.slots.items():
            if name:
                assert slot.owner == (index + 1) % count and not slot.in_use
        observed.append(index)

    hooks = [model.module.head.register_forward_pre_hook(before_head) for model in models]
    try:
        for cycle in range(4):
            pool.prefetch_first()
            for index, model in enumerate(models):
                if cycle % 2 == 0 and index == 0:
                    ready = torch.cuda.Event()
                    ready.record()
                    ids.record_stream(runner.stream)
                    runner.start(lambda m=model: m(ids), ready)
                    actual = runner.finish()
                else:
                    with torch.no_grad():
                        actual = model(ids)
                torch.testing.assert_close(actual.cpu(), reference[index], rtol=0, atol=0)
                assert model.module.embed.weight is model.module.head.weight
                for name, module in pool.modules[index].items():
                    handle = module._handle
                    assert handle.flat_param.device.type == "cpu"
                    for param_name, owner, _ in handle.flat_param._param_infos:
                        assert getattr(owner, param_name).device.type == "cpu"
                    if name:
                        assert module.module.last_pointer == pointers[name]
                assert {name: slot.tensor.data_ptr() for name, slot in pool.slots.items()} == pointers
        assert observed == list(range(count)) * 4
        assert pool.use_counts == [4 * len(pool.slots)] * count
        assert pool.copy_counts == [5 * len(pool.slots)] + [4 * len(pool.slots)] * (count - 1)
    finally:
        runner.close()
        for hook in hooks:
            hook.remove()
        pool.close()
    # Clean teardown restores ordinary offloaded forwards.
    with torch.no_grad():
        torch.testing.assert_close(models[0](ids).cpu(), reference[0], rtol=0, atol=0)


def test_layout_and_cap_rejected_before_patching(gpu):
    first = _model()
    other = _model(width=24)
    with pytest.raises(ValueError, match="identical FSDP"):
        TeacherLayerPipeline([first, other], ["a", "b"], 1)
    assert "pre_unshard" not in first._handle.__dict__
    large = [_model(width=512, seed=i) for i in range(2)]
    with pytest.raises(ValueError, match="above the"):
        TeacherLayerPipeline(large, ["a", "b"], 1)
    assert "pre_unshard" not in large[0]._handle.__dict__
    with pytest.raises(ValueError, match="positive"):
        TeacherLayerPipeline([first, other], ["a", "b"], 0)


def test_order_grad_mode_and_failed_forward_are_guarded(gpu):
    models = [_model(seed=i) for i in range(2)]
    pool = TeacherLayerPipeline(models, ["a", "b"], 1)
    ids = torch.arange(4, device="cuda").unsqueeze(0)
    try:
        with pytest.raises(RuntimeError, match="no_grad"):
            models[0](ids)
        with torch.no_grad(), pytest.raises(RuntimeError, match="expected teacher a"):
            models[1](ids)
        assert pool.active is None and pool.expected == 0 and not pool.broken
        with torch.no_grad():
            models[0](ids)
        with pytest.raises(RuntimeError, match="cycle has not completed"):
            pool.prefetch_first()
        models[1].module.fail = True
        with torch.no_grad(), pytest.raises(RuntimeError, match="intentional forward failure"):
            models[1](ids)
        assert pool.broken and pool.active is None
        with torch.no_grad(), pytest.raises(RuntimeError, match="closed or failed"):
            models[0](ids)
    finally:
        pool.close()


def test_skipped_layers_fail_without_reusing_wrong_teacher(gpu):
    models = [_model(seed=i) for i in range(2)]
    pool = TeacherLayerPipeline(models, ["a", "b"], 1)
    try:
        models[0].module.skip = True
        with torch.no_grad(), pytest.raises(RuntimeError, match="skipped FSDP layers"):
            models[0](torch.arange(4, device="cuda").unsqueeze(0))
        assert pool.broken
    finally:
        pool.close()


@pytest.mark.parametrize("batch_size,capacity,dynamic", [(2, 1, False), (4, 2, False), (2, 4, True), (0, 4, False)])
def test_multiple_microbatches_rejected_before_scoring(batch_size, capacity, dynamic):
    worker = SimpleNamespace(
        _teacher_layer_pipeline=object(),
        config=SimpleNamespace(use_dynamic_bsz=dynamic, micro_batch_size_per_gpu=capacity),
    )
    batch = DataProto.from_dict({"input_ids": torch.zeros(batch_size, 4, dtype=torch.long)})
    with pytest.raises(ValueError, match="exactly one fixed micro-batch"):
        RewardModelWorker._compute_rm_score(worker, batch)


@pytest.mark.parametrize("student_capacity,teacher_capacity", [(1, 2), (2, 1)])
def test_joint_scoring_rejects_split_before_launch(student_capacity, teacher_capacity):
    from omegaconf import OmegaConf

    teacher = SimpleNamespace(config=SimpleNamespace(micro_batch_size_per_gpu=teacher_capacity))
    worker = SimpleNamespace(
        config=OmegaConf.create(
            {
                "rollout": {
                    "teacher_forward_overlap": True,
                    "log_prob_micro_batch_size_per_gpu": student_capacity,
                    "log_prob_use_dynamic_bsz": False,
                }
            }
        ),
        get_fused_worker_by_name=lambda name: teacher,
    )
    batch = DataProto.from_dict(
        {"input_ids": torch.zeros(2, 4, dtype=torch.long)}, meta_info={"reward_mode": "mt_opd", "log_prob_top_k": 256}
    )
    with pytest.raises(ValueError, match="exactly one fixed"):
        ActorRolloutRefWorker.compute_log_prob_and_teacher.__wrapped__(worker, batch)


def test_entropy_handles_noncontiguous_batched_response_slice():
    logits = torch.randn(4, 12, 32)[:, 3:-1, :]
    assert not logits.is_contiguous()
    result = RewardModelWorker._compute_entropy_safe(None, logits, chunk_size=7)
    log_probs = torch.log_softmax(logits, dim=-1)
    torch.testing.assert_close(result, -(log_probs.exp() * log_probs).sum(-1))
