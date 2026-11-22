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
"""CPU data parallel (single node): rank-local FSDP plus one shared host copy.

Enabled by ``actor_rollout_ref.rollout.cpu_dp=true``. The driver exports
``OPENMOPD_CPU_DP=1`` and a shared-memory namespace to every worker, so colocated
teacher workers (which do not see the rollout config) follow the same mode.

Every model is wrapped by FSDP1 over a process group containing only its own rank:
FSDP sees world_size 1 (NO_SHARD) and never communicates across GPUs, so all
single-GPU slot/overlap machinery applies unchanged per rank. The only cross-rank
paths are the CPU gradient reduction (verl.utils.dp_cpu_adam), the scalar loss
denominator (gloo), and the trainer's existing dispatch/collect.
"""

import hashlib
import os

import torch

ENV = "OPENMOPD_CPU_DP"
_local_group = None


def enabled():
    return os.environ.get(ENV, "0") == "1"


def local_process_group():
    """Collective on first call (every rank creates every singleton group in the same order)."""
    global _local_group
    if _local_group is None:
        import torch.distributed as dist

        rank, world = dist.get_rank(), dist.get_world_size()
        groups = [dist.new_group([r]) for r in range(world)]
        _local_group = groups[rank]
    return _local_group


def fsdp_kwargs():
    """FSDP1 constructor overrides: rank-local NO_SHARD, every rank loads its own weights."""
    from torch.distributed.fsdp import ShardingStrategy

    return {
        "process_group": local_process_group(),
        "sharding_strategy": ShardingStrategy.NO_SHARD,
        "sync_module_states": False,
    }


def bind_numa():
    """Pin this worker's CPUs to its GPU's NUMA node (OPENMOPD_CPU_DP_BIND_NUMA=0 disables)."""
    if os.environ.get("OPENMOPD_CPU_DP_BIND_NUMA", "1") != "1":
        return None
    from verl.utils import shared_host

    node = shared_host.gpu_numa_node()
    shared_host.bind_process_to_node(node)
    return node


@torch.no_grad()
def share_offloaded_fsdp_params(model, label):
    """Replace each CPU-offloaded FSDP1 flat parameter by one host copy shared by all ranks.

    Collective. Must run before FSDP lazy initialization (FSDP keeps an already
    pinned shard; registered shared pages report pinned). Rank 0 fills the copy.
    Teachers are read by every GPU, so pages are interleaved across NUMA nodes;
    OPENMOPD_CPU_DP_TEACHER_POLICY=node:<n> overrides for experiments.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from verl.utils import shared_host

    if getattr(model, "_openmopd_shared_host", False):
        return
    if model._is_root is not None:
        raise RuntimeError("share_offloaded_fsdp_params must run before FSDP lazy initialization")
    handles = []
    for index, module in enumerate(FSDP.fsdp_modules(model)):
        handle = module._handle
        if handle is None:
            continue
        flat = handle.flat_param
        if (handle.uses_sharded_strategy or not handle._offload_params or handle._use_orig_params
                or flat.device.type != "cpu" or handle.world_size != 1):
            raise ValueError("shared teacher weights require rank-local CPU-offloaded NO_SHARD, orig_params=false")
        handles.append((f"m{index}", flat))
    specs = [(key, tuple(flat.shape), flat.dtype) for key, flat in handles]

    def fill(tensors):
        for key, flat in handles:
            tensors[key].copy_(flat)

    name = "teacher_" + hashlib.sha1(label.encode()).hexdigest()[:16]
    policy = os.environ.get("OPENMOPD_CPU_DP_TEACHER_POLICY", "interleave")
    shared = shared_host.shared_arena(name, specs, creator=0, policy=policy, fill=fill)
    for key, flat in handles:
        flat.data = shared[key]
        flat._openmopd_shared_cpu = True
    model._openmopd_shared_host = True
    # The module's parameter views still alias the private CPU flat parameters and would
    # keep ~1 model of private host memory per rank alive until FSDP lazy init. Re-point
    # them at the shared copy now so the private copies are freed immediately.
    from verl.utils.fsdp_utils import release_fsdp_cpu_offloaded_param_views

    release_fsdp_cpu_offloaded_param_views(model)


@torch.no_grad()
def check_replicas(gpu_weights, step):
    """Verify every rank's GPU student is identical: exact int64 sums of the BF16 bit patterns, over gloo.

    Two independent checksums per flat parameter (plain and position-weighted over 64 strides)
    without materializing a widened copy of the weights.
    """
    import torch.distributed as dist

    rows = []
    for _, weight in sorted(gpu_weights.items()):
        bits = weight.view(torch.int16)
        rows.append(bits.sum(dtype=torch.int64))
        usable = bits.numel() // 64 * 64
        rows.append((bits[:usable].view(-1, 64).sum(dim=0, dtype=torch.int64)
                     * torch.arange(1, 65, device=bits.device)).sum())
    local = torch.stack(rows).cpu()
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    if not all(torch.equal(gathered[0], other) for other in gathered[1:]):
        raise RuntimeError(f"CPU DP replicas diverged at update {step}")
    return int(local.sum())
