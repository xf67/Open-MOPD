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

from collections import Counter
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedLinearMapping:
    actor_modules: tuple[str, ...]
    output_sizes: tuple[int, ...]


@dataclass(frozen=True)
class SharedWeightBinding:
    name: str
    actor: torch.nn.Parameter
    rollout: torch.nn.Parameter
    # Keep the original storage alive: graph capture must never see rebinding.
    bound_tensor: torch.Tensor
    row_offset: int | None = None
    rollout_shape: tuple[int, ...] | None = None


def get_vllm_packed_linear_mapping(rollout_model: torch.nn.Module) -> dict[str, PackedLinearMapping]:
    """Read instance-specific fusion metadata from the vLLM Transformers backend.

    Do not infer fusion from suffixes: a model can contain fused and unfused
    modules with the same short name. Only dense, single-rank QKV/GLU layouts
    are supported; unexpected layouts fail before any parameter is rebound.
    """
    mapper = getattr(rollout_model, "hf_to_vllm_mapper", None)
    grouped = {}
    for source, (target, shard) in getattr(mapper, "orig_to_new_stacked", {}).items():
        shards = grouped.setdefault(target, {})
        if shard in shards:
            raise RuntimeError(f"Duplicate fused shard {shard!r} for {target}")
        shards[shard] = source

    result = {}
    for target, shards in grouped.items():
        if set(shards) == {"q", "k", "v"}:
            order = ("q", "k", "v")
        elif set(shards) == {0, 1}:
            order = (0, 1)
        else:
            raise RuntimeError(f"Unsupported shared fusion shards for {target}: {tuple(shards)}")
        module = rollout_model.get_submodule(target)
        if getattr(module, "tp_size", None) != 1 or getattr(module, "quant_config", None) is not None:
            raise RuntimeError(f"Shared fused weights require TP=1 and no quantization: {target}")
        sizes = tuple(module.output_sizes)
        if len(sizes) != len(order) or any(size <= 0 for size in sizes):
            raise RuntimeError(f"Invalid fused output sizes for {target}: {sizes}")
        result[target] = PackedLinearMapping(tuple(shards[shard] for shard in order), sizes)
    return result


def _without_fsdp_wrappers(name: str) -> str:
    return ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module")


def _storage_key(tensor: torch.Tensor) -> tuple[torch.device, int]:
    return tensor.device, tensor.untyped_storage().data_ptr()


def _contiguous_packed_view(params: tuple[torch.Tensor, ...], target: torch.Tensor) -> torch.Tensor | None:
    first = params[0]
    offset = first.storage_offset()
    for param in params:
        if _storage_key(param) != _storage_key(first) or param.storage_offset() != offset:
            return None
        offset += param.numel()
    return first.detach().as_strided(target.shape, target.stride(), first.storage_offset())


def bind_shared_weights(
    rollout_model: torch.nn.Module,
    actor_parameters: dict[str, torch.nn.Parameter],
    *,
    required_dtype: torch.dtype | None = None,
    packed_modules: dict[str, PackedLinearMapping] | None = None,
    allow_actor_rebind: bool = False,
) -> list[SharedWeightBinding]:
    """Share direct parameters and row-stacked QKV/GLU parameters before capture.

    Existing adjacent actor views are reused without moving either the actor
    storage or its Parameter objects. This also preserves external flat-buffer
    owners, such as the single-rank CPU optimizer pipeline.

    For ordinary independent parameters, allow_actor_rebind permits a one-time
    copy into the already allocated fused rollout weight, then points the same
    actor Parameter objects at its slices. Later optimizer/checkpoint updates
    are in-place and require no packing or actor-to-rollout synchronization.
    Call only before backward/graph capture, with no concurrent model users.
    """
    rollout_parameters = dict(rollout_model.named_parameters(remove_duplicate=False))
    actor = {}
    for actor_name, param in actor_parameters.items():
        name = _without_fsdp_wrappers(actor_name)
        if name in actor:
            raise RuntimeError(f"Duplicate actor parameter after removing FSDP wrappers: {name}")
        actor[name] = param
    packed_modules = packed_modules or {}
    aliases = Counter(id(param) for param in actor.values())
    storage_uses = Counter(_storage_key(param) for param in actor.values())
    consumed = set()
    plans = []

    # Validate the complete mapping before mutating even a direct parameter.
    for name, rollout_param in rollout_parameters.items():
        module_name, _, suffix = name.rpartition(".")
        mapping = packed_modules.get(module_name)
        if mapping is None:
            sources = (name,)
        else:
            if suffix not in {"weight", "bias"}:
                raise RuntimeError(f"Unsupported shared fused parameter: {name}")
            sources = tuple(f"{module}.{suffix}" for module in mapping.actor_modules)
            if len(sources) < 2 or len(sources) != len(mapping.output_sizes) or len(set(sources)) != len(sources):
                raise RuntimeError(f"Invalid shared fusion mapping for {name}")
        missing = set(sources) - actor.keys()
        repeated = consumed.intersection(sources)
        if missing or repeated:
            raise RuntimeError(
                f"Shared actor/vLLM parameter names do not match for {name}: "
                f"missing_in_actor={sorted(missing)}, repeated={sorted(repeated)}"
            )
        consumed.update(sources)
        params = tuple(actor[source] for source in sources)
        if rollout_param.device.type == "meta" or rollout_param.numel() == 0 or not rollout_param.is_contiguous():
            raise RuntimeError(f"Cannot share unmaterialized, empty or non-contiguous rollout parameter {name}")
        for index, (source, param) in enumerate(zip(sources, params, strict=True)):
            shape = tuple(rollout_param.shape)
            if mapping is not None:
                expected_ndim = 2 if suffix == "weight" else 1
                if rollout_param.ndim != expected_ndim or rollout_param.size(0) != sum(mapping.output_sizes):
                    raise RuntimeError(f"Invalid fused parameter shape for {name}: {shape}")
                shape = (mapping.output_sizes[index], *shape[1:])
                if aliases[id(param)] != 1:
                    raise RuntimeError(f"Cannot pack tied actor parameter {source}")
            if tuple(param.shape) != shape:
                raise RuntimeError(f"Cannot share {source}: actor shape {tuple(param.shape)} != vLLM shape {shape}")
            if param.dtype != rollout_param.dtype or (required_dtype is not None and param.dtype != required_dtype):
                raise RuntimeError(
                    f"Cannot share {source}: actor dtype {param.dtype}, vLLM dtype {rollout_param.dtype}, "
                    f"required dtype {required_dtype}"
                )
            if param.device != rollout_param.device:
                raise RuntimeError(
                    f"Cannot share {source}: actor device {param.device} != vLLM {rollout_param.device}"
                )
            if not param.is_contiguous():
                raise RuntimeError(f"Cannot share non-contiguous parameter {source}")
        packed = None if mapping is None else _contiguous_packed_view(params, rollout_param)
        if mapping is not None and packed is None:
            if not allow_actor_rebind:
                raise RuntimeError(
                    f"Fused shared weights require adjacent actor storage for {name}; rebinding disabled"
                )
            for source, param in zip(sources, params, strict=True):
                # A larger storage may have an external owner. Never detach its
                # parameter views from a pipeline/FSDP buffer behind its back.
                if (
                    param.storage_offset() != 0
                    or param.untyped_storage().nbytes() != param.nbytes
                    or storage_uses[_storage_key(param)] != 1
                ):
                    raise RuntimeError(f"Cannot repack actor storage view {source}; prepare its layout before binding")
                if param.grad is not None:
                    raise RuntimeError(f"Cannot repack {source} after gradients have been created")
            if _storage_key(rollout_param) in {_storage_key(param) for param in actor.values()}:
                raise RuntimeError(f"Cannot repack overlapping actor/rollout storage for {name}")
        plans.append((name, rollout_param, sources, params, mapping, packed))
    unused = actor.keys() - consumed
    unused_mappings = packed_modules.keys() - {name.rpartition(".")[0] for name in rollout_parameters}
    if unused or unused_mappings:
        raise RuntimeError(
            f"Shared actor/vLLM parameter names do not match: missing_in_rollout={sorted(unused)[:8]}, "
            f"unused_fusions={sorted(unused_mappings)[:8]}"
        )

    bindings = []
    with torch.no_grad():
        for name, rollout_param, sources, params, mapping, packed in plans:
            if mapping is None:
                rollout_param.data = params[0].detach()
                bindings.append(SharedWeightBinding(name, params[0], rollout_param, params[0].detach()))
                continue
            if packed is not None:
                rollout_param.data = packed
            offset = 0
            for source, param in zip(sources, params, strict=True):
                view = rollout_param.detach().narrow(0, offset, param.size(0))
                if packed is None:
                    view.copy_(param)
                    param.data = view
                bindings.append(
                    SharedWeightBinding(source, param, rollout_param, view, offset, tuple(rollout_param.shape))
                )
                offset += param.size(0)

    validate_shared_weights(bindings)
    return bindings


def validate_shared_weights(bindings: list[SharedWeightBinding]) -> None:
    """Validate live fused slices, including their offsets and captured storage."""
    for binding in bindings:
        actor = binding.actor
        rollout = binding.rollout
        bound = binding.bound_tensor
        if binding.row_offset is not None:
            if tuple(rollout.shape) != binding.rollout_shape:
                raise RuntimeError(f"Shared fused weight shape changed for {binding.name}")
            rollout = rollout.narrow(0, binding.row_offset, bound.size(0))
        if any(
            param.shape != bound.shape
            or param.dtype != bound.dtype
            or param.device != bound.device
            or param.stride() != bound.stride()
            for param in (actor, rollout)
        ):
            raise RuntimeError(f"Shared weight metadata changed for {binding.name}")
        if any(
            param.data_ptr() != bound.data_ptr() or param.storage_offset() != bound.storage_offset()
            for param in (actor, rollout)
        ):
            raise RuntimeError(
                f"Shared weight storage changed for {binding.name}: "
                f"actor_ptr={actor.data_ptr()}, rollout_ptr={rollout.data_ptr()}, bound_ptr={bound.data_ptr()}"
            )
