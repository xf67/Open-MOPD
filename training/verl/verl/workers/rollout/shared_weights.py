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

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SharedWeightBinding:
    name: str
    actor: torch.nn.Parameter
    rollout: torch.nn.Parameter


def _without_fsdp_wrappers(name: str) -> str:
    return ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module")


def bind_shared_weights(
    rollout_model: torch.nn.Module,
    actor_parameters: dict[str, torch.nn.Parameter],
) -> list[SharedWeightBinding]:
    """Make rollout parameters alias the actor's CUDA storage.

    This intentionally supports only a one-to-one parameter layout.  Failing
    closed here is important: a copied or partially shared model would silently
    become stale after the first optimizer step.
    """
    rollout_parameters = dict(rollout_model.named_parameters(remove_duplicate=False))
    normalized_actor_parameters = {}
    for actor_name, actor_param in actor_parameters.items():
        normalized_name = _without_fsdp_wrappers(actor_name)
        if normalized_name in normalized_actor_parameters:
            raise RuntimeError(f"Duplicate actor parameter after removing FSDP wrappers: {normalized_name}")
        normalized_actor_parameters[normalized_name] = actor_param

    missing_in_actor = sorted(set(rollout_parameters) - set(normalized_actor_parameters))
    missing_in_rollout = sorted(set(normalized_actor_parameters) - set(rollout_parameters))
    if missing_in_actor or missing_in_rollout:
        raise RuntimeError(
            "Shared actor/vLLM parameter names do not match: "
            f"missing_in_actor={missing_in_actor[:8]}, missing_in_rollout={missing_in_rollout[:8]}"
        )

    bindings = []
    with torch.no_grad():
        for name, rollout_param in rollout_parameters.items():
            actor_param = normalized_actor_parameters[name]
            if actor_param.shape != rollout_param.shape:
                raise RuntimeError(
                    f"Cannot share {name}: actor shape {tuple(actor_param.shape)} != "
                    f"vLLM shape {tuple(rollout_param.shape)}"
                )
            if actor_param.dtype != rollout_param.dtype:
                raise RuntimeError(
                    f"Cannot share {name}: actor dtype {actor_param.dtype} != vLLM dtype {rollout_param.dtype}"
                )
            if actor_param.device != rollout_param.device:
                raise RuntimeError(
                    f"Cannot share {name}: actor device {actor_param.device} != vLLM device {rollout_param.device}"
                )
            if not actor_param.is_contiguous() or not rollout_param.is_contiguous():
                raise RuntimeError(f"Cannot share non-contiguous parameter {name}")

            rollout_param.data = actor_param.detach()
            bindings.append(SharedWeightBinding(name=name, actor=actor_param, rollout=rollout_param))

    validate_shared_weights(bindings)
    return bindings


def validate_shared_weights(bindings: list[SharedWeightBinding]) -> None:
    """Verify that every binding still references the exact same storage."""
    for binding in bindings:
        actor = binding.actor
        rollout = binding.rollout
        if actor.shape != rollout.shape or actor.dtype != rollout.dtype or actor.device != rollout.device:
            raise RuntimeError(f"Shared weight metadata changed for {binding.name}")
        if actor.data_ptr() != rollout.data_ptr() or actor.storage_offset() != rollout.storage_offset():
            raise RuntimeError(
                f"Shared weight storage changed for {binding.name}: "
                f"actor_ptr={actor.data_ptr()}, rollout_ptr={rollout.data_ptr()}"
            )
