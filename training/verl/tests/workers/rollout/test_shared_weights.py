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

import pytest
import torch

from verl.workers.rollout.shared_weights import bind_shared_weights, validate_shared_weights


class _FSDPWrappedModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self._fsdp_wrapped_module = module

    def forward(self, inputs):
        return self._fsdp_wrapped_module(inputs)


def test_bind_shared_weights_tracks_actor_optimizer_step():
    actor = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    rollout = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))

    bindings = bind_shared_weights(
        rollout,
        dict(actor.named_parameters(remove_duplicate=False)),
    )
    before = rollout[0].weight.detach().clone()

    optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
    actor(torch.ones(2, 4)).sum().backward()
    optimizer.step()

    validate_shared_weights(bindings)
    assert actor[0].weight.data_ptr() == rollout[0].weight.data_ptr()
    assert not torch.equal(before, rollout[0].weight)


def test_bind_shared_weights_ignores_nested_fsdp_wrapper_names():
    actor = torch.nn.Sequential(_FSDPWrappedModule(torch.nn.Linear(4, 3)))
    rollout = torch.nn.Sequential(torch.nn.Linear(4, 3))

    bindings = bind_shared_weights(
        rollout,
        dict(actor.named_parameters(remove_duplicate=False)),
    )

    assert actor[0]._fsdp_wrapped_module.weight.data_ptr() == rollout[0].weight.data_ptr()
    validate_shared_weights(bindings)


def test_bind_shared_weights_rejects_different_parameter_layout():
    actor = torch.nn.Linear(4, 3)
    rollout = torch.nn.Linear(4, 2)

    with pytest.raises(RuntimeError, match="shape"):
        bind_shared_weights(
            rollout,
            dict(actor.named_parameters(remove_duplicate=False)),
        )
