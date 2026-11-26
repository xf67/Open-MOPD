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
"""Large-batch helpers: token-budget micro-batches and per-micro-batch column compaction.

Rollout tensors use a fixed layout: a left-padded prompt of width P followed by a
right-padded response of width R (``input_ids``/``attention_mask``/``position_ids`` are
[B, P + R]; ``responses`` and every per-response-token tensor are [B, R, ...]). A
micro-batch only needs the columns that hold real tokens of at least one of its rows:
the last ``p`` prompt columns and the first ``r`` response columns. Slicing those
columns changes no value of any real token, and padded positions are masked
everywhere, so per-micro-batch compaction is exact. Outputs are re-padded to R.
"""

import torch

__all__ = [
    "valid_lengths",
    "split_by_budget",
    "compact_columns",
    "repad_responses",
]


def valid_lengths(attention_mask: torch.Tensor, response_width: int):
    """Per-row real prompt and response lengths (CPU int lists)."""
    prompt = attention_mask[:, :-response_width].sum(-1)
    response = attention_mask[:, -response_width:].sum(-1)
    return prompt.tolist(), response.tolist()


def split_by_budget(prompt_lengths, response_lengths, budget, padded):
    """Contiguous micro-batches (lists of row indices) under a token budget.

    ``padded``: cost = rows x (longest prompt + longest response) in the micro-batch,
    i.e. the positions of a compacted padded layout. Otherwise cost = sum of real
    tokens (remove-padding layout). A row that alone exceeds the budget forms its own
    micro-batch.
    """
    if budget < 1:
        raise ValueError("token budget must be positive")
    batches, current = [], []
    max_prompt = max_response = total = 0
    for row, (p, r) in enumerate(zip(prompt_lengths, response_lengths, strict=True)):
        if padded:
            cost = (len(current) + 1) * (max(max_prompt, p) + max(max_response, r))
        else:
            cost = total + p + r
        if current and cost > budget:
            batches.append(current)
            current, max_prompt, max_response, total = [], 0, 0, 0
        current.append(row)
        max_prompt, max_response, total = max(max_prompt, p), max(max_response, r), total + p + r
    if current:
        batches.append(current)
    return batches


def compact_columns(tensors: dict, response_width: int, keep_prompt: int, keep_response: int):
    """Slice a micro-batch to its last ``keep_prompt`` prompt and first ``keep_response`` response columns.

    Full-sequence tensors are recognized by width P + R and per-response tensors by width R
    (dim 1). Other tensors (per-sequence) are returned unchanged.
    """
    keep_response = max(int(keep_response), 1)
    keep_prompt = max(int(keep_prompt), 1)
    out = {}
    full_width = None
    for key in ("input_ids", "attention_mask"):
        if key in tensors:
            full_width = tensors[key].shape[1]
            break
    for key, value in tensors.items():
        if not torch.is_tensor(value) or value.dim() < 2:
            out[key] = value
            continue
        if key == "position_ids" and value.dim() == 3:  # mrope [B, 3, S]
            prompt_width = value.shape[-1] - response_width
            out[key] = value[..., prompt_width - keep_prompt: prompt_width + keep_response].contiguous()
        elif full_width is not None and value.shape[1] == full_width and key in (
            "input_ids", "attention_mask", "position_ids",
        ):
            prompt_width = full_width - response_width
            out[key] = value[:, prompt_width - keep_prompt: prompt_width + keep_response].contiguous()
        elif value.shape[1] == response_width:
            out[key] = value[:, :keep_response].contiguous()
        else:
            out[key] = value
    return out


def repad_responses(value: torch.Tensor, response_width: int, pad_value=0):
    """Right-pad dim 1 of a per-response-token tensor back to ``response_width``."""
    if value.shape[1] == response_width:
        return value
    if value.shape[1] > response_width:
        raise ValueError(f"cannot re-pad width {value.shape[1]} to {response_width}")
    shape = list(value.shape)
    shape[1] = response_width - value.shape[1]
    pad = torch.full(shape, pad_value, dtype=value.dtype, device=value.device)
    return torch.cat([value, pad], dim=1)
