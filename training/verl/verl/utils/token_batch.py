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
"""Merge complete vLLM rollout batches until valid sequence tokens exceed a floor."""

from verl import DataProto


def compact_rollout_padding(data):
    """Remove common prompt-left/response-right padding, preserving token alignment."""
    tensors = dict(data.batch.items())
    mask = tensors["attention_mask"]
    response_length = tensors["responses"].shape[-1]
    prompt_length = mask.shape[-1] - response_length
    if prompt_length <= 0 or response_length <= 0:
        raise ValueError("token batching requires nonempty prompt and response dimensions")
    prompt_columns = mask[:, :prompt_length].any(dim=0).nonzero().flatten()
    response_columns = mask[:, prompt_length:].any(dim=0).nonzero().flatten()
    if not len(prompt_columns):
        raise ValueError("token batching encountered an empty prompt batch")
    left = int(prompt_columns[0])
    response_end = int(response_columns[-1]) + 1 if len(response_columns) else 1
    for key in ("input_ids", "attention_mask", "position_ids"):
        if tensors[key].shape[-1] != mask.shape[-1]:
            raise ValueError(f"Misaligned full-sequence tensor: {key}")
        tensors[key] = tensors[key][..., left : prompt_length + response_end].contiguous()
    if "prompts" in tensors:
        tensors["prompts"] = tensors["prompts"][..., left:].contiguous()
    for key in ("responses", "response_mask", "rollout_log_probs"):
        if key in tensors:
            if tensors[key].shape[-1] != response_length:
                raise ValueError(f"Misaligned response tensor: {key}")
            tensors[key] = tensors[key][..., :response_end].contiguous()
    meta_info = dict(data.meta_info)
    # Clip-rate metrics must use the original rollout limits, not compact widths.
    meta_info["token_batch_original_prompt_length"] = prompt_length
    meta_info["token_batch_original_response_length"] = response_length
    return DataProto.from_dict(tensors, dict(data.non_tensor_batch), meta_info)


class RolloutTokenAccumulator:
    """Keep whole rollout groups and their metadata, with no sample reuse/splitting.

    All groups must use one published policy version and the same fixed padding
    layout. The trainer completes collection before scoring, optimization or a
    checkpoint, so there is no partially accumulated batch to checkpoint.
    """

    def __init__(self, min_tokens, max_batch_size):
        for name, value in (("min_tokens", min_tokens), ("max_batch_size", max_batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"token_batching.{name} must be a positive integer")
        self.min_tokens = min_tokens
        self.max_batch_size = max_batch_size
        self.parts = []
        self.tokens = 0
        self.sequences = 0
        self.prompt_tokens = 0
        self.policy_version = None

    @property
    def ready(self):
        return self.tokens > self.min_tokens

    def add(self, batch):
        if self.ready:
            raise ValueError("Token batch is already ready; consume it before adding more rollouts")
        mask = batch.batch["attention_mask"]
        if mask.ndim != 2 or not ((mask == 0) | (mask == 1)).all().item():
            raise ValueError("token batching requires a binary two-dimensional attention_mask")
        if len(batch) == 0 or not (mask.sum(-1) > 0).all().item():
            raise ValueError("token batching encountered an empty rollout sequence")
        version = batch.meta_info.get("rollout_policy_version")
        if version is None:
            raise ValueError("token batching requires rollout_policy_version on every vLLM batch")
        if self.parts and version != self.policy_version:
            raise ValueError("Cannot merge rollouts from different policy versions")
        if self.sequences + len(batch) > self.max_batch_size:
            raise ValueError(f"Merging rollout would exceed token_batching.max_batch_size={self.max_batch_size}")
        count = int(mask.sum().item())
        response_length = batch.batch["responses"].shape[-1]
        self.parts.append(batch)
        self.policy_version = version
        self.tokens += count
        self.sequences += len(batch)
        self.prompt_tokens += int(mask[:, : mask.shape[-1] - response_length].sum().item())
        if not self.ready and self.sequences == self.max_batch_size:
            raise ValueError(
                f"token_batching.max_batch_size={self.max_batch_size} reached with {self.tokens} valid "
                f"prompt+response tokens; need strictly more than {self.min_tokens}"
            )

    def finish(self):
        if not self.ready:
            raise ValueError(f"Only {self.tokens} rollout tokens; need strictly more than {self.min_tokens}")
        batch = compact_rollout_padding(DataProto.concat(self.parts))
        if int(batch.batch["attention_mask"].sum().item()) != self.tokens:
            raise ValueError("Rollout merge changed the effective token count")
        batch.meta_info["pipeline_token_batch"] = True
        metrics = {
            "token_batch/rollout_batches": len(self.parts),
            "token_batch/rollout_batch_size_min": min(map(len, self.parts)),
            "token_batch/rollout_batch_size_max": max(map(len, self.parts)),
            "token_batch/rollout_policy_version": self.policy_version,
            "token_batch/sequences": self.sequences,
            "token_batch/prompt_tokens": self.prompt_tokens,
            "token_batch/response_tokens": self.tokens - self.prompt_tokens,
            "token_batch/scoring_tokens": self.tokens,
            "token_batch/min_tokens": self.min_tokens,
        }
        return batch, metrics


def validate_token_batching(config):
    """Validate before workers start; rollout chunks keep their configured size."""
    from omegaconf import OmegaConf

    def select(key, default=None):
        return OmegaConf.select(config, key, default=default)

    if not select("data.token_batching.enabled", False):
        return
    for key in ("min_tokens", "max_batch_size"):
        value = select(f"data.token_batching.{key}")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"data.token_batching.{key} must be a positive integer")
    if not select("actor_rollout_ref.rollout.student_teacher_pipeline", False):
        raise ValueError("token batching currently requires student_teacher_pipeline=true")
    if select("trainer.n_gpus_per_node") != 1 or select("trainer.nnodes") != 1:
        raise ValueError("token batching currently requires a single GPU")
    if select("actor_rollout_ref.actor.ppo_epochs") != 1:
        raise ValueError("token batching requires one optimizer step per batch (ppo_epochs=1)")
    if select("algorithm.filter_groups.enable", False):
        raise ValueError("token batching cannot be combined with filter_groups")
    if select("actor_rollout_ref.rollout.mode", "sync") != "sync":
        raise ValueError("token batching currently requires synchronous rollout")
    for key in ("actor_rollout_ref.actor.use_dynamic_bsz", "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz"):
        if select(key, False):
            raise ValueError(f"token batching requires {key}=false to keep the pipeline batch intact")
    steps = select("trainer.total_training_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("token batching requires explicit positive trainer.total_training_steps")
    chunk = select("data.gen_batch_size") or select("data.train_batch_size", 1)
    if chunk * select("actor_rollout_ref.rollout.n", 1) > select("data.token_batching.max_batch_size"):
        raise ValueError("One rollout batch already exceeds token_batching.max_batch_size after rollout.n")
