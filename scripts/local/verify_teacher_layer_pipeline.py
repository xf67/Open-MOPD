#!/usr/bin/env python3
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
"""Compare real Math/Code scoring against ordinary FSDP, using fixed inputs.

Exercises joint student/Math/Code scoring against separate scoring, Code's
remove-padding path, and repeated slot handoffs. No Ray, rollout, optimizer
update or checkpoint writes. Compares the fields consumed by MT-OPD training.
Run with PYTHONPATH=training/verl and CUDA_DEVICE_MAX_CONNECTIONS=8.
"""

import argparse
import json
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM
from verify_teacher_forward_overlap import bind

from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker, RewardModelWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True)
    parser.add_argument("--math", required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--forward-overlap", choices=("alternate", "true", "false"), default="alternate")
    parser.add_argument("--profile", action="store_true", help="Capture only warmed scoring via CUDA profiler API")
    parser.add_argument("--sequence-length", type=int, default=1248)
    parser.add_argument("--response-length", type=int, default=1024)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cycles < 2 or not 0 < args.response_length < args.sequence_length - 8:
        parser.error(
            "require cycles >= 2 and 0 < response-length < sequence-length - 8"
        )
    torch.cuda.set_device(0)
    torch.manual_seed(31)
    teachers = []
    pool = None
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group(
            "nccl", init_method=Path(directory, "init").as_uri(), rank=0, world_size=1
        )
        try:

            def load(path, offload, remove_padding=False):
                print(f"Loading {path}", flush=True)
                raw = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    local_files_only=True,
                ).eval()
                apply_monkey_patch(
                    raw, use_remove_padding=remove_padding, ulysses_sp_size=1
                )
                wrap = get_fsdp_wrap_policy(
                    raw, OmegaConf.create({"wrap_policy": {"min_num_params": 0}})
                )
                return FSDP(
                    raw,
                    auto_wrap_policy=wrap,
                    device_id=0,
                    use_orig_params=False,
                    cpu_offload=CPUOffload(offload_params=offload),
                ).eval()

            student_model = load(args.student, False)
            actor = SimpleNamespace(
                actor_module=student_model,
                config=SimpleNamespace(entropy_checkpointing=False),
                use_remove_padding=False,
                use_fused_kernels=False,
                use_ulysses_sp=False,
                device_name="cuda",
            )
            bind(
                actor,
                DataParallelPPOActor,
                ("_log_prob_model_forward", "_forward_micro_batch", "compute_log_prob"),
            )
            for index, path in enumerate((args.math, args.code)):
                teacher = SimpleNamespace(
                    reward_module=load(path, True, index == 1),
                    use_remove_padding=index == 1,
                    use_fused_kernels=False,
                    use_ulysses_sp=False,
                    _do_switch_chat_template=False,
                    world_size=1,
                    ulysses_sequence_parallel_size=1,
                    ulysses_sharding_manager=nullcontext(),
                    _teacher_forward_overlap=None,
                    _teacher_handle_prefetch=None,
                    _teacher_layer_pipeline=None,
                    config=OmegaConf.create(
                        {
                            "use_dynamic_bsz": False,
                            "micro_batch_size_per_gpu": args.batch_size,
                            "model": {"path": path},
                        }
                    ),
                )
                bind(
                    teacher,
                    RewardModelWorker,
                    (
                        "_forward_model_logits",
                        "_forward_micro_batch",
                        "_compute_entropy_safe",
                        "_compute_teacher_top_k_log_probs",
                        "prepare_forward_overlap",
                        "_compute_rm_score",
                        "prepare_param_prefetch",
                        "start_param_prefetch",
                    ),
                )
                teachers.append(teacher)
            fused = dict(zip(("rm", "mt_rm_1"), teachers, strict=True))
            worker = SimpleNamespace(
                actor=actor,
                _is_actor=True,
                _is_offload_param=False,
                world_size=1,
                ulysses_sequence_parallel_size=1,
                ulysses_sharding_manager=nullcontext(),
                get_fused_worker_by_name=fused.get,
                fused_worker_dict=fused,
                config=OmegaConf.create(
                    {
                        "rollout": {
                            "teacher_forward_overlap": True,
                            "teacher_param_prefetch": False,
                            "teacher_layer_pipeline": False,
                            "teacher_param_prefetch_max_mb": 768,
                            "teacher_layer_pipeline_max_mb": 8192,
                            "log_prob_micro_batch_size_per_gpu": args.batch_size,
                            "log_prob_max_token_len_per_gpu": args.sequence_length,
                            "log_prob_use_dynamic_bsz": False,
                            "temperature": 1.0,
                            "log_prob_top_k": 256,
                        }
                    }
                ),
            )
            bind(
                worker,
                ActorRolloutRefWorker,
                (
                    "_compute_log_prob",
                    "compute_log_prob_and_teacher",
                    "prepare_teacher_layer_pipeline",
                ),
            )
            ids = torch.randint(
                0, student_model.config.vocab_size, (args.batch_size, args.sequence_length)
            )
            mask = torch.ones_like(ids)
            mask[:, :7] = 0  # Exercise actual left padding in Code's packed input path.

            def score(overlap):
                data = DataProto.from_dict(
                    {
                        "input_ids": ids.clone(),
                        "responses": ids[:, -args.response_length :].clone(),
                        "attention_mask": mask.clone(),
                        "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
                        "response_mask": torch.ones(
                            args.batch_size, args.response_length, dtype=torch.long
                        ),
                    },
                    meta_info={
                        "reward_mode": "mt_opd",
                        "log_prob_top_k": 256,
                        "top_k_strategy": "only_stu",
                    },
                )
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                if overlap:
                    output = worker.compute_log_prob_and_teacher(data)
                else:
                    student = worker._compute_log_prob(data)
                    data.union(student)
                    output = student.union(teachers[0]._compute_rm_score(data))
                    code = teachers[1]._compute_rm_score(data)
                    output.union(DataProto.from_dict(tensors={
                        "mt_teacher_1_on_student_log_probs": code.batch["teacher_on_student_log_probs"],
                    }))
                    del code
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                tensors = {
                    key: value.cpu().clone() for key, value in output.batch.items()
                }
                return tensors, elapsed

            baseline, baseline_ms = score(False)
            # Compare warmed baselines as well; cold setup/compilation is not a speedup.
            baseline_times = [score(False)[1] for _ in range(2)]
            worker.prepare_teacher_layer_pipeline()
            worker.config.rollout.teacher_layer_pipeline = True
            pool = teachers[0]._teacher_layer_pipeline
            pointers = [slot.tensor.data_ptr() for slot in pool.slots.values()]
            records = []
            warm, _ = score(True)
            for key in baseline:
                torch.testing.assert_close(warm[key], baseline[key], rtol=0, atol=0, msg=key)
            if args.profile:
                torch.cuda.profiler.start()
            for cycle in range(args.cycles):
                overlap = cycle % 2 == 0 if args.forward_overlap == "alternate" else args.forward_overlap == "true"
                previous_copies = pool.copy_counts.copy()
                previous_uses = pool.use_counts.copy()
                with torch.cuda.nvtx.range(f"openmopd::step::{cycle + 1}"):
                    actual, elapsed = score(overlap=overlap)
                assert actual.keys() == baseline.keys()
                for key in baseline:
                    torch.testing.assert_close(
                        actual[key], baseline[key], rtol=0, atol=0, msg=key
                    )
                assert [
                    slot.tensor.data_ptr() for slot in pool.slots.values()
                ] == pointers
                assert all(slot.owner == 0 for slot in pool.slots.values())
                assert [a - b for a, b in zip(pool.copy_counts, previous_copies, strict=True)] == [len(pool.slots)] * 2
                assert [a - b for a, b in zip(pool.use_counts, previous_uses, strict=True)] == [len(pool.slots)] * 2
                record = dict(
                    cycle=cycle,
                    forward_overlap=overlap,
                    elapsed_ms=elapsed,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                    teacher_parameter_H2D_MiB=pool.bytes * 2 / 2**20,
                    exact_outputs=True,
                )
                records.append(record)
                print(json.dumps(record), flush=True)
            if args.profile:
                torch.cuda.profiler.stop()
            result = dict(
                scope=__doc__,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                response_length=args.response_length,
                left_padding=7,
                checked_keys=sorted(baseline),
                baseline_cold_ms=baseline_ms,
                baseline_warm_ms=baseline_times,
                records=records,
                slots=len(pool.slots),
                pool_MiB=pool.bytes / 2**20,
                copy_counts=pool.copy_counts,
                use_counts=pool.use_counts,
                same_slot_addresses=True,
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        finally:
            for teacher in teachers:
                if teacher._teacher_forward_overlap is not None:
                    teacher._teacher_forward_overlap.close()
            if pool is not None:
                pool.close()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
