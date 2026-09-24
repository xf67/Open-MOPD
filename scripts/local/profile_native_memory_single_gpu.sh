#!/usr/bin/env bash
# Same workload as profile_native_single_gpu.sh, using PyTorch memory snapshots.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/local/profile_native_memory_single_gpu.sh [--dry-run] [OUTPUT_DIR] [Hydra overrides...]

Writes CUDA allocation snapshots, interactive HTML timelines, and a peak report.
Defaults match profile_native_single_gpu.sh (shared BF16 student, two teachers,
validation before training, export at steps 3 and 4). No Nsight installation needed.

Environment:
  CUDA_VISIBLE_DEVICES              GPU to train on (default: 0)
  SHARE_STUDENT_WEIGHTS             true (default) or false
  BF16_STUDENT_WEIGHTS              true (default) or false for non-shared runs
  OPENMOPD_PROFILE_STEP             First step to export (default: 3)
  OPENMOPD_PROFILE_STEP_COUNT       Number of consecutive exports (default: 2)
  OPENMOPD_NUM_STEPS                Total steps (default: last export step)
  OPENMOPD_VAL_BEFORE_TRAIN          true (default) or false
  OPENMOPD_MEMORY_MAX_ENTRIES        Allocation history capacity (default: 1000000)
  OPENMOPD_MEMORY_SAMPLE_MS          nvidia-smi interval; 0 disables (default: 100)
  OPENMOPD_PYTHON                   Python executable in the mopd environment

History starts at worker initialization, not OPENMOPD_PROFILE_STEP. Snapshots
are cumulative, bounded by MAX_ENTRIES; initialization/validation can be the peak.
See scripts/local/README.md for viewing and measurement limits.
EOF
}

dry_run=false
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=true; shift; fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
run_dir="${repo_root}/output/native_memory_$(date +%Y%m%d_%H%M%S)_$$"
if (($# > 0)) && [[ "$1" != *=* ]]; then run_dir="$1"; shift; fi
run_dir="$(realpath -m "${run_dir}")"
if [[ "${run_dir}" == *"'"* || "${run_dir}" == *\\* || "${run_dir}" == *$'\n'* ]]; then
  printf 'Output path cannot contain single quotes, backslashes or newlines\n' >&2
  exit 2
fi
python_bin="${OPENMOPD_PYTHON:-/home/xxf/anaconda3/envs/mopd/bin/python}"
SHARE_STUDENT_WEIGHTS="${SHARE_STUDENT_WEIGHTS:-true}"
BF16_STUDENT_WEIGHTS="${BF16_STUDENT_WEIGHTS:-true}"
profile_step="${OPENMOPD_PROFILE_STEP:-3}"
profile_step_count="${OPENMOPD_PROFILE_STEP_COUNT:-2}"
val_before_train="${OPENMOPD_VAL_BEFORE_TRAIN:-true}"
max_entries="${OPENMOPD_MEMORY_MAX_ENTRIES:-1000000}"
sample_ms="${OPENMOPD_MEMORY_SAMPLE_MS:-100}"

for var in profile_step profile_step_count max_entries; do
  if [[ ! "${!var}" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer, got: %s\n' "${var}" "${!var}" >&2
    exit 2
  fi
done
for var in SHARE_STUDENT_WEIGHTS BF16_STUDENT_WEIGHTS val_before_train; do
  if [[ "${!var}" != true && "${!var}" != false ]]; then
    printf '%s must be true or false, got: %s\n' "${var}" "${!var}" >&2
    exit 2
  fi
done
if [[ ! "${sample_ms}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  printf 'OPENMOPD_MEMORY_SAMPLE_MS must be a non-negative integer\n' >&2
  exit 2
fi
profile_end_step=$((profile_step + profile_step_count - 1))
num_steps="${OPENMOPD_NUM_STEPS:-${profile_end_step}}"
if [[ ! "${num_steps}" =~ ^[1-9][0-9]*$ ]] || ((num_steps < profile_end_step)); then
  printf 'OPENMOPD_NUM_STEPS must be >= %s\n' "${profile_end_step}" >&2
  exit 2
fi
profile_steps=()
for ((step = profile_step; step <= profile_end_step; step++)); do profile_steps+=("${step}"); done
profile_steps_csv="$(IFS=,; printf '%s' "${profile_steps[*]}")"
student_training_args=()
if [[ "${BF16_STUDENT_WEIGHTS}" == true && "${SHARE_STUDENT_WEIGHTS}" != true ]]; then
  student_training_args+=(
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.bf16_optimizer
    actor_rollout_ref.actor.optim.optimizer=BF16StochasticAdamW
  )
fi
if [[ "${python_bin}" != */* ]]; then python_bin="$(command -v "${python_bin}" || true)"; fi
if [[ ! -x "${python_bin}" ]]; then
  printf 'Python executable not found: %s\n' "${python_bin}" >&2
  exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V1=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export PATH="$(dirname "${python_bin}"):${PATH}"
export PYTHONPATH="${repo_root}/training/verl${PYTHONPATH:+:${PYTHONPATH}}"

cmd=("${python_bin}" -m verl.trainer.main_ppo
  algorithm.adv_estimator=token_reward_direct
  data.train_files=/home/xxf/Distill/models/OPD/data/rl_prompt_mix/train.parquet
  data.val_files=/home/xxf/Distill/models/OPD/data/rl_prompt_mix/eval.parquet
  data.train_batch_size=1 data.max_prompt_length=512 data.max_response_length=1024
  data.filter_overlong_prompts=True data.truncation=error
  actor_rollout_ref.model.path=/home/xxf/Distill/models/OPD/MixSFT
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.share_weights="${SHARE_STUDENT_WEIGHTS}"
  "${student_training_args[@]}"
  +actor_rollout_ref.rollout.reward_mode=mt_opd
  actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.max_model_len=1536
  actor_rollout_ref.actor.ppo_mini_batch_size=1
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.optim.override_optimizer_config='{foreach:false}'
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  +actor_rollout_ref.rollout.log_prob_top_k=256
  "custom_reward_function.path=${repo_root}/training/verl/verl/utils/reward_score/opd_val_dispatch.py"
  custom_reward_function.name=reward_func
  reward_model.micro_batch_size_per_gpu=1 reward_model.enable=True
  reward_model.model.path=/home/xxf/Distill/models/OPD/Math
  reward_model.model.input_tokenizer=null
  +reward_model.reward_kwargs.compute_true_reward=false
  '+mt_opd.teacher_domains=[math,code]' +mt_opd.n_additional_teachers=1
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.total_epochs=1
  trainer.total_training_steps="${num_steps}" trainer.val_before_train="${val_before_train}"
  trainer.resume_mode=disable
  "trainer.default_local_dir='${run_dir}/checkpoints'"
  trainer.project_name=OpenOPD-local trainer.experiment_name=mt-opd-memory
  "trainer.logger=['console']"
  +mt_reward_model_1.enable=True
  +mt_reward_model_1.model.path=/home/xxf/Distill/models/OPD/Code
  +mt_reward_model_1.model.input_tokenizer=null
  +mt_reward_model_1.model.use_remove_padding=True
  +mt_reward_model_1.model.fsdp_config.param_offload=True
  global_profiler.tool=torch_memory
  "global_profiler.steps=[${profile_steps_csv}]"
  global_profiler.profile_continuous_steps=false
  "global_profiler.save_path='${run_dir}/snapshots'"
  global_profiler.global_tool_config.torch_memory.trace_alloc_max_entries="${max_entries}"
  actor_rollout_ref.actor.profiler.enable=true
  actor_rollout_ref.actor.profiler.all_ranks=true
  "$@"
)

printf 'CUDA_VISIBLE_DEVICES=%s; export steps=[%s]; output=%s\n' \
  "${CUDA_VISIBLE_DEVICES}" "${profile_steps_csv}" "${run_dir}"
if [[ "${dry_run}" == true ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
# A fresh directory prevents old snapshots from being mistaken for this run.
if [[ -d "${run_dir}" && -n "$(ls -A "${run_dir}")" ]]; then
  printf 'Output directory must be new or empty: %s\n' "${run_dir}" >&2
  exit 2
fi
mkdir -p "${run_dir}/snapshots"
printf '%q ' "${cmd[@]}" >"${run_dir}/command.sh"
printf '\n' >>"${run_dir}/command.sh"

monitor_pid=""
stop_monitor() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    monitor_pid=""
  fi
}
trap stop_monitor EXIT
if ((sample_ms > 0)) && command -v nvidia-smi >/dev/null; then
  # All physical GPUs are recorded with UUID/bus ID; CUDA ordinals can differ.
  nvidia-smi --query-gpu=timestamp,index,uuid,pci.bus_id,memory.used,memory.total \
    --format=csv,nounits --loop-ms="${sample_ms}" \
    >"${run_dir}/device_memory.csv" 2>"${run_dir}/device_memory_monitor.log" &
  monitor_pid=$!
fi
set +e
"${cmd[@]}" 2>&1 | tee "${run_dir}/run.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
stop_monitor
printf '%s\n' "${pipeline_status[0]}" >"${run_dir}/training_exit_code.txt"

# Preserve completed snapshots and the training exit code if a later step fails.
report_rc=0
"${python_bin}" "${repo_root}/scripts/local/analyze_torch_memory.py" \
  "${run_dir}/snapshots" --output-dir "${run_dir}/memory_report" \
  --max-entries "${max_entries}" --device-csv "${run_dir}/device_memory.csv" || report_rc=$?
for step in "${profile_steps[@]}"; do
  found=false
  for snapshot in "${run_dir}/snapshots/step${step}"/torch_memory*.pickle; do
    if [[ -s "${snapshot}" ]]; then found=true; break; fi
  done
  if [[ "${found}" != true ]]; then
    printf 'Missing memory snapshot for step %s; see %s/run.log\n' "${step}" "${run_dir}" >&2
    report_rc=1
  fi
done
printf 'Peak report: %s/memory_report/summary.md\n' "${run_dir}"
printf 'Viewer: https://pytorch.org/memory_viz (drag in a snapshots/step*/torch_memory*.pickle file)\n'
((pipeline_status[0] == 0)) || exit "${pipeline_status[0]}"
((pipeline_status[1] == 0)) || exit "${pipeline_status[1]}"
exit "${report_rc}"
