#!/usr/bin/env bash
# Profile consecutive steps; set SHARE_STUDENT_WEIGHTS=false for native weights.
set -euo pipefail

SHARE_STUDENT_WEIGHTS="${SHARE_STUDENT_WEIGHTS:-true}"
BF16_STUDENT_WEIGHTS="${BF16_STUDENT_WEIGHTS:-true}"
# Overlap the primary teacher (Math) with student log-prob scoring.
TEACHER_PARAM_PREFETCH="${TEACHER_PARAM_PREFETCH:-true}"
TEACHER_PARAM_PREFETCH_MAX_MB="${TEACHER_PARAM_PREFETCH_MAX_MB:-768}"
TEACHER_FORWARD_OVERLAP="${TEACHER_FORWARD_OVERLAP:-true}"
# verl defaults to one CUDA work queue, which can serialize independent streams.
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"

# share=true configures BF16 actor parameters and FP32 optimizer state in the
# worker.  This array exposes the same training precision for non-shared runs.
student_training_args=()
if [[ "${BF16_STUDENT_WEIGHTS}" == "true" && "${SHARE_STUDENT_WEIGHTS}" != "true" ]]; then
  student_training_args+=(
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.bf16_optimizer
    actor_rollout_ref.actor.optim.optimizer=BF16StochasticAdamW
  )
fi

repo_root="/home/xxf/Distill/Open-MOPD"
run_stamp="$(date +%Y%m%d_%H%M%S)"
run_dir="${1:-${repo_root}/output/native_nsys_${run_stamp}}"
profile_step="${OPENMOPD_PROFILE_STEP:-3}"
profile_step_count="${OPENMOPD_PROFILE_STEP_COUNT:-2}"
val_before_train="${OPENMOPD_VAL_BEFORE_TRAIN:-true}"
trace_wait_seconds="${OPENMOPD_TRACE_WAIT_SECONDS:-300}"
python_bin="${OPENMOPD_PYTHON:-/home/xxf/anaconda3/envs/mopd/bin/python}"
nsys_bin="${OPENMOPD_NSYS:-/usr/local/cuda-12/bin/nsys}"
num_gpus="${OPENMOPD_N_GPUS_PER_NODE:-1}"
train_batch_size="${OPENMOPD_TRAIN_BATCH_SIZE:-${num_gpus}}"

if [[ ! "${num_gpus}" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "${train_batch_size}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'OPENMOPD_N_GPUS_PER_NODE and OPENMOPD_TRAIN_BATCH_SIZE must be positive integers\n' >&2
  exit 2
fi
if ((train_batch_size % num_gpus != 0)); then
  printf 'OPENMOPD_TRAIN_BATCH_SIZE must be divisible by OPENMOPD_N_GPUS_PER_NODE\n' >&2
  exit 2
fi
if ((num_gpus > 1)) && [[ "${SHARE_STUDENT_WEIGHTS}" == "true" ]]; then
  printf 'Shared actor/vLLM weights require one GPU; set SHARE_STUDENT_WEIGHTS=false for multi-GPU profiling.\n' >&2
  exit 2
fi
if [[ "${TEACHER_FORWARD_OVERLAP}" == "true" ]] && ((num_gpus != 1 || train_batch_size != 1)); then
  printf 'Teacher forward overlap requires one GPU and train batch size 1; set TEACHER_FORWARD_OVERLAP=false otherwise.\n' >&2
  exit 2
fi

if [[ ! "${profile_step}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'OPENMOPD_PROFILE_STEP must be a positive integer, got: %s\n' "${profile_step}" >&2
  exit 2
fi
if [[ ! "${profile_step_count}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'OPENMOPD_PROFILE_STEP_COUNT must be a positive integer, got: %s\n' \
    "${profile_step_count}" >&2
  exit 2
fi
profile_end_step=$((profile_step + profile_step_count - 1))
profile_steps=()
for ((step = profile_step; step <= profile_end_step; step++)); do
  profile_steps+=("${step}")
done
profile_steps_csv="$(IFS=,; printf '%s' "${profile_steps[*]}")"
num_steps="${OPENMOPD_NUM_STEPS:-${profile_end_step}}"
if [[ ! "${num_steps}" =~ ^[1-9][0-9]*$ ]] || ((num_steps < profile_end_step)); then
  printf 'OPENMOPD_NUM_STEPS must be an integer >= final profile step, got: %s < %s\n' \
    "${num_steps}" "${profile_end_step}" >&2
  exit 2
fi
if [[ "${val_before_train}" != "true" && "${val_before_train}" != "false" ]]; then
  printf 'OPENMOPD_VAL_BEFORE_TRAIN must be true or false, got: %s\n' "${val_before_train}" >&2
  exit 2
fi
if [[ ! "${trace_wait_seconds}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  printf 'OPENMOPD_TRACE_WAIT_SECONDS must be a non-negative integer, got: %s\n' \
    "${trace_wait_seconds}" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  printf 'Python executable not found: %s\n' "${python_bin}" >&2
  exit 2
fi
if [[ "${nsys_bin}" != */* ]]; then
  nsys_bin="$(command -v "${nsys_bin}" || true)"
fi
if [[ ! -x "${nsys_bin}" ]]; then
  printf 'Nsight Systems executable not found: %s\n' "${nsys_bin}" >&2
  exit 2
fi

mkdir -p "${run_dir}/traces" "${run_dir}/profile"
run_dir="$(cd "${run_dir}" && pwd -P)"
# Ray interpolates the Nsight output path into a shell command.
if [[ ! "${run_dir}" =~ ^/[a-zA-Z0-9_./-]+$ ]]; then
  printf 'Use an output path containing only letters, digits, / . _ -: %s\n' "${run_dir}" >&2
  exit 2
fi
raw_dir="${run_dir}/profile"
profile_marker="${run_dir}/profile.started"
touch "${profile_marker}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PATH="$(dirname "${nsys_bin}"):$(dirname "${python_bin}"):${PATH}"
export PYTHONPATH="${repo_root}/training/verl"

printf 'CUDA_VISIBLE_DEVICES=%s; profile steps=[%s]; raw reports=%s\n' \
  "${CUDA_VISIBLE_DEVICES}" "${profile_steps_csv}" "${raw_dir}"
printf 'Teacher param prefetch=%s; prefetch cap=%s MiB; teacher forward overlap=%s\n' \
  "${TEACHER_PARAM_PREFETCH}" "${TEACHER_PARAM_PREFETCH_MAX_MB}" "${TEACHER_FORWARD_OVERLAP}"
printf 'CUDA_DEVICE_MAX_CONNECTIONS=%s\n' "${CUDA_DEVICE_MAX_CONNECTIONS}"
# Collect completed captures even if training fails later, retaining its exit code.
set +e
"${python_bin}" -m verl.trainer.main_ppo \
algorithm.adv_estimator=token_reward_direct \
data.train_files=/home/xxf/Distill/models/OPD/data/rl_prompt_mix/train.parquet \
data.val_files=/home/xxf/Distill/models/OPD/data/rl_prompt_mix/eval.parquet \
data.train_batch_size="${train_batch_size}" \
data.max_prompt_length=512 \
data.max_response_length=1024 \
data.filter_overlong_prompts=True \
data.truncation=error \
actor_rollout_ref.model.path=/home/xxf/Distill/models/OPD/MixSFT \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.share_weights="${SHARE_STUDENT_WEIGHTS}" \
actor_rollout_ref.rollout.teacher_param_prefetch="${TEACHER_PARAM_PREFETCH}" \
actor_rollout_ref.rollout.teacher_param_prefetch_max_mb="${TEACHER_PARAM_PREFETCH_MAX_MB}" \
actor_rollout_ref.rollout.teacher_forward_overlap="${TEACHER_FORWARD_OVERLAP}" \
"${student_training_args[@]}" \
+actor_rollout_ref.rollout.reward_mode=mt_opd \
actor_rollout_ref.rollout.n=1 \
actor_rollout_ref.rollout.max_model_len=1536 \
actor_rollout_ref.actor.ppo_mini_batch_size="${train_batch_size}" \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.actor.optim.override_optimizer_config='{foreach:false}' \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
+actor_rollout_ref.rollout.log_prob_top_k=256 \
custom_reward_function.path=/home/xxf/Distill/Open-MOPD/training/verl/verl/utils/reward_score/opd_val_dispatch.py \
custom_reward_function.name=reward_func \
reward_model.micro_batch_size_per_gpu=1 \
reward_model.enable=True \
reward_model.model.path=/home/xxf/Distill/models/OPD/Math \
reward_model.model.input_tokenizer=null \
+reward_model.reward_kwargs.compute_true_reward=false \
+mt_opd.teacher_domains=\[math\,code\] \
+mt_opd.n_additional_teachers=1 \
trainer.n_gpus_per_node="${num_gpus}" \
trainer.nnodes=1 \
trainer.total_epochs=1 \
trainer.total_training_steps="${num_steps}" \
trainer.val_before_train="${val_before_train}" \
trainer.resume_mode=disable \
trainer.default_local_dir="${run_dir}/checkpoints" \
trainer.project_name=OpenOPD-local \
trainer.experiment_name=mt-opd-local \
trainer.logger=\[\'console\'\] \
"+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS='${CUDA_DEVICE_MAX_CONNECTIONS}'" \
+mt_reward_model_1.enable=True \
+mt_reward_model_1.model.path=/home/xxf/Distill/models/OPD/Code \
+mt_reward_model_1.model.input_tokenizer=null \
+mt_reward_model_1.model.use_remove_padding=True \
+mt_reward_model_1.model.fsdp_config.param_offload=True \
global_profiler.tool=nsys \
"global_profiler.steps=[${profile_steps_csv}]" \
global_profiler.profile_continuous_steps=true \
global_profiler.save_path="${run_dir}/profile" \
global_profiler.global_tool_config.nsys.discrete=false \
global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range-end=repeat-shutdown:1 \
global_profiler.global_tool_config.nsys.worker_nsight_options.kill=none \
"+global_profiler.global_tool_config.nsys.worker_nsight_options.o=${raw_dir}/worker_%p" \
"+global_profiler.global_tool_config.nsys.controller_nsight_options.o=${raw_dir}/controller_%p" \
actor_rollout_ref.actor.profiler.enable=true \
actor_rollout_ref.actor.profiler.all_ranks=true \
 2>&1 | tee "${run_dir}/run.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
train_rc="${pipeline_status[0]}"
tee_rc="${pipeline_status[1]}"
printf '%s\n' "${train_rc}" >"${run_dir}/training_exit_code.txt"
if ((train_rc != 0 || tee_rc != 0)); then
  printf 'Training exit=%s, tee exit=%s; collecting any completed reports. See %s/run.log\n' \
    "${train_rc}" "${tee_rc}" "${run_dir}" >&2
fi

# Absolute Nsight -o paths isolate reports from other jobs changing Ray's
# session_latest symlink. repeat-shutdown:1 finalizes our single continuous
# capture at the final selected step, without waiting for worker teardown.
# A newly-created .nsys-rep can
# remain at zero bytes for several polls, so file-size stability alone is not a
# completion signal.  Wait until nsys can read a non-empty report containing
# every requested step marker.
trace_files=()
worker_source=""
declare -A checked_trace_states=()
deadline=$((SECONDS + trace_wait_seconds))
collector_log="${run_dir}/collector.log"
printf 'Waiting up to %s seconds for steps [%s] in %s\n' \
  "${trace_wait_seconds}" "${profile_steps_csv}" "${raw_dir}" | tee "${collector_log}"
while ((SECONDS <= deadline)); do
  mapfile -d '' trace_files < <(
    find "${raw_dir}" -maxdepth 1 -type f \
      \( -name 'worker_*.nsys-rep' -o -name 'worker_*.qdrep' \) -newer "${profile_marker}" -print0 | sort -z
  )
  for trace_file in "${trace_files[@]}"; do
    [[ -s "${trace_file}" ]] || continue
    trace_state="$(stat -c '%i:%s:%y' "${trace_file}")"
    if [[ "${checked_trace_states[${trace_file}]:-}" == "${trace_state}" ]]; then
      continue
    fi
    remaining=$((deadline - SECONDS))
    ((remaining > 0)) || remaining=1
    printf 'Checking %s\n' "${trace_file}" >>"${collector_log}"
    if ! nvtx_summary="$(
      timeout --kill-after=5s "${remaining}s" \
        "${nsys_bin}" stats --force-export=true --report nvtx_pushpop_sum \
        --format csv "${trace_file}" 2>>"${collector_log}"
    )"; then
      # Retry transient export failures even if size/mtime did not change.
      continue
    fi
    # Reject a report that changed while nsys was reading it; it is still being
    # finalized and will be retried after its size/mtime changes.
    if [[ "$(stat -c '%i:%s:%y' "${trace_file}")" != "${trace_state}" ]]; then
      continue
    fi
    checked_trace_states["${trace_file}"]="${trace_state}"
    # Match complete CSV range names so step 30 cannot satisfy step 3.
    if "${python_bin}" -c '
import csv, sys
ranges = {row[-1].lstrip(":") for row in csv.reader(sys.stdin) if row}
expected = {"openmopd::step::" + step for step in sys.argv[1].split(",")}
sys.exit(0 if expected <= ranges else 1)
' "${profile_steps_csv}" <<<"${nvtx_summary}"; then
      worker_source="${trace_file}"
      break
    fi
  done
  if [[ -n "${worker_source}" ]]; then
    break
  fi
  sleep 2
done

if [[ -z "${worker_source}" ]]; then
  printf 'No finalized Nsight report containing steps [%s] appeared within %s seconds in %s\n' \
    "${profile_steps_csv}" "${trace_wait_seconds}" "${raw_dir}" >&2
  printf 'See %s/run.log and %s/collector.log\n' "${run_dir}" "${run_dir}" >&2
  ((train_rc == 0)) || exit "${train_rc}"
  ((tee_rc == 0)) || exit "${tee_rc}"
  exit 1
fi

worker_report="${run_dir}/traces/$(basename "${worker_source}")"
cp -f "${worker_source}" "${worker_report}"

"${nsys_bin}" export --type sqlite --force-overwrite=true --quiet=true \
  --output "${run_dir}/traces/worker.sqlite" "${worker_report}"
"${python_bin}" "${repo_root}/scripts/local/analyze_nsys_memory.py" \
  "${run_dir}/traces/worker.sqlite" >"${run_dir}/memory_analysis.md"
"${python_bin}" "${repo_root}/scripts/local/analyze_teacher_forward_overlap.py" \
  "${run_dir}" >"${run_dir}/teacher_overlap.json"
"${nsys_bin}" stats \
  --report nvtx_pushpop_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_gpu_sum,cuda_api_sum \
  --format csv --output "${run_dir}/traces/worker_stats" "${worker_report}"

printf 'Selected worker report: %s\n' "${worker_report}"
printf 'Profiled continuous steps: %s\n' "${profile_steps_csv}"
printf 'Profile output: %s\n' "${run_dir}"
((train_rc == 0)) || exit "${train_rc}"
((tee_rc == 0)) || exit "${tee_rc}"
