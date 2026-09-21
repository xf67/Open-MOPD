#!/usr/bin/env bash
# Profile consecutive Open-MOPD training steps (native/non-shared by default).
set -euo pipefail

SHARE_STUDENT_WEIGHTS="${SHARE_STUDENT_WEIGHTS:-false}"
BF16_STUDENT_WEIGHTS="${BF16_STUDENT_WEIGHTS:-true}"

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
trace_wait_seconds="${OPENMOPD_TRACE_WAIT_SECONDS:-120}"
python_bin="${OPENMOPD_PYTHON:-/home/xxf/anaconda3/envs/mopd/bin/python}"
nsys_bin="${OPENMOPD_NSYS:-/usr/local/cuda-12/bin/nsys}"

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
profile_marker="${run_dir}/profile.started"
touch "${profile_marker}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PATH="$(dirname "${nsys_bin}"):/home/xxf/anaconda3/envs/mopd/bin:${PATH}"
export PYTHONPATH="${repo_root}/training/verl"

"${python_bin}" -m verl.trainer.main_ppo \
algorithm.adv_estimator=token_reward_direct \
data.train_files=/home/xxf/Distill/models/OPD/data/rl_prompt_mix/train.parquet \
data.val_files=/home/xxf/Distill/models/OPD/data/rl_prompt_mix/eval.parquet \
data.train_batch_size=1 \
data.max_prompt_length=512 \
data.max_response_length=1024 \
data.filter_overlong_prompts=True \
data.truncation=error \
actor_rollout_ref.model.path=/home/xxf/Distill/models/OPD/MixSFT \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.share_weights="${SHARE_STUDENT_WEIGHTS}" \
"${student_training_args[@]}" \
+actor_rollout_ref.rollout.reward_mode=mt_opd \
actor_rollout_ref.rollout.n=1 \
actor_rollout_ref.rollout.max_model_len=1536 \
actor_rollout_ref.actor.ppo_mini_batch_size=1 \
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
trainer.n_gpus_per_node=1 \
trainer.nnodes=1 \
trainer.total_epochs=1 \
trainer.total_training_steps="${num_steps}" \
trainer.val_before_train="${val_before_train}" \
trainer.resume_mode=disable \
trainer.default_local_dir="${run_dir}/checkpoints" \
trainer.project_name=OpenOPD-local \
trainer.experiment_name=mt-opd-local \
trainer.logger=\[\'console\'\] \
+mt_reward_model_1.enable=True \
+mt_reward_model_1.model.path=/home/xxf/Distill/models/OPD/Code \
+mt_reward_model_1.model.input_tokenizer=null \
+mt_reward_model_1.model.use_remove_padding=True \
+mt_reward_model_1.model.fsdp_config.param_offload=True \
global_profiler.tool=nsys \
"global_profiler.steps=[${profile_steps_csv}]" \
global_profiler.profile_continuous_steps=true \
global_profiler.save_path="${run_dir}/profile" \
actor_rollout_ref.actor.profiler.enable=true \
actor_rollout_ref.actor.profiler.all_ranks=true \
 2>&1 | tee "${run_dir}/run.log"

# Ray's Nsight integration writes beside its worker logs and finalizes reports
# asynchronously as the Python driver exits.  A newly-created .nsys-rep can
# remain at zero bytes for several polls, so file-size stability alone is not a
# completion signal.  Wait until nsys can read a non-empty report containing
# every requested step marker.
ray_nsight_dir="$(readlink -f /tmp/ray/session_latest/logs/nsight 2>/dev/null || true)"
if [[ -z "${ray_nsight_dir}" || ! -d "${ray_nsight_dir}" ]]; then
  printf 'Ray Nsight output directory was not created. See %s/run.log\n' "${run_dir}" >&2
  exit 1
fi

trace_files=()
worker_source=""
declare -A checked_trace_states=()
deadline=$((SECONDS + trace_wait_seconds))
while ((SECONDS <= deadline)); do
  mapfile -d '' trace_files < <(
    find -L "${ray_nsight_dir}" -maxdepth 1 -type f \
      \( -name '*.nsys-rep' -o -name '*.qdrep' \) -newer "${profile_marker}" -print0 2>/dev/null | sort -z
  )
  for trace_file in "${trace_files[@]}"; do
    [[ -s "${trace_file}" ]] || continue
    trace_state="$(stat -c '%s:%Y' "${trace_file}")"
    if [[ "${checked_trace_states[${trace_file}]:-}" == "${trace_state}" ]]; then
      continue
    fi
    checked_trace_states["${trace_file}"]="${trace_state}"
    nvtx_summary="$(
      "${nsys_bin}" stats --force-export=true --report nvtx_pushpop_sum \
        --format csv "${trace_file}" 2>/dev/null || true
    )"
    # Reject a report that changed while nsys was reading it; it is still being
    # finalized and will be retried after its size/mtime changes.
    if [[ "$(stat -c '%s:%Y' "${trace_file}")" != "${trace_state}" ]]; then
      continue
    fi
    contains_all_profile_steps=true
    for step in "${profile_steps[@]}"; do
      if [[ "${nvtx_summary}" != *"openmopd::step::${step}"* ]]; then
        contains_all_profile_steps=false
        break
      fi
    done
    if [[ "${contains_all_profile_steps}" == "true" ]]; then
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
    "${profile_steps_csv}" "${trace_wait_seconds}" "${ray_nsight_dir}" >&2
  exit 1
fi

worker_report="${run_dir}/traces/$(basename "${worker_source}")"
cp -f "${worker_source}" "${worker_report}"

"${nsys_bin}" export --type sqlite --force-overwrite=true --quiet=true \
  --output "${run_dir}/traces/worker.sqlite" "${worker_report}"
"${repo_root}/scripts/local/analyze_nsys_memory.py" \
  "${run_dir}/traces/worker.sqlite" >"${run_dir}/memory_analysis.md"
"${nsys_bin}" stats \
  --report nvtx_pushpop_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_gpu_sum,cuda_api_sum \
  --format csv --output "${run_dir}/traces/worker_stats" "${worker_report}"

printf 'Selected worker report: %s\n' "${worker_report}"
printf 'Profiled continuous steps: %s\n' "${profile_steps_csv}"
printf 'Profile output: %s\n' "${run_dir}"
