#!/usr/bin/env bash
# Sourced by the Nsight and CUDA-memory launchers. Keep their workload identical.

profile_die() { printf '%s\n' "$*" >&2; exit 2; }

profile_usage() {
  printf 'Usage: bash %s [--dry-run] [OUTPUT_DIR] [Hydra overrides...]\n\n' "${0}"
  cat <<'EOF'
Both launchers use the same training configuration and environment options.
Without --dry-run, training starts immediately. Use a new or empty output directory.
Trailing Hydra overrides take precedence over the common training defaults.

Common environment:
  CUDA_VISIBLE_DEVICES              GPU selection (default: 0)
  CUDA_DEVICE_MAX_CONNECTIONS       CUDA work queues, also forwarded to Ray (default: 8)
  SHARE_STUDENT_WEIGHTS             Share BF16 actor/vLLM weights (default: true)
  BF16_STUDENT_WEIGHTS              Use BF16 when sharing is disabled (default: true)
  OPTIMIZER_OFFLOAD_PER_LAYER       Stage optimizer states per layer (default: true)
  STUDENT_TEACHER_PIPELINE          CPU Adam + shared student/teacher slots (default: false)
  PIPELINE_OVERLAP                  Overlap CPU/copies; false is serial reference (default: true)
  CPU_OPTIMIZER_THREADS             CPU Adam worker threads (default: 8)
  PIPELINE_GRADIENT_OFFLOAD        Offload completed gradients during backward (default: true)
  TEACHER_LAYER_PIPELINE            Share slots among teachers only (default: false)
  TEACHER_PARAM_PREFETCH            Prefetch primary teacher parameters (default: true)
  TEACHER_PARAM_PREFETCH_MAX_MB      Teacher prefetch cap in MiB (default: 768)
  TEACHER_FORWARD_OVERLAP           Overlap student/teacher scoring (default: true)
  OPENMOPD_N_GPUS_PER_NODE          GPU count (default: 1)
  OPENMOPD_TRAIN_BATCH_SIZE         Train and PPO minibatch size (default: GPU count)
  OPENMOPD_PROFILE_STEP             First profiled/exported step (default: 3)
  OPENMOPD_PROFILE_STEP_COUNT       Consecutive steps (default: 2)
  OPENMOPD_NUM_STEPS                Total training steps (default: last profile step)
  OPENMOPD_VAL_BEFORE_TRAIN          Run initial validation (default: true)
  OPENMOPD_MEMORY_SAMPLE_MS         Whole-device sampling interval; 0 disables (default: 100)
  OPENMOPD_PYTHON                   Python executable (default: mopd environment)
  OPENMOPD_REPO_ROOT                Repository root (default: relative to this helper)
  OPENMOPD_MODEL_ROOT               Model/data root (default: /home/xxf/Distill/models/OPD)

Nsight-only options:
  OPENMOPD_NSYS                     nsys executable (default: /usr/local/cuda-12/bin/nsys)
  OPENMOPD_TRACE_WAIT_SECONDS       Wait for a finalized capture (default: 300)
  OPENMOPD_NSYS_CPU_SAMPLING        process-tree or none; none also disables CPU scheduling trace
                                  (default: process-tree; CUDA/NVTX remain enabled)
  OPENMOPD_NSYS_SAMPLING_PERIOD     CPU reference cycles per sample, 237500..30400000 (default: 19000000)
  OPENMOPD_NSYS_SAMPLES_PER_BACKTRACE  CPU samples per call stack, 1..32 (default: 4)

CUDA-memory-only options:
  OPENMOPD_MEMORY_MAX_ENTRIES        Allocation history capacity (default: 1000000)

Shared weights require one GPU. Teacher forward overlap requires one GPU and
train batch size 1; disable it for larger batches or multiple GPUs.
Use OPENMOPD_PROFILE_* to select steps: profiler tool, steps, save path and
continuous capture mode are managed by the launchers, not trailing overrides.

Nsight records NVTX/kernel/copy timing; CUDA-memory exports allocation snapshots
and HTML reports. Memory history starts at worker initialization and is cumulative,
so initialization/validation can be the peak. Nsight is not needed for --dry-run
or for the CUDA-memory launcher. See scripts/local/README.md for details.
EOF
}

profile_resolve_executable() {
  local executable="$1"
  if [[ "${executable}" != */* ]]; then executable="$(command -v "${executable}" || true)"; fi
  [[ -f "${executable}" && -x "${executable}" ]] || profile_die "Executable not found: $1"
  # Preserve the executable symlink: resolving a virtualenv's python to its
  # base interpreter would lose that environment's packages.
  printf '%s/%s\n' "$(cd "$(dirname "${executable}")" && pwd -P)" "$(basename "${executable}")"
}

profile_hydra_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "${value}"
}

profile_init() {
  profile_tool="$1"
  shift
  dry_run=false
  while (($# > 0)); do
    case "$1" in
      --help|-h) profile_usage; exit 0 ;;
      --dry-run) dry_run=true; shift ;;
      --) shift; break ;;
      -*) profile_die "Unknown option: $1" ;;
      *) break ;;
    esac
  done

  repo_root="${OPENMOPD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
  repo_root="$(realpath -m "${repo_root}")"
  model_root="${OPENMOPD_MODEL_ROOT:-/home/xxf/Distill/models/OPD}"
  python_bin="${OPENMOPD_PYTHON:-/home/xxf/anaconda3/envs/mopd/bin/python}"
  nsys_bin="${OPENMOPD_NSYS:-/usr/local/cuda-12/bin/nsys}"
  SHARE_STUDENT_WEIGHTS="${SHARE_STUDENT_WEIGHTS:-true}"
  BF16_STUDENT_WEIGHTS="${BF16_STUDENT_WEIGHTS:-true}"
  STUDENT_TEACHER_PIPELINE="${STUDENT_TEACHER_PIPELINE:-true}"
  PIPELINE_OVERLAP="${PIPELINE_OVERLAP:-true}"
  CPU_OPTIMIZER_THREADS="${CPU_OPTIMIZER_THREADS:-8}"
  PIPELINE_GRADIENT_OFFLOAD="${PIPELINE_GRADIENT_OFFLOAD:-true}"
  TEACHER_LAYER_PIPELINE="${TEACHER_LAYER_PIPELINE:-false}"
  local optimizer_layer_default=true
  if [[ "${STUDENT_TEACHER_PIPELINE}" == true ]]; then optimizer_layer_default=false; fi
  OPTIMIZER_OFFLOAD_PER_LAYER="${OPTIMIZER_OFFLOAD_PER_LAYER:-${optimizer_layer_default}}"
  TEACHER_PARAM_PREFETCH="${TEACHER_PARAM_PREFETCH:-true}"
  TEACHER_PARAM_PREFETCH_MAX_MB="${TEACHER_PARAM_PREFETCH_MAX_MB:-768}"
  TEACHER_FORWARD_OVERLAP="${TEACHER_FORWARD_OVERLAP:-true}"
  export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
  num_gpus="${OPENMOPD_N_GPUS_PER_NODE:-1}"
  train_batch_size="${OPENMOPD_TRAIN_BATCH_SIZE:-${num_gpus}}"
  profile_step="${OPENMOPD_PROFILE_STEP:-3}"
  profile_step_count="${OPENMOPD_PROFILE_STEP_COUNT:-2}"
  val_before_train="${OPENMOPD_VAL_BEFORE_TRAIN:-true}"
  sample_ms="${OPENMOPD_MEMORY_SAMPLE_MS:-100}"
  max_entries="${OPENMOPD_MEMORY_MAX_ENTRIES:-1000000}"
  trace_wait_seconds="${OPENMOPD_TRACE_WAIT_SECONDS:-300}"
  nsys_cpu_sampling="${OPENMOPD_NSYS_CPU_SAMPLING:-process-tree}"
  nsys_sampling_period="${OPENMOPD_NSYS_SAMPLING_PERIOD:-19000000}"
  nsys_samples_per_backtrace="${OPENMOPD_NSYS_SAMPLES_PER_BACKTRACE:-4}"

  local var prefix override key
  for var in SHARE_STUDENT_WEIGHTS BF16_STUDENT_WEIGHTS OPTIMIZER_OFFLOAD_PER_LAYER \
    TEACHER_PARAM_PREFETCH TEACHER_FORWARD_OVERLAP STUDENT_TEACHER_PIPELINE PIPELINE_OVERLAP \
    TEACHER_LAYER_PIPELINE PIPELINE_GRADIENT_OFFLOAD val_before_train; do
    [[ "${!var}" == true || "${!var}" == false ]] || profile_die "${var} must be true or false"
  done
  for var in num_gpus train_batch_size profile_step profile_step_count \
    TEACHER_PARAM_PREFETCH_MAX_MB CUDA_DEVICE_MAX_CONNECTIONS CPU_OPTIMIZER_THREADS; do
    [[ "${!var}" =~ ^[1-9][0-9]*$ ]] || profile_die "${var} must be a positive integer"
  done
  [[ "${sample_ms}" =~ ^(0|[1-9][0-9]*)$ ]] || profile_die "OPENMOPD_MEMORY_SAMPLE_MS must be non-negative"
  ((train_batch_size % num_gpus == 0)) || profile_die "Train batch size must be divisible by GPU count"
  if ((num_gpus > 1)) && [[ "${SHARE_STUDENT_WEIGHTS}" == true ]]; then
    profile_die "Shared weights require one GPU; set SHARE_STUDENT_WEIGHTS=false"
  fi
  if [[ "${TEACHER_FORWARD_OVERLAP}" == true ]] && ((num_gpus != 1 || train_batch_size != 1)); then
    profile_die "Teacher forward overlap requires one GPU and batch size 1; set TEACHER_FORWARD_OVERLAP=false"
  fi
  case "${profile_tool}" in
    nsys)
      prefix=native_nsys
      [[ "${trace_wait_seconds}" =~ ^(0|[1-9][0-9]*)$ ]] || profile_die "OPENMOPD_TRACE_WAIT_SECONDS must be non-negative"
      [[ "${nsys_cpu_sampling}" == process-tree || "${nsys_cpu_sampling}" == none ]] || \
        profile_die "OPENMOPD_NSYS_CPU_SAMPLING must be process-tree or none"
      [[ "${nsys_sampling_period}" =~ ^[1-9][0-9]{0,7}$ ]] && \
        ((nsys_sampling_period >= 237500 && nsys_sampling_period <= 30400000)) || \
        profile_die "OPENMOPD_NSYS_SAMPLING_PERIOD must be an integer in 237500..30400000 reference cycles"
      [[ "${nsys_samples_per_backtrace}" =~ ^[1-9][0-9]?$ ]] && \
        ((nsys_samples_per_backtrace <= 32)) || \
        profile_die "OPENMOPD_NSYS_SAMPLES_PER_BACKTRACE must be an integer in 1..32"
      ;;
    torch_memory)
      prefix=native_memory
      [[ "${max_entries}" =~ ^[1-9][0-9]*$ ]] || profile_die "OPENMOPD_MEMORY_MAX_ENTRIES must be positive"
      ;;
    *) profile_die "Unknown profiler: ${profile_tool}" ;;
  esac

  profile_end_step=$((profile_step + profile_step_count - 1))
  num_steps="${OPENMOPD_NUM_STEPS:-${profile_end_step}}"
  [[ "${num_steps}" =~ ^[1-9][0-9]*$ ]] || profile_die "OPENMOPD_NUM_STEPS must be positive"
  ((num_steps >= profile_end_step)) || profile_die "OPENMOPD_NUM_STEPS must be >= ${profile_end_step}"
  profile_steps=()
  local step
  for ((step = profile_step; step <= profile_end_step; step++)); do profile_steps+=("${step}"); done
  profile_steps_csv="$(IFS=,; printf '%s' "${profile_steps[*]}")"
  run_dir="${repo_root}/output/${prefix}_$(date +%Y%m%d_%H%M%S)_$$"
  if (($# > 0)) && [[ "$1" != *=* ]]; then run_dir="$1"; shift; fi
  if [[ "${1:-}" == -- ]]; then shift; fi
  profile_overrides=("$@")
  for override in "${profile_overrides[@]}"; do
    key="${override%%=*}"
    key="${key#++}"; key="${key#+}"; key="${key#~}"
    case "${key}" in
      global_profiler.tool|global_profiler.steps|global_profiler.save_path|global_profiler.profile_continuous_steps)
        profile_die "${key} is managed by the launcher; use OUTPUT_DIR and OPENMOPD_PROFILE_* instead" ;;
    esac
  done
  run_dir="$(realpath -m "${run_dir}")"
  if [[ "${run_dir}" == *"'"* || "${run_dir}" == *\\* || "${run_dir}" == *$'\n'* ]]; then
    profile_die "Output path cannot contain single quotes, backslashes or newlines"
  fi
  # Ray interpolates the Nsight output path into a shell command.
  if [[ "${profile_tool}" == nsys && ! "${run_dir}" =~ ^/[a-zA-Z0-9_./-]+$ ]]; then
    profile_die "Nsight output paths may contain only letters, digits, / . _ -"
  fi
  raw_dir="${run_dir}/profile"
  profile_marker="${run_dir}/profile.started"
  python_bin="$(profile_resolve_executable "${python_bin}")" || exit 2
  if [[ "${profile_tool}" == nsys && "${dry_run}" == false ]]; then
    nsys_bin="$(profile_resolve_executable "${nsys_bin}")" || exit 2
    export PATH="$(dirname "${nsys_bin}"):${PATH}"
  fi
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export VLLM_USE_V1=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
  export PATH="$(dirname "${python_bin}"):${PATH}"
  export PYTHONPATH="${repo_root}/training/verl${PYTHONPATH:+:${PYTHONPATH}}"
}

profile_build_command() {
  cmd=("${python_bin}" -m verl.trainer.main_ppo
    algorithm.adv_estimator=token_reward_direct
    "data.train_files=$(profile_hydra_string "${model_root}/data/rl_prompt_mix/train.parquet")"
    "data.val_files=$(profile_hydra_string "${model_root}/data/rl_prompt_mix/eval.parquet")"
    data.train_batch_size="${train_batch_size}" data.max_prompt_length=1024 data.max_response_length=4096
    data.filter_overlong_prompts=True data.truncation=error
    "actor_rollout_ref.model.path=$(profile_hydra_string "${model_root}/MixSFT")"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.share_weights="${SHARE_STUDENT_WEIGHTS}"
    actor_rollout_ref.rollout.teacher_param_prefetch="${TEACHER_PARAM_PREFETCH}"
    actor_rollout_ref.rollout.teacher_param_prefetch_max_mb="${TEACHER_PARAM_PREFETCH_MAX_MB}"
    actor_rollout_ref.rollout.teacher_forward_overlap="${TEACHER_FORWARD_OVERLAP}"
    actor_rollout_ref.rollout.student_teacher_pipeline="${STUDENT_TEACHER_PIPELINE}"
    actor_rollout_ref.rollout.pipeline_overlap="${PIPELINE_OVERLAP}"
    actor_rollout_ref.rollout.cpu_optimizer_threads="${CPU_OPTIMIZER_THREADS}"
    actor_rollout_ref.rollout.pipeline_gradient_offload="${PIPELINE_GRADIENT_OFFLOAD}"
    actor_rollout_ref.rollout.teacher_layer_pipeline="${TEACHER_LAYER_PIPELINE}"
    +actor_rollout_ref.rollout.reward_mode=mt_opd
    actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.max_model_len=5120
    actor_rollout_ref.actor.ppo_mini_batch_size="${train_batch_size}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload_per_layer="${OPTIMIZER_OFFLOAD_PER_LAYER}"
    actor_rollout_ref.actor.optim.override_optimizer_config='{foreach:false}'
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    +actor_rollout_ref.rollout.log_prob_top_k=256
    "custom_reward_function.path=$(profile_hydra_string "${repo_root}/training/verl/verl/utils/reward_score/opd_val_dispatch.py")"
    custom_reward_function.name=reward_func
    reward_model.micro_batch_size_per_gpu=1 reward_model.enable=True
    "reward_model.model.path=$(profile_hydra_string "${model_root}/Math")"
    reward_model.model.input_tokenizer=null
    +reward_model.reward_kwargs.compute_true_reward=false
    '+mt_opd.teacher_domains=[math,code]' +mt_opd.n_additional_teachers=1
    trainer.n_gpus_per_node="${num_gpus}" trainer.nnodes=1 trainer.total_epochs=1
    trainer.total_training_steps="${num_steps}" trainer.val_before_train="${val_before_train}"
    trainer.resume_mode=disable
    "trainer.default_local_dir=$(profile_hydra_string "${run_dir}/checkpoints")"
    trainer.project_name=OpenOPD-local trainer.experiment_name=mt-opd-local
    "trainer.logger=['console']"
    "+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS='${CUDA_DEVICE_MAX_CONNECTIONS}'"
    +mt_reward_model_1.enable=True
    "+mt_reward_model_1.model.path=$(profile_hydra_string "${model_root}/Code")"
    +mt_reward_model_1.model.input_tokenizer=null
    +mt_reward_model_1.model.use_remove_padding=True
    +mt_reward_model_1.model.fsdp_config.param_offload=True
    global_profiler.tool="${profile_tool}"
    "global_profiler.steps=[${profile_steps_csv}]"
    actor_rollout_ref.actor.profiler.enable=true
    actor_rollout_ref.actor.profiler.all_ranks=true
  )
  if [[ "${BF16_STUDENT_WEIGHTS}" == true && "${SHARE_STUDENT_WEIGHTS}" != true ]]; then
    cmd+=(actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
      actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.bf16_optimizer
      actor_rollout_ref.actor.optim.optimizer=BF16StochasticAdamW)
  fi
  if [[ "${profile_tool}" == nsys ]]; then
    cmd+=(global_profiler.profile_continuous_steps=true
      "global_profiler.save_path=$(profile_hydra_string "${raw_dir}")"
      global_profiler.global_tool_config.nsys.discrete=false
      global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range-end=repeat-shutdown:1
      global_profiler.global_tool_config.nsys.worker_nsight_options.kill=none
      "+global_profiler.global_tool_config.nsys.worker_nsight_options.o=${raw_dir}/worker_%p"
      "+global_profiler.global_tool_config.nsys.controller_nsight_options.o=${raw_dir}/controller_%p")
    # Linux x86 Nsight uses reference cycles, not Hz. Reduce both IP sample
    # frequency and backtrace volume to avoid overflowing the Perf buffers.
    # Ray launches separate profilers for the controller and the GPU worker.
    local role
    for role in controller worker; do
      cmd+=("++global_profiler.global_tool_config.nsys.${role}_nsight_options.sample=${nsys_cpu_sampling}"
        "++global_profiler.global_tool_config.nsys.${role}_nsight_options.cpuctxsw=${nsys_cpu_sampling}"
        "++global_profiler.global_tool_config.nsys.${role}_nsight_options.sampling-period=${nsys_sampling_period}"
        "++global_profiler.global_tool_config.nsys.${role}_nsight_options.samples-per-backtrace=${nsys_samples_per_backtrace}")
    done
  else
    cmd+=(global_profiler.profile_continuous_steps=false
      "global_profiler.save_path=$(profile_hydra_string "${run_dir}/snapshots")"
      global_profiler.global_tool_config.torch_memory.trace_alloc_max_entries="${max_entries}")
  fi
  cmd+=("${profile_overrides[@]}")
  printf 'CUDA_VISIBLE_DEVICES=%s; profile steps=[%s]; output=%s\n' \
    "${CUDA_VISIBLE_DEVICES}" "${profile_steps_csv}" "${run_dir}"
  printf 'Teacher prefetch=%s; cap=%s MiB; forward overlap=%s; optimizer per layer=%s; CUDA connections=%s\n' \
    "${TEACHER_PARAM_PREFETCH}" "${TEACHER_PARAM_PREFETCH_MAX_MB}" "${TEACHER_FORWARD_OVERLAP}" \
    "${OPTIMIZER_OFFLOAD_PER_LAYER}" "${CUDA_DEVICE_MAX_CONNECTIONS}"
  if [[ "${profile_tool}" == nsys ]]; then
    printf 'Nsight CPU sampling/scheduling=%s; period=%s reference cycles; samples per backtrace=%s (before Hydra overrides)\n' \
      "${nsys_cpu_sampling}" "${nsys_sampling_period}" "${nsys_samples_per_backtrace}"
  fi
  if [[ "${dry_run}" == true ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
}

profile_prepare_run() {
  [[ ! -e "${run_dir}" || -d "${run_dir}" ]] || profile_die "Not a directory: ${run_dir}"
  if [[ -d "${run_dir}" && -n "$(ls -A "${run_dir}")" ]]; then
    profile_die "Output directory must be new or empty: ${run_dir}"
  fi
  if [[ "${profile_tool}" == nsys ]]; then
    mkdir -p "${run_dir}/traces" "${raw_dir}"
    touch "${profile_marker}"
  else
    mkdir -p "${run_dir}/snapshots"
  fi
  printf '%q ' "${cmd[@]}" >"${run_dir}/command.sh"
  printf '\n' >>"${run_dir}/command.sh"
}

profile_stop_monitor() {
  if [[ -n "${monitor_pid:-}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    monitor_pid=""
  fi
}

profile_run_training() {
  monitor_pid=""
  trap profile_stop_monitor EXIT
  if ((sample_ms > 0)) && command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=timestamp,index,uuid,pci.bus_id,memory.used,memory.total \
      --format=csv,nounits --loop-ms="${sample_ms}" \
      >"${run_dir}/device_memory.csv" 2>"${run_dir}/device_memory_monitor.log" &
    monitor_pid=$!
  fi
  set +e
  "${cmd[@]}" 2>&1 | tee "${run_dir}/run.log"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  profile_stop_monitor
  train_rc="${pipeline_status[0]}"
  tee_rc="${pipeline_status[1]}"
  printf '%s\n' "${train_rc}" >"${run_dir}/training_exit_code.txt"
}
