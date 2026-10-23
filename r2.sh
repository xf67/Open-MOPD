#!/usr/bin/env bash
# Normal training with the workload settings from scripts/local/profile_native_common.sh.
set -euo pipefail

die() { printf '%s\n' "$*" >&2; exit 2; }

dry_run=false
while (($# > 0)); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --help|-h)
            printf 'Usage: bash %s [--dry-run] [Hydra overrides...]\n' "$0"
            printf 'Activate conda environment mopd first. Defaults to one epoch without profiling.\n'
            printf 'Use OPENMOPD_NUM_STEPS to limit steps; trailing Hydra overrides take precedence.\n'
            exit 0 ;;
        --) shift; break ;;
        -*) die "Unknown option: $1" ;;
        *) break ;;
    esac
done

repo_root="${OPENMOPD_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
python_bin="${OPENMOPD_PYTHON:-python3}"
checkpoint_dir="${repo_root}/ckpt"

# Support both models/OPD/... and models/... layouts.
model_root="${OPENMOPD_MODEL_ROOT:-/home/xxf/Distill/models}"
if [[ -d "${model_root}/OPD" ]]; then
    model_root="${model_root}/OPD"
fi

resolve_executable() {
  local executable="$1"
  if [[ "${executable}" != */* ]]; then executable="$(command -v "${executable}" || true)"; fi
  [[ -f "${executable}" && -x "${executable}" ]] || die "Executable not found: $1"
  # Preserve the executable symlink: resolving a virtualenv's python to its
  # base interpreter would lose that environment's packages.
  printf '%s/%s\n' "$(cd "$(dirname "${executable}")" && pwd -P)" "$(basename "${executable}")"
}

hydra_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "${value}"
}

SHARE_STUDENT_WEIGHTS="${SHARE_STUDENT_WEIGHTS:-true}"
BF16_STUDENT_WEIGHTS="${BF16_STUDENT_WEIGHTS:-true}"
STUDENT_TEACHER_PIPELINE="${STUDENT_TEACHER_PIPELINE:-true}"
PIPELINE_OVERLAP="${PIPELINE_OVERLAP:-true}"
CPU_OPTIMIZER_THREADS="${CPU_OPTIMIZER_THREADS:-8}"
PIPELINE_GRADIENT_OFFLOAD="${PIPELINE_GRADIENT_OFFLOAD:-true}"
TEACHER_LAYER_PIPELINE="${TEACHER_LAYER_PIPELINE:-false}"
optimizer_layer_default=true
if [[ "${STUDENT_TEACHER_PIPELINE}" == true ]]; then optimizer_layer_default=false; fi
OPTIMIZER_OFFLOAD_PER_LAYER="${OPTIMIZER_OFFLOAD_PER_LAYER:-${optimizer_layer_default}}"
TEACHER_PARAM_PREFETCH="${TEACHER_PARAM_PREFETCH:-true}"
TEACHER_PARAM_PREFETCH_MAX_MB="${TEACHER_PARAM_PREFETCH_MAX_MB:-768}"
TEACHER_FORWARD_OVERLAP="${TEACHER_FORWARD_OVERLAP:-true}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
num_gpus="${OPENMOPD_N_GPUS_PER_NODE:-1}"
train_batch_size="${OPENMOPD_TRAIN_BATCH_SIZE:-${num_gpus}}"
gpu_memory_limit_gb="${OPENMOPD_GPU_MEMORY_LIMIT_GB:-24}"
kv_cache_mb="${OPENMOPD_KV_CACHE_MB:-4096}"
max_num_seqs="${OPENMOPD_MAX_NUM_SEQS:-8}"
val_batch_size="${OPENMOPD_VAL_BATCH_SIZE:-8}"
val_reward_workers="${OPENMOPD_VAL_REWARD_WORKERS:-8}"
val_before_train="${OPENMOPD_VAL_BEFORE_TRAIN:-true}"
# Keep r2.sh's full-epoch default instead of the profiler's four-step capture.
num_steps="${OPENMOPD_NUM_STEPS:-null}"

for var in SHARE_STUDENT_WEIGHTS BF16_STUDENT_WEIGHTS OPTIMIZER_OFFLOAD_PER_LAYER \
  TEACHER_PARAM_PREFETCH TEACHER_FORWARD_OVERLAP STUDENT_TEACHER_PIPELINE PIPELINE_OVERLAP \
  TEACHER_LAYER_PIPELINE PIPELINE_GRADIENT_OFFLOAD val_before_train; do
  [[ "${!var}" == true || "${!var}" == false ]] || die "${var} must be true or false"
done
for var in num_gpus train_batch_size \
  TEACHER_PARAM_PREFETCH_MAX_MB CUDA_DEVICE_MAX_CONNECTIONS CPU_OPTIMIZER_THREADS \
  gpu_memory_limit_gb kv_cache_mb max_num_seqs val_batch_size val_reward_workers; do
  [[ "${!var}" =~ ^[1-9][0-9]*$ ]] || die "${var} must be a positive integer"
done
((gpu_memory_limit_gb > 1 && kv_cache_mb < (gpu_memory_limit_gb - 1) * 1024)) || \
  die "GPU budget must exceed 1 GiB plus the KV cache"
((num_gpus == 1)) || die "The GPU memory budget currently requires one GPU"
cudagraph_sizes_csv=1
for ((capture_size = 2; capture_size < max_num_seqs; capture_size *= 2)); do
  cudagraph_sizes_csv+=",${capture_size}"
done
if ((max_num_seqs > 1)); then cudagraph_sizes_csv+=",${max_num_seqs}"; fi
((train_batch_size % num_gpus == 0)) || die "Train batch size must be divisible by GPU count"
if [[ "${TEACHER_FORWARD_OVERLAP}" == true ]] && ((num_gpus != 1 || train_batch_size != 1)); then
  die "Teacher forward overlap requires one GPU and batch size 1; set TEACHER_FORWARD_OVERLAP=false"
fi

[[ "${num_steps}" == null || "${num_steps}" =~ ^[1-9][0-9]*$ ]] || \
    die "OPENMOPD_NUM_STEPS must be a positive integer"

python_bin="$(resolve_executable "${python_bin}")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V1=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export PATH="$(dirname "${python_bin}"):${PATH}"
export PYTHONPATH="${repo_root}/training/verl${PYTHONPATH:+:${PYTHONPATH}}"

cmd=("${python_bin}" -m verl.trainer.main_ppo
  algorithm.adv_estimator=token_reward_direct
  "data.train_files=$(hydra_string "${model_root}/data/rl_prompt_mix/train.parquet")"
  "data.val_files=$(hydra_string "${model_root}/data/rl_prompt_mix/eval.parquet")"
  data.train_batch_size="${train_batch_size}" data.max_prompt_length=1024 data.max_response_length=4096
  data.val_batch_size="${val_batch_size}"
  data.filter_overlong_prompts=True data.truncation=error
  "actor_rollout_ref.model.path=$(hydra_string "${model_root}/MixSFT")"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.share_weights="${SHARE_STUDENT_WEIGHTS}"
  actor_rollout_ref.rollout.gpu_memory_limit_gb="${gpu_memory_limit_gb}"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=$((kv_cache_mb * 1024 * 1024))"
  actor_rollout_ref.rollout.max_num_seqs="${max_num_seqs}"
  actor_rollout_ref.rollout.cudagraph_capture_sizes="[${cudagraph_sizes_csv}]"
  actor_rollout_ref.rollout.max_num_batched_tokens=5120
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
  "custom_reward_function.path=$(hydra_string "${repo_root}/training/verl/verl/utils/reward_score/opd_val_dispatch.py")"
  custom_reward_function.name=reward_func
  reward_model.micro_batch_size_per_gpu=1 reward_model.enable=True
  "reward_model.model.path=$(hydra_string "${model_root}/Math")"
  reward_model.model.input_tokenizer=null
  +reward_model.reward_kwargs.compute_true_reward=false
  +reward_model.reward_kwargs.distributed_validation=true
  +reward_model.reward_kwargs.distributed_num_shards="${val_reward_workers}"
  +reward_model.reward_kwargs.distributed_cpus_per_shard=1
  +reward_model.reward_kwargs.distributed_max_workers_per_shard=1
  '+mt_opd.teacher_domains=[math,code,if]' +mt_opd.n_additional_teachers=2
  trainer.n_gpus_per_node="${num_gpus}" trainer.nnodes=1 trainer.total_epochs=1
  trainer.total_training_steps="${num_steps}" trainer.val_before_train="${val_before_train}"
  trainer.resume_mode=disable
  "trainer.default_local_dir=$(hydra_string "${checkpoint_dir}")"
  trainer.project_name=OpenOPD-local trainer.experiment_name=mt-opd-local
  "trainer.logger=['console','tensorboard']"
  "+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS='${CUDA_DEVICE_MAX_CONNECTIONS}'"
  +mt_reward_model_1.enable=True
  "+mt_reward_model_1.model.path=$(hydra_string "${model_root}/Code")"
  +mt_reward_model_1.model.input_tokenizer=null
  +mt_reward_model_1.model.use_remove_padding=True
  +mt_reward_model_1.model.fsdp_config.param_offload=True
  +mt_reward_model_2.enable=True
  "+mt_reward_model_2.model.path=$(hydra_string "${model_root}/IF")"
  +mt_reward_model_2.model.input_tokenizer=null
  +mt_reward_model_2.model.use_remove_padding=True
  +mt_reward_model_2.model.fsdp_config.param_offload=True
)
if [[ "${BF16_STUDENT_WEIGHTS}" == true && "${SHARE_STUDENT_WEIGHTS}" != true ]]; then
  cmd+=(actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.bf16_optimizer
    actor_rollout_ref.actor.optim.optimizer=BF16StochasticAdamW)
fi
cmd+=("$@")
if [[ "${dry_run}" == true ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi
exec "${cmd[@]}"
