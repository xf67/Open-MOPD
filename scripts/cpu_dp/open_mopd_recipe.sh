#!/usr/bin/env bash
# Open-MOPD stage-3 recipe (arXiv 2608.19098 Table 8, "Open-MOPD" column) on one node with
# CPU data parallel (rollout.cpu_dp) in large-batch mode. Wraps r2.sh.
#
#   global batch 1024 (256 rollouts per GPU on 4 GPUs), mini-batch 256 -> 4 synchronous
#   updates per step, lr 1.5e-6 constant, clip 0.2/0.28, no KL / entropy loss,
#   dense top-16 reward with student_p weights restricted to the routed teacher's 0.99
#   nucleus (top_p_intersec), 3-teacher hard routing, M1 equal token share, M2
#   gap-following (alpha=1, multiply, unanchored mean), M4 reward refresh,
#   math:code:if = 2:2:1 prompts, prompt 2048 / response 16384 (math, code) / 2048 (if).
#
# Usage (conda env mopd active):
#   bash scripts/cpu_dp/open_mopd_recipe.sh [--dry-run] [extra Hydra overrides]
# Common environment overrides:
#   OPENMOPD_NUM_STEPS=200  RUN_NAME=open-mopd-recipe  RESUME_MODE=auto
#   SCORING_TOKEN_BUDGET=24576  TRAIN_TOKEN_BUDGET=24576  ACCUM_RING_SLOTS=4
#   MAX_PROMPT_LENGTH=2048  MAX_RESPONSE_LENGTH=16384  SHORT_RESPONSE_LENGTH=2048  MINI_BATCH_SIZE=256
#   OPENMOPD_VAL_BEFORE_TRAIN=false  TEST_FREQ=20  SAVE_FREQ=10
#   MAX_CKPT_TO_KEEP=2             (older checkpoints are deleted; null = keep all, ~60 GB each)
#   DOMAIN_MAX_TOKENS=512,512,512  (training max_tokens for math,code,if; the padded width stays
#                                   MAX_RESPONSE_LENGTH: a fast full-batch host-memory check)
#   OPENMOPD_PHASE_LOG=1           (driver prints phase boundaries with its RSS and MemAvailable)
#   OPENMOPD_TOPK_ANALYSIS_ROWS=256 (rows sampled for the val-topk diagnostics; 0 = all rows, +~12 GB host)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
run_name="${RUN_NAME:-open-mopd-recipe}"
base="${OPENMOPD_OUTPUT_ROOT:-/mnt/sda1/xxf}"
log_dir="${base}/cpu_dp_logs"
mkdir -p "${log_dir}" "${base}/ray_tmp" "${base}/tensorboard/${run_name}" "${base}/ckpt"

export RAY_TMPDIR="${RAY_TMPDIR:-${base}/ray_tmp}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${base}/tensorboard/${run_name}}"
export OPENMOPD_N_GPUS_PER_NODE="${OPENMOPD_N_GPUS_PER_NODE:-4}"
export OPENMOPD_TRAIN_BATCH_SIZE="${OPENMOPD_TRAIN_BATCH_SIZE:-1024}"
export OPENMOPD_NUM_STEPS="${OPENMOPD_NUM_STEPS:-200}"
export OPENMOPD_KV_CACHE_MB="${OPENMOPD_KV_CACHE_MB:-10240}"
export OPENMOPD_MAX_NUM_SEQS="${OPENMOPD_MAX_NUM_SEQS:-64}"
export OPENMOPD_VAL_BEFORE_TRAIN="${OPENMOPD_VAL_BEFORE_TRAIN:-false}"
# All 128 validation prompts in one generate call (r2.sh's default of 8 would serialize
# 16 calls, each bounded by its longest 16K-token response).
export OPENMOPD_VAL_BATCH_SIZE="${OPENMOPD_VAL_BATCH_SIZE:-128}"
export OPENMOPD_CHUNKED_HEAD="${OPENMOPD_CHUNKED_HEAD:-2048}"
export OPENMOPD_RELEASE_CACHE="${OPENMOPD_RELEASE_CACHE:-1}"
export CPU_DP=true LARGE_BATCH=true CPU_ADAM_ASYNC=false STUDENT_TEACHER_PIPELINE=true
export TOP_K_STRATEGY=top_p_intersec REWARD_WEIGHT_MODE=student_p LOG_PROB_TOP_K=16
export TOKEN_BATCHING_ENABLED=false

mini_batch="${MINI_BATCH_SIZE:-256}"
max_response="${MAX_RESPONSE_LENGTH:-16384}"   # math / code (and the padded width)
short_response="${SHORT_RESPONSE_LENGTH:-2048}" # if
max_prompt="${MAX_PROMPT_LENGTH:-2048}"
mini_batches=$((OPENMOPD_TRAIN_BATCH_SIZE / mini_batch))
((OPENMOPD_TRAIN_BATCH_SIZE % mini_batch == 0)) || { echo "batch must be a multiple of the mini-batch" >&2; exit 2; }
echo "Open-MOPD recipe: batch ${OPENMOPD_TRAIN_BATCH_SIZE}, mini-batch ${mini_batch} (${mini_batches} updates/step)," \
     "${OPENMOPD_NUM_STEPS} steps, ${OPENMOPD_N_GPUS_PER_NODE} GPUs, run ${run_name}"

overrides=(
  # lengths: per-domain training limit, one validation limit (response_length)
  data.max_prompt_length="${max_prompt}" data.max_response_length="${max_response}"
  actor_rollout_ref.rollout.max_model_len=$((max_prompt + max_response))
  '+data.response_length_domains=[math,code,if]'
  "+data.response_length_values=[${DOMAIN_MAX_TOKENS:-${max_response},${max_response},${short_response}}]"
  # rollout engine
  # verl's vLLM wrapper requires max_num_batched_tokens >= max_model_len with chunked prefill
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-$((max_prompt + max_response))}"
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0
  # batch / optimizer / PPO
  actor_rollout_ref.actor.ppo_mini_batch_size="${mini_batch}" actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.optim.lr=1.5e-6 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
  actor_rollout_ref.actor.optim.lr_warmup_steps=-1
  actor_rollout_ref.actor.grad_clip=1.0
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  # reward: top-16, teacher nucleus 0.99
  +actor_rollout_ref.rollout.top_p_intersec_p=0.99
  # MT-OPD M1 equal token share, M2 gap-following (m_d / m_ref, multiply, unanchored mean)
  '+mt_opd.target_share_domains=[math,code,if]' '+mt_opd.target_share_values=[0.3333,0.3333,0.3334]'
  +mt_opd.normalize_reward_scale=1.0 +mt_opd.reward_scale_direction=multiply
  +mt_opd.reward_scale_stat=mean +mt_opd.reward_scale_anchored=false
  # data mix 2:2:1 over the full parquet (84 batches of 1024 per pass)
  data.sampler.class_path=pkg://verl.utils.dataset.domain_weighted_sampler
  data.sampler.class_name=DomainWeightedSampler
  +data.domain_weights.math=2 +data.domain_weights.code=2 +data.domain_weights.if=1
  data.dataloader_num_workers=0 data.seed=1
  # memory
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.rollout.scoring_token_budget="${SCORING_TOKEN_BUDGET:-24576}"
  actor_rollout_ref.rollout.train_token_budget="${TRAIN_TOKEN_BUDGET:-24576}"
  actor_rollout_ref.rollout.accum_ring_slots="${ACCUM_RING_SLOTS:-4}"
  # trainer: validation (128 fixed random prompts), checkpoints, logging
  trainer.total_epochs="${TOTAL_EPOCHS:-3}"
  trainer.test_freq="${TEST_FREQ:-20}"
  data.val_max_samples=128 data.validation_shuffle=True
  trainer.save_freq="${SAVE_FREQ:-10}" trainer.max_actor_ckpt_to_keep="${MAX_CKPT_TO_KEEP:-2}"
  trainer.resume_mode="${RESUME_MODE:-auto}"
  "trainer.default_local_dir=${CKPT_DIR:-${base}/ckpt/${run_name}}"
  "trainer.logger=['console','tensorboard']"
  trainer.experiment_name="${run_name}"
)

log_file="${LOG_FILE:-${log_dir}/${run_name}.log}"
if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  exec bash "${repo_root}/r2.sh" --dry-run "${overrides[@]}" "$@"
fi
echo "log: ${log_file}"
bash "${repo_root}/r2.sh" "${overrides[@]}" "$@" 2>&1 | tee -a "${log_file}"
exit "${PIPESTATUS[0]}"
