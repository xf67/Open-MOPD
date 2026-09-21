#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
SHARE_STUDENT_WEIGHTS="${SHARE_STUDENT_WEIGHTS:-false}"
BF16_STUDENT_WEIGHTS="${BF16_STUDENT_WEIGHTS:-false}"

# Shared weights already force this configuration in fsdp_workers.py.  These
# overrides make the same BF16-parameter/FP32-optimizer path available to the
# non-shared baseline without changing its default behavior.
student_training_args=()
if [[ "${BF16_STUDENT_WEIGHTS}" == "true" && "${SHARE_STUDENT_WEIGHTS}" != "true" ]]; then
    student_training_args+=(
        actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
        actor_rollout_ref.actor.optim.optimizer_impl=verl.utils.bf16_optimizer
        actor_rollout_ref.actor.optim.optimizer=BF16StochasticAdamW
    )
fi

python3 -m verl.trainer.main_ppo \
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
trainer.default_local_dir=/home/xxf/Distill/Open-MOPD/output/checkpoints \
trainer.project_name=OpenOPD-local \
trainer.experiment_name=mt-opd-local \
trainer.logger=\[\'console\'\] \
+mt_reward_model_1.enable=True \
+mt_reward_model_1.model.path=/home/xxf/Distill/models/OPD/Code \
+mt_reward_model_1.model.input_tokenizer=null \
+mt_reward_model_1.model.use_remove_padding=True \
+mt_reward_model_1.model.fsdp_config.param_offload=True \
# +mt_reward_model_2.enable=True \
# +mt_reward_model_2.model.path=/home/xxf/Distill/models/OPD/IF \
# +mt_reward_model_2.model.input_tokenizer=null \
# +mt_reward_model_2.model.use_remove_padding=True \
# +mt_reward_model_2.model.fsdp_config.param_offload=True
