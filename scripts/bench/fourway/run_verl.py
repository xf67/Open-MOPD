"""Launch one verl-based speed case inside the open-mopd container.

METHOD=ours : Open-MOPD + our system (shared student weights, student/teacher
              slot pipeline, progressive gradient offload, async CPU Adam,
              vLLM CUDA graphs), optional dynamic token batching.
METHOD=orig : the same code base with every one of our switches turned off,
              i.e. the original Open-MOPD/verl execution path: separate vLLM
              student copy + weight sync, FSDP actor with fp32 params and
              torch AdamW (optimizer state offloaded by verl's native FSDP
              offload so that it fits in 48 GB), teachers offloaded by verl's
              native FSDP param offload, vLLM CUDA graphs on (verl default).
"""
import json
import os
from pathlib import Path
import runpy
import sys

method = os.environ['METHOD']
assert method in ('ours', 'orig', 'zero')
bs = int(os.environ['BATCH'])            # prompts per rollout chunk
dyn = int(os.environ.get('DYN_MIN_TOKENS', '0'))  # 0 = fixed batching
micro = int(os.environ.get('MICRO', str(bs)))
out = Path(os.environ['RUN_DIR'])
args = json.loads(Path('/study/base-argv.json').read_text())
args = [a for a in args if not a.lstrip('+').startswith('ray_kwargs.ray_init.runtime_env.')]


def setarg(key, value):
    global args
    args = [a for a in args if a.split('=', 1)[0].lstrip('+') != key.lstrip('+')]
    args.append(key + '=' + str(value))


ours = method == 'ours'
seqs = max(bs, 1)
caps = [c for c in (1, 2, 4, 8, 16, 32) if c <= seqs]
if caps[-1] != seqs:
    caps.append(seqs)
common = {
    'data.train_files': '/study/data/train.parquet',
    'data.val_files': '/study/data/eval.parquet',
    'data.train_batch_size': bs,
    'data.train_max_samples': -1, 'data.dataloader_num_workers': 0,
    'data.max_response_length': 1024, 'data.max_prompt_length': 1024,
    'data.shuffle': 'false', 'data.seed': 42,
    'data.token_batching.enabled': str(dyn > 0).lower(),
    'data.token_batching.min_tokens': dyn if dyn > 0 else 8192,
    'data.token_batching.max_batch_size': 64,
    'actor_rollout_ref.actor.ppo_mini_batch_size': bs,
    'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': micro,
    'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': micro,
    'reward_model.micro_batch_size_per_gpu': micro,
    '+mt_reward_model_1.micro_batch_size_per_gpu': micro,
    '+mt_reward_model_2.micro_batch_size_per_gpu': micro,
    'actor_rollout_ref.rollout.max_model_len': 2048,
    'actor_rollout_ref.rollout.max_num_batched_tokens': 4096,
    'actor_rollout_ref.rollout.max_num_seqs': seqs,
    'actor_rollout_ref.rollout.cudagraph_capture_sizes': '[' + ','.join(map(str, caps)) + ']',
    'actor_rollout_ref.rollout.enforce_eager': 'false',
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes': 4294967296,
    'actor_rollout_ref.model.enable_gradient_checkpointing': 'true',
    'actor_rollout_ref.actor.optim.lr': '1e-6',
    'actor_rollout_ref.actor.optim.lr_scheduler_type': 'constant',
    'actor_rollout_ref.actor.optim.lr_warmup_steps': 0,
    'actor_rollout_ref.actor.loss_agg_mode': 'token-mean',
    'trainer.total_training_steps': 1000000, 'trainer.total_epochs': 100,
    'trainer.save_freq': -1,
    'trainer.val_before_train': 'false', 'trainer.test_freq': -1,
    'trainer.step_memory.enabled': 'false',
    'trainer.default_local_dir': str(out / 'checkpoints'),
    'trainer.experiment_name': out.name,
    'trainer.project_name': 'fourway-speed-a6000-20261118',
    'hydra.run.dir': str(out / 'hydra'),
    '++trainer.suite_token_budget': int(os.environ['TOKEN_BUDGET']),
    '++trainer.suite_min_steps': 1,
    '+ray_kwargs.ray_init.runtime_env.worker_process_setup_hook': 'verl.utils.speed_bench.seed_worker',
}
if ours:
    switches = {
        'actor_rollout_ref.rollout.share_weights': 'true',
        'actor_rollout_ref.rollout.student_teacher_pipeline': 'true',
        'actor_rollout_ref.rollout.teacher_forward_overlap': 'true',
        'actor_rollout_ref.rollout.teacher_layer_pipeline': 'false',
        'actor_rollout_ref.rollout.pipeline_overlap': 'true',
        'actor_rollout_ref.rollout.pipeline_gradient_offload': 'true',
        'actor_rollout_ref.rollout.cpu_adam_enabled': 'true',
        'actor_rollout_ref.rollout.cpu_adam_async': 'true',
        'actor_rollout_ref.rollout.pipeline_fp32_gradients': 'true',
        'actor_rollout_ref.rollout.teacher_param_prefetch': 'false',
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': 'true',
        'actor_rollout_ref.actor.fsdp_config.param_offload': 'false',
        'actor_rollout_ref.actor.fsdp_config.model_dtype': 'fp32',
        'actor_rollout_ref.actor.fsdp_config.use_orig_params': 'true',
        'actor_rollout_ref.rollout.gpu_memory_utilization': 0.35,
    }
else:
    switches = {
        'actor_rollout_ref.rollout.share_weights': 'false',
        'actor_rollout_ref.rollout.student_teacher_pipeline': 'false',
        'actor_rollout_ref.rollout.teacher_forward_overlap': 'false',
        'actor_rollout_ref.rollout.teacher_layer_pipeline': 'false',
        'actor_rollout_ref.rollout.pipeline_overlap': 'false',
        'actor_rollout_ref.rollout.pipeline_gradient_offload': 'false',
        'actor_rollout_ref.rollout.cpu_adam_enabled': 'false',
        'actor_rollout_ref.rollout.cpu_adam_async': 'null',
        'actor_rollout_ref.rollout.pipeline_fp32_gradients': 'false',
        'actor_rollout_ref.rollout.teacher_param_prefetch': 'false',
        # verl-native single-GPU fit: FSDP2 CPUOffloadPolicy keeps fp32 params,
        # grads and AdamW state on the host and runs the optimizer step there.
        'actor_rollout_ref.actor.strategy': 'fsdp2',
        'actor_rollout_ref.actor.fsdp_config.offload_policy': 'true',
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': 'false',
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload_per_layer': 'false',
        'actor_rollout_ref.actor.fsdp_config.param_offload': 'false',
        'actor_rollout_ref.actor.fsdp_config.model_dtype': 'fp32',
        'actor_rollout_ref.actor.fsdp_config.use_orig_params': 'false',
        'actor_rollout_ref.rollout.gpu_memory_utilization': 0.35,
    }
if method == 'zero':
    # Original Open-MOPD path + ZeRO-Offload optimizer: DeepSpeed's C++ CPU Adam on the
    # host-resident fp32 master weights (FSDP2 CPUOffloadPolicy), synchronous.
    switches.update({
        'actor_rollout_ref.actor.optim.optimizer_impl': 'verl.utils.zero_offload_adam',
        'actor_rollout_ref.actor.optim.optimizer': 'ZeroOffloadCPUAdam',
        'actor_rollout_ref.actor.optim.override_optimizer_config': 'null',
    })
for k, v in {**common, **switches}.items():
    setarg(k, v)
for key in ['RUN_DIR', 'TOKEN_BUDGET', 'WARMUP_STEPS', 'MIN_MEASURED_STEPS', 'MAX_STEPS',
            'PYTHONHASHSEED', 'OPENMOPD_PINNED_ALIGNMENT_MB']:
    setarg('+ray_kwargs.ray_init.runtime_env.env_vars.' + key, "'" + os.environ[key] + "'")
if os.environ.get('OPENMOPD_CHUNKED_HEAD'):  # chunked LM head (source_chunked only)
    setarg('+ray_kwargs.ray_init.runtime_env.env_vars.OPENMOPD_CHUNKED_HEAD', "'" + os.environ['OPENMOPD_CHUNKED_HEAD'] + "'")
(out / 'argv.json').write_text(json.dumps(args, indent=2))
is_config = '--cfg' in sys.argv
if not is_config:
    from verl.utils import speed_bench as long_observer
    long_observer.seed_worker()
sys.argv = ['verl.trainer.main_ppo', *args, *sys.argv[1:]]
runpy.run_module('verl.trainer.main_ppo', run_name='__main__')
if not is_config:
    long_observer.finish_run()
