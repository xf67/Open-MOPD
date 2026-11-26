# CPU data parallel (single node, no NVLink)

`actor_rollout_ref.rollout.cpu_dp=true` runs MT-OPD data parallel over the GPUs of one
node without any GPU-to-GPU traffic in the training path. `r2.sh` turns it on whenever
`OPENMOPD_N_GPUS_PER_NODE > 1`:

```bash
export RAY_TMPDIR=/mnt/sda1/xxf/ray_tmp   # G4090j: root fs is full
OPENMOPD_N_GPUS_PER_NODE=4 bash r2.sh     # train batch = 4 (one sequence per GPU)
```

## Design

* **Rank-local FSDP.** Every model (student and teachers) is wrapped by FSDP1 over a
  process group that contains only its own rank, so FSDP sees world size 1 (NO_SHARD)
  and never communicates. All single-GPU machinery (shared actor/vLLM weights, teacher
  forward overlap, student/teacher slot pipeline, early gradient offload) runs unchanged
  per rank. Each rank runs its own vLLM engine (TP=1) on its share of the prompts.
* **One host copy** (`verl/utils/shared_host.py`). Named files in
  `/dev/shm/openmopd-<run>/`, mapped `MAP_SHARED` and `cudaHostRegister`-ed by every
  process that DMAs them, so async H2D/D2H work without private pinned replicas.
  Pages are placed with `mbind` before first touch:
  * teachers and the student's BF16 weights: interleaved across NUMA nodes (every GPU reads them);
  * per-rank gradient staging: on that rank's GPU node (D2H stays on-socket);
  * FP32 master / Adam moments / reduced gradient: split by parameter blocks, one block per node.
  The driver creates the namespace and removes it when training returns.
* **CPU gradient reduction + one CPUAdam** (`verl/utils/dp_cpu_adam.py`,
  `SharedCPUAdamPipeline`). Ranks copy layer gradients into their staging during
  backward (existing hooks). The owner rank (`cpu_dp_owner_rank`) runs one worker process
  per NUMA node, pinned to that node: it sums the four staging buffers for its block (rank
  order, deterministic), returns its squared norm, then applies the global clip
  coefficient and DeepSpeed CPUAdam, and writes BF16 weights into the shared copy. Every
  rank restores its GPU weights from that copy, so replicas are bitwise identical by
  construction. A non-finite reduced norm skips the update on all ranks.
* **Loss** (`core_algos.global_loss_denominator`). Ranks' gradients are summed, so each
  rank's token-mean / seq-mean denominator is replaced by the sum over ranks (one gloo
  scalar all-reduce per `agg_loss` call): the result is exactly the global-batch loss.
* **Memory.** Each rank's private teacher copy is released as soon as the shared copy is bound
  (parameter views re-pointed), and teacher lazy-init (which allocates a transient model-sized
  pinned gradient) runs one rank at a time.

Not supported with `cpu_dp`: FSDP2, LoRA, token batching, sharded or GPU-offloaded actor,
non-CPU optimizer backends, multi-node.

## Knobs

| variable / key | default | |
|---|---|---|
| `CPU_DP` / `rollout.cpu_dp` | true if >1 GPU | |
| `CPU_DP_OWNER_RANK` | 0 | rank whose process hosts the optimizer workers |
| `CPU_DP_NODE_THREADS` | max(cpu_optimizer_threads, 16) | threads per NUMA optimizer worker |
| `OPENMOPD_CPU_DP_BIND_NUMA` | 1 | pin each GPU worker's CPUs to its GPU's node |
| `OPENMOPD_CPU_DP_TEACHER_POLICY` | interleave | or `node:<n>` |
| `OPENMOPD_CPU_DP_CHECK_EVERY` | 0 | every N updates, verify replicas are identical (`cpu_dp/replicas_identical`) |

## Tests and tools

```bash
python training/verl/tests/cpu_dp/test_shared_host.py --gpus 0,1,2,3   # sharing, NUMA, H2D/D2H bandwidth
python training/verl/tests/cpu_dp/test_dp_adam.py --gpus 0,1,2,3       # vs FP32 reference, replicas, NaN skip
python scripts/cpu_dp/bench_reduce.py --numel 1.5e9 --threads 8,16     # reduction kernel
python scripts/cpu_dp/mem_sampler.py LOG 5                             # per-process host memory
python scripts/cpu_dp/summarize_run.py LOG [LOG ...]                   # step metrics summary
```

## Large-batch mode (Open-MOPD recipe: batch 1024, mini-batch 256, 16K responses)

`actor_rollout_ref.rollout.large_batch=true` (`LARGE_BATCH=true` in `r2.sh`; requires
`cpu_adam_async=false` and the student/teacher pipeline) lifts the one-sequence-per-GPU limits.
Default single-GPU and cpu_dp behaviour is unchanged when it is off.

* **Scoring** (`fsdp_workers._score_large_batch`): each rank's batch is cut into contiguous
  micro-batches with `rows x (longest prompt + longest response) <= rollout.scoring_token_budget`
  (padded positions: the primary teacher's forward-overlap path uses the padded layout).
  Each micro-batch is compacted to its own columns, scored by one S->T0->T1->T2->S slot cycle
  (`_score_cycle`, the former single-cycle body), moved to host and re-padded to the batch's
  response width. Teacher top-k ids/log-probs are not returned; student top-k ids travel as int32,
  masks as bool. `OPENMOPD_SCORING_CHECK=1` re-scores the rank batch in one cycle and logs
  `scoring_check/*` (test 4).
* **FP32 host accumulation** (`dp_cpu_adam.SharedCPUAdamPipeline(accumulate=True)`): the per-rank
  stage is an FP32 arena `student_accum{r}` (node-local, not CUDA-registered). Backward hooks copy
  each layer's BF16 gradient into a free slot of a pinned landing ring (`rollout.accum_ring_slots`,
  each slot = largest FSDP flat parameter), and one accumulate thread per rank copies (first
  micro-batch of the mini-batch) or adds the slot into the accumulator. `step()` drains the thread,
  zeroes never-written parameters and signals the owner, whose node workers reduce the four FP32
  accumulators in rank order, clip by the global norm and run CPUAdam (unchanged). A layer whose
  hook fires twice in one backward raises (checks gradient checkpointing).
* **Several synchronous updates per step** (`dp_actor.update_policy`): mini-batches of
  `ppo_mini_batch_size / world` sequences per rank, micro-batches of at most
  `rollout.train_token_budget` real tokens (remove-padding layout), each compacted to its columns.
  Per mini-batch one gloo all-reduce gives the global token / sequence counts of `response_mask`
  and `response_mask * format_mask`; `global_loss_denominator(fixed=...)` then makes the sum of all
  micro-batch losses on all ranks the exact global mini-batch token-mean (no collective inside the
  micro-batch loop). The PPO ratio uses the scoring-time student top-k log-probs (clip 0.2/0.28 act
  on updates 2..K) and the M4 reward refresh runs on every update. `actor/grad_norm` is the owner's
  global norm; `actor/grad_norm_local` the root-sum-square of this rank's micro-batch norms.
* **top_p_intersec**: every teacher returns its nucleus mask over the student top-k ids; the driver
  keeps the routed teacher's mask per sample (`mt_opd.select_routed_teacher_masks`). The MT-OPD
  reward uses the released `intersection` formula on that mask (the released MT-OPD reward has no
  `top_p_intersec` branch; the released Direct-OPD/ExOPD rewards treat it the same way), and the
  refresh applies the same fixed mask (`refresh_opd_advantage(valid_mask=...)`).
* **Per-domain response limit**: `+data.response_length_domains=[...] +data.response_length_values=[...]`
  set a per-prompt vLLM `max_tokens` during training (validation keeps one limit); responses are
  still padded to `data.max_response_length`.
* **GPU budget while vLLM sleeps**: `gpu_memory_limit_gb` caps PyTorch's allocator, and vLLM's sleep
  allocator unmaps the KV cache's pages but leaves its blocks counted by PyTorch, so training got
  ~10 GiB less than the budget (the first full-size trial OOMed on a 24,576-token micro-batch with
  12.4 GiB physically in use). In large-batch mode `update_actor` runs inside
  `memory_utils.unbacked_memory_allowance()`, which raises the cap by `memory_reserved - device used`
  (a lower bound on the unbacked bytes, so backed memory stays within the budget) and restores it on
  exit; the cached blocks are returned before vLLM wakes (`OPENMOPD_RELEASE_CACHE=1`, set by the
  recipe). Metrics: `memory/update_kv_allowance_gib`, `memory/update_backed_gib_max`.
* **Driver host memory** (`ray_trainer._release_step_data`): at batch 1024 x 16K x top-16 every
  [rows, T, K] tensor is 1 GiB. The driver holds the batch (~10 GB), the scoring output (~6 GB,
  including keys already popped from the batch) and ~16 metric temporaries of the top-k analysis
  (`student_p`, `adv_for_max`, ...), all bound to fit() locals that are rebound only partway into the
  next step, so two steps' data coexisted (driver RSS 38 GB at the start of step 2 vs 8 GB at step 1;
  host MemAvailable down to 13.5 GiB of 251). In large-batch mode, right before the update_actor
  dispatch everything the rest of the step does not read is freed (DataProtos other than `batch`,
  [rows, ...] tensors >= 64 MiB not sharing storage with `batch`), and at the end of the step all of it.
  The top-k diagnostics at the end of `adv` (`val-topk/*`, `val-extrema/*`, logged every step) read
  every 4th row in large-batch mode (`OPENMOPD_TOPK_ANALYSIS_ROWS=256` rows, `0` = all rows;
  `mt_opd/topk_analysis_rows` records it): their temporaries were the step's host peak (+16 GB on the
  driver, ~224-227 GiB of 251 in use in the full-size trial). They are diagnostics only; the loss,
  rewards and advantages use every row.
  `OPENMOPD_PHASE_LOG=1` prints phase boundaries with driver RSS / MemAvailable and the locals still
  holding >= 64 MiB.

Launcher: `scripts/cpu_dp/open_mopd_recipe.sh` (all Table-8 settings; TensorBoard under
`/mnt/sda1/xxf/tensorboard/<run>`, checkpoints under `/mnt/sda1/xxf/ckpt/<run>`, log tee'd to
`/mnt/sda1/xxf/cpu_dp_logs/<run>.log`).

```bash
python training/verl/tests/cpu_dp/test_dp_accumulate.py --gpus 0,1,2,3  # test 1: accumulation x updates vs FP32 reference
python training/verl/tests/cpu_dp/test_large_batch_loss.py              # test 2: global loss over micro-batches x ranks (CPU)
python training/verl/tests/cpu_dp/test_top_p_intersec.py                # test 3: nucleus mask, routing, reward, refresh (CPU)
```

Fast checks before a full-size run (~15-20 min each, so a failure shows up before an hour-long
rollout does):

```bash
# full-batch host memory, dispatch and checkpoint: batch 1024 at the full padded width, generation capped at 512 tokens
DOMAIN_MAX_TOKENS=512,512,512 OPENMOPD_NUM_STEPS=2 SAVE_FREQ=2 TEST_FREQ=-1 RESUME_MODE=disable OPENMOPD_PHASE_LOG=1 \
  RUN_NAME=lb_hostmem bash scripts/cpu_dp/open_mopd_recipe.sh
# GPU memory at the full training token budget (24,576-token micro-batches): batch 128, 8K responses
OPENMOPD_TRAIN_BATCH_SIZE=128 MINI_BATCH_SIZE=64 MAX_RESPONSE_LENGTH=8192 OPENMOPD_NUM_STEPS=2 SAVE_FREQ=-1 TEST_FREQ=-1 \
  RESUME_MODE=disable RUN_NAME=lb_memfix bash scripts/cpu_dp/open_mopd_recipe.sh
```

Test results (G4090j, 2026-11-23; logs in `/mnt/sda1/xxf/cpu_dp_logs/large_batch/`):

| test | result |
|---|---|
| 1 accumulation x updates (4 GPUs, 3 micro-batches x 4 updates, ring of 2, NaN in one micro-batch of one rank) | masters vs a one-step reference built from the exact BF16 micro-batch grads the pipeline received (FP32 rank-order sum, FP64 norm, clip, AdamW from the pipeline's previous state): max abs error 1.2e-7; replicas bitwise identical every update; the NaN update is skipped on every rank. (The end-to-end trajectory vs a plain non-FSDP model drifts on single elements after a few updates: BF16 kernel differences, amplified by Adam on near-zero gradients; informational.) |
| 2 loss over micro-batches x ranks (4 gloo ranks, ragged/empty responses, format mask, top-p-masked 3D advantages) | loss and gradient equal to the single-batch `compute_policy_loss_vanilla` exactly (max abs grad error 0) for token-mean, seq-mean-token-sum, seq-mean-token-mean |
| 3 top_p_intersec | row-blocked teacher nucleus mask == released implementation bit for bit (and == an FP64 exclusive-cumsum nucleus); routed mask/log-probs == the domain teacher's; MT-OPD reward == per-sample intersection formula on the routed teacher (float rounding only); refresh with the mask == reward x response_mask exactly |
| 4 scoring loop, N micro-batches vs 1 cycle (in-run, `OPENMOPD_SCORING_CHECK=1`, 7-8 cycles per rank) | widths/shapes identical; sampled-token log-prob mean abs diff 0.017-0.019 (max 0.49), entropy 0.013; student top-16 ids 82% positionally equal (near-tie order swaps); teacher log-probs on matching ids mean abs 0.075; nucleus masks 99.3-99.4% equal. These are BF16 batch-shape differences (different packed lengths change the GEMM shapes), not layout errors (a misalignment would give ~0% id agreement). |
| 7 checkpoint save + resume (batch 32, mini-batch 8, save at step 2, resume, train-token budget 2048 to force several micro-batches) | owner optimizer restored (version 8, seq 8, Adam steps 8, lr); replicas identical after the resumed updates; published version 12; resumed step 3 drew the same prompts as the original step 3 (`DomainWeightedSampler` resumes its position inside the epoch through StatefulDataLoader. It has no `state_dict`, so its epoch counter, which seeds each epoch's order, is not saved: a resume in epoch 2 or 3 would presumably reuse epoch 1's order; not tested across an epoch boundary). Save: ~200 s for 60 GB; load: ~5 min (mmap). |
| GPU budget fix (`lb_memfix`: batch 128, mini-batch 64, 8K responses, train budget 24,576, 2 steps) | passes. PyTorch-counted update peak 24.4 GiB, above the old 22.53 GiB cap (the unpatched code OOMs here); backed peak <= 15.2 GiB; slept-KV allowance 8.8-9.3 GiB; nvidia-smi peak 20.5 GiB of 23.5; replicas identical; 5.0-5.2 min per step |
| full-batch host memory (`lb_hostmem*`: batch 1024, padded width 16384, generation capped at 512 tokens, 2-3 steps) | before the release: driver RSS 8 GB at step 1 vs 38 GB at step 2, MemAvailable down to 13.5-21 GiB. With the end-of-step release only: every step starts at 16 GB, MemAvailable 17.7 GiB during the dispatch. With the pre-dispatch release as well (`lb_hostmem4`): driver 22.6 GB at the dispatch, lowest MemAvailable 30-31 GiB (~220 GiB in use; the remaining peak is the top-k metric analysis at the end of `adv`); replicas identical, 4 updates per step. Full-size checkpoint save: 3.7-3.9 min, no host peak |
| 5 long sequence + gradient checkpointing | covered by test 8: 16K-token responses train on the 24 GB cards (GPU0/1) with 24,576-token micro-batches, nvidia-smi peak 21.1 GiB |
| 6 equivalence vs single GPU with accumulation | not run: large-batch accumulation is implemented for cpu_dp only (the single-GPU path has no accumulation mode to compare against) |
| 8 full-size trial (`lb_trial2`: the exact recipe, validation first, 2 steps, no checkpoint; profile in `/mnt/sda1/xxf/cpu_dp_logs/profile_report.md`) | passes. Validation 6.7 min; step 80.4 / 81.9 min = rollout 48.7-49.7 (slowest vs fastest rank 3.7-4.6 min apart) + scoring 17.6-17.9 + reward/adv 1.5 + update_actor 12.7-12.8 (4 updates, ~37.6 micro-batches per rank per update, CPU update 2.5 s). Math/code responses mean 11.7-11.8K tokens (28% at 16,384), IF 0.4K. Replicas identical, no skipped update, versions in step (policy lag 0). GPU: nvidia-smi peak 21.1 GiB (GPU0), update backed <= 15.8 GiB. Host: peak 225.9 GiB in use (sampler) / 228.5 GiB (0.5 s telemetry) of 251 at the end of `adv`, the plan's ~225 abort line -> top-k diagnostics sampled (next row). /dev/shm 131.8 GiB. 200 steps ~11.4 days incl. 10 validations + 20 checkpoints |
| top-k diagnostics on 256 sampled rows (`lb_topk`: as `lb_hostmem`, 2 steps, no checkpoint; compare `lb_hostmem4`) | driver at the end of `adv` 36.0 GB vs 46.1-46.3; host telemetry peak 211-212 GiB vs 219-221 (sampler: lowest MemAvailable 38 vs 30-31 GiB); `adv` 7-9 s vs 19-23 s; the sampled metrics match the full-batch ones within step-to-step noise (overlap 0.322 / 0.296 vs 0.315 / 0.306, student p-sum 0.969 / 0.979 vs 0.971 / 0.976); replicas identical. Expected full-recipe host peak ~219 GiB (from 228.5) |

## Results on G4090j (4x RTX 4090, all treated as 24 GiB; 2026-11-22)

Same r2.sh workload, 30 steps each, validation off, steps 2..30 averaged
(logs: `/mnt/sda1/xxf/cpu_dp_logs/{base1_30,dp4_30}.log`).

| | 1 GPU (dev 973458c) | 4 GPU cpu_dp |
|---|---|---|
| sequences / step | 1 | 4 |
| mean step time | 24.7 s | 33.3 s |
| sequences / s | 0.041 | 0.120 (**2.96x**) |
| prompt+response tokens / s | 121 | 302 (**2.49x**) |
| generation (max over ranks) | 20.6 s | 28.2 s |
| scoring (student + 3 teachers) | 1.68 s | 1.63 s |
| update_actor | 1.90 s | 2.35 s |
| CPU reduce / CPUAdam (hidden behind rollout) | – / 1.48 s | 0.95 s / 0.98 s |
| peak GPU reserved | 22.3 GB | 22.8 GB |

Replicas were verified bitwise identical every 5 updates (`OPENMOPD_CPU_DP_CHECK_EVERY=5`).
The remaining gap to 4x is the synchronous rollout: with one sequence per GPU each step
waits for the longest of four responses (often the 4096-token cap) while the other GPUs
idle. Host memory: ~92 GB in /dev/shm (one copy of teachers, student weights, optimizer
state and four staging buffers). Startup peak: 150.3 GiB of 251 GiB in use (see the overview).

## Training equivalence vs single GPU (same global batch)

Single GPU vs 4-GPU cpu_dp, both with global batch 4 sequences/step (single GPU: one
4-prompt token-batch chunk, i.e. `TOKEN_BATCHING_ENABLED=true TOKEN_BATCHING_MIN_TOKENS=1
TOKEN_BATCHING_MAX_BATCH_SIZE=4 OPENMOPD_TRAIN_BATCH_SIZE=4`), same prompt order (`data.seed=1`),
40 steps, `OPENMOPD_CHUNKED_HEAD=2048 OPENMOPD_RELEASE_CACHE=1
actor_rollout_ref.model.use_remove_padding=True` in both (4 sequences do not fit on one
24 GiB card otherwise). Rollout sampling differs between runs, so agreement is statistical.
Logs: `/mnt/sda1/xxf/cpu_dp_logs/cmp2_{1gpu,dp4}.log`, compared with `scripts/cpu_dp/compare_runs.py`.

| 10-step window means | 1-10 | 11-20 | 21-30 | 31-40 | all (diff / pooled sd) |
|---|---|---|---|---|---|
| pg_loss 1 GPU | 0.0558 | 0.0349 | 0.0317 | 0.0190 | |
| pg_loss 4 GPU | 0.0509 | 0.0316 | 0.0302 | 0.0198 | +0.11 sd |
| rm_scores_mean 1 GPU | -0.0569 | -0.0385 | -0.0317 | -0.0196 | |
| rm_scores_mean 4 GPU | -0.0520 | -0.0351 | -0.0308 | -0.0204 | -0.10 sd |
| entropy 1 GPU / 4 GPU | 0.90 / 0.94 | 0.65 / 0.69 | 0.52 / 0.51 | 0.46 / 0.44 | -0.03 sd |
| response length 1 GPU / 4 GPU | 2208 / 2148 | 2267 / 2221 | 2137 / 2132 | 1746 / 1786 | +0.02 sd |

Validation after 40 steps (fixed random 128-prompt subset of eval.parquet):

| | ifeval | LCB v5 | LCB v6 | AIME24 | AIME25 |
|---|---|---|---|---|---|
| start (same checkpoint), 1 GPU / 4 GPU | 0.720 / 0.720 | 0.053 / 0.053 | 0.125 / 0.167 | 0 / 0 | 0 / 0 |
| step 40, 1 GPU | 0.760 | 0.053 | 0.167 | 0 | 0.25 |
| step 40, 4 GPU | 0.773 | 0.053 | 0.125 | 0 | 0.25 |

Re-validating the same checkpoint twice moved ifeval by ~0.013 and AIME25 by 0.25 (a
handful of AIME prompts), so the step-40 differences are within run-to-run noise. Replicas
were identical at every check.

Note on speed at equal global batch: the single GPU generates its 4 rollouts concurrently
in one vLLM engine, so it took 41.3 s/step vs 34.1 s/step for 4 GPUs (1.2x). The 2.96x
figure above compares against the default single-GPU setting of 1 sequence per step.
With the chunked head, `OPENMOPD_RELEASE_CACHE=1` is required (otherwise PyTorch keeps
~22.5 GB cached after training and the vLLM KV cache cannot be re-allocated).

## vs the original Open-MOPD / verl FSDP path on 4 GPUs

`orig` = the `scripts/bench/fourway` METHOD=orig switches on 4 GPUs: FSDP2 sharded over the 4
ranks with verl-native CPUOffloadPolicy (fp32 params/grads/AdamW on host), a separate vLLM
student copy with weight sync, teachers with verl-native FSDP param offload, NCCL collectives
over PCIe. Same r2.sh workload, batch 4 (1/GPU), 30 steps, steps 2..30 averaged. Both runs drew
the same prompts (steps 1-2 had identical response lengths); launcher:
`/mnt/sda1/xxf/cpu_dp_logs/orig4.sh`, logs `orig4_30.log` / `dp4_30.log`.

| mean per step | orig FSDP2 4 GPU | cpu_dp 4 GPU | |
|---|---|---|---|
| step | 44.3 s | 33.3 s | **1.33x** |
| prompt+response tokens / s | 235 | 302 | **1.28x** |
| gen (rollout, incl. weight sync) | 29.8 s | 28.2 s | |
| scoring: student log-prob + 3 teachers | 0.79 + 4.38 s | 1.63 + 2.27 s | 1.3x |
| update_actor | 9.62 s | 2.35 s | **4.1x** |
| everything except rollout | 14.5 s | 5.1 s | **2.8x** |
| torch peak allocated (GPU) | 18.9 GB | 19.5 GB | both fit 24 GiB |

Rollout (the longest of four 4096-token-capped responses) is ~29 s in both, so the end-to-end
gain is bounded by it; the non-rollout part of the step is 2.8x faster.

### Training loss / accuracy vs the original FSDP path

Same setup as the single-GPU equivalence runs (40 steps, batch 4, `data.seed=1`, remove-padding,
validation on the fixed 128-prompt subset at step 40); log `cmp2_orig4.log`.

| | orig FSDP2 4 GPU | cpu_dp 4 GPU | single GPU |
|---|---|---|---|
| rm_scores_mean (teacher-student signal, driver, token mean) | -0.0387 | -0.0346 | -0.0367 |
| entropy | 0.613 | 0.645 | 0.634 |
| response length | 2066 | 2072 | 2089 |
| pg_loss (reported) | **0.0555** | 0.0331 | 0.0354 |
| pre-clip grad norm | **16.2** | 3.2 | 4.0 |
| val @40 ifeval / LCB v5 / LCB v6 / AIME25 | 0.733 / 0.053 / 0.167 / 0 | 0.773 / 0.053 / 0.125 / 0.25 | 0.760 / 0.053 / 0.167 / 0.25 |

The policy statistics and validation agree within run-to-run noise. pg_loss and the gradient
norm differ by construction: verl's original data parallel takes a token-mean on each rank and
averages the ranks (FSDP gradient averaging), so each token of a short response on one GPU
weighs up to `N_total / (world * n_rank)` (often ~5-8x) more than in a single-GPU batch. cpu_dp
divides by the global token count instead, which is why its pg_loss ≈ -rm_scores_mean like the
single-GPU run, while the original path's is not.

### vs upstream Open-MOPD (BytedTsinghua-SIA main @4809a96) data parallel

Upstream code (copy in `~/Distill/Open-MOPD-upstream`, launcher `/mnt/sda1/xxf/cpu_dp_logs/up4.sh`)
with only environment fixes from dev (transformers 5 `AutoModelForVision2Seq` shim, prompt-length
filter and long-parquet loading, vLLM compile isolation). Launched like `scripts/local/mt_opd.sh`
(FSDP1 FULL_SHARD over 4 GPUs, separate vLLM copy, NCCL) with the same workload and
`reward_model.model.input_tokenizer=null`, `compute_true_reward=false`. Upstream defaults peak at
30.7 GB and OOM on the 24 GB cards, so the standard verl switches
`actor.fsdp_config.param_offload=True optimizer_offload=True` and
`reward_model.model.fsdp_config.param_offload=True` were added (~19.9 GB per GPU). 30 steps,
same prompts (step-2 responses identical to the cpu_dp run).

| mean per step (steps 2..30) | upstream DP | orig switches (FSDP2 offload) | cpu_dp |
|---|---|---|---|
| step | 41.2 s | 44.3 s | **33.3 s** |
| prompt+response tokens / s | 255 | 235 | **302** |
| gen (rollout + weight sync) | 31.0 s | 29.8 s | 28.2 s |
| scoring (student + 3 teachers) | 1.41 + 5.68 s | 0.79 + 4.38 s | 1.63 + 2.27 s |
| update_actor | 3.95 s | 9.62 s | 2.35 s |
| everything except rollout | 10.1 s | 14.5 s | 5.1 s |

cpu_dp is 1.24x faster per step (1.18x tokens/s) than upstream DP, and 2.0x on the non-rollout
part; the rollout (~28-31 s, bounded by the longest of four 4096-token responses) dominates all three.

## Overview: Open-MOPD FSDP4 vs Open-MOPD (upstream) DP4 vs cpu_dp DP4 (G4090j, 24 GiB per GPU)

* **FSDP4**: this repo with every single-GPU switch off (`scripts/bench/fourway` METHOD=orig): FSDP2 +
  verl CPUOffloadPolicy, separate vLLM copy, NCCL.
* **Upstream DP4**: BytedTsinghua-SIA main @4809a96 + environment-only fixes, FSDP1 FULL_SHARD with
  verl param/optimizer offload (its defaults OOM at 24 GiB).
* **Ours**: `rollout.cpu_dp=true`.

| | FSDP4 | upstream DP4 | ours |
|---|---|---|---|
| step time, r2.sh workload, 30 steps | 44.3 s | 41.2 s | **33.3 s** |
| prompt+response tokens/s | 235 | 255 | **302** |
| step time, remove-padding runs, 40 steps | 44.0 s | 40.5 s | **34.1 s** |
| everything except rollout | 14.5 s | 10.1 s | **5.1 s** |
| GPU peak per card (nvidia-smi, GPU0) | **14.3 GiB** | 23.5 GiB (at capacity) | 19.0 GiB |
| host memory in training (MemTotal-MemAvailable) | 169.7 GiB | 175.7 GiB | **147.6 GiB** (92 GiB of it /dev/shm) |
| host memory peak (incl. startup) | 170.1 GiB | 177.6 GiB | **150.3 GiB** |
| val@40 ifeval / LCB v5 / LCB v6 / AIME25 | 0.733 / 0.053 / 0.167 / 0 | 0.760 / 0.053 / 0.208 / 0.25 | 0.773 / 0.053 / 0.125 / 0.25 |
| teacher-student signal, 40-step mean | -0.0387 | -0.0416 | -0.0346 |
| pg_loss / pre-clip grad norm | 0.055 / 16.1 | 0.063 / 20.5 | 0.033 / 3.2 |

Single-GPU reference (batch 4): val 0.760 / 0.053 / 0.167 / 0.25, signal -0.0367, pg_loss 0.035,
grad norm 4.0. Starting checkpoint: 0.720 / 0.053 / 0.125-0.167 / 0. AIME24 is 0 everywhere.
Memory logs: `mem_{orig4,up4,dp4}.log`, `gpu_{orig4,up4,dp4}.log` (`scripts/cpu_dp/memory_peaks.py`);
the machine's idle baseline is ~21.5 GiB. cpu_dp's startup peak was 218.7 GiB until the
teachers' module parameter views were re-pointed at the shared copy right after sharing (they
kept each rank's private teacher copies alive until lazy init); it is now 150.3 GiB, the same
as steady-state training (`mem_dp4b.log`).

## Overview: Open-MOPD FSDP4 vs Open-MOPD DP4 vs cpu_dp DP4 (G4090j, 2026-11-23)

FSDP4 = dev code with every Open-MOPD speed switch off (fourway `orig`: FSDP2 + CPUOffloadPolicy);
DP4 = upstream Open-MOPD main (FSDP1 FULL_SHARD + param/optimizer offload, env fixes only);
cpu_dp = this work. All on 4 GPUs, batch 4 (1/GPU), same prompts, same workload.

| | Open-MOPD FSDP4 | Open-MOPD DP4 (upstream) | cpu_dp DP4 |
|---|---|---|---|
| step time, 30-step run | 44.3 s | 41.2 s | **33.3 s** |
| tokens/s, 30-step run | 235 | 255 | **302** |
| step time / tokens/s, 40-step validation run | 44.0 s / 217 | 40.5 s / 232 | **34.1 s / 281** |
| non-rollout time per step | 14.5 s | 10.1 s | **5.1 s** |
| GPU peak per card (nvidia-smi, 24 GB card) | 14.6 GB | 24.1 GB (98% of card) | 19.4 GB |
| host RAM in use, steady (incl. /dev/shm) | ~170 GB | ~176 GB | **~148 GB** (101 GB shm) |
| host RAM in use, peak | ~170 GB | ~177 GB | ~207 GB (init only) |
| val @40 ifeval / LCB5 / LCB6 / AIME24 / AIME25 | 0.733 / 0.053 / 0.167 / 0 / 0 | 0.760 / 0.053 / 0.208 / 0 / 0.25 | 0.773 / 0.053 / 0.125 / 0 / 0.25 |
| teacher signal rm_scores_mean (40 steps) | -0.039 | -0.042 | -0.035 |
| reported pg_loss / grad norm | 0.055 / 16.1 | 0.063 / 20.6 | 0.033 / 3.2 |
| loss normalization | per-GPU token mean, averaged | per-GPU token mean, averaged | global token mean (= 1 GPU) |

Single GPU at the same global batch: 41.3 s/step, val 0.760 / 0.053 / 0.167 / 0 / 0.25,
pg_loss 0.035, grad norm 4.0. Host RAM = total - MemAvailable (machine otherwise idle, ~9 GB
baseline). Memory runs: `memrun_{orig4,dp4}.log` + `mem_*.log` / `gpu_*.log`; upstream sampled
during `cmp2_up4.log`.
