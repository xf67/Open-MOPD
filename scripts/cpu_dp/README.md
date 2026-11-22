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
