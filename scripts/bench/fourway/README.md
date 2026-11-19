# Single-GPU MOPD speed benchmark (four-way study, 2026-11-18)

Reference launchers used for the A6000 comparison in `fourway-speed-a6000-20261118`
(results, figures and the full write-up live there).

- `run_verl.py`: Hydra overrides for the three verl-based systems, selected by `METHOD`:
  - `orig`: every Open-MOPD single-GPU switch off, FSDP2 `offload_policy` (verl-native CPU offload).
  - `zero`: `orig` + ZeRO-Offload CPU Adam (`verl.utils.zero_offload_adam.ZeroOffloadCPUAdam`).
  - `ours`: shared student weights, student/teacher slot pipeline, progressive gradient offload, async CPU Adam,
    vLLM CUDA graphs; `DYN_MIN_TOKENS>0` enables dynamic token batching.
  Batch size: `BATCH` (prompts per step), `MICRO` (defaults to `BATCH`).
- Telemetry: `verl.utils.speed_bench` (enabled by `++trainer.suite_token_budget`) writes `RUN_DIR/steps.jsonl`
  and stops after `WARMUP_STEPS` + a window of `TOKEN_BUDGET` tokens and `MIN_MEASURED_STEPS` steps.
- `run_verl_case.sh` / `orchestrator.py`: the docker launcher and two-GPU queue used in the study. They contain
  absolute paths of that experiment (`/study`, `/repo`, image digest); adjust before reuse.
- `analyze.py`, `non_rollout.py`, `make_tables.py`, `paper_figures.py`: summaries, tables and figures.
