"""Training-time profile of a large-batch run from its console log and the GPU / host samplers.

python scripts/cpu_dp/profile_report.py RUN_LOG GPU_LOG MEM_LOG [--out report.md] [--skip 0]
    [--ckpt-seconds 234 224 --ckpt-source lb_hostmem]   # when the profiled run saved no checkpoint

* per-phase time per step: rollout (gen; slowest vs fastest rank = decode tail), scoring
  (student + 3 teachers), reward/advantage, update_actor (all synchronous updates), other;
* peak GPU per card (nvidia-smi samples) and torch peak reserved;
* peak host memory (MemTotal - MemAvailable) and /dev/shm;
* response length per domain;
* projected wall time for 200 and 600 steps incl. validation (every test_freq) and
  checkpoints (every save_freq).
"""
import argparse
import re
import statistics

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PAIR = re.compile(r"([\w/@.\-]+):(?:np\.float64\()?(-?[0-9.]+(?:e[-+]?\d+)?)\)?")


def parse_steps(path):
    steps = {}
    with open(path, errors="replace") as handle:
        for line in handle:
            line = ANSI.sub("", line)
            if " step:" not in line or "timing_s/step" not in line:
                continue
            values = {k: float(v) for k, v in PAIR.findall(line)}
            if "step" in values:
                steps[int(values["step"])] = values
    return [steps[k] for k in sorted(steps)]


def parse_initial_validation(path):
    """timing_s/testing of val_before_train (its own line; the initial metrics are not a step line)."""
    found = []
    with open(path, errors="replace") as handle:
        for line in handle:
            match = re.search(r"\[val_before_train\] timing_s/testing:([0-9.]+)", ANSI.sub("", line))
            if match:
                found.append(float(match.group(1)))
    return found


def parse_gpu(path):
    peaks = None
    with open(path, errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2 or not all(p.isdigit() for p in parts[1:]):
                continue
            used = [int(p) for p in parts[1:]]
            peaks = used if peaks is None else [max(a, b) for a, b in zip(peaks, used, strict=False)]
    return peaks or []


def parse_mem(path, total_gib):
    host_peak = shm_peak = 0.0
    with open(path, errors="replace") as handle:
        for line in handle:
            avail = re.search(r"avail=([0-9.]+)", line)
            shm = re.search(r"shm=([0-9.]+)", line)
            if avail:
                host_peak = max(host_peak, total_gib - float(avail.group(1)))
            if shm:
                shm_peak = max(shm_peak, float(shm.group(1)))
    return host_peak, shm_peak


def fmt_s(seconds):
    return f"{seconds:,.0f} s ({seconds / 60:.1f} min)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_log")
    parser.add_argument("gpu_log")
    parser.add_argument("mem_log")
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip", type=int, default=0, help="steps excluded from the per-step means")
    parser.add_argument("--host-total-gib", type=float, default=251.0)
    parser.add_argument("--test-freq", type=int, default=20)
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--ckpt-seconds", type=float, nargs="*", default=None,
                        help="checkpoint save times measured in another run, used when this run saved none")
    parser.add_argument("--ckpt-source", default="another run", help="where --ckpt-seconds was measured")
    args = parser.parse_args()

    rows = parse_steps(args.run_log)
    if not rows:
        raise SystemExit("no step lines found")
    used = rows[args.skip:] or rows
    mean = lambda key: statistics.mean(r.get(key, 0.0) for r in used)
    out = []
    p = out.append
    p(f"# Large-batch Open-MOPD profile: {args.run_log}\n")
    p(f"Steps measured: {', '.join(str(int(r['step'])) for r in rows)} (means over steps "
      f"{', '.join(str(int(r['step'])) for r in used)}).\n")

    p("## Time per step\n")
    p("| step | rollout (gen) | gen slowest / fastest rank | scoring (S + 3 T) | reward+adv rest | "
      "update_actor (all updates) | other | step excl. ckpt/val | checkpoint + validation |")
    p("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        gen = r.get("timing_s/gen", 0.0)
        scoring = r.get("timing_s/compute_log_prob_and_teacher", 0.0)
        reward = r.get("timing_s/reward", 0.0)
        adv = r.get("timing_s/adv", 0.0)
        update = r.get("timing_s/update_actor", 0.0)
        extra = r.get("timing_s/save_checkpoint", 0.0) + r.get("timing_s/testing", 0.0)
        step = r["timing_s/step"] - extra
        other = step - gen - reward - adv - update
        p(f"| {int(r['step'])} | {gen:,.0f} s | {r.get('timing_s/generation_timing/max', 0):,.0f} / "
          f"{r.get('timing_s/generation_timing/min', 0):,.0f} s | {scoring:,.0f} s | {reward - scoring + adv:,.0f} s | "
          f"{update:,.0f} s | {other:,.0f} s | {step:,.0f} s ({step / 60:.1f} min) | {extra:,.0f} s |")
    p("")
    p("Per update (mean over updates of the measured steps, owner rank where noted):\n")
    for key, label in [
        ("pipeline/updates_per_step", "updates per step"),
        ("actor/microbatches_per_update", "training micro-batches per update (mean over ranks)"),
        ("pipeline/cpu_optimizer_seconds", "CPU update (reduce + clip + Adam + publish), s"),
        ("cpu_dp/reduce_seconds", "  of which 4-rank FP32 reduce, s (owner)"),
        ("pipeline/cpu_adam_native_seconds", "  of which CPUAdam, s (owner)"),
        ("cpu_dp/accumulate_seconds", "host FP32 accumulation per update (overlapped with backward), s"),
        ("cpu_dp/accumulate_wait_slot_seconds", "backward stalled waiting for a landing slot, s"),
        ("scoring/cycles_max", "scoring cycles (slowest rank)"),
        ("scoring/padded_positions_max", "scored padded positions (slowest rank)"),
        ("scoring/real_tokens_sum", "scored real tokens (all ranks)"),
        ("perf/total_num_tokens", "prompt+response tokens per step"),
        ("actor/grad_norm", "global grad norm (pre-clip)"),
        ("actor/pg_clipfrac", "PPO clip fraction"),
    ]:
        values = [r[key] for r in used if key in r]
        if values:
            p(f"* {label}: {statistics.mean(values):,.3f}")
    p("")

    val = [r["timing_s/testing"] for r in rows if "timing_s/testing" in r] + parse_initial_validation(args.run_log)
    ckpt = [r["timing_s/save_checkpoint"] for r in rows if "timing_s/save_checkpoint" in r]
    ckpt_note = ""
    if not ckpt and args.ckpt_seconds:
        ckpt = list(args.ckpt_seconds)
        ckpt_note = f" (not saved in this run; measured in {args.ckpt_source}: " + \
            ", ".join(f"{c:.0f} s" for c in ckpt) + ")"
    p("## Validation and checkpoints\n")
    p(f"* validation (128 prompts): {fmt_s(statistics.mean(val)) if val else 'not measured'}")
    p(f"* checkpoint save: {fmt_s(statistics.mean(ckpt)) if ckpt else 'not measured'}{ckpt_note}\n")

    gpu = parse_gpu(args.gpu_log)
    host, shm = parse_mem(args.mem_log, args.host_total_gib)
    p("## Memory peaks\n")
    if gpu:
        p("* GPU (nvidia-smi, MiB): " + ", ".join(f"GPU{i} {v:,}" for i, v in enumerate(gpu)))
    reserved = [r.get("perf/max_memory_reserved_gb") for r in rows if "perf/max_memory_reserved_gb" in r]
    if reserved:
        p(f"* torch peak reserved during update_actor (max over steps, worst rank; counts the slept "
          f"vLLM KV cache): {max(reserved):.2f} GiB")
    backed = [r.get("memory/update_backed_gib_max") for r in rows if "memory/update_backed_gib_max" in r]
    allowance = [r.get("memory/update_kv_allowance_gib") for r in rows if "memory/update_kv_allowance_gib" in r]
    if backed:
        p(f"* update_actor peak backed by GPU memory (worst rank, upper bound): {max(backed):.2f} GiB; "
          f"slept-KV allowance {statistics.mean(allowance):.2f} GiB")
    step_host = [r.get("memory/step_host_used_peak_gib") for r in rows if "memory/step_host_used_peak_gib" in r]
    p(f"* host in use (MemTotal - MemAvailable, sampler peak incl. startup): {host:.1f} GiB of "
      f"{args.host_total_gib:.0f}" + (f"; per-step telemetry peak {max(step_host):.1f} GiB" if step_host else ""))
    p(f"* /dev/shm peak: {shm:.1f} GiB\n")

    p("## Response length per domain (tokens)\n")
    p("| step | domain | prompts | mean | median | max | token share |")
    p("|---|---|---|---|---|---|---|")
    for r in rows:
        for d in ("math", "code", "if"):
            k = f"mt_opd/domain/{d}"
            if f"{k}/mean_response_length" in r:
                p(f"| {int(r['step'])} | {d} | {r.get(k + '/prompt_count', 0):.0f} | "
                  f"{r[k + '/mean_response_length']:,.0f} | {r.get(k + '/median_response_length', float('nan')):,.0f} | "
                  f"{r.get(k + '/max_response_length', float('nan')):,.0f} | {r.get(k + '/token_share', 0):.3f} |")
    clip = [r.get("response_length/clip_ratio") for r in rows if "response_length/clip_ratio" in r]
    if clip:
        p(f"\nFraction of responses at data.max_response_length (response_length/clip_ratio): "
          + ", ".join(f"{c:.3f}" for c in clip))
    p("")

    step_time = statistics.mean(
        r["timing_s/step"] - r.get("timing_s/save_checkpoint", 0.0) - r.get("timing_s/testing", 0.0) for r in used
    )
    t_val = statistics.mean(val) if val else 0.0
    t_ckpt = statistics.mean(ckpt) if ckpt else 0.0
    p("## Projected wall time\n")
    p(f"Mean step {fmt_s(step_time)}; validation every {args.test_freq} steps ({fmt_s(t_val)} each), "
      f"checkpoint every {args.save_freq} steps ({fmt_s(t_ckpt)} each).\n")
    p("| steps | training | validations | checkpoints | total |")
    p("|---|---|---|---|---|")
    for n in (200, 600):
        n_val, n_ckpt = n // args.test_freq, n // args.save_freq
        total = n * step_time + n_val * t_val + n_ckpt * t_ckpt
        p(f"| {n} | {n * step_time / 3600:.1f} h | {n_val} x = {n_val * t_val / 3600:.1f} h | "
          f"{n_ckpt} x = {n_ckpt * t_ckpt / 3600:.1f} h | **{total / 3600:.1f} h = {total / 86400:.2f} days** |")
    text = "\n".join(out) + "\n"
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
