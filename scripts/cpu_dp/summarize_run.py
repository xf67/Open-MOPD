"""Summarize verl console step metrics from one or more training logs.

python scripts/cpu_dp/summarize_run.py LOG [LOG ...] [--skip 1]

Reports per run: steps, mean step time, prompt+response tokens/s (perf/total_num_tokens),
and the mean/min/max of selected timing / CPU-DP metrics. The first --skip steps
(warm-up, compilation) are excluded from the means.
"""
import argparse
import re
import statistics

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PAIR = re.compile(r"([\w/@.\-]+):(?:np\.float64\()?(-?[0-9.]+(?:e-?\d+)?)\)?")


def parse(path):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--skip", type=int, default=1)
    args = parser.parse_args()
    keys = ["timing_s/step", "timing_s/gen", "timing_s/generation_timing/max", "timing_s/generation_timing/min",
            "timing_s/compute_log_prob_and_teacher", "timing_s/update_actor", "response_length/mean",
            "pipeline/cpu_optimizer_seconds", "cpu_dp/reduce_seconds", "pipeline/cpu_adam_native_seconds",
            "actor/grad_norm", "cpu_dp/global_grad_norm", "actor/pg_loss", "perf/max_memory_reserved_gb"]
    for path in args.logs:
        rows = parse(path)[args.skip:]
        if not rows:
            print(f"{path}: no steps")
            continue
        total_time = sum(r["timing_s/step"] for r in rows)
        # verl: perf/total_num_tokens counts prompt+response tokens of the step's batch.
        tokens = sum(r.get("perf/total_num_tokens", 0.0) for r in rows)
        print(f"== {path}: steps {int(rows[0]['step'])}..{int(rows[-1]['step'])} ({len(rows)})")
        print(f"   time {total_time:.1f}s  mean step {total_time / len(rows):.2f}s  "
              f"all tokens/s {tokens / total_time:.1f}")
        for key in keys:
            values = [r[key] for r in rows if key in r]
            if values:
                print(f"   {key:42s} mean {statistics.mean(values):10.4f}  min {min(values):10.4f}  "
                      f"max {max(values):10.4f}")


if __name__ == "__main__":
    main()
