"""Compare training/validation metrics of two runs (e.g. single-GPU vs CPU data parallel).

python scripts/cpu_dp/compare_runs.py A.log B.log [--window 10] [--names 1gpu,dp4]

Prints window means of the training signal per run, the per-step difference
statistics, and every val-core metric found (initial validation and later ones).
CPU-DP reports the reduced-gradient norm one step late (cpu_dp/global_grad_norm at
step t belongs to the update of step t-1 under the default one-step pipeline); it is
shifted back here so it lines up with the single-GPU actor/grad_norm.
"""
import argparse
import math
import re
import statistics

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PAIR = re.compile(r"([\w/@.\-]+):(?:np\.float64\()?(-?[0-9.]+(?:e-?\d+)?|nan|inf)\)?")
TRAIN_KEYS = ["actor/pg_loss", "mt_opd/rm_scores_mean", "actor/entropy", "actor/ppo_kl",
              "mt_opd/teacher_entropy", "response_length/mean", "response_length/clip_ratio", "grad_norm"]


def parse(path):
    steps, vals = {}, []
    with open(path, errors="replace") as handle:
        for line in handle:
            line = ANSI.sub("", line)
            if " step:" not in line and "val-core" not in line:
                continue
            values = {k: float(v) for k, v in PAIR.findall(line)}
            if any(k.startswith("val-core") for k in values):
                vals.append(values)
            if "step" in values and "timing_s/step" in values:
                steps[int(values["step"])] = values
    dp = any("cpu_dp/global_grad_norm" in v for v in steps.values())
    for step, values in steps.items():
        if dp:
            later = steps.get(step + 1, {})
            if "cpu_dp/global_grad_norm" in later:
                values["grad_norm"] = later["cpu_dp/global_grad_norm"]
        elif "actor/grad_norm" in values:
            values["grad_norm"] = values["actor/grad_norm"]
    return steps, vals


def mean(values):
    values = [v for v in values if math.isfinite(v)]
    return statistics.mean(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--names", default="A,B")
    args = parser.parse_args()
    na, nb = args.names.split(",")
    (sa, va), (sb, vb) = parse(args.a), parse(args.b)
    common = sorted(set(sa) & set(sb))
    print(f"steps: {na} {len(sa)}, {nb} {len(sb)}, common {len(common)}")
    if common:
        last = common[-1]
        for key in TRAIN_KEYS:
            if not any(key in sa[s] for s in common):
                continue
            print(f"\n{key}")
            for lo in range(common[0], last + 1, args.window):
                window = [s for s in common if lo <= s < lo + args.window]
                xa = [sa[s][key] for s in window if key in sa[s]]
                xb = [sb[s][key] for s in window if key in sb[s]]
                print(f"  steps {lo:3d}-{window[-1]:3d}: {na} {mean(xa):10.5f}   {nb} {mean(xb):10.5f}")
            xa = [sa[s][key] for s in common if key in sa[s] and key in sb[s]]
            xb = [sb[s][key] for s in common if key in sa[s] and key in sb[s]]
            if len(xa) > 2:
                diff = [a - b for a, b in zip(xa, xb)]
                sd = statistics.pstdev(xa + xb) or float("nan")
                print(f"  all: {na} {mean(xa):.5f} ± {statistics.pstdev(xa):.5f}   {nb} {mean(xb):.5f} ± "
                      f"{statistics.pstdev(xb):.5f}   mean diff {mean(diff):+.5f} ({mean(diff) / sd:+.2f} pooled sd)")
    for name, vals in ((na, va), (nb, vb)):
        print(f"\nvalidation {name}:")
        for index, values in enumerate(vals):
            core = {k: v for k, v in values.items() if k.startswith("val-core") and "/acc/" in k}
            print(f"  #{index}: " + ", ".join(f"{k.removeprefix('val-core/')}={v:.4f}" for k, v in sorted(core.items())))


if __name__ == "__main__":
    main()
