"""Peak memory of one run from mem_sampler.py and gpu_sampler.sh logs.

python scripts/cpu_dp/memory_peaks.py NAME [LOG_DIR]   (reads mem_NAME.log and gpu_NAME.log)

GPU: per-GPU maximum of nvidia-smi memory.used (MiB -> GiB). Host: peak (MemTotal - MemAvailable),
which includes /dev/shm, and the peak /dev/shm usage; the machine's idle baseline is reported too.
"""
import os
import sys

name = sys.argv[1]
d = sys.argv[2] if len(sys.argv) > 2 else "/mnt/sda1/xxf/cpu_dp_logs"
with open("/proc/meminfo") as h:
    total = int(next(l for l in h if l.startswith("MemTotal")).split()[1]) / 2**20
gpu_max = None
with open(os.path.join(d, f"gpu_{name}.log")) as h:
    for line in h:
        parts = line.split()[1:]
        if not parts:
            continue
        vals = [int(x) / 1024 for x in parts]
        gpu_max = vals if gpu_max is None else [max(a, b) for a, b in zip(gpu_max, vals)]
avail, shm = [], []
with open(os.path.join(d, f"mem_{name}.log")) as h:
    for line in h:
        f = dict(p.split("=") for p in line.split("|")[0].split()[1:])
        avail.append(float(f["avail"]))
        shm.append(float(f["shm"]))
print(f"{name}: GPU peak used (GiB) " + " ".join(f"gpu{i}={v:.1f}" for i, v in enumerate(gpu_max)))
print(f"{name}: host peak used {total - min(avail):.1f} GiB (baseline {total - avail[0]:.1f}), "
      f"peak /dev/shm {max(shm):.1f} GiB (baseline {shm[0]:.1f})")
