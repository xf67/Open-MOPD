"""Microbenchmark of the CPU gradient reduction kernel (4 BF16 staging buffers -> FP32 sum + sumsq).

numactl-free: pins itself to a NUMA node. python scripts/cpu_dp/bench_reduce.py --numel 1.5e9 --threads 8,16,24
"""
import argparse
import os
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--numel", type=float, default=1.5e9)
p.add_argument("--threads", default="8,16,24")
p.add_argument("--node", type=int, default=0)
a = p.parse_args()
n = int(a.numel)
cpus = []
with open(f"/sys/devices/system/node/node{a.node}/cpulist") as f:
    for part in f.read().strip().split(","):
        lo, _, hi = part.partition("-")
        cpus += range(int(lo), int(hi or lo) + 1)
os.sched_setaffinity(0, cpus)
stages = [torch.randn(n, dtype=torch.float32).to(torch.bfloat16) for _ in range(4)]
out = torch.empty(n, dtype=torch.float32)


def v_current(chunk):
    s = 0.0
    for i in range(0, n, chunk):
        e = min(i + chunk, n)
        o = out[i:e]
        o.copy_(stages[0][i:e])
        for st in stages[1:]:
            o.add_(st[i:e])
        s += float(torch.linalg.vector_norm(o, dtype=torch.float64)) ** 2
    return s


def v_fp32norm(chunk):
    s = 0.0
    for i in range(0, n, chunk):
        e = min(i + chunk, n)
        o = out[i:e]
        o.copy_(stages[0][i:e])
        for st in stages[1:]:
            o.add_(st[i:e])
        s += float(torch.dot(o, o))
    return s


for t in [int(x) for x in a.threads.split(",")]:
    torch.set_num_threads(t)
    for name, fn, chunk in [("current16M", v_current, 1 << 24), ("current1M", v_current, 1 << 20),
                            ("dotnorm1M", v_fp32norm, 1 << 20), ("dotnorm4M", v_fp32norm, 1 << 22)]:
        fn(chunk)
        t0 = time.perf_counter()
        fn(chunk)
        dt = time.perf_counter() - t0
        print(f"threads={t} {name}: {dt:.2f}s  ({n * 12 / dt / 1e9:.1f} GB/s effective)", flush=True)
