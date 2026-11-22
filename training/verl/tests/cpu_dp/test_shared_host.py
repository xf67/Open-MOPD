"""Shared host arena checks on real GPUs (no Ray).

python training/verl/tests/cpu_dp/test_shared_host.py [--gpus 0,1,2,3]

Checks: one physical copy across ranks, cross-process visibility, CUDA
registration (is_pinned, async H2D), NUMA placement, H2D bandwidth per rank
when all ranks copy concurrently.
"""
import argparse
import os
import sys
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from verl.utils import shared_host  # noqa: E402

GiB = 2**30


def numa_pages(tensor):
    """Pages per node of the mapping holding ``tensor`` (from /proc/self/numa_maps)."""
    address = tensor.data_ptr()
    with open("/proc/self/numa_maps") as handle:
        for line in handle:
            fields = line.split()
            start = int(fields[0], 16)
            if start <= address < start + tensor.nbytes + (1 << 30) and "/dev/shm" in line:
                return {f.split("=")[0]: int(f.split("=")[1]) for f in fields if f.startswith("N")}
    return {}


def worker(rank, world, gpus, init_file, size_gib, result_queue):
    torch.cuda.set_device(gpus[rank])
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    numel = int(size_gib * GiB) // 2
    node = shared_host.gpu_numa_node()
    shared_host.bind_process_to_node(node)

    def fill(t):
        t["w"].copy_(torch.arange(numel, dtype=torch.float32).to(torch.bfloat16))

    # 1) interleaved arena read by all ranks
    arena = shared_host.shared_arena("w", [("w", (numel,), torch.bfloat16)], creator=0,
                                     policy="interleave", fill=fill)
    w = arena["w"]
    assert w.is_pinned(), "registered shared mapping must report pinned"
    assert float(w[12345]) == float(torch.tensor(12345.0).to(torch.bfloat16))

    # 2) per-rank node-local staging; rank writes, rank 0 reads
    stage = shared_host.shared_arena(f"s{rank}", [("g", (1024,), torch.float32)], creator=rank,
                                     policy=f"node:{node}", register=True)
    # every rank must take part in every arena's barrier
    others = {}
    for r in range(world):
        if r == rank:
            continue
        others[r] = shared_host.shared_arena(f"s{r}", [("g", (1024,), torch.float32)], creator=r,
                                             attach=(rank == 0), register=False)
    stage["g"].fill_(rank + 1)
    shared_host.cpu_barrier()
    if rank == 0:
        for r, t in others.items():
            assert t is not None and torch.all(t["g"] == r + 1), f"rank {r} write not visible"

    # 3) concurrent H2D bandwidth from the shared interleaved copy
    dst = torch.empty(numel, dtype=torch.bfloat16, device="cuda")
    dst.copy_(w, non_blocking=True)
    torch.cuda.synchronize()
    shared_host.cpu_barrier()
    start = time.perf_counter()
    reps = 3
    for _ in range(reps):
        dst.copy_(w, non_blocking=True)
    torch.cuda.synchronize()
    h2d = reps * w.nbytes / (time.perf_counter() - start) / 1e9
    assert torch.equal(dst[:1000].cpu(), w[:1000])

    # 4) D2H into node-local registered staging
    big = shared_host.shared_arena(f"d{rank}", [("g", (numel,), torch.bfloat16)], creator=rank,
                                   policy=f"node:{node}", register=True, attach=False)
    shared_host.cpu_barrier()
    start = time.perf_counter()
    for _ in range(reps):
        big["g"].copy_(dst, non_blocking=True)
    torch.cuda.synchronize()
    d2h = reps * w.nbytes / (time.perf_counter() - start) / 1e9
    result_queue.put((rank, gpus[rank], node, h2d, d2h, numa_pages(w), numa_pages(big["g"])))
    shared_host.cpu_barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--size-gib", type=float, default=2.0)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    world = len(gpus)
    # Per-rank staging arenas d<r> are created without a matching collective; keep names unique.
    name = shared_host.new_namespace()
    try:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        with tempfile.TemporaryDirectory() as tmp:
            procs = [ctx.Process(target=worker, args=(r, world, gpus, os.path.join(tmp, "init"), args.size_gib, queue))
                     for r in range(world)]
            for p in procs:
                p.start()
            for p in procs:
                p.join()
            assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
        rows = sorted(queue.get() for _ in range(world))
        print(f"namespace={name} nodes={ {k: len(v) for k, v in shared_host.numa_nodes().items()} }")
        for rank, gpu, node, h2d, d2h, wp, dp in rows:
            print(f"rank={rank} gpu={gpu} node={node} H2D={h2d:.1f}GB/s D2H={d2h:.1f}GB/s "
                  f"shared_pages={wp} staging_pages={dp}")
        print("SHARED_HOST_PASS")
    finally:
        shared_host.remove_namespace(name)


if __name__ == "__main__":
    main()
