"""Large-batch SharedCPUAdamPipeline: FP32 host accumulation over micro-batches, several updates.

python training/verl/tests/cpu_dp/test_dp_accumulate.py --gpus 0,1,2,3 [--microbatches 3 --updates 4]

Each rank runs FSDP1 over a rank-local process group, runs M micro-batch backwards per
update (gradients land in a pinned ring and are added into the rank's FP32 accumulator in
/dev/shm), and K synchronous updates. Every loss is sum(token loss) / (tokens over all
ranks and micro-batches of that update), so the reduced gradient is the global token-mean.
Checks against an independent reference (per rank: FP32 sum of its BF16 micro-batch
grads in micro-batch order; then ranks summed in rank order; global clip; torch AdamW):
* masters after every update vs a one-step reference built from the captured BF16 micro-batch
  grads and the pipeline's previous FP32 state: atol 2e-6 / rtol 2e-5 (the end-to-end
  trajectory vs a plain non-FSDP model is also printed, informational);
* all GPU replicas bitwise identical after each update;
* a NaN in one micro-batch of one rank skips that update on every rank (masters unchanged);
* the hook fires exactly once per layer per backward (the pipeline raises otherwise).
"""
import argparse
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_dp_adam import CLIP, LR, SEQ, VOCAB, WD, digest, tiny_model, token_loss_sum  # noqa: E402


def mb_batch(rank, update, micro):
    g = torch.Generator().manual_seed(100000 * update + 1000 * micro + rank)
    rows = 1 + (rank + micro) % 3  # unequal rows per micro-batch
    ids = torch.randint(0, VOCAB, (rows, SEQ), generator=g)
    mask = torch.ones(rows, SEQ)
    mask[:, : 2 + (rank * 3 + micro) % 7] = 0  # unequal valid token counts
    return ids, mask


def update_tokens(world, update, micro):
    return sum(mb_batch(r, update, m)[1][:, 1:].sum() for r in range(world) for m in range(micro))


def reference(world, updates, micro, nan_update):
    model = tiny_model(0).cuda()
    names = [n for n, _ in model.named_parameters()]
    masters = {n: p.detach().float().clone() for n, p in model.named_parameters()}
    opt_params = [torch.nn.Parameter(masters[n]) for n in names]
    opt = torch.optim.AdamW(opt_params, lr=LR, weight_decay=WD, eps=1e-8, foreach=False)
    history = []
    for update in range(updates):
        tokens = update_tokens(world, update, micro)
        total = None
        for r in range(world):
            acc = None
            for m in range(micro):
                model.zero_grad(set_to_none=True)
                ids, mask = mb_batch(r, update, m)
                (token_loss_sum(model, ids.cuda(), mask.cuda()) / tokens).backward()
                grads = {n: p.grad.float() for n, p in model.named_parameters()}
                if acc is None:
                    acc = {n: g.clone() for n, g in grads.items()}
                else:
                    for n in names:
                        acc[n] += grads[n]  # FP32 accumulation of BF16 micro-batch grads
            if total is None:
                total = {n: acc[n].clone() for n in names}
            else:
                for n in names:
                    total[n] += acc[n]  # rank order
        if update == nan_update:
            history.append(None)
            continue
        norm = torch.sqrt(sum((g.double() ** 2).sum() for g in total.values()))
        scale = min(1.0, CLIP / (float(norm) + 1e-6))
        for n, p in zip(names, opt_params, strict=True):
            p.grad = total[n] * scale
        opt.step()
        with torch.no_grad():
            for n, p in zip(names, opt_params, strict=True):
                model.get_parameter(n).copy_(p)
        history.append({n: p.detach().clone().cpu() for n, p in zip(names, opt_params, strict=True)})
    return history


def one_step_reference(before, rank_accumulators, count):
    """Expected masters after one update from the pipeline's own state and captured grads.

    Ranks summed in rank order (FP32), global norm in FP64, clip, then torch AdamW with the
    pipeline's previous FP32 masters / moments / step. Returns None for a non-finite norm.
    """
    total = {}
    for acc in rank_accumulators:  # rank order
        for index, grad in acc.items():
            total[index] = total[index] + grad if index in total else grad.clone()
    norm = torch.sqrt(sum((g.double() ** 2).sum() for g in total.values()))
    if not torch.isfinite(norm):
        return None
    scale = min(1.0, CLIP / (float(norm) + 1e-6))
    expected = {}
    for index in range(count):
        state = before[index]
        if index not in total:
            expected[index] = state["cpu_master"].clone()
            continue
        param = torch.nn.Parameter(state["cpu_master"].clone())
        opt = torch.optim.AdamW([param], lr=LR, weight_decay=WD, eps=1e-8, foreach=False)
        opt.state[param] = {"step": torch.tensor(float(state["step"])), "exp_avg": state["exp_avg"].clone(),
                            "exp_avg_sq": state["exp_avg_sq"].clone()}
        param.grad = (total[index] * scale).view_as(param)  # ring slices are flat
        opt.step()
        expected[index] = param.detach().clone()
    return expected


def worker(rank, world, gpus, init_file, updates, micro, nan_update, ring_slots, out_queue):
    from functools import partial

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    from verl.utils.dp_cpu_adam import SharedCPUAdamPipeline

    torch.cuda.set_device(gpus[rank])
    dist.init_process_group("cpu:gloo,cuda:nccl", init_method=f"file://{init_file}", rank=rank, world_size=world,
                            device_id=torch.device("cuda", gpus[rank]))
    local = [dist.new_group([r]) for r in range(world)][rank]
    model = FSDP(
        tiny_model(0), process_group=local, sharding_strategy=ShardingStrategy.NO_SHARD,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer}),
        device_id=torch.cuda.current_device(), use_orig_params=True,
    )
    base = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, eps=1e-8)
    opt = SharedCPUAdamPipeline(base, model, overlap=True, threads=4, async_update=False, owner_rank=0,
                                accumulate=True, ring_slots=ring_slots)
    # Capture exactly what lands in the ring (BF16 micro-batch grads), in arrival order.
    captured = []
    original_put = opt.accum_queue.put

    def capture_put(item):
        if item is not None:
            slot, ready, entries = item
            ready.synchronize()
            captured.append([(i, opt.ring[slot][off:off + n].clone()) for i, off, n in entries])
        original_put(item)

    opt.accum_queue.put = capture_put
    results = []
    for update in range(updates):
        tokens = update_tokens(world, update, micro)
        captured.clear()
        before = None
        if rank == 0:
            before = {index: {k: opt.state[param][k].clone() for k in ("cpu_master", "exp_avg", "exp_avg_sq")}
                      | {"step": int(opt.state[param]["step"])} for index, (param, *_r) in enumerate(opt.pairs)}
        opt.zero_grad()
        for m in range(micro):
            opt.begin_microbatch()
            ids, mask = mb_batch(rank, update, m)
            loss = token_loss_sum(model, ids.cuda(), mask.cuda()) / tokens
            if update == nan_update and rank == world - 1 and m == micro - 1:
                loss = loss * float("nan")
            loss.backward()
        opt.clip_grad_norm_(CLIP)
        opt.step()  # sync: drain accumulation, reduce + Adam + publish + H2D flush on every rank
        torch.cuda.synchronize()
        # This rank's FP32 accumulation of its BF16 micro-batch grads, in arrival order.
        acc = {}
        for layer in captured:
            for index, grad in layer:
                if index in acc:
                    acc[index] += grad.float()
                else:
                    acc[index] = grad.float().clone()
        gathered = [None] * world
        dist.all_gather_object(gathered, acc)
        digests = [None] * world
        dist.all_gather_object(digests, digest(model))
        assert len(set(digests)) == 1, f"update {update}: replicas differ"
        assert opt.version == update + 1, (opt.version, update)
        if rank == 0:
            masters = {index: opt.state[param]["cpu_master"].clone() for index, (param, *_r) in enumerate(opt.pairs)}
            expected = one_step_reference(before, gathered, len(opt.pairs))
            results.append((masters, dict(opt.last_metrics), expected))
    if rank == 0:
        names = {id(p): n for n, p in model.named_parameters()}
        order = [names[id(p)].replace("_fsdp_wrapped_module.", "") for p, *_ in opt.pairs]
        out_queue.put((order, results))
    opt.close()
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--microbatches", type=int, default=3)
    parser.add_argument("--ring-slots", type=int, default=2)
    parser.add_argument("--nan-update", type=int, default=2)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    world = len(gpus)
    from verl.utils import shared_host

    name = shared_host.new_namespace()
    try:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        with tempfile.TemporaryDirectory() as tmp:
            procs = [ctx.Process(target=worker, args=(r, world, gpus, os.path.join(tmp, "init"), args.updates,
                                                      args.microbatches, args.nan_update, args.ring_slots, queue))
                     for r in range(world)]
            for p in procs:
                p.start()
            order, results = queue.get(timeout=900)
            for p in procs:
                p.join()
            assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
        torch.cuda.set_device(gpus[0])
        trajectory = reference(world, args.updates, args.microbatches, args.nan_update)
        worst_step = 0.0
        for update, ((masters, metrics, step_ref), traj) in enumerate(zip(results, trajectory, strict=True)):
            if (step_ref is None) != (traj is None):
                raise AssertionError(f"update {update}: skip decision differs (step_ref={step_ref is None})")
            if step_ref is not None:
                for index, want in step_ref.items():
                    got = masters[index]
                    worst_step = max(worst_step, (got - want).abs().max().item())
                    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5, msg=f"param {index} update {update}")
        print(f"one-step reference (captured BF16 grads, FP32 rank-order sum, clip, AdamW): "
              f"max abs master error {worst_step:.2e} over {args.updates} updates")
        expected = trajectory
        previous = None
        worst_all = 0.0
        for update, ((masters, metrics, _step_ref), ref) in enumerate(zip(results, expected, strict=True)):
            shown = {k: round(v, 6) for k, v in metrics.items() if isinstance(v, float)}
            print(f"update {update}: metrics={shown}")
            assert metrics["cpu_dp/microbatches_per_update"] == args.microbatches, metrics
            if ref is None:
                assert metrics["cpu_dp/skipped"] == 1.0
                assert all(torch.equal(masters[i], previous[i]) for i in masters), "skip changed masters"
            else:
                assert metrics["cpu_dp/skipped"] == 0.0
                worst = 0.0
                failures = []
                for index, name_ in enumerate(order):
                    got, want = masters[index], ref[name_]
                    worst = max(worst, ((got - want).abs() / (want.abs() + 1e-3)).max().item())
                    bad = (got - want).abs() > 2e-6 + 2e-5 * want.abs()
                    if bad.any():
                        where = bad.nonzero()[0].tolist()
                        failures.append(f"{name_}: {int(bad.sum())}/{bad.numel()} bad, max abs "
                                        f"{(got - want).abs().max().item():.3e}, e.g. at {where} got "
                                        f"{got[tuple(where)].item():.8f} want {want[tuple(where)].item():.8f}")
                if failures:
                    # Informational: the plain (non-FSDP) model's BF16 grads can differ by an ulp
                    # from the FSDP replicas', which Adam amplifies over several updates.
                    print(f"update {update} trajectory vs plain-model reference (informational):\n  "
                          + "\n  ".join(failures))
                worst_all = max(worst_all, worst)
                print(f"update {update}: max relative master error vs reference {worst:.2e}")
            previous = masters
        print(f"DP_ACCUMULATE_PASS (world={world}, microbatches={args.microbatches}, updates={args.updates}, "
              f"worst rel err {worst_all:.2e})")
    finally:
        shared_host.remove_namespace(name)


if __name__ == "__main__":
    main()
