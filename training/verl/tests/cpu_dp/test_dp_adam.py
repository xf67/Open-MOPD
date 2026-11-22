"""SharedCPUAdamPipeline on real GPUs with a tiny Qwen2 (no Ray).

python training/verl/tests/cpu_dp/test_dp_adam.py --gpus 0,1,2,3

Each rank runs FSDP1 over a rank-local process group (no gradient all-reduce),
backpropagates sum(token loss)/global_tokens on its own data, and the owner
reduces gradients on CPU and runs one CPUAdam. Checks against an independent
reference (per-rank BF16 grads summed in FP32, global clip, torch AdamW in FP32):
* reduced/clipped update matches the reference (two steps, clipping active);
* all GPU replicas are bitwise identical after each update;
* a non-finite gradient on one rank makes the update skip everywhere.
"""
import argparse
import hashlib
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

SEQ, VOCAB, LR, WD, CLIP = 24, 512, 1e-3, 0.01, 0.05


def tiny_model(seed):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=VOCAB, max_position_embeddings=64,
                         tie_word_embeddings=False, attn_implementation="sdpa")
    return Qwen2ForCausalLM(config).to(torch.bfloat16)


def batch(rank, step):
    g = torch.Generator().manual_seed(1000 * step + rank)
    ids = torch.randint(0, VOCAB, (2, SEQ), generator=g)
    mask = torch.ones(2, SEQ)
    mask[:, : 3 + rank] = 0  # unequal valid token counts per rank
    return ids, mask


def token_loss_sum(model, ids, mask):
    logits = model(input_ids=ids).logits.float()
    logp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return -(logp * mask[:, 1:]).sum()


def global_tokens(world, step):
    return sum(batch(r, step)[1][:, 1:].sum() for r in range(world))


def digest(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def reference(world, steps, nan_step):
    """Independent single-process computation of the expected FP32 masters."""
    model = tiny_model(0).cuda()
    names = [n for n, _ in model.named_parameters()]
    masters = {n: p.detach().float().clone() for n, p in model.named_parameters()}
    opt_params = [torch.nn.Parameter(masters[n]) for n in names]
    opt = torch.optim.AdamW(opt_params, lr=LR, weight_decay=WD, eps=1e-8, foreach=False)
    history = []
    for step in range(steps):
        total = {n: torch.zeros_like(masters[n]) for n in names}
        tokens = global_tokens(world, step)
        for r in range(world):
            model.zero_grad(set_to_none=True)
            ids, mask = batch(r, step)
            (token_loss_sum(model, ids.cuda(), mask.cuda()) / tokens).backward()
            for n, p in model.named_parameters():
                total[n] += p.grad.float()  # rank order, FP32 accumulation
        if step == nan_step:
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


def worker(rank, world, gpus, init_file, steps, nan_step, out_queue):
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
    opt = SharedCPUAdamPipeline(base, model, overlap=True, threads=4, async_update=False, owner_rank=0)
    results = []
    for step in range(steps):
        ids, mask = batch(rank, step)
        opt.zero_grad()
        loss = token_loss_sum(model, ids.cuda(), mask.cuda()) / global_tokens(world, step)
        if step == nan_step and rank == world - 1:
            loss = loss * float("nan")
        loss.backward()
        opt.clip_grad_norm_(CLIP)
        opt.step()  # async_update=False: reduce + Adam + publish + H2D flush on every rank
        torch.cuda.synchronize()
        digests = [None] * world
        dist.all_gather_object(digests, digest(model))
        assert len(set(digests)) == 1, f"step {step}: replicas differ"
        if rank == 0:
            masters = {}
            for index, (param, *_rest) in enumerate(opt.pairs):
                masters[index] = opt.state[param]["cpu_master"].clone()
            results.append((masters, opt.last_metrics))
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
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    world, nan_step = len(gpus), 1
    from verl.utils import shared_host

    name = shared_host.new_namespace()
    try:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        with tempfile.TemporaryDirectory() as tmp:
            procs = [ctx.Process(target=worker, args=(r, world, gpus, os.path.join(tmp, "init"), args.steps,
                                                      nan_step, queue)) for r in range(world)]
            for p in procs:
                p.start()
            order, results = queue.get(timeout=900)
            for p in procs:
                p.join()
            assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
        torch.cuda.set_device(gpus[0])
        expected = reference(world, args.steps, nan_step)
        previous = None
        for step, ((masters, metrics), ref) in enumerate(zip(results, expected, strict=True)):
            print(f"step {step}: metrics={ {k: round(v, 6) for k, v in metrics.items()} }")
            if ref is None:
                assert metrics["cpu_dp/skipped"] == 1.0
                assert all(torch.equal(masters[i], previous[i]) for i in masters), "skip changed masters"
            else:
                assert metrics["cpu_dp/skipped"] == 0.0
                worst = 0.0
                for index, name_ in enumerate(order):
                    got, want = masters[index], ref[name_]
                    err = ((got - want).abs() / (want.abs() + 1e-3)).max().item()
                    worst = max(worst, err)
                    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5, msg=f"{name_} step {step}")
                print(f"step {step}: max relative master error vs reference {worst:.2e}")
            previous = masters
        print("DP_ADAM_PASS")
    finally:
        shared_host.remove_namespace(name)


if __name__ == "__main__":
    main()
