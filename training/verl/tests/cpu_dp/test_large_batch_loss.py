"""Large-batch loss normalization: micro-batches x ranks == one global batch (CPU, gloo).

python training/verl/tests/cpu_dp/test_large_batch_loss.py [--world 4]

Four gloo processes each hold a slice of one global mini-batch with ragged response
lengths, empty responses, a format mask and 3D top-k advantages zeroed outside a top-p
nucleus mask. Each rank splits its slice with the actor's token-budget splitter
(``DataParallelPPOActor._large_batch_microbatches``: one gloo all-reduce of the mask counts),
compacts every micro-batch's columns, computes the PPO loss under
``global_loss_denominator(fixed=...)`` and backpropagates. The sum of all micro-batch losses
over ranks and the gradient w.r.t. every log-prob must equal the single-batch loss / gradient
of ``compute_policy_loss_vanilla`` over the whole batch (token-mean, seq-mean-token-sum,
seq-mean-token-mean).
"""
import argparse
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

PER_RANK, PROMPT, RESP, K = 9, 6, 40, 4
MODES = ("token-mean", "seq-mean-token-sum", "seq-mean-token-mean")


def global_batch(world):
    g = torch.Generator().manual_seed(7)
    n = world * PER_RANK
    prompt_len = torch.randint(1, PROMPT + 1, (n,), generator=g)
    resp_len = torch.randint(0, RESP + 1, (n,), generator=g)
    resp_len[3] = 0  # an empty response
    resp_len[5] = RESP  # a full-width response
    attention = torch.zeros(n, PROMPT + RESP, dtype=torch.long)
    for i in range(n):
        attention[i, PROMPT - prompt_len[i]: PROMPT + resp_len[i]] = 1
    response_mask = attention[:, PROMPT:].float()
    format_mask = (torch.rand(n, generator=g) > 0.25).float()
    top_p_mask = torch.rand(n, RESP, K, generator=g) > 0.3
    advantages = torch.randn(n, RESP, K, generator=g) * top_p_mask * response_mask.unsqueeze(-1)
    old_log_prob = -torch.rand(n, RESP, K, generator=g) * 3
    log_prob = old_log_prob + 0.3 * torch.randn(n, RESP, K, generator=g)  # ratios outside the clip range too
    return dict(attention_mask=attention, input_ids=torch.randint(0, 100, (n, PROMPT + RESP), generator=g),
                position_ids=torch.cumsum(attention, -1), responses=torch.randint(0, 100, (n, RESP), generator=g),
                response_mask=response_mask, format_mask=format_mask, advantages=advantages,
                old_log_probs=old_log_prob, log_prob=log_prob, overlap_mask=top_p_mask)


def config():
    from omegaconf import OmegaConf

    return OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.28, "clip_ratio_c": 3.0})


def reference(world, mode):
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    b = global_batch(world)
    log_prob = b["log_prob"].clone().requires_grad_(True)
    loss, _ = compute_policy_loss_vanilla(
        old_log_prob=b["old_log_probs"], log_prob=log_prob, advantages=b["advantages"],
        response_mask=b["response_mask"], loss_agg_mode=mode, config=config(), format_mask=b["format_mask"],
    )
    loss.backward()
    return loss.detach(), log_prob.grad


def worker(rank, world, init_file, budget, queue):
    from verl import DataProto
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla, global_loss_denominator
    from verl.utils.large_batch import compact_columns
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    b = global_batch(world)
    rows = slice(rank * PER_RANK, (rank + 1) * PER_RANK)
    out = {}
    for mode in MODES:
        local = {k: v[rows].clone() for k, v in b.items()}
        leaf = local["log_prob"].clone().requires_grad_(True)
        local["log_prob"] = leaf
        tensors = {k: v for k, v in local.items() if k != "log_prob"}
        mini = DataProto.from_dict(tensors=tensors)
        micro_batches, counts, widths = DataParallelPPOActor._large_batch_microbatches(None, mini, budget)
        total, offset = torch.zeros(()), 0
        for mb, (keep_prompt, keep_response) in zip(micro_batches, widths, strict=True):
            n = len(mb)
            inputs = {k: v for k, v in mb.batch.items()}
            inputs["log_prob"] = leaf[offset:offset + n]
            offset += n
            inputs = compact_columns(inputs, RESP, keep_prompt, keep_response)
            with global_loss_denominator(fixed=counts) as denominator, denominator.select("pg"):
                loss, _ = compute_policy_loss_vanilla(
                    old_log_prob=inputs["old_log_probs"], log_prob=inputs["log_prob"],
                    advantages=inputs["advantages"], response_mask=inputs["response_mask"],
                    loss_agg_mode=mode, config=config(), format_mask=inputs["format_mask"],
                )
            loss.backward()
            total = total + loss.detach()
        assert offset == PER_RANK
        dist.all_reduce(total)
        grads = [torch.zeros_like(leaf.grad) for _ in range(world)]
        dist.all_gather(grads, leaf.grad)
        out[mode] = (total, torch.cat(grads), len(micro_batches))
    if rank == 0:
        queue.put(out)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--budget", type=int, default=90)
    args = parser.parse_args()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [ctx.Process(target=worker, args=(r, args.world, os.path.join(tmp, "init"), args.budget, queue))
                 for r in range(args.world)]
        for p in procs:
            p.start()
        out = queue.get(timeout=600)
        for p in procs:
            p.join()
        assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    for mode in MODES:
        want_loss, want_grad = reference(args.world, mode)
        got_loss, got_grad, n_micro = out[mode]
        torch.testing.assert_close(got_loss, want_loss, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(got_grad, want_grad, atol=1e-7, rtol=1e-5)
        print(f"{mode}: loss {got_loss.item():.6f} == {want_loss.item():.6f}, grad max abs err "
              f"{(got_grad - want_grad).abs().max().item():.2e}, rank-0 micro-batches {n_micro}")
    print("LARGE_BATCH_LOSS_PASS")


if __name__ == "__main__":
    main()
