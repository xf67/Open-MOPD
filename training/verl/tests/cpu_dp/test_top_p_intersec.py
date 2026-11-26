"""top_p_intersec in the large-batch MT-OPD pipeline (CPU unit test).

python training/verl/tests/cpu_dp/test_top_p_intersec.py

Three teachers, student top-16 ids, ragged response masks, hard domain routing.
1. Teacher nucleus mask: the worker's chunked ``_compute_teacher_top_k_log_probs``
   (strategy=top_p_intersec, row-blocked mask) equals the released unblocked
   implementation bit for bit, and an independent exclusive-cumsum nucleus.
2. Routing: the routed mask / log-probs of each sample are its domain teacher's.
3. Reward: ``compute_distillation_reward`` (top_p_intersec) equals, per sample, the
   released intersection formula on that sample's routed teacher alone.
4. Refresh: ``refresh_opd_advantage(..., valid_mask=routed mask)`` at the scoring-time
   student equals the reward x response_mask (the closed-form M4 refresh is exact).
"""
import os
import sys
import types

# CPU test: the reward code moves its inputs to the current device, so a visible GPU would put the
# result on cuda while the references stay on the CPU (and the exact refresh check compares CPU math).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

B, T, V, K, P = 6, 23, 1000, 16, 0.99
DOMAINS = ["math", "code", "if", "math", "if", "code"]
ORDER = ["math", "code", "if"]


def released_top_p_mask(logits, student_ids, top_p):
    """The released (unblocked) _compute_student_topk_in_teacher_top_p_mask."""
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    sorted_mask = cumulative_probs <= float(top_p)
    sorted_mask[..., 0] = True
    sorted_mask[..., 1:] |= sorted_mask[..., :-1].clone()
    full_mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
    full_mask.scatter_(dim=-1, index=sorted_ids, src=sorted_mask)
    return torch.gather(full_mask, dim=-1, index=student_ids).float()


def independent_top_p_mask(logits, student_ids, top_p):
    probs = torch.softmax(logits.double(), dim=-1)
    order = torch.argsort(probs, dim=-1, descending=True)
    sorted_probs = torch.gather(probs, -1, order)
    exclusive = sorted_probs.cumsum(-1) - sorted_probs  # mass strictly before each token
    inside = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, order, exclusive <= top_p)
    return torch.gather(inside, -1, student_ids)


def main():
    from verl import DataProto
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.actor.mt_opd import (
        build_domain_weights,
        refresh_opd_advantage,
        select_routed_teacher_logprobs,
        select_routed_teacher_masks,
    )
    from verl.workers.fsdp_workers import RewardModelWorker

    g = torch.Generator().manual_seed(3)
    student_logits = torch.randn(B * T, V, generator=g) * 3
    teacher_logits = [torch.randn(B * T, V, generator=g) * s for s in (2.0, 3.5, 5.0)]
    s_logp_full = torch.log_softmax(student_logits, -1)
    s_logp, s_ids = torch.topk(s_logp_full, K, dim=-1)

    os.environ["OPENMOPD_TOP_P_MASK_ROWS"] = "5"  # force several row blocks
    t_on_s, masks = [], []
    for j, logits in enumerate(teacher_logits):
        on_s, counts, overlap, *_ = RewardModelWorker._compute_teacher_top_k_log_probs(
            None, logits, s_ids, top_k=K, strategy="top_p_intersec", chunk_size=37, top_p_intersec_p=P,
        )
        want = released_top_p_mask(logits, s_ids, P)
        assert torch.equal(overlap, want), f"teacher {j}: blocked mask differs from the released implementation"
        independent = independent_top_p_mask(logits, s_ids, P)
        agree = (overlap.bool() == independent).float().mean().item()
        assert agree > 0.999, f"teacher {j}: nucleus agreement {agree}"
        torch.testing.assert_close(on_s, torch.log_softmax(logits, -1).gather(-1, s_ids))
        assert torch.equal(counts, overlap.sum(-1).long())
        print(f"teacher {j}: in-nucleus fraction of student top-{K} {overlap.mean().item():.3f}, "
              f"agreement with fp64 exclusive-cumsum nucleus {agree:.4f}")
        t_on_s.append(on_s.view(B, T, K))
        masks.append(overlap.view(B, T, K))

    weights = build_domain_weights(DOMAINS, ORDER, device="cpu")
    routed_t = select_routed_teacher_logprobs(t_on_s, weights)
    routed_mask = select_routed_teacher_masks(masks, weights)
    for b, d in enumerate(DOMAINS):
        j = ORDER.index(d)
        assert torch.equal(routed_t[b], t_on_s[j][b]) and torch.equal(routed_mask[b], masks[j][b].bool())

    response_mask = torch.zeros(B, T)
    for b, length in enumerate([23, 7, 0, 15, 1, 20]):
        response_mask[b, :length] = 1
    s_logp3 = s_logp.view(B, T, K)
    data = DataProto.from_dict(
        tensors={"student_top_k_ids": s_ids.view(B, T, K).int(), "student_top_k_log_probs": s_logp3,
                 "teacher_on_student_log_probs": routed_t, "overlap_mask": routed_mask},
        meta_info={"log_prob_top_k": K, "top_k_strategy": "top_p_intersec", "reward_weight_mode": "student_p",
                   "micro_batch_size": B, "temperature": 1.0, "use_dynamic_bsz": False},
    )
    actor = types.SimpleNamespace(actor_module=torch.nn.Identity())
    rm = DataParallelPPOActor.compute_distillation_reward(actor, data).batch["rm_scores"]

    for b, d in enumerate(DOMAINS):
        j = ORDER.index(d)
        valid = masks[j][b].bool()
        kl = torch.where(valid, s_logp3[b] - t_on_s[j][b], torch.zeros(()))
        w = torch.softmax(torch.where(valid, s_logp3[b], torch.full((), -float("inf"))), -1)
        w = torch.nan_to_num(w, nan=0.0)
        torch.testing.assert_close(rm[b], -kl * w, atol=1e-6, rtol=1e-5)  # softmax vs exp(x - lse) rounding
    print(f"reward: per-sample routed-teacher intersection formula reproduced (float rounding only) "
          f"(rows with an empty nucleus intersection: {int((routed_mask.sum(-1) == 0).sum())})")

    refreshed = refresh_opd_advantage(s_logp3, routed_t, response_mask, "student_p", valid_mask=routed_mask)
    torch.testing.assert_close(refreshed, rm * response_mask.unsqueeze(-1), atol=0, rtol=0)
    unmasked = refresh_opd_advantage(s_logp3, routed_t, response_mask, "student_p")
    assert not torch.allclose(unmasked, refreshed), "the mask must matter in this test"
    print("refresh: closed-form M4 refresh with the routed nucleus mask == reward x response_mask (exact)")
    print("TOP_P_INTERSEC_PASS")


if __name__ == "__main__":
    main()
