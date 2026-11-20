
"""Chunked LM-head reductions for MOPD scoring and training.

Enabled with OPENMOPD_CHUNKED_HEAD=<tokens per chunk> (0/unset = original full-logits path).

The model's ``lm_head`` is temporarily replaced for one forward call:
  * student (``student_head``): the replacement receives the final hidden states and computes, chunk by chunk,
    exactly the per-token quantities the full-logits code computes (target log-prob, top-k ids/log-probs,
    entropy), returning a dict of compact tensors as ``output.logits``.
    With grad enabled, the whole head is one autograd node (``_ChunkedStudentHead``): forward keeps no logits, and
    backward recomputes one chunk, differentiates it and frees it before the next, so at most one chunk's
    [chunk, vocab] logits exist at any time in forward or backward. OPENMOPD_CHUNKED_HEAD_BACKWARD=checkpoint
    selects the earlier per-chunk activation-checkpoint variant, whose recomputed chunks all stay alive until the
    end of backward.
  * teacher (``hidden_head``): returns the hidden states and keeps a private GPU copy of the head weight
    (teacher parameters may be CPU-offloaded right after the forward); callers reduce them with ``head_logits``
    chunk by chunk.
All per-token math is unchanged; only the token dimension is split.
"""
import contextlib
import os

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunk_size():
    return int(os.environ.get("OPENMOPD_CHUNKED_HEAD", "0") or 0)


def backward_mode():
    mode = os.environ.get("OPENMOPD_CHUNKED_HEAD_BACKWARD", "function")
    if mode not in ("function", "checkpoint"):
        raise ValueError(f"OPENMOPD_CHUNKED_HEAD_BACKWARD must be function or checkpoint, got {mode!r}")
    return mode


def find_lm_head(module):
    for name, sub in module.named_modules():
        if name.split(".")[-1] == "lm_head" and hasattr(sub, "weight"):
            return sub
    raise RuntimeError("chunked head: model has no lm_head")


def head_logits(hidden, weight, bias, temperature):
    logits = F.linear(hidden, weight, bias)
    if temperature != 1.0:
        logits = logits.div_(temperature)
    return logits


def _student_chunk(hidden, weight, bias, labels, ids, temperature, top_k, entropy_fn):
    logits = head_logits(hidden, weight, bias, temperature)
    topk_ids = topk_log_probs = None
    if top_k > 0:
        log_probs_all = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        topk_ids = ids if ids is not None else torch.topk(logits, k=top_k, dim=-1)[1]
        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)
    else:
        from verl.utils.torch_functional import logprobs_from_logits
        log_probs = logprobs_from_logits(logits=logits, labels=labels, inplace_backward=False)
    entropy = entropy_fn(logits) if entropy_fn is not None else None
    return log_probs, entropy, topk_ids, topk_log_probs


def _cat(outs, i):
    return None if outs[0][i] is None else torch.cat([o[i] for o in outs], dim=0)


class _ChunkedStudentHead(torch.autograd.Function):
    """Outputs (log_probs, entropy, topk_ids, topk_log_probs) for all tokens; one chunk alive at a time."""

    @staticmethod
    def forward(ctx, flat, weight, bias, labels, ids, temperature, top_k, entropy_fn, chunk):
        outs = []
        for start in range(0, flat.size(0), chunk):
            end = min(start + chunk, flat.size(0))
            outs.append(_student_chunk(flat[start:end], weight, bias, labels[start:end],
                                       None if ids is None else ids[start:end], temperature, top_k, entropy_fn))
        log_probs, entropy, topk_ids, topk_log_probs = (_cat(outs, i) for i in range(4))
        del outs
        # Backward reuses the forward's top-k ids (fixed candidates), so recomputation selects nothing again.
        ctx.save_for_backward(flat, weight, bias, labels, topk_ids if top_k > 0 else None)
        ctx.temperature, ctx.top_k, ctx.entropy_fn, ctx.chunk = temperature, top_k, entropy_fn, chunk
        ctx.set_materialize_grads(False)
        if topk_ids is not None:
            ctx.mark_non_differentiable(topk_ids)
        return log_probs, entropy, topk_ids, topk_log_probs

    @staticmethod
    def backward(ctx, g_log_probs, g_entropy, g_topk_ids, g_topk_log_probs):
        flat, weight, bias, labels, ids = ctx.saved_tensors
        need_h, need_w, need_b = ctx.needs_input_grad[:3]
        grad_h = torch.zeros_like(flat) if need_h else None
        grad_w = torch.zeros_like(weight) if need_w else None
        grad_b = torch.zeros_like(bias) if need_b and bias is not None else None
        entropy_fn = ctx.entropy_fn if g_entropy is not None else None
        for start in range(0, flat.size(0), ctx.chunk):
            end = min(start + ctx.chunk, flat.size(0))
            with torch.enable_grad():
                h = flat[start:end].detach().requires_grad_(need_h)
                w = weight.detach().requires_grad_(need_w)
                b = None if bias is None else bias.detach().requires_grad_(need_b)
                lp, ent, _, tlp = _student_chunk(h, w, b, labels[start:end], None if ids is None else ids[start:end],
                                                 ctx.temperature, ctx.top_k, entropy_fn)
                outputs, grads = [], []
                for out, grad in ((lp, g_log_probs), (ent, g_entropy), (tlp, g_topk_log_probs)):
                    if out is not None and grad is not None and out.requires_grad:
                        outputs.append(out)
                        grads.append(grad[start:end])
                inputs = [t for t, need in ((h, need_h), (w, need_w), (b, need_b)) if need and t is not None]
                if not outputs or not inputs:
                    continue
                result = list(torch.autograd.grad(outputs, inputs, grads, allow_unused=True))
            del lp, ent, tlp, outputs
            if need_h:
                gh = result.pop(0)
                if gh is not None:
                    grad_h[start:end] = gh
            if need_w:
                gw = result.pop(0)
                if gw is not None:
                    grad_w.add_(gw)
                del gw
            if need_b and bias is not None:
                gb = result.pop(0)
                if gb is not None:
                    grad_b.add_(gb)
        return grad_h, grad_w, grad_b, None, None, None, None, None, None


@contextlib.contextmanager
def student_head(module, *, labels, ids, temperature, top_k, entropy_fn, chunk):
    """labels: [T] next-token ids; ids: [T, K] fixed candidate ids or None (select top-k)."""
    head = find_lm_head(module)
    mode = backward_mode()

    def forward(hidden):
        flat = hidden.reshape(-1, hidden.size(-1))
        if flat.size(0) != labels.size(0):
            raise ValueError(f"chunked head: {flat.size(0)} hidden rows vs {labels.size(0)} labels")
        if torch.is_grad_enabled() and mode == "function":
            lp, ent, tid, tlp = _ChunkedStudentHead.apply(flat, head.weight, head.bias, labels, ids, temperature,
                                                          top_k, entropy_fn, chunk)
            return {"log_probs": lp, "entropy": ent, "topk_ids": tid, "topk_log_probs": tlp}
        outs = []
        for start in range(0, flat.size(0), chunk):
            end = min(start + chunk, flat.size(0))
            lab = labels[start:end]
            cid = None if ids is None else ids[start:end]

            def fn(h, w, lab=lab, cid=cid):
                return _student_chunk(h, w, head.bias, lab, cid, temperature, top_k, entropy_fn)

            if torch.is_grad_enabled():
                outs.append(checkpoint(fn, flat[start:end], head.weight, use_reentrant=False))
            else:
                outs.append(fn(flat[start:end], head.weight))
        return {"log_probs": _cat(outs, 0), "entropy": _cat(outs, 1), "topk_ids": _cat(outs, 2),
                "topk_log_probs": _cat(outs, 3)}

    head.forward = forward
    try:
        yield
    finally:
        del head.forward


@contextlib.contextmanager
def hidden_head(module):
    """lm_head returns its input; the returned tensor carries ._head_weight/._head_bias GPU copies."""
    head = find_lm_head(module)
    stash = {}

    def forward(hidden):
        stash["weight"] = head.weight.detach().clone()
        stash["bias"] = None if head.bias is None else head.bias.detach().clone()
        return hidden

    head.forward = forward
    try:
        yield stash
    finally:
        del head.forward


def attach(hidden, stash):
    hidden._head_weight = stash["weight"]
    hidden._head_bias = stash["bias"]
    return hidden


def is_hidden(tensor):
    return getattr(tensor, "_head_weight", None) is not None
