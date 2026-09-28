"""Sharpness lambda_max: top Hessian eigenvalue of the training loss, from HVPs.

Stability ratio S = eta * lambda_max / (2 + 2*beta): the edge of stability for SGD with heavy-ball
momentum beta (Kalra & Barkeshli, NeurIPS 2024; Cohen et al. 2021). S >= 1 means unstable.

Rules (CLAUDE.md section 5):
- fixed 2,048-image batch, no augmentation, fp32 outside autocast, TF32 off;
- BatchNorm in train mode (batch statistics) but running stats restored afterwards;
- restricted to surviving weights by masking the probe vector and every HVP;
- lambda_max by Lanczos with full reorthogonalization (Lanczos 1950; as in Cohen et al. 2021,
  Ghorbani et al. 2019). It deviates from the spec's "power iteration + negative shift": at dense
  ResNet-20 init the Hessian has lambda_min ~ -152, and after the shift almost the whole spectrum
  sits near +152, so 20 (even 60) shifted power iterations returned ~0 instead of lambda_max
  (NOTES.md, bug O2; pilot finding P6). Lanczos gets the algebraically largest Ritz value
  directly, needs no shift, and costs one HVP per step like power iteration;
- HVP summed over micro-batches of 512 to fit in 8 GB (each micro-batch uses its own BN stats).
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.prune import Mask

Vec = list[torch.Tensor]
HVP = Callable[[Vec], Vec]


def stability_ratio(lr: float, lam: float, momentum: float) -> float:
    """S = eta * lambda_max / (2 + 2*beta)."""
    return lr * lam / (2 + 2 * momentum)


def _dot(a: Vec, b: Vec) -> float:
    return sum((x * y).sum() for x, y in zip(a, b)).item()


def power_iteration(hvp: HVP, v: Vec, iters: int, shift: float = 0.0) -> float:
    """Rayleigh-quotient estimate of the largest-magnitude eigenvalue of (H - shift*I), plus shift."""
    lam = 0.0
    for _ in range(iters):
        norm = math.sqrt(_dot(v, v))
        v = [x / norm for x in v]
        hv = [h - shift * x for h, x in zip(hvp(v), v)]
        lam = _dot(hv, v)
        v = [h.detach() for h in hv]
    return lam + shift


def top_eigenvalue(hvp: HVP, make_v0: Callable[[], Vec], iters: int) -> float:
    """Power iteration with the negative-eigenvalue shift (the spec's original method).
    Kept for reference only: unreliable when |lambda_min| >> lambda_max (see module docstring)."""
    lam1 = power_iteration(hvp, make_v0(), iters)
    if lam1 >= 0:
        return lam1
    return power_iteration(hvp, make_v0(), iters, shift=lam1)


def lanczos_top(hvp: HVP, v0: Vec, steps: int) -> float:
    """Largest eigenvalue of a symmetric operator by `steps` Lanczos iterations (one HVP each),
    with full reorthogonalization against all previous Lanczos vectors for numerical stability."""
    norm = math.sqrt(_dot(v0, v0))
    q = [x / norm for x in v0]
    basis, alphas, betas = [], [], []
    for _ in range(steps):
        basis.append(q)
        w = [h.detach() for h in hvp(q)]
        a = _dot(w, q)
        alphas.append(a)
        for qi in basis:  # full Gram-Schmidt (includes the three-term recurrence terms)
            c = _dot(w, qi)
            w = [wi - c * qj for wi, qj in zip(w, qi)]
        b = math.sqrt(_dot(w, w))
        if b < 1e-8 * max(abs(a), 1.0):  # invariant subspace found: Ritz values are exact
            break
        betas.append(b)
        q = [wi / b for wi in w]
    k = len(alphas)
    T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    off = torch.tensor(betas[:k - 1], dtype=torch.float64)
    T += torch.diag(off, 1) + torch.diag(off, -1)
    return torch.linalg.eigvalsh(T).max().item()


@contextmanager
def exact_fp32(model: nn.Module):
    """fp32 without TF32 or autocast; BN uses batch stats but its buffers are restored.
    The model is switched to NCHW for the measurement: fp32 double-backward convolutions are
    ~1.7x slower in channels_last on the RTX 4060 (measured), while bf16 training is not."""
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    was_training = model.training
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model.to(memory_format=torch.contiguous_format)  # parameters keep their identity
    model.train()
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        model.to(memory_format=torch.channels_last)
        with torch.no_grad():
            for n, b in model.named_buffers():
                b.copy_(buffers[n])
        model.train(was_training)


def make_hvp(model: nn.Module, x: torch.Tensor, y: torch.Tensor, mask: Mask, micro_batch: int) -> HVP:
    """Masked Hessian-vector product of the mean cross-entropy over (x, y), in micro-batches."""
    names, params = zip(*model.named_parameters())
    masks = [mask.get(n) for n in names]

    def hvp(v: Vec) -> Vec:
        out = [torch.zeros_like(p) for p in params]
        for i in range(0, len(x), micro_batch):
            xb, yb = x[i:i + micro_batch], y[i:i + micro_batch]
            loss = F.cross_entropy(model(xb), yb) * (len(xb) / len(x))  # average of micro-losses
            grads = torch.autograd.grad(loss, params, create_graph=True)
            hv = torch.autograd.grad(grads, params, grad_outputs=v)  # graph freed per micro-batch
            for o, h in zip(out, hv):
                o.add_(h)
        return [o * m if m is not None else o for o, m in zip(out, masks)]

    return hvp


def lambda_max(model: nn.Module, x: torch.Tensor, y: torch.Tensor, mask: Mask,
               iters: int, micro_batch: int, seed: int = 0) -> float:
    """Top Hessian eigenvalue of the loss on (x, y), restricted to surviving weights
    (`iters` Lanczos steps = `iters` Hessian-vector products)."""
    names, params = zip(*model.named_parameters())
    gen = torch.Generator(device=params[0].device).manual_seed(seed)

    def make_v0() -> Vec:
        v = [torch.randn(p.shape, device=p.device, generator=gen) for p in params]
        return [t * mask[n] if n in mask else t for n, t in zip(names, v)]

    with exact_fp32(model):
        x = x.contiguous()
        lam = lanczos_top(make_hvp(model, x, y, mask, micro_batch), make_v0(), iters)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return lam
