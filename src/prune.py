"""Masks, global magnitude pruning and mask enforcement (Frankle & Carbin, ICLR 2019).

Pruned: all conv and linear weights except the first conv. Never pruned: biases, BatchNorm
parameters, the first conv. The output linear layer is pruned separately at half the rate.
"""
from __future__ import annotations

import torch
import torch.nn as nn

Mask = dict[str, torch.Tensor]


def _weight_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def prunable_names(model: nn.Module) -> list[str]:
    """Parameter names of all conv/linear weights except the first conv, in module order."""
    layers = _weight_layers(model)
    first_conv = next(n for n, m in layers if isinstance(m, nn.Conv2d))
    return [f"{n}.weight" for n, _ in layers if n != first_conv]


def output_name(model: nn.Module) -> str:
    """Name of the final linear layer's weight (pruned at half the global rate)."""
    return [f"{n}.weight" for n, m in _weight_layers(model) if isinstance(m, nn.Linear)][-1]


def ones_mask(model: nn.Module) -> Mask:
    params = dict(model.named_parameters())
    return {n: torch.ones_like(params[n]) for n in prunable_names(model)}


@torch.no_grad()
def apply_mask(model: nn.Module, mask: Mask, optimizer: torch.optim.Optimizer | None = None) -> None:
    """Zero the pruned weights and, if given, their SGD momentum buffers."""
    params = dict(model.named_parameters())
    for n, m in mask.items():
        params[n].mul_(m)
        if optimizer is not None:
            buf = optimizer.state.get(params[n], {}).get("momentum_buffer")
            if buf is not None:
                buf.mul_(m)


@torch.no_grad()
def mask_grads(model: nn.Module, mask: Mask) -> None:
    params = dict(model.named_parameters())
    for n, m in mask.items():
        if params[n].grad is not None:
            params[n].grad.mul_(m)


def _prune_group(names: list[str], weights: dict[str, torch.Tensor], mask: Mask, rate: float) -> Mask:
    """Remove round(rate * alive) of the smallest-magnitude surviving weights, pooled over `names`."""
    alive = [mask[n].flatten().nonzero().squeeze(1) for n in names]
    scores = torch.cat([weights[n].flatten()[i].abs().float() for n, i in zip(names, alive)])
    k = round(rate * scores.numel())
    keep = torch.ones_like(scores, dtype=torch.bool)
    if k > 0:
        keep[torch.topk(scores, k, largest=False).indices] = False
    out, start = {}, 0
    for n, idx in zip(names, alive):
        flat = torch.zeros(mask[n].numel(), device=mask[n].device, dtype=mask[n].dtype)
        flat[idx[keep[start:start + len(idx)]]] = 1
        out[n] = flat.view_as(mask[n])
        start += len(idx)
    return out


def global_magnitude_prune(weights: dict[str, torch.Tensor], mask: Mask, rate: float,
                           out_name: str, output_scale: float) -> Mask:
    """One IMP round: prune `rate` of the surviving weights globally by |w|, and the output
    layer at `rate * output_scale` (Frankle & Carbin 2019, Sec. 2 and App. on ResNet)."""
    for n in mask:
        if not torch.isfinite(weights[n]).all():
            raise ValueError(f"non-finite weights in {n}: never build a mask from a diverged run")
    body = [n for n in mask if n != out_name]
    new = _prune_group(body, weights, mask, rate)
    new.update(_prune_group([out_name], weights, mask, rate * output_scale))
    return new


def density(mask: Mask) -> float:
    """Fraction of prunable weights that survive."""
    return sum(m.sum().item() for m in mask.values()) / sum(m.numel() for m in mask.values())


def remaining_fraction(model: nn.Module, mask: Mask) -> float:
    """Fraction of all conv + linear weights (including the unpruned first conv) that survive."""
    params = dict(model.named_parameters())
    names = [f"{n}.weight" for n, _ in _weight_layers(model)]
    alive = sum(mask[n].sum().item() if n in mask else params[n].numel() for n in names)
    return alive / sum(params[n].numel() for n in names)
