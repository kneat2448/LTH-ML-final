"""Control initializations for a fixed mask (CLAUDE.md section 5, pilot finding P4).

- reinit: same mask, fresh default init with seed + 1000 (Frankle & Carbin 2019, Sec. 2).
- rescaled_reinit: reinit with each layer scaled by 1/sqrt(density), so the fan-in variance
  matches dense He init (cf. Evci et al. 2022, "Gradient Flow in Sparse Neural Networks").
- shuffle: theta_0 values permuted within each layer among the surviving positions; keeps
  each layer's weight distribution exactly (layerwise shuffle, Frankle et al. 2021, "Pruning
  Neural Networks at Initialization: Why Are We Missing the Mark?").
"""
from __future__ import annotations

import torch

from src.models import init_model
from src.prune import Mask

State = dict[str, torch.Tensor]

REINIT_SEED_OFFSET = 1000
SHUFFLE_SEED_OFFSET = 2000


def reinit(model_name: str, seed: int, mask: Mask) -> State:
    """Fresh default init with seed + 1000, pruned with `mask`."""
    state = init_model(model_name, seed + REINIT_SEED_OFFSET).state_dict()
    for n, m in mask.items():
        state[n] = state[n] * m.to(state[n].device)
    return state


def rescaled_reinit(model_name: str, seed: int, mask: Mask) -> State:
    """`reinit` with each pruned layer multiplied by 1 / sqrt(layer density)."""
    state = reinit(model_name, seed, mask)
    for n, m in mask.items():
        d = m.float().mean().clamp_min(1e-12)
        state[n] = state[n] / d.sqrt().to(state[n].device)
    return state


def shuffle(theta0: State, mask: Mask, seed: int) -> State:
    """Permute the surviving theta_0 values of each layer among that layer's surviving positions.
    Unpruned tensors (first conv, BatchNorm, biases) keep their theta_0 values."""
    gen = torch.Generator().manual_seed(seed + SHUFFLE_SEED_OFFSET)
    state = {k: v.clone() for k, v in theta0.items()}
    for n, m in mask.items():
        keep = m.bool().to(state[n].device)
        vals = state[n][keep]
        perm = torch.randperm(vals.numel(), generator=gen).to(vals.device)
        w = torch.zeros_like(state[n])
        w[keep] = vals[perm]
        state[n] = w
    return state
