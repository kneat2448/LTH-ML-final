"""Shared fixtures: a tiny synthetic dataset so the trainer can be tested without CIFAR-10."""
import pytest
import torch

from src.data import Data


def make_data(n: int = 256, channels: int = 3, size: int = 32, device: str = "cpu") -> Data:
    g = torch.Generator().manual_seed(0)
    x = lambda k: torch.randint(0, 256, (k, channels, size, size), generator=g, dtype=torch.uint8).to(device)
    y = lambda k: torch.randint(0, 10, (k,), generator=g).to(device)
    return Data(xtr=x(n), ytr=y(n), xval=x(64), yval=y(64), xte=x(64), yte=y(64),
                mean=torch.full((1, channels, 1, 1), 0.5, device=device),
                std=torch.full((1, channels, 1, 1), 0.25, device=device),
                hessian_idx=torch.arange(64, device=device), augment=True)


@pytest.fixture
def tiny_data() -> Data:
    return make_data()


def tiny_config(**over) -> dict:
    cfg = {
        "name": "test", "model": "resnet20", "dataset": "synthetic", "lr": 0.1, "momentum": 0.9,
        "weight_decay": 1e-4, "batch_size": 32, "iters": 10, "warmup_iters": 0, "milestones": [8],
        "lr_decay": 0.1, "sam_rho": None, "anchor_lam": None, "prune_rate": 0.3,
        "output_prune_scale": 0.5, "rounds": 1, "seeds": [0], "baselines": {},
        "sharpness": {"batch": 64, "iters": 3, "micro_batch": 32, "every": 5, "dense_until": 10, "then_every": 100},
        "divergence_val_acc": 0.0, "trained_acc": 0.2, "eval_batch": 32,
    }
    cfg.update(over)
    return cfg
