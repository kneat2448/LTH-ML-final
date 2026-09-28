"""lambda_max (Lanczos) vs the exact eigendecomposition (CLAUDE.md section 12), at the
production setting of 20 HVPs, including a dominant negative eigenvalue."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from src.sharpness import lambda_max, lanczos_top


def _tiny_mlp() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 3)).double()


def _exact_hessian(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    named = list(model.named_parameters())
    flat0 = torch.cat([p.detach().flatten() for _, p in named])

    def loss(flat: torch.Tensor) -> torch.Tensor:
        parts, i = {}, 0
        for n, p in named:
            parts[n] = flat[i:i + p.numel()].view(p.shape)
            i += p.numel()
        return F.cross_entropy(functional_call(model, parts, (x,)), y)

    return torch.autograd.functional.hessian(loss, flat0)


def _batch():
    g = torch.Generator().manual_seed(0)
    return torch.randn(32, 4, dtype=torch.double, generator=g), torch.randint(0, 3, (32,), generator=g)


def test_lambda_max_matches_eigvalsh_on_mlp():
    model, (x, y) = _tiny_mlp(), _batch()
    exact = torch.linalg.eigvalsh(_exact_hessian(model, x, y)).max().item()
    est = lambda_max(model, x, y, {}, iters=20, micro_batch=16)  # 2 micro-batches
    assert abs(est - exact) <= 0.02 * abs(exact)


def test_lambda_max_masked():
    model, (x, y) = _tiny_mlp(), _batch()
    g = torch.Generator().manual_seed(1)
    mask = {"0.weight": (torch.rand(6, 4, generator=g) > 0.5).double(),
            "2.weight": (torch.rand(3, 6, generator=g) > 0.5).double()}
    keep = torch.cat([mask.get(n, torch.ones_like(p)).flatten().bool() for n, p in model.named_parameters()])
    exact = torch.linalg.eigvalsh(_exact_hessian(model, x, y)[keep][:, keep]).max().item()
    est = lambda_max(model, x, y, mask, iters=20, micro_batch=32)
    assert abs(est - exact) <= 0.02 * abs(exact)


def _spectrum_matrix(evals: torch.Tensor, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(len(evals), len(evals), dtype=torch.double, generator=g))
    return q @ torch.diag(evals) @ q.T


def _resnet_like_spectrum() -> torch.Tensor:
    """What dense ResNet-20 at init looks like: lambda_min ~ -152 dominates, lambda_max ~ 30,
    and a large bulk near zero (measured on CIFAR-10, NOTES.md bug O2)."""
    g = torch.Generator().manual_seed(1)
    bulk = 0.5 * torch.randn(400, dtype=torch.double, generator=g)
    return torch.cat([torch.tensor([-152.0, -60.0, 30.0, 22.0, 15.0], dtype=torch.double), bulk])


def test_lanczos_negative_dominant_spectrum():
    A = _spectrum_matrix(_resnet_like_spectrum())
    g = torch.Generator().manual_seed(2)
    est = lanczos_top(lambda v: [A @ v[0]], [torch.randn(len(A), dtype=torch.double, generator=g)], steps=20)
    assert abs(est - 30.0) <= 0.02 * 30.0

