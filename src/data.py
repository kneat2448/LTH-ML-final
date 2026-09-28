"""Datasets held on the GPU as uint8, with GPU random-crop + flip (CLAUDE.md section 4b).

CIFAR-10: 45k train / 5k val carved from the official 50k (seed 0), official 10k test.
Fashion-MNIST (Conv-4 de-risking chain only): 55k train / 5k val from the official 60k.
Splits, per-channel statistics and the fixed Hessian batch indices are cached in data/.
The raw CIFAR-10 archive goes to $LTH_RAW_DATA if set (on Colab: fast local disk instead of
Google Drive), else to data/ as well.
"""
from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from src.utils import ROOT

DATA = ROOT / "data"
RAW_DATA = Path(os.environ.get("LTH_RAW_DATA", DATA))
FMNIST_RAW = ROOT / "pilot" / "data"
SPLIT_SEED = 0
N_VAL = 5000
N_HESSIAN = 2048


@dataclass
class Data:
    """All splits on one device. Images are uint8 (N, C, H, W); labels are int64."""
    xtr: torch.Tensor
    ytr: torch.Tensor
    xval: torch.Tensor
    yval: torch.Tensor
    xte: torch.Tensor
    yte: torch.Tensor
    mean: torch.Tensor  # (1, C, 1, 1), in [0, 1] pixel units
    std: torch.Tensor
    hessian_idx: torch.Tensor  # positions inside the train split
    augment: bool
    num_classes: int = 10

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 or [0, 1] float images -> standardized float32, channels_last."""
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = (x - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last)

    def train_batches(self, batch_size: int, seed: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Endless stream of augmented training batches; a fresh permutation every epoch."""
        gen = torch.Generator(device=self.xtr.device).manual_seed(seed)
        n = len(self.xtr)
        while True:
            perm = torch.randperm(n, device=self.xtr.device, generator=gen)
            for i in range(0, n - batch_size + 1, batch_size):  # drop the last partial batch
                idx = perm[i:i + batch_size]
                x = self.xtr[idx].float() / 255.0
                if self.augment:
                    x = random_crop_flip(x, pad=4, gen=gen)
                yield self.normalize(x), self.ytr[idx]

    def hessian_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The fixed 2,048-image Hessian batch: training images, no augmentation."""
        return self.normalize(self.xtr[self.hessian_idx]), self.ytr[self.hessian_idx]

    def split(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        return {"val": (self.xval, self.yval), "test": (self.xte, self.yte)}[name]


def random_crop_flip(x: torch.Tensor, pad: int, gen: torch.Generator) -> torch.Tensor:
    """Per-image random crop with `pad` px zero padding, then random horizontal flip, on GPU."""
    b, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))  # zero = black pixel, as torchvision's RandomCrop
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device, generator=gen)
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device, generator=gen)
    rows = (oy[:, None] + torch.arange(h, device=x.device))[:, :, None]  # (B, H, 1)
    cols = (ox[:, None] + torch.arange(w, device=x.device))[:, None, :]  # (B, 1, W)
    batch = torch.arange(b, device=x.device)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[batch, rows, cols].permute(0, 3, 1, 2)  # (B, C, H, W)
    flip = torch.rand(b, device=x.device, generator=gen) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


def _cached(file: str, key: str, compute) -> object:
    """Read data/<file>[key], computing and saving it on first use (never recomputed)."""
    path = DATA / file
    store = json.loads(path.read_text()) if path.exists() else {}
    if key not in store:
        store[key] = compute()
        DATA.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(store))
    return store[key]


def _load_cifar10_raw() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from torchvision.datasets import CIFAR10  # MD5 of the archive is checked by torchvision
    tr = CIFAR10(root=str(RAW_DATA), train=True, download=True)
    te = CIFAR10(root=str(RAW_DATA), train=False, download=True)
    to_nchw = lambda a: np.ascontiguousarray(a.transpose(0, 3, 1, 2))
    return to_nchw(tr.data), np.array(tr.targets), to_nchw(te.data), np.array(te.targets)


def _load_fmnist_raw() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def read(kind: str) -> tuple[np.ndarray, np.ndarray]:
        with gzip.open(FMNIST_RAW / f"{kind}-images-idx3-ubyte.gz") as f:
            x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28).copy()
        with gzip.open(FMNIST_RAW / f"{kind}-labels-idx1-ubyte.gz") as f:
            y = np.frombuffer(f.read(), np.uint8, offset=8).copy()
        return x, y.astype(np.int64)
    (xtr, ytr), (xte, yte) = read("train"), read("t10k")
    return xtr, ytr, xte, yte


def load_data(name: str, device: str) -> Data:
    """Load `cifar10` or `fmnist` onto `device` with the fixed splits and cached statistics."""
    raw = {"cifar10": _load_cifar10_raw, "fmnist": _load_fmnist_raw}[name]
    x_all, y_all, xte, yte = raw()

    def make_split() -> dict:
        perm = np.random.default_rng(SPLIT_SEED).permutation(len(x_all))
        return {"train": perm[:-N_VAL].tolist(), "val": perm[-N_VAL:].tolist()}
    split = _cached("splits.json", name, make_split)
    tr, va = np.array(split["train"]), np.array(split["val"])

    def make_stats() -> dict:
        x = x_all[tr].astype(np.float64) / 255.0  # statistics of the train split only
        return {"mean": x.mean(axis=(0, 2, 3)).tolist(), "std": x.std(axis=(0, 2, 3)).tolist()}
    stats = _cached("stats.json", name, make_stats)

    def make_hessian_idx() -> list:
        return np.random.default_rng(SPLIT_SEED).choice(len(tr), N_HESSIAN, replace=False).tolist()
    hidx = _cached("hessian_idx.json", name, make_hessian_idx)

    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    shape = (1, -1, 1, 1)
    return Data(
        xtr=t(x_all[tr], torch.uint8), ytr=t(y_all[tr], torch.long),
        xval=t(x_all[va], torch.uint8), yval=t(y_all[va], torch.long),
        xte=t(xte, torch.uint8), yte=t(yte, torch.long),
        mean=t(stats["mean"], torch.float32).view(shape), std=t(stats["std"], torch.float32).view(shape),
        hessian_idx=t(hidx, torch.long), augment=(name == "cifar10"),
    )
