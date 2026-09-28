"""SGD-momentum training loop with four modes (CLAUDE.md section 5):

- plain:  step LR (x lr_decay at each milestone);
- warmup: linear warmup from lr/warmup_iters to lr over warmup_iters (Goyal et al. 2017;
          Frankle & Carbin 2019 use it for ResNet), then the step schedule;
- sam:    Sharpness-Aware Minimization (Foret et al., ICLR 2021, Alg. 1): ascend to
          w + rho * g/||g||, take the gradient there, descend from w;
- anchor: adds (lam/2) * ||m * (theta - theta_init)||^2, which keeps weights correlated with
          their initialization (raises R_p, Liu et al. 2021) without targeting sharpness.

After every optimizer step the mask is re-applied to the weights and momentum buffers.
Mixed precision: bf16 autocast on GPUs with compute capability >= 8 (RTX 4060, Colab A100/L4/G4);
fp16 autocast + GradScaler on older GPUs such as the T4, which has no bf16 (CLAUDE.md section 4).
The sharpness trajectory lambda_max(t) and S(t) are logged on the schedule in the config.
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import Data
from src.metrics import classification_metrics, loss_spike, overlap_ratio
from src.prune import Mask, apply_mask, mask_grads
from src.sharpness import lambda_max, stability_ratio

REQUIRED_KEYS = {
    "name", "model", "dataset", "lr", "momentum", "weight_decay", "batch_size", "iters",
    "warmup_iters", "milestones", "lr_decay", "sam_rho", "anchor_lam", "prune_rate",
    "output_prune_scale", "rounds", "seeds", "baselines", "sharpness", "divergence_val_acc",
    "trained_acc", "eval_batch",
}
H4_KEYS = {"mask_from", "variants", "anchor_lam_grid", "select_iters", "select_round", "select_target"}
SHARPNESS_KEYS = {"batch", "iters", "micro_batch", "every", "dense_until", "then_every"}
EARLY_STEPS = 200      # window of the early loss spike
LOSS_CHECK_EVERY = 50  # divergence check / loss-curve resolution
S_WINDOW = 3000        # max_t S(t) is reported over the first 3k steps


def check_config(cfg: dict) -> dict:
    """Every hyper-parameter must be in the YAML (no hidden defaults). Echo it."""
    missing = REQUIRED_KEYS - cfg.keys()
    unknown = cfg.keys() - REQUIRED_KEYS - H4_KEYS
    if missing or unknown:
        raise ValueError(f"config: missing {sorted(missing)}, unknown {sorted(unknown)}")
    if SHARPNESS_KEYS != cfg["sharpness"].keys():
        raise ValueError(f"config.sharpness must have exactly {sorted(SHARPNESS_KEYS)}")
    if cfg["sam_rho"] is not None and cfg["anchor_lam"] is not None:
        raise ValueError("SAM and anchor are separate conditions; set only one")
    print("config:", cfg, flush=True)
    return cfg


def amp_dtype(device: torch.device) -> torch.dtype:
    """bf16 where the GPU supports it natively (compute capability >= 8), else fp16."""
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        return torch.float16
    return torch.bfloat16


def mode_of(cfg: dict) -> str:
    if cfg["sam_rho"] is not None:
        return "sam"
    if cfg["anchor_lam"] is not None:
        return "anchor"
    return "warmup" if cfg["warmup_iters"] > 0 else "plain"


def sharpness_steps(cfg: dict, iters: int) -> list[int]:
    """0, every `every` steps up to `dense_until`, then every `then_every`, and the final step."""
    s = cfg["sharpness"]
    steps = {0, iters} | set(range(s["every"], min(s["dense_until"], iters) + 1, s["every"]))
    steps |= set(range(s["dense_until"] + s["then_every"], iters, s["then_every"]))
    return sorted(steps)


def lr_at(cfg: dict, step: int) -> float:
    """Linear warmup (if any) times step decay at each milestone."""
    lr = cfg["lr"]
    if cfg["warmup_iters"] > 0:
        lr *= min(1.0, (step + 1) / cfg["warmup_iters"])
    return lr * cfg["lr_decay"] ** sum(step >= m for m in cfg["milestones"])


@contextmanager
def frozen_bn_stats(model: nn.Module):
    """BatchNorm momentum 0 so a forward pass does not update running stats (SAM's 2nd pass)."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, mom in zip(bns, saved):
            m.momentum = mom


class Trainer:
    """Trains `model` (already initialized and masked) under one config and one mask."""

    def __init__(self, cfg: dict, data: Data, model: nn.Module, mask: Mask, iters: int | None = None):
        self.cfg, self.data, self.model, self.mask = cfg, data, model, mask
        self.iters = cfg["iters"] if iters is None else iters
        self.mode = mode_of(cfg)
        self.device = next(model.parameters()).device
        self.opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                                   weight_decay=cfg["weight_decay"])
        apply_mask(model, mask)
        self.params = dict(model.named_parameters())
        self.init = {n: self.params[n].detach().clone() for n in mask}  # theta_init * m
        self.hx, self.hy = (t[:cfg["sharpness"]["batch"]] for t in data.hessian_batch())
        self.sharpness_s = 0.0
        self.amp_dtype = amp_dtype(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.device.type == "cuda")

    def _loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with self._autocast():
            return F.cross_entropy(self.model(x), y)

    def _anchor_grad(self) -> None:
        """Gradient of (lam/2) * ||m * (theta - theta_init)||^2 over surviving weights."""
        lam = self.cfg["anchor_lam"]
        with torch.no_grad():
            for n, m in self.mask.items():
                self.params[n].grad.add_(lam * m * (self.params[n] - self.init[n]))

    def _sam_perturb(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Move to w + rho * g / ||g|| (Foret et al. 2021, Eq. 2); return the offsets.
        g/||g|| does not depend on the GradScaler's loss scale, so scaled grads are fine."""
        grads = [(p, p.grad) for p in self.model.parameters() if p.grad is not None]
        norm = torch.sqrt(sum((g.float() ** 2).sum() for _, g in grads))
        if not torch.isfinite(norm):  # fp16 overflow: the scaler will skip this step anyway
            return []
        scale = self.cfg["sam_rho"] / (norm + 1e-12)
        offsets = []
        with torch.no_grad():
            for p, g in grads:
                e = g * scale
                p.add_(e)
                offsets.append((p, e))
        return offsets

    def _step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """One optimizer step in the configured mode; returns the (unperturbed) loss."""
        self.opt.zero_grad(set_to_none=True)
        loss = self._loss(x, y)
        self.scaler.scale(loss).backward()
        mask_grads(self.model, self.mask)
        if self.mode == "sam":
            offsets = self._sam_perturb()
            self.opt.zero_grad(set_to_none=True)
            with frozen_bn_stats(self.model):
                self.scaler.scale(self._loss(x, y)).backward()
            with torch.no_grad():
                for p, e in offsets:
                    p.sub_(e)
        self.scaler.unscale_(self.opt)  # no-op without fp16
        mask_grads(self.model, self.mask)
        if self.mode == "anchor":
            self._anchor_grad()
        self.scaler.step(self.opt)
        self.scaler.update()
        apply_mask(self.model, self.mask, self.opt)
        return loss.detach()

    def _measure(self, step: int) -> dict:
        """lambda_max, S and R_0.2 at `step` (weights after `step` updates, eta about to be used)."""
        t0 = time.time()
        sh = self.cfg["sharpness"]
        lam = lambda_max(self.model, self.hx, self.hy, self.mask, sh["iters"], sh["micro_batch"])
        lr = lr_at(self.cfg, min(step, self.iters - 1))
        wT = {n: self.params[n].detach() for n in self.mask}
        row = {"step": step, "lr": lr, "lam": lam, "S": stability_ratio(lr, lam, self.cfg["momentum"]),
               "R02": overlap_ratio(self.init, wT, self.mask, 0.2)}
        self.sharpness_s += time.time() - t0
        return row

    def _set_lr(self, step: int) -> float:
        lr = lr_at(self.cfg, step)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    def train(self, seed: int) -> dict:
        """Run the loop; stop at the first non-finite loss. Returns the training record."""
        batches = self.data.train_batches(self.cfg["batch_size"], seed)
        measure_at = set(sharpness_steps(self.cfg, self.iters))
        early, window, curve, traj = [], [], [], []
        diverged_step = None
        t0 = time.time()
        self.model.train()
        for step in range(self.iters):
            if step in measure_at:
                traj.append(self._measure(step))
            self._set_lr(step)
            x, y = next(batches)
            loss = self._step(x, y)
            if step < EARLY_STEPS:
                early.append(loss.item())
            window.append(loss)
            if len(window) == LOSS_CHECK_EVERY or step == self.iters - 1:
                losses = torch.stack(window).float()
                bad = (~torch.isfinite(losses)).nonzero()
                if len(bad):
                    diverged_step = step - len(window) + 1 + int(bad[0])
                    break
                curve.append({"step": step, "loss": losses.mean().item()})
                window = []
        if diverged_step is None and self.iters in measure_at:
            traj.append(self._measure(self.iters))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0
        return self._record(early, curve, traj, diverged_step, wall)

    @torch.no_grad()
    def evaluate(self, split: str) -> dict:
        """Metrics on 'val' or 'test' (eval-mode BN, bf16 autocast, fp32 logits)."""
        self.model.eval()
        x, y = self.data.split(split)
        bs = self.cfg["eval_batch"]
        with self._autocast():
            logits = torch.cat([self.model(self.data.normalize(x[i:i + bs])).float()
                                for i in range(0, len(x), bs)])
        self.model.train()
        return classification_metrics(logits, y)

    def _record(self, early: list, curve: list, traj: list, diverged_step: int | None, wall: float) -> dict:
        test, val = self.evaluate("test"), self.evaluate("val")
        chance = not (val["acc"] >= self.cfg["divergence_val_acc"])  # NaN counts as chance
        wT = {n: self.params[n].detach() for n in self.mask}
        finite = diverged_step is None
        early_S = [r["S"] for r in traj if r["step"] <= S_WINDOW]
        return {
            "mode": self.mode, "iters": self.iters, "amp_dtype": str(self.amp_dtype).replace("torch.", ""),
            "diverged": diverged_step is not None or chance, "diverged_step": diverged_step,
            "ended_at_chance": chance, "trained": bool(test["acc"] > self.cfg["trained_acc"]),
            "test": test, "val_acc": val["acc"],
            "R01": overlap_ratio(self.init, wT, self.mask, 0.1) if finite else float("nan"),
            "R02": overlap_ratio(self.init, wT, self.mask, 0.2) if finite else float("nan"),
            "loss_spike": loss_spike(early), "early_losses": early, "loss_curve": curve,
            "sharpness": traj, "max_S_3k": max(early_S, default=float("nan")),
            "max_S": max((r["S"] for r in traj), default=float("nan")),
            "wall_s": wall, "sharpness_s": self.sharpness_s,
            "sharpness_overhead": self.sharpness_s / max(wall - self.sharpness_s, 1e-9),
            "it_per_s": (diverged_step or self.iters) / max(wall - self.sharpness_s, 1e-9),
        }


def nan_to_none(obj):
    """JSON-safe copy: NaN / inf -> None (strict JSON has no NaN)."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    return obj
