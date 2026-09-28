"""Evaluation metrics, written from scratch and tested against sklearn (tests/test_metrics.py).

- accuracy, macro precision / recall / F1, macro one-vs-rest ROC-AUC, confusion matrix;
- R_p overlap ratio between theta_0*m and theta_T*m (Liu et al., ICML 2021, Eq. 1);
- early loss spike = max training loss in the first 200 steps / loss at step 0.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from src.prune import Mask


def confusion_matrix(y: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    """cm[i, j] = number of samples with true class i predicted as j."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def macro_prf(cm: np.ndarray) -> tuple[float, float, float]:
    """Macro-averaged precision, recall and F1 (0 where undefined, as sklearn zero_division=0)."""
    tp = np.diag(cm).astype(float)
    pred_pos, true_pos = cm.sum(0).astype(float), cm.sum(1).astype(float)
    prec = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    rec = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    denom = pred_pos + true_pos  # 2TP + FP + FN
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(prec.mean()), float(rec.mean()), float(f1.mean())


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given their average rank."""
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def binary_auc(is_pos: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic."""
    n_pos, n_neg = is_pos.sum(), (~is_pos).sum()
    ranks = _average_ranks(score)
    return float((ranks[is_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_ovr_auc(y: np.ndarray, prob: np.ndarray) -> float:
    """Unweighted mean over classes of the one-vs-rest AUC."""
    return float(np.mean([binary_auc(y == c, prob[:, c]) for c in range(prob.shape[1])]))


def classification_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict:
    """Test-set metrics from logits. All NaN if the logits are non-finite (diverged run)."""
    num_classes = logits.shape[1]
    if not torch.isfinite(logits).all():
        nan = float("nan")
        return {"acc": nan, "precision": nan, "recall": nan, "f1": nan, "auc": nan, "confusion": None}
    prob = torch.softmax(logits.double(), 1).cpu().numpy()
    yt = y.cpu().numpy()
    pred = prob.argmax(1)
    cm = confusion_matrix(yt, pred, num_classes)
    p, r, f1 = macro_prf(cm)
    return {"acc": float((pred == yt).mean()), "precision": p, "recall": r, "f1": f1,
            "auc": macro_ovr_auc(yt, prob), "confusion": cm.tolist()}


def overlap_ratio(w0: dict[str, torch.Tensor], wT: dict[str, torch.Tensor], mask: Mask, p: float) -> float:
    """R_p (Liu et al. 2021, Eq. 1): share of the top-p fraction of surviving weights by |theta_0|
    that is also in the top-p fraction by |theta_T|, counted per layer and pooled. Chance ~ p."""
    num = den = 0
    for n, m in mask.items():
        keep = m.bool().flatten()
        a0, aT = w0[n].flatten()[keep].abs(), wT[n].flatten()[keep].abs()
        k = max(1, int(p * keep.sum().item()))
        top0 = torch.zeros(len(a0), dtype=torch.bool, device=a0.device)
        topT = torch.zeros_like(top0)
        top0[torch.topk(a0, k).indices] = True
        topT[torch.topk(aT, k).indices] = True
        num += (top0 & topT).sum().item()
        den += k
    return num / den


def loss_spike(early_losses: list[float]) -> float:
    """Max training loss in the first 200 steps divided by the loss at step 0."""
    if not early_losses or not math.isfinite(early_losses[0]):
        return float("nan")
    window = early_losses[:200]
    if not all(math.isfinite(v) for v in window):
        return float("inf")  # the loss blew up inside the window
    return max(window) / early_losses[0]
