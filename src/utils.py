"""Shared helpers: seeding, run metadata, config loading, stdout tee and the compute log."""
from __future__ import annotations

import csv
import datetime as dt
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
COMPUTE_LOG = RESULTS / "compute_log.csv"

# Laptop budget (CLAUDE.md section 9), in GPU-hours on the RTX 4060.
BUDGET_GPU_H = {"4060": 18.0}
# Colab budget: 30 h of runtime (wall-clock usage, whatever the GPU), not GPU-hours. Parallel chains
# under MPS share the same runtime, so usage is the union of the training intervals, not their sum.
RUNTIME_BUDGET_H = 30.0
# Runtime already used before RUNTIME_SINCE (stated by the user: session 4 27 h left; session 5 20 h left, 15 h left at 09:30 UTC).
# Only trainings that end after RUNTIME_SINCE are added on top. Re-calibrate both from the Colab
# usage page when they drift: idle runtime (setup, analysis, gaps between runs) is not logged.
RUNTIME_USED_BEFORE_H = 16.0
RUNTIME_SINCE = "2026-10-01T03:37:00"


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA). cudnn.benchmark stays on, so runs are
    reproducible in distribution but not bit-exact (CLAUDE.md section 11)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def git_hash() -> str:
    """Current commit hash, with '-dirty' if the tree has uncommitted changes."""
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def run_metadata() -> dict:
    """Environment facts recorded in every run JSON (CLAUDE.md section 4a)."""
    return {
        "git_hash": git_hash(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu_name(),
        "python": platform.python_version(),
        "started": dt.datetime.now().isoformat(timespec="seconds"),
    }


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Tee:
    """Duplicate stdout into a log file (long runs are started from a terminal)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()


def is_laptop(gpu: str) -> bool:
    return any(key in gpu for key in BUDGET_GPU_H)


def budget_hours(gpu: str) -> float:
    """Budget of the machine in use: 18 GPU-h on the RTX 4060 (section 9), else 30 h of Colab runtime."""
    for key, hours in BUDGET_GPU_H.items():
        if key in gpu:
            return hours
    return RUNTIME_BUDGET_H


def _compute_rows(gpu: str | None = None) -> list[dict]:
    if not COMPUTE_LOG.exists():
        return []
    with open(COMPUTE_LOG, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if gpu is None or r["gpu"] == gpu]


def cumulative_gpu_hours(gpu: str) -> float:
    """Sum of logged GPU time on this exact GPU model (inflated for parallel runs)."""
    return sum(float(r["wall_s"]) / 3600 for r in _compute_rows(gpu))


def _colab_intervals() -> list[tuple[dt.datetime, dt.datetime]]:
    """[start, end] of each non-laptop training that ended after RUNTIME_SINCE, sorted.
    The log's `time` is written when a training finishes, so it is the end of the interval."""
    since = dt.datetime.fromisoformat(RUNTIME_SINCE)
    out = []
    for r in _compute_rows():
        end = dt.datetime.fromisoformat(r["time"])
        if not is_laptop(r["gpu"]) and end > since:
            out.append((max(end - dt.timedelta(seconds=float(r["wall_s"])), since), end))
    return sorted(out)


def union_hours(intervals: list[tuple[dt.datetime, dt.datetime]]) -> float:
    """Total length of the union of sorted intervals, in hours."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += (cur_e - cur_s).total_seconds()
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += (cur_e - cur_s).total_seconds()
    return total / 3600


def runtime_hours_used() -> float:
    """Colab runtime used: the calibrated amount before RUNTIME_SINCE plus the union of later trainings."""
    return RUNTIME_USED_BEFORE_H + union_hours(_colab_intervals())


def effective_minutes_per_training() -> float | None:
    """Runtime per finished training since RUNTIME_SINCE (accounts for parallel chains), if any."""
    iv = _colab_intervals()
    return 60 * union_hours(iv) / len(iv) if iv else None


def budget_used(gpu: str) -> float:
    """GPU-h on the laptop, runtime hours on Colab (same units as budget_hours)."""
    return cumulative_gpu_hours(gpu) if is_laptop(gpu) else runtime_hours_used()


def append_compute_log(row: dict) -> None:
    """Append one training to results/compute_log.csv and warn at 80% / 95% of the budget."""
    COMPUTE_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not COMPUTE_LOG.exists()
    with open(COMPUTE_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    used, budget = budget_used(row["gpu"]), budget_hours(row["gpu"])
    for frac in (0.95, 0.80):
        if used >= frac * budget:
            print(f"WARNING: {used:.2f} h used, past {frac:.0%} of the {budget:.0f} h budget")
            break
