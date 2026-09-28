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

# GPU-hour budget per machine (CLAUDE.md section 9).
BUDGET_GPU_H = {"4060": 18.0, "T4": 30.0}


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


def budget_hours(gpu: str) -> float | None:
    """GPU-hour budget of the machine in use (section 9): 18 h on the RTX 4060, 30 h on a T4.
    Any other GPU (e.g. a Colab G4) needs the budget in the env var LTH_BUDGET_H."""
    for key, hours in BUDGET_GPU_H.items():
        if key in gpu:
            return hours
    if os.environ.get("LTH_BUDGET_H"):
        return float(os.environ["LTH_BUDGET_H"])
    print(f"WARNING: no GPU-hour budget known for {gpu}; set LTH_BUDGET_H (see NOTES.md)")
    return None


def cumulative_gpu_hours(gpu: str) -> float:
    """Sum of logged GPU time on this exact GPU model."""
    if not COMPUTE_LOG.exists():
        return 0.0
    with open(COMPUTE_LOG, newline="", encoding="utf-8") as f:
        return sum(float(r["wall_s"]) / 3600 for r in csv.DictReader(f) if r["gpu"] == gpu)


def append_compute_log(row: dict) -> None:
    """Append one training to results/compute_log.csv and warn at 80% / 95% of budget."""
    COMPUTE_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not COMPUTE_LOG.exists()
    with open(COMPUTE_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    used, budget = cumulative_gpu_hours(row["gpu"]), budget_hours(row["gpu"])
    if budget is None:
        return
    for frac in (0.95, 0.80):
        if used >= frac * budget:
            print(f"WARNING: {used:.2f} GPU-h used, past {frac:.0%} of the {budget:.0f} h budget")
            break
