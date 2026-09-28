"""Iterative magnitude pruning with reset to theta_0 (Frankle & Carbin, ICLR 2019), resumable.

Usage:  python -m src.imp configs/resnet20_warm03.yaml [--seeds 0 1] [--fresh]

Per round r: train the ticket theta_0 * m_r, run the scheduled baselines on m_r, then prune
30% of the surviving weights of the trained ticket to get m_{r+1}. Every finished training
writes results/<name>/seed<k>/round<r>_<variant>.json; a rerun skips finished trainings.
Divergence rule (pilot P3): if the ticket diverges, run this round's baselines on its (good)
mask and stop the chain; a mask is never built from a diverged run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from src import baselines
from src.data import Data, load_data
from src.models import init_model
from src.prune import Mask, density, global_magnitude_prune, ones_mask, output_name, remaining_fraction
from src.train import Trainer, check_config, mode_of, nan_to_none
from src.utils import (CHECKPOINTS, COMPUTE_LOG, RESULTS, Tee, append_compute_log, budget_hours,
                       cumulative_gpu_hours, gpu_name, load_yaml, run_metadata, set_seed, timestamp)

# Section 9 assumption, used until results/compute_log.csv has measurements for this GPU.
ASSUMED_MIN_PER_15K = {"4060": 6.0, "other": 10.0}


def run_dirs(cfg: dict, seed: int, fresh: bool) -> tuple[Path, Path]:
    """results/ and checkpoints/ dirs for (cfg, seed). An existing dir is only resumed if its
    saved config is identical; otherwise a timestamped sibling is used (never overwrite)."""
    name = cfg["name"]
    out = RESULTS / name / f"seed{seed}"
    saved = out / "config.json"
    if fresh or (saved.exists() and json.loads(saved.read_text()) != cfg):
        name = f"{name}_{timestamp()}"
        out = RESULTS / name / f"seed{seed}"
        print(f"config differs from the existing run (or --fresh): writing to {out}")
    out.mkdir(parents=True, exist_ok=True)
    saved = out / "config.json"
    if not saved.exists():
        saved.write_text(json.dumps(cfg, indent=1))
    return out, CHECKPOINTS / name / f"seed{seed}"


def minutes_per_training(cfg: dict) -> float:
    """Measured minutes per training on this GPU (from compute_log.csv), else the assumption."""
    gpu = gpu_name()
    per_it = []
    if COMPUTE_LOG.exists():
        with open(COMPUTE_LOG, newline="", encoding="utf-8") as f:
            per_it = [float(r["wall_s"]) / float(r["iters"]) for r in csv.DictReader(f)
                      if r["gpu"] == gpu and r["model"] == cfg["model"] and r["mode"] != "sam"
                      and r["diverged"] == "False" and float(r["iters"]) >= 1000]
    if per_it:
        minutes = sum(per_it) / len(per_it) * cfg["iters"] / 60
    else:
        minutes = ASSUMED_MIN_PER_15K["4060" if "4060" in gpu else "other"] * cfg["iters"] / 15000
    return minutes * (2 if mode_of(cfg) == "sam" else 1)


def announce_cost(cfg: dict, n_trainings: int) -> None:
    """Estimate and log the cost before launching (CLAUDE.md section 0)."""
    gpu = gpu_name()
    hours = n_trainings * minutes_per_training(cfg) / 60
    used, budget = cumulative_gpu_hours(gpu), budget_hours(gpu)
    print(f"COST ESTIMATE {cfg['name']}: {n_trainings} trainings x {minutes_per_training(cfg):.1f} min"
          f" = {hours:.2f} GPU-h on {gpu}; used so far {used:.2f} / {budget} h", flush=True)
    if budget is not None and used + hours > budget:
        print("WARNING: this run would exceed the budget", flush=True)


def train_one(cfg: dict, data: Data, state: dict, mask: Mask, seed: int, variant: str, rnd: int,
              out_json: Path, ckpt: Path, iters: int | None = None) -> dict:
    """Train one (variant, round) from `state` under `mask`; save weights, JSON and compute log."""
    device = data.xtr.device
    set_seed(seed)
    model = init_model(cfg["model"], seed, device=str(device))
    model.load_state_dict(state)
    mask = {n: m.to(device) for n, m in mask.items()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(cfg, data, model, mask, iters=iters)
    print(f"[{cfg['name']} seed{seed} r{rnd} {variant}] mode={trainer.mode} "
          f"remaining={remaining_fraction(model, mask):.4f}", flush=True)
    rec = trainer.train(seed)
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    rec.update(condition=cfg["name"], seed=seed, round=rnd, variant=variant,
               remaining=remaining_fraction(model, mask), density=density(mask),
               nominal_remaining=(1 - cfg["prune_rate"]) ** rnd, peak_vram_mb=peak_mb,
               config=cfg, env=run_metadata())
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    out_json.write_text(json.dumps(nan_to_none(rec), indent=1))
    t = rec["test"]
    print(f"   acc={t['acc']:.4f} f1={t['f1']:.4f} R02={rec['R02']:.3f} maxS3k={rec['max_S_3k']:.3f} "
          f"diverged={rec['diverged']} wall={rec['wall_s'] / 60:.1f}min "
          f"sharp_overhead={rec['sharpness_overhead']:.1%} peakVRAM={peak_mb:.0f}MB", flush=True)
    append_compute_log({
        "time": rec["env"]["started"], "condition": cfg["name"], "seed": seed, "round": rnd,
        "variant": variant, "model": cfg["model"], "mode": rec["mode"], "gpu": rec["env"]["gpu"],
        "iters": rec["iters"], "wall_s": round(rec["wall_s"], 1), "sharpness_s": round(rec["sharpness_s"], 1),
        "it_per_s": round(rec["it_per_s"], 1), "peak_vram_mb": round(peak_mb), "diverged": rec["diverged"],
    })
    return rec


def load_record(path: Path) -> dict:
    return json.loads(path.read_text())


def baseline_state(variant: str, cfg: dict, theta0: dict, mask: Mask, seed: int, rnd: int) -> dict:
    if variant == "reinit":
        return baselines.reinit(cfg["model"], seed, mask)
    if variant == "rescaled_reinit":
        return baselines.rescaled_reinit(cfg["model"], seed, mask)
    if variant == "shuffle":
        return baselines.shuffle(theta0, mask, seed + 100 * rnd)
    raise ValueError(f"unknown baseline {variant}")


def next_mask(cfg: dict, model_name: str, ckpt_dir: Path, rnd: int, mask: Mask) -> Mask:
    """m_{r+1} from the trained ticket of round r (cached as mask_r{r+1}.pt)."""
    path = ckpt_dir / f"mask_r{rnd + 1}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu")
    trained = torch.load(ckpt_dir / f"r{rnd}_ticket.pt", map_location="cpu")
    new = global_magnitude_prune(trained, mask, cfg["prune_rate"],
                                 output_name(init_model(model_name, 0)), cfg["output_prune_scale"])
    torch.save(new, path)
    return new


def run_chain(cfg: dict, seed: int, data: Data, fresh: bool) -> None:
    """One IMP chain for one seed, resuming from the last finished training."""
    out, ck = run_dirs(cfg, seed, fresh)
    sys.stdout = Tee(out / "log.txt")
    n = (cfg["rounds"] + 1) + sum(len(v) for v in cfg["baselines"].values())
    todo = n - len(list(out.glob("round*_*.json")))
    announce_cost(cfg, todo)
    t0_path = ck / "theta_0.pt"
    if not t0_path.exists():
        init_model(cfg["model"], seed, save_path=t0_path)
    theta0 = torch.load(t0_path, map_location="cpu")
    mask = ones_mask(init_model(cfg["model"], seed))
    for rnd in range(cfg["rounds"] + 1):
        if not (ck / f"mask_r{rnd}.pt").exists():
            torch.save(mask, ck / f"mask_r{rnd}.pt")
        f =out / f"round{rnd}_ticket.json"
        rec = load_record(f) if f.exists() else train_one(cfg, data, theta0, mask, seed, "ticket", rnd, f, ck / f"r{rnd}_ticket.pt")
        for variant, rounds in cfg["baselines"].items():
            fb = out / f"round{rnd}_{variant}.json"
            if rnd in rounds and not fb.exists():
                state = baseline_state(variant, cfg, theta0, mask, seed, rnd)
                train_one(cfg, data, state, mask, seed, variant, rnd, fb, ck / f"r{rnd}_{variant}.pt")
        if rec["diverged"]:
            print(f"ticket diverged at round {rnd} (step {rec['diverged_step']}): chain stopped", flush=True)
            (out / "chain_stopped.json").write_text(json.dumps({"round": rnd, "step": rec["diverged_step"]}))
            break
        if rnd < cfg["rounds"]:
            mask = next_mask(cfg, cfg["model"], ck, rnd, mask)
    sys.stdout = sys.stdout.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--seeds", type=int, nargs="*", help="override the config's seeds")
    ap.add_argument("--fresh", action="store_true", help="new timestamped run directory")
    args = ap.parse_args()
    cfg = check_config(load_yaml(args.config))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_data(cfg["dataset"], device)
    for seed in args.seeds if args.seeds is not None else cfg["seeds"]:
        run_chain(cfg, seed, data, args.fresh)
    from analysis.plots import build_summary
    build_summary()


if __name__ == "__main__":
    main()
