"""H4 causal test: train stored IMP masks under a new training condition (masks held fixed).

Usage:
    python -m src.fixed_mask configs/h4_high_anchor.yaml --select-lam   # pick anchor lam first
    python -m src.fixed_mask configs/h4_high_sam.yaml

The masks and theta_0 come from `mask_from` (the warm03 chain). For each stored round, the
ticket (theta_0 * m) and each baseline in `variants` are trained under this config.
Anchor lam selection (`anchor_lam: auto`): for each lam in `anchor_lam_grid`, one short run of
`select_iters` steps on the `select_round` mask; pick the smallest lam whose R_0.2 reaches the
`select_target` ticket's R_0.2 at the same step (read from its logged trajectory).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from src.data import load_data
from src.imp import announce_cost, baseline_state, load_record, run_dirs, train_one
from src.train import check_config
from src.utils import CHECKPOINTS, RESULTS, Tee, load_yaml


def source(cfg: dict, seed: int) -> tuple[Path, dict]:
    """Checkpoint dir of the source chain and its theta_0."""
    src = cfg["mask_from"]
    ck = CHECKPOINTS / src["condition"] / f"seed{seed}"
    if not (ck / "theta_0.pt").exists():
        raise FileNotFoundError(f"{ck}/theta_0.pt missing: run the {src['condition']} chain first")
    return ck, torch.load(ck / "theta_0.pt", map_location="cpu")


def lam_selection_path(cfg: dict) -> Path:
    return RESULTS / cfg["name"] / "lam_selection.json"


def target_R02(cfg: dict, seed: int, step: int) -> float:
    """R_0.2 of the target condition's ticket at `select_round`, at the logged step nearest `step`."""
    rec = load_record(RESULTS / cfg["select_target"] / f"seed{seed}" / f"round{cfg['select_round']}_ticket.json")
    row = min(rec["sharpness"], key=lambda r: abs(r["step"] - step))
    return row["R02"]


def select_lam(cfg: dict, seed: int, data) -> float:
    """Short runs over anchor_lam_grid; choose the smallest lam reaching the target R_0.2."""
    ck, theta0 = source(cfg, seed)
    rnd = cfg["select_round"]
    mask = torch.load(ck / f"mask_r{rnd}.pt", map_location="cpu")
    target = target_R02(cfg, seed, cfg["select_iters"])
    out = RESULTS / cfg["name"] / "lam_select"
    out.mkdir(parents=True, exist_ok=True)
    announce_cost({**cfg, "iters": cfg["select_iters"]}, len(cfg["anchor_lam_grid"]))
    scores = {}
    for lam in sorted(cfg["anchor_lam_grid"]):
        c = {**cfg, "anchor_lam": lam}
        rec = train_one(c, data, theta0, mask, seed, f"lam{lam:g}", rnd, out / f"lam{lam:g}.json",
                        CHECKPOINTS / cfg["name"] / "lam_select" / f"lam{lam:g}.pt", iters=cfg["select_iters"])
        scores[lam] = rec["R02"]
    reached = [lam for lam in sorted(scores) if scores[lam] >= target]
    chosen = reached[0] if reached else max(scores, key=scores.get)
    result = {"target": cfg["select_target"], "target_R02": target, "R02": scores, "chosen": chosen,
              "reached_target": bool(reached)}
    lam_selection_path(cfg).write_text(json.dumps(result, indent=1))
    print("anchor lam selection:", result, flush=True)
    return chosen


def resolve_lam(cfg: dict) -> dict:
    if cfg["anchor_lam"] != "auto":
        return cfg
    path = lam_selection_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run with --select-lam first")
    return {**cfg, "anchor_lam": json.loads(path.read_text())["chosen"]}


def run_fixed(cfg: dict, seed: int, data, fresh: bool) -> None:
    """Train ticket + baselines for every stored mask round under this condition."""
    ck_src, theta0 = source(cfg, seed)
    out, ck = run_dirs(cfg, seed, fresh)
    sys.stdout = Tee(out / "log.txt")
    rounds = cfg["mask_from"]["rounds"]
    todo = [(r, v) for r in rounds for v in cfg["variants"] if not (out / f"round{r}_{v}.json").exists()]
    announce_cost(cfg, len(todo))
    for rnd, variant in todo:
        mask = torch.load(ck_src / f"mask_r{rnd}.pt", map_location="cpu")
        state = theta0 if variant == "ticket" else baseline_state(variant, cfg, theta0, mask, seed, rnd)
        train_one(cfg, data, state, mask, seed, variant, rnd, out / f"round{rnd}_{variant}.json",
                  ck / f"r{rnd}_{variant}.pt")
    sys.stdout = sys.stdout.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--select-lam", action="store_true", help="run the anchor lam selection only")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    cfg = check_config(load_yaml(args.config))
    if "mask_from" not in cfg:
        raise ValueError("fixed_mask needs a config with mask_from (see configs/h4_*.yaml)")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    data = load_data(cfg["dataset"], "cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds if args.seeds is not None else cfg["seeds"]:
        if args.select_lam:
            select_lam(cfg, seed, data)
        else:
            run_fixed(resolve_lam(cfg), seed, data, args.fresh)
    from analysis.plots import build_summary
    build_summary()


if __name__ == "__main__":
    main()
