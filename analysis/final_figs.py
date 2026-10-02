"""Figures for the final report and presentation, regenerated from results/ only.

Usage:  python -m analysis.final_figs   (after python -m analysis.plots, which writes summary.csv)

Writes results/figures/final/{advantage,max_s,h4_masks,h4_interventions}.png (300 dpi).
Palette: validated categorical slots (CVD + normal-vision checks pass); every series is direct-labelled,
so identity never rests on colour alone.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import RESULTS

OUT = RESULTS / "figures" / "final"
INK, INK2, GRID, SURFACE = "#14213D", "#52514E", "#E3E6EC", "#FFFFFF"
COND = {  # entity -> (colour, label); high is the failing condition, in the attention hue
    "low": ("#2a78d6", "low  (η 0.01)"),
    "warm03": ("#1baf7a", "warm03  (η 0.03 + warmup)"),
    "high_warm": ("#4a3aa7", "high_warm  (η 0.1 + warmup)"),
    "high": ("#eb6834", "high  (η 0.1, no warmup)"),
}
H4 = {"h4_high": ("#2a78d6", "(a) plain"), "h4_high_sam": ("#008300", "(b) SAM"), "h4_high_anchor": ("#e87ba4", "(c) L2 anchor")}
BASE_ROUNDS = [2, 4, 6, 8]
H4_ROUNDS = [3, 6, 8]


REMAINING: dict[int, float] = {}  # round -> actual fraction remaining (the output layer is pruned at half rate)


def pct(r: int) -> str:
    return f"{100 * REMAINING[r]:.1f}%"


def spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label y-positions apart (in data units) so end-of-line labels never overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    })


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {OUT / name}")


def advantage(df: pd.DataFrame, cond: str, rnd: int, seed: int | None = None) -> list[float]:
    """Ticket minus best trained baseline (pp), one value per seed."""
    d = df[(df.condition == cond) & (df["round"] == rnd)]
    if seed is not None:
        d = d[d.seed == seed]
    vals = []
    for _, g in d.groupby("seed"):
        tk, bl = g[g.variant == "ticket"].acc, g[g.variant != "ticket"].acc
        if len(tk) and len(bl):
            vals.append(100 * (tk.iloc[0] - bl.max()))
    return vals


def fig_advantage(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(BASE_ROUNDS))
    ax.axhline(0, color=INK2, lw=1)
    ends, labels = [], []
    for cond, (col, lab) in COND.items():
        vals = [advantage(df, cond, r) for r in BASE_ROUNDS]
        m = np.array([np.mean(v) for v in vals])
        se = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for v in vals])
        ax.plot(x, m, color=col, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
        ax.fill_between(x, m - se, m + se, color=col, alpha=0.15, lw=0)
        n = max(len(v) for v in vals)
        ends.append(m[-1])
        labels.append((f"{lab}" + ("  · 2 seeds" if n > 1 else "  · 1 seed"), col))
    for y0, y, (lab, col) in zip(ends, spread(ends, 0.6), labels):
        ax.plot([x[-1] + 0.06, x[-1] + 0.16], [y0, y], color=col, lw=1, clip_on=False)
        ax.plot([x[-1] + 0.16, x[-1] + 0.28], [y, y], color=col, lw=2.5, clip_on=False, solid_capstyle="round")
        ax.annotate(lab, (x[-1] + 0.34, y), va="center", color=INK, fontsize=10, annotation_clip=False)
    ax.set_xticks(x, [pct(r) for r in BASE_ROUNDS])
    ax.set_xlim(-0.2, len(x) - 0.4)
    ax.set_xlabel("weights remaining")
    ax.set_ylabel("ticket − best trained baseline (pp)")
    save(fig, "advantage.png")


def fig_max_s(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    rng = np.random.default_rng(0)
    order = ["low", "warm03", "high_warm", "high"]
    for i, cond in enumerate(order):
        col, lab = COND[cond]
        v = df[(df.condition == cond) & (df.variant == "ticket") & (df["round"] > 0)].max_S_train.dropna()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=46, color=col, edgecolor=SURFACE, lw=1.2, zorder=3)
        ax.annotate(f"max {v.max():.2f}", (i, v.max()), xytext=(0, 9), textcoords="offset points", ha="center",
                    color=INK2, fontsize=9)
    ax.axhline(1.0, color=INK, lw=1.2, ls="--")
    ax.text(-0.4, 1.02, "stability limit  S = 1", ha="left", va="bottom", color=INK, fontsize=10)
    ax.set_xticks(range(len(order)), [COND[c][1].replace("  (", "\n(") for c in order])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("max S(t), steps 25–3,000")
    save(fig, "max_s.png")


def fig_h4_masks(df: pd.DataFrame) -> None:
    """Ticket accuracy at the three masks: same eta = 0.1 schedule, different mask (and the reverse swap)."""
    series = [
        ("high", 0, "high's own mask, η 0.1", COND["high"][0], "o"),
        ("h4_high", 0, "warm03's mask, η 0.1  (H4a)", COND["warm03"][0], "o"),
        ("h4_swap_high_warm", 0, "high's mask + warmup  (F2)", COND["high_warm"][0], "s"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(H4_ROUNDS))
    dense = 100 * df[(df.condition == "high") & (df.seed == 0) & (df["round"] == 0)].acc.iloc[0]
    ax.axhline(dense, color=INK2, lw=1, ls=":")
    ax.text(-0.15, dense + 0.15, f"dense high {dense:.2f}%", color=INK2, fontsize=9, va="bottom")
    for cond, seed, lab, col, mk in series:
        d = df[(df.condition == cond) & (df.seed == seed) & (df.variant == "ticket")].set_index("round").acc
        if not set(H4_ROUNDS) <= set(d.index):
            continue
        y = 100 * d.reindex(H4_ROUNDS).values
        ax.plot(x, y, color=col, lw=2, marker=mk, ms=8, mec=SURFACE, mew=2, zorder=3)
        ax.annotate(lab, (x[-1], y[-1]), xytext=(10, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
    ax.set_xticks(x, [pct(r) for r in H4_ROUNDS])
    ax.set_xlim(-0.2, len(x) - 0.3)
    ax.set_xlabel("weights remaining")
    ax.set_ylabel("ticket test accuracy (%)")
    save(fig, "h4_masks.png")


def fig_h4_interventions(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(len(H4_ROUNDS))
    w = 0.26
    for i, (cond, (col, lab)) in enumerate(H4.items()):
        xs = x + (i - 1) * w
        vals = [advantage(df, cond, r, seed=0) for r in H4_ROUNDS]
        v = np.array([a[0] if a else np.nan for a in vals])
        ax.bar(xs, v, w - 0.03, color=col, label=lab, zorder=3)
        for xi, vi in zip(xs, v):
            if vi == vi:
                ax.text(xi, vi + 0.2, f"{vi:+.1f}", ha="center", va="bottom", fontsize=9, color=INK)
        if cond == "h4_high":  # seed-1 replication (F1) as open markers on the plain bars
            s1 = [advantage(df, cond, r, seed=1) for r in H4_ROUNDS]
            pts = [(xi, a[0]) for xi, a in zip(xs, s1) if a]
            if pts:
                ax.scatter(*zip(*pts), s=40, facecolor=SURFACE, edgecolor=INK, lw=1.5, zorder=4, label="(a) seed 1  (F1)")
    ax.axhline(2.0, color=INK, lw=1.2, ls="--")
    ax.text(-0.5, 2.15, "rescue bar +2.0 pp", ha="left", va="bottom", fontsize=9, color=INK)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x, [pct(r) for r in H4_ROUNDS])
    ax.set_xlabel("weights remaining (warm03's masks)")
    ax.set_ylabel("ticket − shuffle (pp)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=False)
    save(fig, "h4_interventions.png")


def main() -> None:
    style()
    df = pd.read_csv(RESULTS / "summary.csv")
    REMAINING.update(df.groupby("round").remaining.mean().to_dict())
    fig_advantage(df)
    fig_max_s(df)
    fig_h4_masks(df)
    fig_h4_interventions(df)


if __name__ == "__main__":
    main()
