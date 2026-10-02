"""Every figure and table of CLAUDE.md section 8, regenerated from results/ only.

Usage:  python -m analysis.plots

Writes results/summary.csv, results/figures/fig{1..5}_*.png, results/metrics_table.{md,csv},
results/ticket_advantage.csv, results/gate_a.md and (with H4 runs) results/h4_results.md.
Winning ticket (section 5): ticket acc >= dense acc - 0.5 pp at the same condition, and the
advantage over the best *trained* baseline is > 0 (with >= 2 seeds: mean > 2 SE).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import h4_tables
from src.metrics import stability_summary
from src.utils import RESULTS

FIG = RESULTS / "figures"
IMP_CONDITIONS = ["low", "high", "warm03", "high_warm", "conv4_fmnist_high"]
H4_CONDITIONS = ["h4_high", "h4_high_sam", "h4_high_anchor"]
H4_ROUNDS = [3, 6, 8]
DENSE_TOLERANCE = 0.005
BASELINES = ["reinit", "shuffle", "rescaled_reinit"]


def _records() -> list[dict]:
    """All per-training JSONs: results/<condition>/seed<k>/round<r>_<variant>.json."""
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*/seed*/round*_*.json"))]


def build_summary() -> pd.DataFrame:
    """results/summary.csv: one row per (condition, seed, round, variant)."""
    rows = []
    for r in _records():
        t = r["test"]
        rows.append({
            "condition": r["condition"], "seed": r["seed"], "round": r["round"], "variant": r["variant"],
            "nominal_remaining": r["nominal_remaining"], "remaining": r["remaining"], "mode": r["mode"],
            "lr": r["config"]["lr"], "warmup_iters": r["config"]["warmup_iters"],
            "acc": t["acc"], "precision": t["precision"], "recall": t["recall"], "f1": t["f1"], "auc": t["auc"],
            "val_acc": r["val_acc"], "R01": r["R01"], "R02": r["R02"], **stability_summary(r["sharpness"]),
            "max_S_3k": r["max_S_3k"], "max_S": r["max_S"], "loss_spike": r["loss_spike"], "diverged": r["diverged"],
            "diverged_step": r["diverged_step"], "trained": r["trained"], "wall_s": r["wall_s"],
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["condition", "seed", "round", "variant"])
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "summary.csv", index=False)
    print(f"summary.csv: {len(df)} rows")
    return df


def _se(x: pd.Series) -> float:
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def advantage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (condition, round): ticket advantage over the best trained baseline, and the verdict."""
    out = []
    for (cond, rnd), g in df.groupby(["condition", "round"]):
        dense = df[(df.condition == cond) & (df["round"] == 0) & (df.variant == "ticket")].acc.mean()
        tk = g[g.variant == "ticket"].set_index("seed").acc
        base = g[(g.variant != "ticket") & g.trained.astype(bool)]
        best = base.groupby("seed").acc.max()
        adv = (tk - best).dropna()
        n = len(adv)
        mean, se = (adv.mean() if n else np.nan), _se(adv)
        significant = mean > 0 and (n < 2 or mean > 2 * se)
        win = bool(n and tk.mean() >= dense - DENSE_TOLERANCE and significant)
        out.append({"condition": cond, "round": rnd, "nominal_remaining": g.nominal_remaining.iloc[0],
                    "dense_acc": dense, "ticket_acc": tk.mean(), "best_baseline_acc": best.mean(),
                    "advantage": mean, "advantage_se": se, "n_seeds": n,
                    "untrained_baselines": int((~g[g.variant != "ticket"].trained.astype(bool)).sum()),
                    "winning": win})
    return pd.DataFrame(out)


def _panels(n: int, title: str):
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    fig.suptitle(title)
    return fig, axes[0]


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print(f"figure: {FIG / name}")


def fig1_accuracy(df: pd.DataFrame, conds: list[str]) -> None:
    """Test accuracy vs % remaining: ticket vs each baseline, dense line; untrained flagged."""
    fig, axes = _panels(len(conds), "Fig. 1  Test accuracy vs weights remaining")
    for ax, cond in zip(axes, conds):
        d = df[df.condition == cond]
        for variant, style in [("ticket", "o-"), ("reinit", "s--"), ("shuffle", "^--"), ("rescaled_reinit", "d--")]:
            g = d[d.variant == variant].groupby("nominal_remaining").acc.agg(["mean", "std"]).reset_index()
            if len(g):
                ax.errorbar(100 * g.nominal_remaining, 100 * g["mean"], yerr=100 * g["std"].fillna(0),
                            fmt=style, capsize=2, label=variant)
        bad = d[(d.variant != "ticket") & ~d.trained.astype(bool)]
        if len(bad):
            ax.scatter(100 * bad.nominal_remaining, 100 * bad.acc.fillna(10 / 100), marker="x", s=60,
                       color="red", zorder=5, label="baseline did not train")
        dense = d[(d.variant == "ticket") & (d["round"] == 0)].acc.mean()
        ax.axhline(100 * dense, color="gray", ls=":", label="dense")
        ax.set(xscale="log", title=cond, xlabel="% weights remaining", ylabel="test accuracy (%)")
        ax.invert_xaxis()
        ax.legend(fontsize=7)
    _save(fig, "fig1_accuracy.png")


def fig2_sharpness(records: list[dict], conds: list[str]) -> None:
    """lambda_max(t) of the ticket at 100 / 34 / 5.8 % remaining, with the (2+2beta)/eta_t limit."""
    fig, axes = _panels(len(conds), "Fig. 2  Sharpness trajectory (ticket, seed 0)")
    for ax, cond in zip(axes, conds):
        recs = {r["round"]: r for r in records
                if r["condition"] == cond and r["variant"] == "ticket" and r["seed"] == min(
                    x["seed"] for x in records if x["condition"] == cond)}
        limit = None
        for rnd in (0, 3, 8):
            if rnd in recs and recs[rnd]["sharpness"]:
                tr = recs[rnd]["sharpness"]
                ax.plot([t["step"] for t in tr], [t["lam"] for t in tr], "o-", ms=3,
                        label=f"{100 * recs[rnd]['nominal_remaining']:.1f}% remaining")
                beta = recs[rnd]["config"]["momentum"]
                limit = [(t["step"], (2 + 2 * beta) / t["lr"]) for t in tr]
        if limit:
            ax.plot(*zip(*limit), "k--", lw=1, label="(2+2β)/η_t")
        ax.set(yscale="log", title=cond, xlabel="step", ylabel="λ_max")
        ax.legend(fontsize=7)
    _save(fig, "fig2_sharpness.png")


def fig3_max_s(df: pd.DataFrame, adv: pd.DataFrame, conds: list[str]) -> None:
    """H2: max S(t) over steps 25..3k (training-time sharpness) vs % remaining, colour = sign of the
    ticket advantage. S at step 0 (sharpness at init) is drawn separately as hollow markers."""
    fig, axes = _panels(len(conds), "Fig. 3  max S(t), steps 25–3k (H2)")
    for ax, cond in zip(axes, conds):
        tk = df[(df.condition == cond) & (df.variant == "ticket")]
        g = tk.groupby("round").agg(rem=("nominal_remaining", "first"), s=("max_S_train", "mean"),
                                    s0=("S0", "mean"))
        a = adv[adv.condition == cond].set_index("round").advantage
        colors = ["gray" if pd.isna(a.get(r)) else ("tab:green" if a[r] > 0 else "tab:red") for r in g.index]
        ax.plot(100 * g.rem, g.s, "-", color="lightgray")
        ax.scatter(100 * g.rem, g.s, c=colors, zorder=3)
        ax.scatter(100 * g.rem, g.s0, facecolors="none", edgecolors="gray", marker="D", s=20, zorder=2)
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set(xscale="log", title=cond, xlabel="% weights remaining", ylabel="max S (steps 25–3k)")
        ax.set_ylim(bottom=0)
        ax.invert_xaxis()
    axes[0].scatter([], [], c="tab:green", label="ticket advantage > 0")
    axes[0].scatter([], [], c="tab:red", label="≤ 0")
    axes[0].scatter([], [], c="gray", label="no trained baseline")
    axes[0].scatter([], [], facecolors="none", edgecolors="gray", marker="D", s=20, label="S at step 0 (init)")
    axes[0].legend(fontsize=7)
    _save(fig, "fig3_max_s.png")


def fig4_overlap(df: pd.DataFrame, conds: list[str]) -> None:
    """R_0.1 and R_0.2 of the ticket vs % remaining; chance level R_p ~ p."""
    fig, axes = _panels(len(conds), "Fig. 4  Weight correlation R_p (Liu et al. 2021)")
    for ax, cond in zip(axes, conds):
        tk = df[(df.condition == cond) & (df.variant == "ticket")]
        g = tk.groupby("nominal_remaining")[["R01", "R02"]].mean().reset_index()
        for col, p in (("R01", 0.1), ("R02", 0.2)):
            line, = ax.plot(100 * g.nominal_remaining, g[col], "o-", label=col.replace("R0", "R_0."))
            ax.axhline(p, color=line.get_color(), ls=":", lw=1)
        ax.set(xscale="log", title=cond, xlabel="% weights remaining", ylabel="R_p", ylim=(0, 1))
        ax.invert_xaxis()
        ax.legend(fontsize=7)
    _save(fig, "fig4_overlap.png")


def fig5_h4(df: pd.DataFrame) -> None:
    """H4: ticket - shuffle advantage, max S and R_0.2 at the three fixed warm03 masks."""
    conds = [c for c in H4_CONDITIONS + ["warm03"] if c in set(df.condition)]
    fig, axes = _panels(3, "Fig. 5  H4: fixed warm03 masks trained at η = 0.1")
    width = 0.8 / max(len(conds), 1)
    for i, cond in enumerate(conds):
        d = df[(df.condition == cond) & df["round"].isin(H4_ROUNDS)]
        tk = d[d.variant == "ticket"].groupby("round")
        sh = d[d.variant == "shuffle"].groupby("round").acc.mean()
        adv = (tk.acc.mean() - sh).reindex(H4_ROUNDS)
        vals = [adv, tk.max_S_train.mean().reindex(H4_ROUNDS), tk.R02.mean().reindex(H4_ROUNDS)]
        x = np.arange(len(H4_ROUNDS)) + i * width
        for ax, v in zip(axes, vals):
            ax.bar(x, v.values, width, label=cond)
    labels = [f"{100 * 0.7 ** r:.1f}%" for r in H4_ROUNDS]
    for ax, yl in zip(axes, ["ticket − shuffle acc", "max S (steps 25–3k)", "R_0.2"]):
        ax.set_xticks(np.arange(len(H4_ROUNDS)) + 0.4 - width / 2, labels)
        ax.set(ylabel=yl, xlabel="weights remaining")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[2].axhline(0.2, color="k", ls=":", lw=1)
    axes[0].legend(fontsize=7)
    _save(fig, "fig5_h4.png")


def _markdown(t: pd.DataFrame) -> str:
    """Plain markdown table (avoids the optional `tabulate` dependency)."""
    rows = [list(t.columns), ["---"] * len(t.columns)] + t.astype(str).values.tolist()
    return "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"


def metrics_table(df: pd.DataFrame, conds: list[str]) -> None:
    """acc, macro P/R/F1, AUC for dense and tickets at 34.3% (round 3) and 5.8% (round 8)."""
    rows = []
    for cond in conds:
        for rnd, label in ((0, "dense"), (3, "ticket 34.3%"), (8, "ticket 5.8%")):
            g = df[(df.condition == cond) & (df.variant == "ticket") & (df["round"] == rnd)]
            if len(g):
                row = {"condition": cond, "network": label, "seeds": len(g)}
                for m in ("acc", "precision", "recall", "f1", "auc"):
                    sd = g[m].std(ddof=1) if len(g) > 1 else float("nan")
                    row[m] = f"{g[m].mean():.4f}" + (f" ± {sd:.4f}" if len(g) > 1 else "")
                rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "metrics_table.csv", index=False)
    (RESULTS / "metrics_table.md").write_text(_markdown(t) if len(t) else "(no results)\n")
    print(f"table: {RESULTS / 'metrics_table.md'}")


def gate_a(adv: pd.DataFrame) -> None:
    """Gate A: tickets beat trained baselines for low and warm03 at >= 3 sparsities, and high has no
    *winning* ticket. For high a win uses the section-5 winning-ticket definition (the `winning` column:
    advantage > 0 and ticket acc >= dense - 0.5 pp), not any positive advantage; decided by the user in
    session 4, because high's tiny positive advantages (+0.36, +0.13 pp) came with tickets below dense."""
    lines = ["# Gate A (H1 replication)", ""]
    ok = True
    for cond, want in (("low", True), ("warm03", True), ("high", False)):
        a = adv[(adv.condition == cond) & (adv["round"] > 0) & adv.advantage.notna()]
        wins = a[a.advantage > 0] if want else a[a.winning.astype(bool)]
        n_win = len(wins)
        passed = (n_win >= 3) if want else (n_win == 0)
        ok &= passed and len(a) > 0
        what = "ticket beats the best trained baseline" if want else "winning ticket (advantage > 0 and acc >= dense - 0.5 pp)"
        lines.append(f"- **{cond}**: {what} at {n_win}/{len(a)} sparsities "
                     f"({', '.join(f'{100 * r:.1f}%' for r in wins.nominal_remaining)}); "
                     f"expected {'>= 3' if want else 'none'} -> {'PASS' if passed and len(a) else 'FAIL'}")
        if not want:
            pos = a[a.advantage > 0]
            lines.append(f"  (positive advantage without a winning ticket at {len(pos)}/{len(a)}: "
                         + ", ".join(f"{100 * r.nominal_remaining:.1f}% {100 * r.advantage:+.2f} pp, ticket "
                                     f"{100 * (r.ticket_acc - r.dense_acc):+.2f} pp vs dense" for r in pos.itertuples()) + ")")
    lines += ["", f"**Gate A: {'PASSED' if ok else 'NOT PASSED'}** (all verdicts are in ticket_advantage.csv)", ""]
    (RESULTS / "gate_a.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    df = build_summary()
    if df.empty:
        print("no results yet")
        return
    adv = advantage_table(df)
    adv.to_csv(RESULTS / "ticket_advantage.csv", index=False)
    conds = [c for c in IMP_CONDITIONS if c in set(df.condition)] or sorted(set(df.condition))
    records = _records()
    fig1_accuracy(df, conds)
    fig2_sharpness(records, conds)
    fig3_max_s(df, adv, conds)
    fig4_overlap(df, conds)
    if set(H4_CONDITIONS) & set(df.condition):
        fig5_h4(df)
        h4_tables.main()
    metrics_table(df, conds)
    gate_a(adv)


if __name__ == "__main__":
    main()
