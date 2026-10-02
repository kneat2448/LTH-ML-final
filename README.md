# Why do lottery tickets need learning-rate warmup?

A university ML course project on the lottery ticket hypothesis (Frankle & Carbin, 2019). Winning
tickets in ResNet-20 on CIFAR-10 appear at a low learning rate or with warmup, but not at a high
learning rate. We test two explanations for this:

- **Sharpness.** Sparse subnetworks train close to the stability limit. The stability ratio
  `S = eta * lambda_max / (2 + 2*beta)` reaches about 1, and warmup keeps S below 1
  (Kalra & Barkeshli, 2024).
- **Weight correlation.** Tickets only work when the final weights stay correlated with their
  initialization (Liu et al., 2021).

**Status: complete (October 2026).** Report: [`report/FINAL_REPORT.pdf`](report/FINAL_REPORT.pdf) ·
slides: [`report/LTH_Warmup_Final.pptx`](report/LTH_Warmup_Final.pptx) · verdicts: [`results/FINDINGS.md`](results/FINDINGS.md).
The research plan is in [`src/CLAUDE_1.md`](src/CLAUDE_1.md); the session log in [`NOTES.md`](NOTES.md).

## Findings

**Neither explanation holds. A high learning rate breaks the mask that iterative magnitude pruning (IMP) selects,
not the training of a good mask.**

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** Tickets win at η = 0.01 and with warmup, not at η = 0.1 | **Supported** (2 seeds) | Ticket − best trained baseline: +2.0 to +6.4 pp at η = 0.01 / η = 0.03 + warmup; ≤ +0.5 pp at η = 0.1, no winning ticket. Warmup at η = 0.1 restores it (+1.5 to +5.9 pp). |
| **H2** Tickets fail where S reaches ≈ 1 | **Not supported** | Max S over the 105 pruning-chain runs is 0.54 (0.63 with the H4 runs). Only the failing condition has an early spike (step ≈ 25), but H4 shows it is not the cause. |
| **H3** Warmup keeps S below 1 as the advantage returns | **Partly supported** | The coincidence holds, but the failing runs are also below 1, and warmup neither is needed to train a good mask nor fixes a bad one. |
| **H4** Fix good masks, train at η = 0.1: does SAM or an L2 anchor rescue them? | **Mask, not dynamics** | The good masks **already win** without any intervention (+3.6 / +4.8 pp; seed 1: +3.5 / +4.4 pp). SAM lowers sharpness with no effect; the anchor raises correlation and makes tickets ~3 pp worse. High's own masks **still fail with warmup** (≤ +0.7 pp), even though warmup removes the spike. |

Under identical training (same θ₀, η = 0.1, no warmup), warmup's 6% mask reaches 87.2% and the high-LR chain's own 6% mask 82.7%,
while their shuffled controls are equal. The H4 decision rule (ticket − shuffle ≥ +2.0 pp at ≥ 2 of 3 masks) and both follow-ups were
committed before their runs ([`results/H4.md`](results/H4.md) §5, §8).

**Caveats:** one seed for SAM, the anchor, high_warm and the reverse swap; good masks at η = 0.1 still sit 0.6–2.6 pp below dense;
shortened protocol (15k steps, 30% pruning per round, no rewinding, one model and dataset).

Scale: 137 trainings on one Colab T4 (26.1 of 30 runtime hours), three runs in parallel under CUDA MPS.

## What's here

| Folder | Contents |
|---|---|
| `src/` | data, ResNet-20, pruning (IMP), baselines, sharpness (Lanczos), metrics, trainer, run scripts |
| `configs/` | one YAML per experimental condition |
| `analysis/` | `plots.py`: every figure and table from `results/`; `h4_tables.py`: H4 tables and verdicts; `final_figs.py`: report figures |
| `tests/` | unit tests (`pytest`) |
| `colab/` | notebook for running on Google Colab |
| `pilot/` | earlier CPU pilot (Conv-4 on Fashion-MNIST), for reference |
| `results/` | per-run JSON files, `summary.csv`, `compute_log.csv`, write-ups (`FINDINGS.md`, `PHASE_A.md`, `H2_H3.md`, `H4.md`) |
| `report/` | final report (`.md` / `.pdf` / `.docx`), slides (`.pptx`), `build.sh` to rebuild both from `results/` |

## Setup

```bash
conda create -n lth python=3.11 -y
conda activate lth
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy scikit-learn pyyaml matplotlib pandas pytest tqdm
python -m pytest
```

CIFAR-10 downloads automatically on first use. The Conv-4 pilot chain also needs the four
Fashion-MNIST `.gz` files in `pilot/data/`, from
[zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist).

On Colab, use the preinstalled packages and open `colab/run_on_colab.ipynb`.

## Running

```bash
python -m src.imp configs/smoke.yaml             # quick benchmark (300 iterations)
python -m src.imp configs/resnet20_low.yaml      # one IMP chain (resumable)
python -m src.fixed_mask configs/h4_high_sam.yaml
python -m analysis.plots                         # figures + tables in results/
bash report/build.sh                             # report PDF/DOCX + slides (pip install weasyprint python-pptx)
```

Runs can be stopped and restarted at any time: finished trainings are skipped. `NOTES.md` has the
run order, the compute log and every decision made along the way.
