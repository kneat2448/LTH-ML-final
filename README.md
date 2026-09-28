# Why do lottery tickets need learning-rate warmup?

A university ML course project on the lottery ticket hypothesis (Frankle & Carbin, 2019). Winning
tickets in ResNet-20 on CIFAR-10 appear at a low learning rate or with warmup, but not at a high
learning rate. We test two explanations for this:

- **Sharpness.** Sparse subnetworks train close to the stability limit. The stability ratio
  `S = eta * lambda_max / (2 + 2*beta)` reaches about 1, and warmup keeps S below 1
  (Kalra & Barkeshli, 2024).
- **Weight correlation.** Tickets only work when the final weights stay correlated with their
  initialization (Liu et al., 2021).

The full research plan, hypotheses and rules are in [`src/CLAUDE_1.md`](src/CLAUDE_1.md).
The current status and next steps are in [`NOTES.md`](NOTES.md).

## What's here

| Folder | Contents |
|---|---|
| `src/` | data, ResNet-20, pruning (IMP), baselines, sharpness (Lanczos), metrics, trainer, run scripts |
| `configs/` | one YAML per experimental condition |
| `analysis/` | `plots.py`: builds every figure and table from `results/` |
| `tests/` | unit tests (`pytest`) |
| `colab/` | notebook for running on Google Colab |
| `pilot/` | earlier CPU pilot (Conv-4 on Fashion-MNIST), for reference |
| `results/` | per-run JSON files, `summary.csv`, `compute_log.csv` |

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
```

Runs can be stopped and restarted at any time: finished trainings are skipped. See `NOTES.md`
for the run order, the compute budget and open decisions.
