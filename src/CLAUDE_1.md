# CLAUDE.md: Why do lottery tickets need learning-rate warmup? Sharpness vs. weight correlation

Read this whole file before writing any code. It explains what the repository is for, how the
experiments must be built, how much compute we have, and what the CPU pilot already found.
If a request in chat conflicts with this file, ask before deviating.

## 0. Caliber and working style

This is a **university ML course research project** (mid-review, then final report). The bar
is a careful workshop-style empirical study, not a large benchmark sweep:

- One sharp question, a small number of well-controlled conditions, and honest reporting,
  including negative results.
- Every claim comes with seeds and error bars and a baseline that is itself sane. Every number
  in the report must be regenerable from `results/` with one command.
- Replicate the known phenomenon before testing the new explanation.
- We are not chasing state-of-the-art accuracy. Budget goes to controls and seeds, not to
  bigger models.
- Code should be readable by a grader: short functions, docstrings that cite the paper or
  equation implemented, no clever abstractions.
- **Compute is the binding constraint.** Primary machine: the team's Windows laptop with an
  RTX 4060 (8 GB VRAM) and 16 GB RAM, estimated at ~18 GPU-hours for the whole plan. Fallback:
  ~30 T4 GPU-hours on Colab/Kaggle (section 9). Estimate and log the cost of every run before
  launching it. Never start an open-ended sweep.

## 1. Research question

> Do lottery tickets fail at high learning rates because sparse subnetworks train in a
> sharper, less stable regime (learning rate times top Hessian eigenvalue near the
> stability limit)? And does warmup rescue them by controlling sharpness, rather than by
> keeping final weights correlated with their initialization?

Papers this builds on (cite these in docstrings and the report):
- **Frankle & Carbin, ICLR 2019.** Iterative magnitude pruning (IMP) with reset to theta_0
  finds winning tickets in VGG-19 and "Resnet-18" (a 20-layer, ~0.27M-parameter CIFAR ResNet,
  i.e. our `resnet20`) only at eta = 0.01, or with linear warmup (ResNet: eta = 0.03, 20k
  warmup iterations). Even with warmup, no winning tickets were found at eta = 0.1. The paper
  leaves "why warmup?" open (Sec. 6).
- **Liu et al., ICML 2021 ("Lottery Ticket Preserves Weight Correlation").** Winning tickets
  appear only when theta_0 and theta_T stay correlated. They measure this with the overlap
  ratio R_p (their Eq. 1), and it happens when the learning rate is too low. They do not
  test warmup.
- **Kalra & Barkeshli, NeurIPS 2024 ("Why Warmup the Learning Rate?").** Warmup's main
  benefit is letting the network tolerate a larger target learning rate by reducing sharpness
  lambda_max. The stability limit is eta * lambda_max < 2 for GD, and (2 + 2*beta) with
  momentum beta.
- **Lange & Sprekeler, ICML 2023 (ES lottery tickets).** Sharpness of GD-trained sparse
  optima rises quickly with sparsity. ES optima stay flat.
- Closest related work, to position against in the report: Frankle et al. 2020 (linear mode
  connectivity and stability to SGD noise), Paul et al. ICLR 2023 (landscape geometry of IMP
  masks), Sakamoto & Sato NeurIPS 2022 (PAC-Bayes: winning tickets land in sharper minima),
  and "Finding Stable Subnetworks at Initialization with Dataset Distillation" (2025).
  None of these relates the ticket-failure boundary to eta * lambda_max during early training.

## 2. Hypotheses (revised after the pilot; see section 10)

- **H1 (replication).** On ResNet-20 / CIFAR-10, tickets beat random reinit at eta = 0.01 and
  with warmup (eta = 0.03), but not at eta = 0.1 without warmup.
- **H2 (stability ratio).** Where tickets fail, the stability ratio
  S(t) = eta_t * lambda_max(t) / (2 + 2*beta), measured over early training, reaches ~1.
  Where tickets win, S stays clearly below 1. Sharpness at initialization alone is **not**
  expected to predict this (pilot finding P2).
- **H3 (warmup mechanism).** Warmup keeps S below 1 through the high-learning-rate phase, and
  this coincides with the return of the ticket advantage.
- **H4 (causal test, masks held fixed).** Take the good masks from the warm03 chain and train
  them at eta = 0.1 in three ways:
  - (a) no warmup;
  - (b) with SAM, which lowers sharpness, and no warmup;
  - (c) with an L2 anchor to theta_0, which raises R_p without lowering sharpness, and no warmup.

  If (b) rescues the ticket and (c) does not, that supports sharpness. The reverse supports
  Liu et al. If both rescue, both mechanisms contribute.

## 3. Repository layout

```
configs/            # one YAML per condition (section 6); never edit after results exist
src/
  data.py           # CIFAR-10 held on GPU as uint8; GPU random-crop + flip
  models.py         # resnet20 (BN), conv4 (pilot reference only)
  prune.py          # masks, global magnitude pruning, mask enforcement
  baselines.py      # random reinit, density-rescaled reinit, layerwise-shuffled theta_0
  train.py          # SGD-momentum loop: step LR, linear warmup, SAM, L2-anchor
  sharpness.py      # lambda_max via HVP power iteration with negative-eigenvalue shift
  metrics.py        # acc, macro P/R/F1, OvR AUC, confusion matrix, R_p, loss spike
  imp.py            # IMP driver, resumable per round
  fixed_mask.py     # H4: train stored masks under a new training condition
analysis/plots.py   # every figure in section 8, from results/ only
pilot/lt_sharp.py   # CPU pilot (Fashion-MNIST, Conv-4). Reference only; do not edit
results/            # JSON per run + summary.csv + compute_log.csv (no checkpoints in git)
tests/
```

## 4. Hardware and environment

**Primary: local laptop.**
- Windows 11 x64, NVIDIA RTX 4060 Laptop GPU (8 GB VRAM, Ada), 16 GB system RAM. Anaconda is
  already installed.
- ResNet-20 on CIFAR-10 needs well under 1 GB of VRAM to train. The 8 GB limit only matters
  for the Hessian computation (see the VRAM rules below).

**Fallback: one T4 on Colab/Kaggle.** Same code. The only change is fp16 instead of bf16
autocast. Kaggle sessions stop after 12 h, which is one more reason to keep runs resumable.

**Windows-specific rules**
- Use native Windows with the conda env below. WSL2 is optional, not required.
- Do not use `torch.compile`: Triton is not supported on native Windows.
- Keep `num_workers=0`. Data lives on the GPU anyway, and Windows multiprocessing uses
  spawn, which is slow and fragile.
- Every script entry point is guarded by `if __name__ == "__main__":`.
- Use `pathlib.Path` everywhere. No hard-coded `/` or `\` paths.
- Long runs are started from a terminal, never from a notebook: `python -m src.imp configs/X.yaml`,
  with stdout teed to `results/<run>/log.txt`.
- Runs must survive being stopped at any time, because a laptop can sleep, update or
  overheat. Checkpoint after every finished training, and resume from the last finished one.

**Laptop hygiene (tell the user; do not automate)**
- Plugged in, power mode set to "Best performance".
- Sleep disabled while running: `powercfg /change standby-timeout-ac 0`.
- Good airflow. Watch `nvidia-smi -l 30`. If the GPU sits above ~85 °C for long, runs slow
  down; log the throughput per run so throttling shows up in `compute_log.csv`.

**Numerics**
- Train under `torch.autocast("cuda", dtype=torch.bfloat16)`. Ada supports bf16, so no
  GradScaler is needed.
- Use `channels_last`. Set `torch.backends.cuda.matmul.allow_tf32 = True` for training.
- **All Hessian computations run in fp32, outside autocast, with TF32 disabled** (flip the
  flag inside `sharpness.py` and restore it afterwards).

**VRAM rules for the Hessian**
- A double-backward pass on 2,048 images at once can exceed 8 GB.
- Compute each Hessian-vector product as the sum over micro-batches of 512 images (average the
  losses), `retain_graph` only per micro-batch, and `torch.cuda.empty_cache()` after each
  measurement.
- The smoke test must print peak VRAM (`torch.cuda.max_memory_allocated`). Keep it under 6 GB.

## 4a. Code necessities

**Environment (create once)**
```
conda create -n lth python=3.11 -y
conda activate lth
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # use the current CUDA 12.x wheel from the pytorch.org selector
pip install numpy scikit-learn pyyaml matplotlib pandas pytest tqdm
```
- Record exact versions in `requirements.txt` with `pip freeze`, and commit it.
- Verify with `python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.

**Dependency policy**
- Allowed: PyTorch, torchvision, numpy, scikit-learn, pyyaml, matplotlib, pandas, pytest,
  tqdm. Optional: wandb (off by default).
- Write everything else from scratch, in this repo, so graders can read it:
  - IMP and masking;
  - the SAM optimizer (~30 lines, Foret et al. 2021, two-step ascent/descent);
  - the L2 anchor;
  - power iteration (use `torch.autograd.grad` with `create_graph=True`, or `torch.func`);
  - the R_p overlap.
- Do not pull in lottery-ticket or Hessian libraries (e.g. PyHessian, open_lth). You may read
  them as a reference and cite them in comments.

**Modules that must exist** (section 3), each with the stated responsibilities:
- `resnet20`: He et al. 2016 CIFAR ResNet, 3 stages × 3 basic blocks, 16/32/64 channels,
  option-A (zero-pad) shortcuts, BatchNorm, ~0.27M params. Assert the parameter count in a
  test.
- A seeded `init_model(seed)` that returns the model and saves `theta_0.pt`.
- A single `Trainer` that takes a config and a mask and supports four modes: plain, warmup,
  SAM and anchor.

**Reproducibility**
- One `set_seed(seed)` call covering python, numpy, torch and cuda.
- Record every run's config, seed, git hash, torch/CUDA version, GPU name and wall-clock
  time in its JSON.

## 4b. Data necessities

**CIFAR-10 (main study)**
- Download: `torchvision.datasets.CIFAR10(root="data", download=True)`. This is the
  ~163 MB `cifar-10-python.tar.gz` (~180 MB extracted), and torchvision checks its MD5
  automatically.
- If the university network blocks the download, fetch the same archive by hand from
  https://www.cs.toronto.edu/~kriz/cifar.html into `data/`.
- Splits (fixed, seed 0, saved to `data/splits.json`):
  - 45,000 train / 5,000 validation, carved from the official 50,000 training images;
  - the official 10,000 test images.
- Use the validation set **only** for choices, such as the anchor lam in H4. Use the test set
  only for reported numbers.
- Normalization: per-channel mean/std computed on the 45k train split. Cache it in
  `data/stats.json`. Do not hard-code values copied from the internet.
- Augmentation (training only): random crop 32 with 4-px zero padding, and horizontal flip,
  both done on the GPU.
- Hessian batch: 2,048 fixed images from the 45k train split, no augmentation. Save the
  indices in `data/hessian_idx.json`, and use the same batch for every run.
- Memory: the full train set as uint8 on the GPU is ~140 MB.

**Fashion-MNIST (pilot only)**
- 60,000 train / 10,000 test images, 28×28×1, 10 classes. It is already covered by
  `pilot/lt_sharp.py`, which reads the four `.gz` files from `pilot/data/` (from the
  zalandoresearch/fashion-mnist GitHub repo).
- Not used in the main study.

**Git and disk**
- Do not commit `data/` or checkpoints. Add `data/`, `checkpoints/` and `*.pt` to
  `.gitignore`.
- Disk: ~0.2 GB data plus ~1.1 MB per ResNet-20 checkpoint (theta_0, masks, final weights;
  under 1 GB for the whole plan).
- Results JSON/CSV files (a few MB) **are** committed.

## 5. Implementation rules (non-negotiable)

**IMP**
- Prune globally over all conv and linear weights by magnitude among surviving weights.
- Do not prune biases, BatchNorm parameters or the first conv layer.
- Prune the final linear layer at half the global rate.
- Rate: 30% of remaining weights per round, 8 rounds, giving 100, 70, 49, 34.3, 24, 16.8,
  11.8, 8.2 and 5.8% remaining. This is coarser than Frankle's 20% to fit the budget; state
  this in the report.
- Reset to the exact saved theta_0 (no rewinding).
- **Divergence rule (pilot finding P3).** If a training run produces a non-finite loss or
  ends at chance accuracy:
  - record it as `diverged: true` with the step at which it happened;
  - stop pruning that chain, since the next mask would be built from NaN weights;
  - keep the last good mask for the remaining baselines.

  Never compute a mask from a diverged run.
- After every optimizer step: re-apply the mask to the weights and to the momentum buffers.
- Save theta_0, every mask and the final state_dict per round (outside git).
- Log the run's git hash, seed and config.

**Baselines** (pilot finding P4: a naive baseline can collapse and fake a ticket effect)
- `reinit`: same mask, fresh default init with seed + 1000.
- `shuffle`: permute theta_0 values within each layer among the surviving positions. This
  keeps each layer's weight distribution, so it is the fairest control.
- Report whether each baseline actually trained (test accuracy > 20%). A baseline stuck at
  chance must be flagged in the plots, never silently averaged in.

**Training (ResNet-20, CIFAR-10)**

Settings: SGD with momentum 0.9, batch 128, weight decay 1e-4, 15,000 iterations
(~38 epochs), learning rate x0.1 at 10k and 12.5k. This is half of Frankle's 30k schedule,
for budget reasons.

| Condition | Learning rate | Warmup | Other |
|---|---|---|---|
| `low` | 0.01 | none | |
| `high` | 0.1 | none | |
| `warm03` | 0.03 | linear, 10k iterations | Frankle's successful ResNet setting, rescaled to 15k |
| `high_warm` | 0.1 | linear, 10k iterations | Frankle reports failure here, so H2/H3 must predict it |
| `high_sam` | 0.1 | none | SAM, rho = 0.05 (costs about 2x per step) |
| `high_anchor` | 0.1 | none | Adds (lam/2) * \|\|theta - theta_0\|\|^2 over surviving weights |

For `high_anchor`: pick lam from {1e-3, 1e-2} using one short run, as the value that brings
R_0.2 at 34% remaining up to the `low` level.

**Sharpness (`sharpness.py`)**
- Compute lambda_max of the training loss on a fixed batch of 2,048 training images: no
  augmentation, fp32. Save the indices.
- BatchNorm: use batch statistics on that batch, without updating the running stats. Restore
  the module state afterwards.
- Restrict to surviving weights by masking both the probe vector and each Hessian-vector
  product.
- Power iteration: 20 iterations. If the estimate is negative, re-run on H - lambda_1 I and add
  lambda_1 back (pilot finding P6).
- Unit-test the power iteration against an exact eigendecomposition on a tiny net.
- Log the **trajectory**, not a single point: step 0, every 250 steps up to 3,000, then every
  1,500 steps, and the final step.
- Log S(t) = eta_t * lambda_max(t) / (2 + 2*beta) alongside it, plus max_t S(t) in the first
  3k steps.
- Cost: each measurement is about 20 Hessian-vector products. Keep total sharpness overhead
  under 15% of training time, and check this in the smoke test.

**Metrics (`metrics.py`)**
- On the test set: accuracy, macro precision, macro recall, macro F1 and macro one-vs-rest
  ROC-AUC, all at the end of training. Save the confusion matrix too.
- Ticket advantage = acc(ticket) - acc(best of the trained baselines) at matched sparsity.
- Call it a **winning ticket** only if both hold:
  - ticket accuracy >= dense accuracy at the same condition - 0.5 pp;
  - the advantage > 0, with mean > 2 SE over seeds where we have 2 seeds.
- R_p overlap (Liu et al., Eq. 1) between theta_0*m and theta_T*m, for p in {0.1, 0.2}.
  Chance level is R_p ~ p.
- Early loss spike = max training loss in the first 200 steps / loss at step 0.

## 6. Config example (`configs/resnet20_warm03.yaml`)

```yaml
model: resnet20
dataset: cifar10
lr: 0.03
warmup_iters: 10000
iters: 15000
milestones: [10000, 12500]
prune_rate: 0.3
rounds: 8
seeds: [0]
baselines: {reinit: [2, 4, 6, 8], shuffle: [4, 8]}   # rounds at which to run them
sharpness: {batch: 2048, iters: 20, every: 250, dense_until: 3000, then_every: 1500}
```

## 7. Run order, with gates

1. **Benchmark (≤ 0.5 h).** Run `pytest tests/` and `configs/smoke.yaml` (300 iterations,
   2 rounds). Measure minutes per 15k-iteration training on the RTX 4060 (or T4), including the
   sharpness overhead and peak VRAM. Write the result to `results/compute_log.csv` and **re-derive the budget table
   in section 9** from it.
   - If one training takes > 12 min, cut `iters` to 10k (milestones 6.7k/8.3k, warmup 6.7k)
     before cutting any condition or seed.
2. **Phase A: preliminary (≈ 11.5 h).** First a Conv-4 / Fashion-MNIST chain at eta = 0.1 with
   the full S(t) trajectory (≤ 0.5 h; de-risks H2, see P3). Then seed 0 of `low`, `high` and
   `warm03`, then `high_warm`.
   - **Gate A:** H1 must replicate, i.e. the ticket beats the trained baselines for `low` and
     `warm03` at ≥ 3 sparsity levels, and does not for `high`.
   - If it does not replicate, stop and write up why before spending more compute.
3. **Phase B: findings (≈ 13 h).**
   - Seed 1 of `low`, `high` and `warm03`, for error bars.
   - H4 fixed-mask experiment: warm03 masks at 34.3%, 11.8% and 5.8% remaining, trained under
     `high`, `high_sam` and `high_anchor` (ticket + shuffle baseline each).
4. **Reserve (≈ 5 h)** for reruns and crashed sessions. Do not spend it on new conditions
   without asking.
5. **Out of scope unless the user adds budget:** VGG, the ES arm, learning-rate sweeps, ImageNet.

## 8. Required outputs

- `results/<condition>/seed<k>/round<r>_<ticket|reinit|shuffle>.json` containing every metric
  above and the full sharpness trajectory.
- `results/summary.csv`: one row per (condition, seed, round, variant).
- Figures, regenerated by `python -m analysis.plots`:
  1. Test accuracy vs % weights remaining: ticket vs each baseline, one panel per condition,
     with the dense line.
  2. Sharpness trajectory lambda_max(t) with the (2 + 2*beta)/eta_t limit, for 100%, 34% and
     5.8% remaining in each condition.
  3. max_t S(t) vs % remaining, per condition, with the ticket-advantage sign overlaid (the
     H2 plot).
  4. R_p vs % remaining, per condition.
  5. H4 bars: ticket advantage, max S and R_0.2 for high / high_sam / high_anchor / warm03 at
     the three fixed masks.
- A metrics table (acc, P, R, F1, AUC) for dense and for the tickets at 34% and 5.8%, as
  required by the course rubric.
- `results/FINDINGS.md`: one paragraph per hypothesis with its verdict (supported / not
  supported / inconclusive) and the numbers behind it.

## 9. Compute budget

Assumption, to be replaced by the step-1 measurement: one 15k-iteration ResNet-20 training
takes ~6 min on the RTX 4060 laptop (bf16, GPU-resident data) and ~10 min on a T4.
- One IMP chain is 9 ticket runs plus 6 baseline runs: ~15 trainings, which is ~1.7 h on the
  4060 (or ~2.6 h on a T4) with sharpness logging.
- SAM runs cost ~2x.

| Phase | Runs | Trainings | 4060 GPU-h | T4 GPU-h (fallback) |
|---|---|---|---|---|
| Benchmark + tests | smoke | — | 0.5 | 0.5 |
| A: preliminary | low, high, warm03, high_warm (seed 0) | ~60 | ~7 | ~11.5 |
| B: findings | seed 1 of low/high/warm03 (~45) + H4 fixed masks (18, SAM at 2x) | ~63 | ~8 | ~13 |
| Reserve | reruns / crashes / throttling | — | ~3 | ~5 |
| **Total** | | | **~18** | **~30** |

- On the laptop, run Phase A as four overnight chains, one per condition. Never run two
  trainings at once on the laptop GPU.
- Log actual GPU time per run to `results/compute_log.csv`, and warn when cumulative use
  passes 80% and 95% of the budget for the machine in use.

## 10. Pilot findings (CPU, already done; the plan above already reflects them)

Pilot setup: `pilot/lt_sharp.py`. Conv-4 without BatchNorm on Fashion-MNIST (15k-image train
subset, 10k test), SGD momentum 0.9, 2 epochs, IMP at 60%/round for 5 rounds, 1 seed.
Treat these as directional only: one seed, a very short schedule and a shallow network.

- **P1. Dense probe at eta = 0.1 is NOT reproducible; do not cite it.** A first dense-only
  probe (2 epochs, 30k images) stuck at chance at eta = 0.1 without warmup and reached 85.1%
  with warmup (S = 0.80). The team re-run reached **87.1%** without warmup, with only a 3x
  early loss spike.

  **Implication:** one-seed probes with short schedules are unstable. Any dense-network
  claim needs ≥ 2 seeds and a saved JSON.
- **P2. Sharpness at initialization does not grow with sparsity. It shrinks.**
  - lambda_max(theta_0*m) fell from 0.84 (dense) to 0.12–0.20 at 1% remaining in every
    condition.
  - At eta = 0.05, sharpness at step 100 also fell (22.9 dense → 7.7–12.7 at ≤ 2.6%), so
    S ≤ 0.3 for all tickets.

  **Implication:** the original "sparse init is too sharp" version of H1 is not supported.
  The hypothesis now concerns the training-time stability ratio (H2), and must be tested
  on ResNet-20 where tickets actually fail.
- **P3. The high-learning-rate failure DOES appear in Conv-4 at eta = 0.1 (team re-run).**
  - At eta = 0.05 tickets still beat reinit (e.g. 85.3% vs 71.7% at 6.4% remaining).
  - At eta = 0.1 without warmup, the ticket lost to random reinit at 50% remaining (86.0% vs
    90.0%) and diverged at 25%.
  - With warmup, and at eta = 0.01, tickets beat reinit at every sparsity. H1's pattern is
    therefore already reproduced in a shallow network.
  - Step-100 sharpness of the losing ticket was ~2.4x the reinit's (16.0 vs 6.7, i.e.
    S ≈ 0.42 vs 0.18).
  - At 12.5% remaining S reached 1.01, but that mask came from a diverged network.
  - With warmup, S at step 100 was 0.49–0.55 (at peak LR) and tickets won.
  - Winning and losing tickets had the same R_0.2 ≈ 0.2 (chance).

  **Implication:** H2 is suggestive but not shown, because only step 100 was measured. Before
  ResNet-20, run one cheap Conv-4 chain at eta = 0.1 with the full S(t) trajectory to check
  whether the loss spike and divergence coincide with S crossing 1.
- **P4. The naive random-reinit baseline collapses at high sparsity.** Sparse Conv-4 without
  BatchNorm stayed at or near chance (10–19%) from 16% remaining at eta = 0.01, and from
  2.6–6.4% at eta = 0.05 with or without warmup. That inflates the apparent ticket advantage.

  **Implication:** use BatchNorm models, add the layerwise `shuffle` baseline, and flag
  untrained baselines (section 5).
- **P5. Weight correlation behaves as Liu et al. describe, but was not necessary here.**
  - Dense R_0.2 was 0.76 at eta = 0.01, 0.41 at 0.05, and 0.35 at 0.05 with warmup.
  - At eta = 0.05, tickets still beat trained reinit baselines (e.g. 83.8% vs 79.4% at 40%)
    while R_0.2 ~ 0.20, which is chance.

  **Implication:** H4 must separate the two mechanisms explicitly. Do not assume either one.
- **P6. Measurement pitfalls.**
  - Plain power iteration returned a negative eigenvalue at init. The shift fix is required.
  - Single-point lambda at step 100 was noisy (low-rate chain: 106, 44, 47, 45, 109, 7.5).
    Use the trajectory and max S, not one snapshot.
- **P7. Warmup length must fit the schedule.** A half-epoch warmup in a 2-epoch run cost
  ~2–3 pp of dense accuracy (81.1% vs 83.7%). Keep the warmup fraction of total iterations
  matched to Frankle (≈ 2/3 for warm03) and document it.

Pilot metrics for reference (dense / ticket at 6.4% remaining, eta = 0.05, no warmup):
- Dense: accuracy 83.7%, macro P 0.835, R 0.837, F1 0.834, AUC 0.984.
- Ticket at 6.4%: accuracy 85.3%, P 0.851, R 0.853, F1 0.851, AUC 0.985.
- Reinit at 6.4%: accuracy 71.7%, F1 0.709, AUC 0.959.

## 11. Coding conventions

- Seed python, numpy and torch (including cuda). Set `cudnn.benchmark = True`, and note
  that this makes runs not bit-exact.
- Every hyper-parameter comes from the YAML and is echoed at start. No hidden defaults.
- Functions under ~50 lines, with type hints and docstrings citing the source paper or
  equation.
- Never overwrite a results directory. Append a timestamp.
- Ask before adding a dependency, a condition, or anything that costs > 1 T4-hour and is not
  in section 9.

## 12. Tests (`tests/`)

- After 10 optimizer steps, masked weights and masked momentum entries are exactly 0.
- Global pruning removes the requested fraction to within ±1 weight, and skips the excluded
  layers.
- Power iteration matches `torch.linalg.eigvalsh` of the full Hessian on a tiny MLP within 2%,
  including when the most negative eigenvalue has the largest magnitude.
- Reloading theta_0 is bit-exact. The shuffle baseline preserves each layer's multiset of
  values.
- Metric functions match sklearn on a synthetic example.

## 13. Definition of done

- Tests pass.
- Gate A is documented.
- All figures and the metrics table regenerate from `results/` with one command.
- `results/FINDINGS.md` gives a verdict and numbers for H1–H4.
- `compute_log.csv` shows the plan stayed within budget (≤ ~18 h on the 4060 or ≤ 30 T4-hours).
