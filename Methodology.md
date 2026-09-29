# Methodology

*Why do lottery tickets need learning-rate warmup? Sharpness vs. weight correlation.*

This file describes the method **as it is actually run** and the steps of the study, with their
status. The original plan is the spec `src/CLAUDE_1.md`. Every place where this file differs from
the spec is listed in section 9, which points to the decisions (D1–D11) in `NOTES.md`. The day-to-day log is `NOTES.md`.

*Last updated: 2026-09-29 (session 3).*

---

## 1. Research question

Frankle & Carbin (2019) found winning tickets in ResNet-20 only at a low learning rate
(eta = 0.01) or with linear warmup. Liu et al. (2021) explain ticket success by weight correlation:
theta_T stays correlated with theta_0. Kalra & Barkeshli (2024) show that warmup works by lowering
sharpness lambda_max so that a large learning rate becomes stable. We ask:

> Do lottery tickets fail at high learning rates because sparse subnetworks train in a sharper,
> less stable regime (eta * lambda_max near the stability limit)? And does warmup rescue them by
> controlling sharpness, rather than by keeping the final weights correlated with their
> initialization?

## 2. Hypotheses

| | Hypothesis | Tested by |
|---|---|---|
| **H1** | Replication: tickets beat trained baselines at eta = 0.01 and with warmup (eta = 0.03), but not at eta = 0.1 without warmup | Phase A, Gate A |
| **H2** | Where tickets fail, the stability ratio S(t) reaches ~1 in early training. Where they win, S stays clearly below 1. Sharpness at init alone does not predict this | max S(t) over steps 25–3k vs ticket advantage (fig. 3); S(0) reported separately |
| **H3** | Warmup keeps S below 1 during the high-LR phase, and this coincides with the return of the ticket advantage | `warm03` / `high_warm` vs `high` trajectories (fig. 2) |
| **H4** | Causal test with masks held fixed: warm03 masks trained at eta = 0.1 (a) plain, (b) with SAM (lowers sharpness), (c) with an L2 anchor to theta_0 (raises R_p without lowering sharpness) | Phase B, fig. 5 |

H4 decision rule: if (b) rescues the ticket and (c) does not, that supports sharpness. The
reverse supports weight correlation (Liu et al.). If both rescue, both mechanisms contribute.

## 3. Experimental setup

**Model.** ResNet-20 (He et al. 2016 CIFAR variant): 3 stages × 3 basic blocks, 16/32/64 channels,
option-A (zero-pad) shortcuts, BatchNorm, 269,722 parameters. Seeded `init_model` saves theta_0.

**Data.** CIFAR-10:
- 45,000 train / 5,000 validation images, split from the official 50k with seed 0 (`data/splits.json`);
- the official 10,000-image test set.

Other data details:
- Per-channel normalization is computed on the 45k split.
- Augmentation: random crop 32 with 4-px zero padding plus horizontal flip, done on the GPU.
- The validation set is used only for choices (the divergence rule, the H4 anchor lam). The test set is used only for reported numbers.

**Training.** SGD, momentum beta = 0.9, batch 128, weight decay 1e-4, 15,000 iterations
(~38 epochs; half of Frankle's 30k). The learning rate is multiplied by 0.1 at 10k and 12.5k.
Mixed precision: fp16 + GradScaler on the T4 (bf16 on Ampere+ GPUs). The Hessian is always fp32.

**Conditions.**

| Condition | eta | Warmup | Extra | Role |
|---|---|---|---|---|
| `low` | 0.01 | none | | tickets expected to win (Frankle) |
| `high` | 0.1 | none | | tickets expected to fail |
| `warm03` | 0.03 | linear, 10k | | Frankle's successful ResNet setting, rescaled to 15k |
| `high_warm` | 0.1 | linear, 10k | | Frankle reports failure here; H2/H3 must predict it |
| `high_sam` (H4) | 0.1 | none | SAM, rho = 0.05 | lowers sharpness |
| `high_anchor` (H4) | 0.1 | none | + (lam/2)·‖theta − theta_0‖² on surviving weights | raises R_p, not sharpness |

## 4. Iterative magnitude pruning (IMP)

1. Train the dense network from theta_0.
2. Prune 30% of the surviving weights **globally** by magnitude over all conv and linear weights.
   The first conv layer, biases and BatchNorm parameters are never pruned. The final linear layer is pruned at half the rate.
3. Reset the survivors to the **exact** theta_0 (no rewinding) and retrain.
4. Repeat for 8 rounds: 100, 70, 49, 34.3, 24, 16.8, 11.8, 8.2 and 5.8% of weights remaining.
   (30% per round is coarser than Frankle's 20%, for budget reasons.)

After every optimizer step the mask is re-applied to the weights **and** to the momentum buffers.

**Divergence rule.** A run counts as diverged if its loss becomes non-finite or it ends at chance
(validation accuracy < 0.15). A diverged run is recorded with the step at which it diverged. The chain then
stops pruning, because a mask is never computed from a diverged network. That round's baselines still run on the last good mask.

## 5. Baselines (at matched masks)

- **reinit:** the same mask with a fresh default initialization (seed + 1000). Rounds 2, 4, 6 and 8.
- **shuffle:** theta_0 values permuted within each layer among the surviving positions. This keeps
  each layer's weight distribution, so it is the fairest control. Rounds 4 and 8.

A baseline counts as *trained* only if its test accuracy is > 20%. Untrained baselines are flagged, never
averaged in, because a collapsed naive baseline can fake a ticket effect (pilot P4).

## 6. Measurements

**Test metrics (end of training):** accuracy, macro precision / recall / F1, macro one-vs-rest
ROC-AUC, and the confusion matrix. All are implemented from scratch and tested against sklearn.

**Ticket advantage** = acc(ticket) − acc(best trained baseline) at the same mask.
A **winning ticket** requires both:
- ticket accuracy ≥ the dense accuracy of the same condition − 0.5 pp;
- advantage > 0, and with 2 seeds the mean > 2 SE.

**Sharpness.** lambda_max of the training loss on a fixed batch of 2,048 non-augmented training
images (`data/hessian_idx.json`):
- **Lanczos**, 10 steps with full reorthogonalization (D1, D11);
- fp32, no TF32, NCHW;
- Hessian-vector products summed over micro-batches of 512;
- BatchNorm in batch-stat mode, with running stats restored afterwards;
- the probe vector and every HVP are masked to the surviving weights.

Schedule (D11): steps 0, 25, 50, 100, 200, every 500 up to 3,000, then every 3,000, plus the final step
(15 points).

**Stability ratio.** S(t) = eta_t · lambda_max(t) / (2 + 2·beta) = eta_t · lambda_max(t) / 3.8.
S = 1 is the edge of stability for SGD with heavy-ball momentum. We record the trajectory. The H2 statistic is
**max S over steps 25–3,000** (training-time sharpness). **S(0)**, sharpness at initialization before any
update, is reported separately (see section 9).

**Weight correlation.** R_p (Liu et al. 2021, Eq. 1): the share of the top-p surviving weights by
|theta_0| that are also in the top-p by |theta_T|. It is computed per layer and pooled, for p ∈ {0.1, 0.2}.
Chance is ≈ p. R_0.2 is also logged at every sharpness step, which is needed for choosing the H4 anchor lam.

**Early loss spike** = max training loss in the first 200 steps / loss at step 0. In the pilot,
a spike > 8x preceded every collapse (PILOT_FINDINGS F7).

**Collapse rate** per condition: the share of runs that diverged or ended at chance. It is reported
next to the ticket advantage, because the pilot showed that high-LR failure can be a random
collapse rather than a systematic ticket deficit (F1, F2).

## 7. H4 procedure (fixed masks)

1. Masks: the warm03 chain's masks at 34.3%, 11.8% and 5.8% remaining (rounds 3, 6 and 8).
2. Anchor lam: for each lam in {1e-3, 1e-2}, run 3k steps at the round-3 mask. Pick the lam whose
   R_0.2 at step 3,000 comes closest to the `low` ticket's R_0.2 at the same step (D6).
   Command: `python -m src.fixed_mask configs/h4_high_anchor.yaml --select-lam`.
3. Train ticket + shuffle at each mask under `high`, `high_sam` and `high_anchor` (18 trainings).
4. Compare ticket advantage, max S and R_0.2 against warm03 (fig. 5).

## 8. Steps of the study and their status

| # | Step | Status |
|---|---|---|
| 0 | Pilot on Conv-4 / Fashion-MNIST (CPU + GPU re-run); re-analysis → `pilot_results/PILOT_FINDINGS.md` | **done** (sessions 1–2) |
| 1 | Code + tests (21 tests pass on the T4) | **done** |
| 2 | Benchmark / smoke run on the T4: ~9 min per 15k training, 2.46 GB peak VRAM; budget re-derived | **done** (session 2) |
| 3a | Phase A: `low` seed 0 (15 trainings) | **running** (session 3; r0 dense 86.66%) |
| 3b | Phase A: `high`, `warm03`, `high_warm` seed 0 | pending |
| 3c | Conv-4 chain at eta = 0.1 with the full S(t) trajectory | optional; needs `pilot/data/*.gz` on Drive (O8) |
| 4 | **Gate A:** the ticket beats trained baselines at ≥ 3 sparsities for `low` and `warm03`, and not for `high` (`results/gate_a.md`). If it fails, stop and write up why | pending |
| 5 | Phase B: seed 1 of `low`, `high` and `warm03` (error bars) | pending (after Gate A) |
| 6 | Phase B: H4 (lam selection, then 3 conditions × 3 masks × ticket + shuffle) | pending |
| 7 | Analysis: `python -m analysis.plots` → summary.csv, figs 1–5, metrics table, gate_a.md | script ready |
| 8 | `results/FINDINGS.md`: verdict + numbers for H1–H4; final report | pending |

**Compute (T4, D11 schedule):** Phase A ≈ 9 GPU-h and Phase B ≈ 10 GPU-h of training. The budget is **30 h of Colab
runtime** (wall-clock usage, not GPU-hours; 27 h were left at the start of session 4). Every training goes to
`results/compute_log.csv`. Runtime used is counted as the union of training intervals, because parallel chains share the runtime. If trainings take longer than 12 min, `iters` is cut
to 10k before any condition or seed is cut.

**Analysis outputs** (all regenerate from `results/` with `python -m analysis.plots`):
1. Test accuracy vs % remaining: ticket vs baselines, one panel per condition, dense line (H1).
2. lambda_max(t) with the (2 + 2·beta)/eta_t limit at 100%, 34% and 5.8% remaining (H2, H3).
3. max S(t) over steps 25–3k vs % remaining, with the sign of the ticket advantage, and S(0) as hollow markers (H2).
4. R_p vs % remaining per condition (weight correlation).
5. H4 bars: ticket advantage, max S and R_0.2 for high / high_sam / high_anchor / warm03.
6. A metrics table (acc, P, R, F1, AUC) for dense and for the tickets at 34% and 5.8%.

## 9. Deviations from the spec (to state in the report)

- **Lanczos instead of power iteration + shift (D1).** On dense ResNet-20 theta_0, power iteration converged to
  lambda_min ≈ −152, and the shifted re-run returned wrong values (0.04 or 2.6). Lanczos gives 25.6.
  Cost per step is unchanged.
- **Sharpness schedule (D11).** 10 Lanczos steps and 15 points, including early steps 25/50/100/200, instead of 20 steps × 21
  points. The early points were added because pilot collapses happen before step 200. Overhead is ≈ 150% of
  training time, against the spec's 15% target, which no GPU can meet at this Hessian batch size.
- **fp16 + GradScaler on the T4 (D2).** The spec assumed bf16 on the RTX 4060.
- **Hessian in NCHW (D3)**, for speed. Training stays channels_last.
- **Divergence threshold (D4):** validation accuracy < 0.15. The anchor and R_p use each run's own init (D5).
- **H2 statistic excludes step 0.** The spec's max S(t ≤ 3k) is replaced by max S over steps 25–3,000, and S(0) is reported
  on its own. Without warmup, S(0) = eta · lambda_max(theta_0 ⊙ m) sets the old maximum (0.67 for dense `high`, while
  lambda_max drops from 25.5 to 6.2 within 25 steps). That is sharpness at init, which H2 says should *not* predict failure.
  Only the analysis changed; the raw trajectories were already saved, so no run was repeated. The run JSONs still
  contain the old `max_S_3k` field.
- **Pilot re-interpretation:** P1 is reversed (dense eta = 0.1 collapses in 2 of 3 short runs), P3 is weakened
  (seed-dependent), and P2 is unverified (likely the D1 artifact). See `pilot_results/PILOT_FINDINGS.md`.

## 10. Results so far

*The table is regenerated from `results/` by `python -m analysis.methodology_table` after every finished training.*

<!-- results-table:start -->
| Condition | Seed | Round | % remaining | Variant | Test acc | S(0) | max S(25–3k) | R_0.2 | Diverged |
|---|---|---|---|---|---|---|---|---|---|
| low | 0 | 0 | 100.0 | ticket (dense) | 86.66% | 0.067 | 0.081 | 0.640 | no |
| low | 0 | 1 | 70.1 | ticket | 87.43% | 0.083 | 0.148 | 0.584 | no |
| low | 0 | 2 | 49.1 | reinit | 85.78% | 0.084 | 0.075 | 0.550 | no |
| low | 0 | 2 | 49.1 | ticket | 87.85% | 0.053 | 0.085 | 0.525 | no |
| low | 0 | 3 | 34.5 | ticket | 87.87% | 0.046 | 0.087 | 0.465 | no |
| low | 0 | 4 | 24.2 | reinit | 84.64% | 0.055 | 0.079 | 0.441 | no |
| low | 0 | 4 | 24.2 | ticket | 87.92% | 0.048 | 0.086 | 0.407 | no |
| high | 0 | 0 | 100.0 | ticket (dense) | 89.83% | 0.670 | 0.226 | 0.247 | no |
| high | 0 | 1 | 70.1 | ticket | 89.39% | 0.638 | 0.536 | 0.238 | no |
| high | 0 | 2 | 49.1 | ticket | 89.21% | 0.659 | 0.240 | 0.235 | no |
| warm03 | 0 | 0 | 100.0 | ticket (dense) | 87.48% | 0.000 | 0.083 | 0.558 | no |
| warm03 | 0 | 1 | 70.1 | ticket | 88.22% | 0.000 | 0.051 | 0.496 | no |
| warm03 | 0 | 2 | 49.1 | ticket | 88.57% | 0.000 | 0.045 | 0.447 | no |
<!-- results-table:end -->

**Observations so far (seed 0, single runs, directional only):**
- **Dense networks (r0).** high 89.83% > warm03 87.48% > low 86.66%. The high-LR dense network did not collapse
  (loss spike 1.19x), unlike 2 of 3 pilot Conv-4 runs.
- **Sharpness dynamics (dense).** All three conditions start from the same theta_0 (lambda_max 25.5).
  - `low`: lambda_max stays ~25–31 early, then falls to ~8. S ≤ 0.08 throughout.
  - `high`: lambda_max **drops from 25.5 to 6.2 within 25 steps** and to ~2 by step 1,000, so the network moves itself to a
    flat region right away. S is 0.67 at step 0, then ≤ 0.23.
  - `warm03`: lambda_max **rises during warmup** (progressive sharpening: 25 → 53 at step 2,000) while the small warmup LR keeps
    S ≤ 0.08. It then falls to ~4 once the LR is large.
- **Methodological issue in max S(≤3k).** For runs without warmup, the maximum is set by **step 0**, where S = eta · lambda_max(theta_0 ⊙ m)
  is sharpness at initialization, before any update (0.67 for `high`). H2 is about *training-time* sharpness and
  expects sharpness at init *not* to predict failure. The H2 analysis should therefore use max S over steps 25–3,000
  (the `high` dense value is 0.226), and report step 0 separately. *Adopted in session 4 (section 9); the table above and fig. 3 use it.*
- **Weight correlation.** Dense R_0.2 is 0.64 (low), 0.56 (warm03) and 0.25 (high; chance is 0.2), which agrees with Liu et al. A large LR
  decorrelates theta_T from theta_0, and warmup keeps part of the correlation.
- **Ticket advantage (low):** +2.07 pp over reinit at 49.1% remaining. Tickets stay at or above dense accuracy down to 34.5% (87.87%).
