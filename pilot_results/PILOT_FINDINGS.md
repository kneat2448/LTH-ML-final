# Pilot findings: re-analysis of all Conv-4 / Fashion-MNIST runs

*2026-09-28, session 2. Every number below regenerates with `python -m analysis.pilot_summary`,
which writes `pilot_summary.csv`, `pilot_tables.md`, `fig_pilot_acc.png` and `fig_pilot_spike.png`
next to this file.*

## 0. Status of the project

- **The main study (ResNet-20 / CIFAR-10) has not started.** `results/` holds only the archived
  laptop smoke run. The project is still at Gate 0 (`NOTES.md` section 1). O1 (sharpness cost)
  and O4 (G4 budget) are still open decisions.
- The test suite passes on this Colab T4 (20/20, 66 s, torch 2.11 + cu128).
- The only experimental results are the **three Conv-4 pilot run sets** in `pilot_results/`.
  The spec (`src/CLAUDE_1.md` section 10) summarises only the first two of them:

| Run set | Files | Setup | In spec section 10? |
|---|---|---|---|
| Original pilot | `res_{low,high,high_warm}.json` | 2 epochs, 15k images, 60%/round, eta 0.01 / 0.05 (+0.5-ep warmup) | yes (P2, P4, P5, P7, metrics) |
| Team re-run | `res_full_{low,high_nowarm,high_warm}.json` | 4 epochs, 30k images, 50%/round, eta 0.01 / 0.1 (+1-ep warmup), CPU | yes (P1, P3) |
| **GPU re-run** | `res_gpu_{low,high_nowarm,high_warm}.json` | same as the team re-run, on CUDA | **no, new here** |

**Read the GPU re-run as a second seed, not as a device effect.** `lt_sharp.py` seeds with
`torch.manual_seed(0)`, but the CPU and CUDA RNG streams differ. The GPU run therefore starts
from a different theta_0 (dense lambda_0 = 0.816 vs 0.845) and draws different minibatch orders.
The 4-epoch design now has **2 seeds per condition**, and that is the main value of the new data.

## 1. Where the new data differs from the pilot findings

| Pilot finding | What it said | What the 2-seed data shows | Status |
|---|---|---|---|
| **P1** dense eta = 0.1 | Chance-level collapse "NOT reproducible" (re-run reached 87.1%) | GPU re-run: dense eta = 0.1 **collapsed to chance again** (10.0%, loss spike 13.3x). Now 2 of 3 dense runs collapsed. | **Reversed.** The collapse is real but stochastic (bimodal) |
| **P3** ticket fails at eta = 0.1 | Ticket lost to reinit at 50% (86.0 vs 90.0) and diverged at 25% | CPU: loses at 50 / 12.5 / 6.2% (−4.1, −0.6, −1.9 pp), NaN at 25%. GPU: the chain collapsed at round 0, so every later mask is invalid; valid tickets at ≤ 6.2% *beat* reinit by +1.2 to +2.8 pp | **Weakened.** Seed-dependent. The robust effect is random collapse, not a systematic ticket deficit |
| **P3** warmup rescues | Tickets beat reinit at every sparsity with warmup | CPU: wins at 50–3.1% (+1.2 to +7.7 pp). GPU: **ties at 50% and 25%** (−0.4, −0.2 pp), wins at 12.5–3.1% (+2.0 to +3.5 pp) | **Holds, weaker.** Rescue appears only at ≤ 12.5% on seed 2 |
| **P3** eta = 0.01 | Tickets beat reinit at every sparsity | Both seeds: wins at every sparsity with a trained baseline (+1.5 to +13.5 pp) | **Replicated** |
| **P3** S(100) of the losing ticket | 16.0 vs 6.7 (S ≈ 0.42 vs 0.18); S = 1.01 at 12.5% | The S = 1.01 ticket trained fine (85.7%, −0.6 pp vs reinit). Collapsed runs show S(100) ≈ 0 because a dead network is flat. On GPU, failed and successful runs have overlapping S(100) | **Not supported** by the step-100 snapshot |
| **P2** sharpness at init shrinks with sparsity | lambda_0: 0.84 dense → 0.12–0.20 sparse | Sparse lambda_0 values cluster tightly at 0.11–0.13 (median 0.121, n = 84). That is the same value reported for dead networks, and the power-iteration + shift failure found in D1 produces exactly this | **Unverified (likely a measurement artifact)** |
| **P4** reinit collapses at high sparsity | reinit at chance from ~16% (eta 0.01) | Both 4-epoch seeds: reinit at chance from ≤ 6.2% (CPU) / ≤ 3.1% (GPU) at eta = 0.01, and at 1.6% in every warm/high chain | **Replicated.** Onset is later with the longer schedule |
| **P5** R_0.2 not necessary | Tickets win with R_0.2 ≈ chance | Both seeds: dense R_0.2 = 0.64 / 0.62 at eta 0.01. Every sparse ticket sits at 0.15–0.28, and winning warm tickets sit at 0.15–0.22 | **Replicated** |
| *new* | — | **Early loss spike separates failures cleanly.** All 4 runs with spike > 8x failed (NaN or chance). The largest spike among runs that trained was 6.3x | **New finding** |
| *new* | — | The pilot's divergence check only catches NaN. **All three GPU failures were finite chance-level collapses**, and IMP kept pruning from those dead networks | **New pitfall.** Already handled in `src/` (D4) |

## 2. Findings in detail

### F1. Dense eta = 0.1 without warmup collapses in 2 of 3 runs (revises P1)
- The GPU dense run at eta = 0.1 ended at 10.0% test accuracy (AUC 0.500), with an early loss
  spike of 13.3x and `diverged: false`, because the loss stayed finite.
- With the original 2-epoch probe (chance) and the CPU re-run (87.1%), dense eta = 0.1 has now
  collapsed in 2 of 3 attempts. Both 4-epoch dense runs **with** warmup trained (89.8%, 88.7%).
- **Implication.** P1's conclusion ("the collapse is not reproducible, do not cite it") was too
  strong. The failure is bimodal: a run either collapses in the first ~200 steps or trains
  normally. This is exactly why the spec requires ≥ 2 seeds. It also means the ResNet-20 `high`
  condition may show a *dense* collapse, which Gate A must be ready to interpret.

### F2. The high-LR ticket failure is seed-dependent (weakens P3 / H1 in Conv-4)
- **Seed 1 (CPU).** The ticket loses to reinit at 50%, 12.5% and 6.2% remaining, is NaN at 25%,
  and never meets the section-5 winning-ticket rule.
- **Seed 2 (GPU).** The dense network collapsed, and the pilot script still pruned from it. At
  25% the ticket collapsed again, and pruning went on from that network too. Under the spec's
  divergence rule, this chain should have stopped at round 0, so **none of its sparse verdicts
  count** (marked † in `pilot_tables.md`). For what they are worth, the trained tickets at 6.2%,
  3.1% and 1.6% beat trained reinits by +2.8, +1.2 and +2.9 pp.
- **What is common to both seeds** is not "tickets lose" but "runs at eta = 0.1 without warmup
  randomly collapse": 1 NaN ticket on CPU, and 2 chance-level tickets plus 1 chance-level reinit
  (50%) on GPU. With warmup, **no** ticket failed in either seed.
- **Implication.** In Conv-4 at 4 epochs, H1's "no winning ticket at high eta" is really
  "high eta without warmup is a collapse lottery". The main study should report the **collapse
  rate** per condition next to the ticket advantage.

### F3. Warmup restores the ticket advantage in both seeds, less strongly in seed 2
- CPU: +1.2, +2.2, +1.8, +4.8, +7.7 pp at 50, 25, 12.5, 6.2, 3.1% remaining (all "win").
- GPU: −0.4, −0.2, +2.0, +2.0, +3.5 pp. At 50% and 25% the ticket ties the reinit. Single-seed
  margins of ±0.4 pp are within noise.
- At 1.6% the reinit is at chance in both seeds, so that advantage (~78 pp) is inflated (P4).

### F4. Low learning rate: clean replication in both seeds
- Tickets beat trained reinits at every level where the reinit trained: CPU +1.5 to +3.9 pp,
  GPU +1.5 to +13.5 pp. Ticket accuracy stays within 0.5 pp of dense (86.3% / 86.6%) or above it
  down to 1.6% remaining.
- The reinit collapses to chance from 6.2% (CPU) and 3.1% (GPU), so the advantages there are
  inflated, as in P4. The main study's `shuffle` baseline and BatchNorm exist for exactly this.

### F5. Weight correlation (R_0.2) behaves as in P5
- Dense R_0.2 drops with learning rate: 0.64 / 0.62 at eta 0.01, versus 0.23 / 0.22 at eta 0.1
  (no warmup) and 0.31 / 0.26 at eta 0.1 with warmup.
- For sparse tickets, R_0.2 is 0.15–0.28 in every condition. Winning (warm) and losing or
  collapsing (high) tickets are indistinguishable, and both sit near chance (0.2).
- In Conv-4, therefore, weight correlation is neither necessary for a winning ticket nor what
  separates warm from no-warm. H4's anchor arm is still needed on ResNet-20.

### F6. The pilot's sharpness numbers should not be relied on (revises P2 and P3's S values)
- 27 of the 90 runs that trained to > 20% accuracy (mean 84.6%) report lambda_max at step 100 of
  0.07–0.20. That is the same value reported for dead, chance-level networks, and 100–1000x
  below the dense values (10–120). A network actively learning at eta ≥ 0.01 is very unlikely
  to be that flat.
- Sparse lambda_0 values are suspiciously uniform (IQR 0.114–0.132 across all three run sets).
- `lt_sharp.py` uses plain power iteration with a shift re-run when the estimate is negative.
  That is the method `NOTES.md` D1 showed to fail on ResNet-20: it returned 0.04–2.6 where
  Lanczos gives 25.6. The pilot values are consistent with the same failure.
- **Implication.** P2 ("sparse init is flatter") and the S(100) comparisons in P3 are
  **unverified**, not refuted. The main-study code already uses Lanczos. If P2 goes in the
  report, re-measure it with `src/sharpness.py`.
- Also: S must use the learning rate in effect at the measured step. With warmup, eta at step 100
  is 0.043, not 0.1. The warm dense S(100) is 0.46 (CPU) / 0.54 (GPU), not 1.08 / 1.27.
  `pilot_tables.md` uses eta_t throughout.

### F7. The early loss spike is the cleanest failure signal in the pilot (new)
- Every run with a loss spike > 8x in the first 200 steps failed: 13.3x (dense, GPU), 11.6x
  (25% ticket, GPU), 29.2x (50% reinit, GPU) and 2.8e10 (25% ticket, CPU, NaN).
- The highest spike among runs that trained was 6.3x: the CPU 50% ticket that lost to reinit.
  See `fig_pilot_spike.png`.
- The other failures, with spike ≈ 1, are sparse reinits that never started learning. That is
  a different failure (P4), not an instability.
- **Implication for H2.** The collapse happens before step 200. A sharpness schedule starting at
  step 0 and then every 250 steps (spec), or every 500 steps (the O1 recommendation), **will
  measure the network only after it has already collapsed or stabilised**, and a collapsed
  network reads as flat (S ≈ 0). Two options: add a few dense early points (e.g. steps 25, 50,
  100, 150, 200) when deciding O1, or treat the loss spike as the primary instability outcome
  and S(t) as the mechanism. This is a proposal only; no config was changed.

## 3. Metrics table (pilot, test set)

| Run | Dense acc / F1 / AUC | Ticket at ~6% acc / F1 / AUC | Reinit at ~6% acc |
|---|---|---|---|
| 2-ep, eta 0.05 (spec reference) | 83.7 / 0.834 / 0.984 | 85.3 / 0.851 / 0.985 (6.4%) | 71.7 |
| 4-ep CPU, eta 0.01 | 86.3 / 0.863 / 0.988 | 87.8 / 0.878 / 0.990 (6.2%) | 19.0 (chance) |
| 4-ep GPU, eta 0.01 | 86.6 / 0.865 / 0.988 | 87.9 / 0.879 / 0.990 | 74.4 |
| 4-ep CPU, eta 0.1 | 87.1 / 0.871 / 0.989 | 86.0 / 0.859 / 0.987 | 87.9 |
| 4-ep GPU, eta 0.1 | 10.0 / 0.018 / 0.500 (collapsed) | 87.9 / 0.879 / 0.990 † | 85.1 |
| 4-ep CPU, eta 0.1 + warmup | 89.8 / 0.899 / 0.993 | 90.2 / 0.902 / 0.993 | 85.4 |
| 4-ep GPU, eta 0.1 + warmup | 88.7 / 0.886 / 0.991 | 89.0 / 0.890 / 0.991 | 87.0 |

The first row matches the spec's section 10 reference metrics exactly, which confirms the files
are the ones the spec summarised. Full per-round tables (precision, recall, R_0.2, spike, S) are
in `pilot_tables.md`.

## 4. Caveats

- Each 4-epoch condition now has 2 seeds (1 CPU, 1 GPU). The original 2-epoch pilot has 1.
  Nothing here has error bars, and differences under ~1 pp are noise.
- Conv-4 has no BatchNorm, and its only baseline is `reinit` (no `shuffle`). This is directional
  evidence for designing the main study, not a result to report as a finding.
- Pruning rates differ between run sets (60% vs 50% per round), so sparsity levels do not line up.

## 5. Suggested next steps (need your decision; nothing was run)

1. **Decide O1**, taking F7 into account: add dense early sharpness points (≤ 200 steps), or
   make the loss spike the primary instability measure.
2. **Decide O4** (G4 budget), then run the smoke benchmark (notebook cells 4–5; cheap). This T4
   session could run it too.
3. In the report, cite P1 as "dense eta = 0.1 collapses in 2 of 3 short runs", and drop P2 or
   re-measure it with Lanczos.
4. Add a per-condition **collapse rate** to the main-study summary (the code already records
   `diverged` via validation accuracy < 0.15, D4).
