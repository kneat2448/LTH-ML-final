# Phase A write-up (seed 0): draft

*Status: draft of 2026-10-01, 05:00 UTC. Phase A is 58/60; the high_warm r8 (5.8%) reinit and shuffle are still
running, and those cells are marked "pending". Seed 1 (Phase B) is running, so every number here is a **single run**. Treat
differences under ~0.5 pp as noise until seed 1 is in. All numbers come from `results/summary.csv`
(`python -m analysis.plots`); the per-run table is in `Methodology.md` section 10. Setup and definitions: `Methodology.md` sections 3–6.*

## 1. What Phase A ran

ResNet-20 on CIFAR-10, IMP with 30% global pruning per round and reset to theta_0, 9 sparsity levels
(100% → 6.0% remaining), 15k iterations per training. Four conditions, all starting from the same theta_0 (seed 0):

| Condition | eta | Warmup | Role |
|---|---|---|---|
| `low` | 0.01 | none | tickets expected to win (Frankle & Carbin 2019) |
| `high` | 0.1 | none | tickets expected to fail |
| `warm03` | 0.03 | linear over 10k steps | Frankle's successful ResNet setting |
| `high_warm` | 0.1 | linear over 10k steps | Frankle reports failure |

Each chain has 9 tickets, a reinit baseline at 49.1, 24.2, 12.0 and 6.0% remaining, and a shuffle baseline at 24.2 and 6.0%:
15 trainings per chain, 60 in all. No run diverged, and no run had an early loss spike above 1.19x.
(The pilot's failure signal was a spike above 8x.)

## 2. H1 (replication): Gate A passed

**Ticket accuracy minus dense accuracy of the same condition (pp)**

| % remaining | low (dense 86.66%) | high (dense 89.83%) | warm03 (dense 87.48%) | high_warm (dense 89.46%) |
|---|---|---|---|---|
| 70.1 | +0.77 | −0.44 | +0.74 | +0.64 |
| 49.1 | +1.19 | −0.62 | +1.09 | +0.29 |
| 34.5 | +1.21 | −1.47 | +1.04 | +0.84 |
| 24.2 | +1.26 | −2.55 | +1.63 | +0.89 |
| 17.0 | +1.18 | −3.70 | +1.17 | +0.99 |
| 12.0 | +0.93 | −4.31 | +1.17 | +0.26 |
| 8.4 | +0.51 | −5.58 | +0.46 | −0.33 |
| 6.0 | −0.53 | −7.11 | +0.38 | −1.11 |

**Ticket advantage = ticket − best trained baseline at the same mask (pp).** ✓ marks a winning ticket: advantage > 0 and
ticket ≥ dense − 0.5 pp.

| % remaining | low | high | warm03 | high_warm |
|---|---|---|---|---|
| 49.1 | +2.07 ✓ | +0.36 | +1.91 ✓ | +1.47 ✓ |
| 24.2 | +3.28 ✓ | +0.13 | +3.87 ✓ | +3.45 ✓ |
| 12.0 | +4.54 ✓ | +0.39 | +4.89 ✓ | +5.11 ✓ |
| 6.0 | +6.00 (ticket −0.53 vs dense) | −0.03 | +7.43 ✓ | pending |

The reinit was the best baseline in every case except high_warm at 24.2% (shuffle 86.90% vs reinit 86.67%).
Fig. 1 (`results/figures/fig1_accuracy.png`) shows the curves.

**Gate A: passed** (`results/gate_a.md`). low wins at 4/4 sparsities, warm03 at 4/4 and high at 0/4. At 6.0% the low ticket
beats every baseline by 6 pp but sits 0.53 pp below dense, just outside the tolerance. The gate needs only ≥ 3.

- **low and warm03 replicate Frankle & Carbin.** The tickets stay at or above dense down to 8.4% remaining (warm03 down to 6.0%), and the
  advantage over the baselines grows steadily with sparsity: +2 pp at 49% to +6–7 pp at 6%.
- **high replicates the failure, though in a milder form than "collapse".** The high ticket loses accuracy steadily with sparsity
  (−0.4 → −7.1 pp vs dense). It beats a random reinit by at most 0.4 pp, and at 6.0% not at all (82.72% vs 82.75%).
  We judged high's three small positive advantages (+0.36, +0.13, +0.39 pp) by the winning-ticket definition, not the sign alone (D12, a deviation to report).
  They come with tickets 0.6–4.3 pp below dense and are within the scale of seed noise.
  The dense high network is the most accurate dense network (89.83%), so this is not a weak baseline.
- **Seed 1 so far agrees** (low and high, rounds 0–5). The seed-1 dense accuracies are within 0.3 pp of seed 0 (low 86.48%, high 89.55%).
  low tickets are +0.63 to +1.64 pp vs dense, with advantages of +2.11 (49.1%) and +3.64 pp (24.2%). high tickets are −0.32 to −3.21 pp vs dense,
  with advantages of +0.15 and +0.07 pp. Error bars for the final report come from the two seeds.

## 3. high_warm: winning tickets at eta = 0.1, against Frankle

Frankle & Carbin found no winning tickets at eta = 0.1 even with warmup. **We find winning tickets at 49.1, 24.2 and 12.0% remaining.**
These are the best accuracies in the study (90.45% at 17.0%), with advantages over reinit of +1.5 to +5.1 pp.
Below 12% the high_warm tickets fall under dense (−0.33 pp at 8.4%, −1.11 pp at 6.0%). Whether the 6.0% ticket still beats its
baselines is pending.

**Caveat on what "eta = 0.1 with warmup" means here.** The 10k-step warmup ends at the first LR milestone (`src/train.py::lr_at`).
The LR rises linearly to 0.1 at step 9,999 and is cut to 0.01 at step 10,000, so **high_warm never trains at a sustained eta = 0.1**.
Its LR passes 0.03 only after step 3,000, and its mean LR over the first 10k steps is 0.05.

**Frankle & Carbin's ResNet protocol (checked in arXiv 1803.03635v5, Sec. 4 and App. I.3–I.5).**
- **Schedule.** 30k iterations in three stages of 20k, 5k and 5k (LR ÷10 at 20k and 25k), batch 128, momentum 0.9, weight decay 1e-4.
  Warmup is linear from 0 to the target LR over k iterations.
- **Warmup ends at the first decay in their setting too.** Their successful ResNet setting is eta = 0.03 with k = 20,000, so warmup also ends
  at the first decay (20k). warm03 (10k of 15k) is the same schedule shape compressed 2x, and so is high_warm at eta = 0.1. The
  "never at a sustained high LR" caveat therefore applies equally to Frankle's warmup runs. It is not a difference between the studies.
- **Their eta = 0.1 + warmup result for ResNet is a single sentence, with no figure.** "Even with warmup, however, we could not find
  hyperparameters for which we could identify winning tickets at the original learning rate, 0.1" (Sec. 4). App. I.5: "warmup made it
  possible to increase the learning rate from 0.01 to 0.03, but no further". The warmup-length sweep (Fig. 44: k ∈ {0, 0.5k, 1k, 5k, 10k, 20k})
  is shown only at eta = 0.03. The k values tried at 0.1 are not reported.
- **For VGG-19, warmup does rescue eta = 0.1** (k = 10,000, Fig. 7 / Fig. 45; 112k-iteration schedule, so there warmup ends long before the decay).
- **Their winning-ticket criterion** is matching the accuracy of the unpruned network (Sec. 1) in at most the same number of iterations.
  Ours also requires beating trained baselines at the same mask (Methodology section 6). By their criterion alone, high_warm still has
  winning tickets from 70.1% to 12.0% (+0.26 to +0.99 pp vs dense).

**Differences that could explain the disagreement** (none tested):
- **Half the iterations.** 15k vs 30k, so the ramp to 0.1 is twice as steep per step (10k vs, for a 20k warmup, 20k steps).
- **Pruning rate.** 30% vs 20% per round. Also, they do not prune the output layer; we prune it at half the rate.
- **Initialization.** Kaiming normal (`src/models.py`) vs their Gaussian Glorot. Glorot gives smaller conv weights for these fan-ins.
  That changes lambda_max at theta_0 and the early dynamics, which is exactly what H2 is about.
- **Shortcuts.** Option-A (zero-pad) shortcuts here. They mention 2,560 downsampling parameters, so they used projection shortcuts.
- **Seeds.** One seed here; Frankle report averages of five trials. We do not know which k they tried at 0.1.

Report it as a disagreement with Frankle, with these differences listed, and not as a refutation.

## 4. H2 (stability ratio): not supported as stated

H2 predicts that where tickets fail, S(t) = eta_t · lambda_max(t) / 3.8 reaches about 1 in early training, and stays well below 1 where they win.

**max S over steps 25–3,000 (tickets, all rounds):** low 0.08–0.15, warm03 0.03–0.08, high_warm 0.05–0.09, **high 0.17–0.54**.

- **No run comes near S = 1.** The largest training-time value is 0.54 (high, 70.1%). The high tickets fail at S ≈ 0.2–0.3,
  a factor of 3–5 below the stability edge.
- **S does separate the failing condition from the winning ones**, at the level of conditions: every high ticket has max S ≥ 0.17,
  and every ticket in the other three conditions has ≤ 0.15. A weaker version of H2 survives: failure goes with *higher*
  early S, but not with the edge of stability.
- **The difference lasts only a few hundred steps.** Without warmup, the network flattens itself almost at once: dense high
  goes from lambda_max 25.5 to 6.2 within 25 steps and to ~2 by step 1,000. After step 1,000, S is 0.02–0.09 in *every* condition,
  including high_warm when its LR is near 0.1 (step 9,000: lambda_max 1.3–1.7, S 0.03–0.04, the same as high at that step).
  If sharpness is the mechanism, the damage must happen in roughly the first 200 steps.
  Our schedule samples that window only at steps 25, 50, 100 and 200.
- **The baselines at high have the same S as the tickets** (reinit 0.33 vs ticket 0.24 at 49.1%; 0.31 vs 0.31 at 24.2%;
  0.27 vs 0.18 at 12.0%). So S says nothing about the ticket-vs-baseline difference *within* the high condition. This fits,
  because there is no difference to explain: neither the ticket nor the baselines win.
- **Sharpness at initialization does not predict failure**, as H2 assumed (pilot P2). S(0) for high ranges 0.13–0.67 across masks and
  falls with sparsity (sparser masks start flatter: lambda_max(theta_0 ⊙ m) is 5.1 at 6.0% vs 25.5 dense). The ticket deficit grows
  as S(0) falls. Seed 1's dense high network *starts* above the limit (S(0) = 1.07, lambda_max 40.6) and trains normally (89.55%).

Fig. 3 (`fig3_max_s.png`) is the H2 plot; fig. 2 (`fig2_sharpness.png`) has the trajectories.

## 5. H3 (warmup mechanism): partly supported

- **Warmup does keep S low and coincides with winning tickets.** warm03 and high_warm stay at max S ≤ 0.09 and win (§2–3);
  high does not.
- **But the mechanism is not "keeping S below 1".** high never reaches 1 either (§4). The measurable difference is that
  warmup removes the brief S ≈ 0.2–0.5 transient of the first ~200 steps at full LR.
- **Warmup also lets sharpness grow.** In warm03, lambda_max *rises* during warmup (progressive sharpening, Kalra & Barkeshli 2024),
  from 25.5 to 52.7 at step 2,000. The small LR still holds S ≤ 0.08, and lambda_max then falls to ~4 as the LR grows.
  In high_warm, sharpening stops earlier (peak 36.6 at step 500) and turns down once the LR passes ~0.01.
  In both, the turn-over happens at S ≈ 0.08–0.09, far below 1.

## 6. Weight correlation (Liu et al. 2021): R_p alone does not separate the conditions

**R_0.2 (chance ≈ 0.2), tickets:** low 0.64 → 0.26, warm03 0.56 → 0.21, high_warm 0.33 → 0.20, high 0.25 → 0.22.

- **The dense pattern agrees with Liu et al.** A large LR decorrelates theta_T from theta_0 (high 0.25, near chance; low 0.64). Warmup keeps part
  of the correlation even at eta = 0.1 (high_warm 0.33).
- **The sparse tickets contradict a simple "R_p predicts winning" reading.** high_warm wins at 12.0% with R_0.2 = 0.210, and warm03 is
  still above dense at 6.0% with 0.214. Both are *lower* than the failing high tickets at the same sparsity (0.227, 0.216).
  R_p falls with sparsity in every condition, so across conditions it tracks sparsity more than ticket success.
- So at the condition level, early S separates high from high_warm and R_0.2 does not. This is the comparison H4 tests
  causally, with the masks held fixed.

## 7. Open questions handed to Phase B

1. **Error bars.** Do the high advantages (+0.1 to +0.4 pp) and the high_warm wins hold on seed 1? (Seed 1 of high_warm is not in the plan.)
2. **H4 decides between the two readings in §4 and §6.** Train warm03's masks at eta = 0.1 (a) plain, (b) with SAM, which lowers sharpness, and
   (c) with an L2 anchor to theta_0, which raises R_p. If SAM rescues the ticket and the anchor does not, that supports sharpness, in the weak
   "early transient" form of §4.
3. **The first 200 steps are under-sampled** (4 points). If H4 points to sharpness, a denser early schedule on a few runs is the
   cheapest follow-up. Any such follow-up needs budget approval.

## 8. Limitations (Phase A)

- One seed per condition. Seed 1 covers low/high/warm03 only.
- 15k iterations (half of Frankle's 30k), 30% pruning per round (Frankle 20%), no rewinding.
- high_warm's warmup ends at the LR decay, so it never trains at a sustained eta = 0.1 (§3; Frankle's eta = 0.03, k = 20k warmup has the same structure).
- lambda_max is measured on one fixed 2,048-image batch with 10 Lanczos steps, at 15 points per run. Transients between points are missed.
- The deviations from the spec (D1 Lanczos, D11 schedule, fp16, the H2 window excluding step 0, the D12 Gate A criterion) are listed
  in `Methodology.md` section 9.
