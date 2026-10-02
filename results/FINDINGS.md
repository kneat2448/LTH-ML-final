# FINDINGS: verdicts for H1–H4

*ResNet-20 / CIFAR-10, IMP with 30% pruning per round (8 rounds, down to 6.0% remaining), no rewinding, 15k iterations, batch 128,
momentum 0.9. Seeds 0 and 1 for low / high / warm03, seed 0 for high_warm and H4. All numbers come from `results/summary.csv` and
`results/ticket_advantage.csv` (`python -m analysis.plots`). The details behind each verdict are in `PHASE_A.md` (H1), `H2_H3.md` (H2, H3) and `H4.md` (H4).
"Advantage" is ticket accuracy minus the best trained baseline (reinit or shuffle) at the same mask. A **winning ticket** has advantage > 0
(with two seeds, mean > 2 SE) and accuracy ≥ dense − 0.5 pp (section 5 of the spec; D12).*

## H1 (replication): **supported**

Tickets beat their trained baselines at eta = 0.01 and at eta = 0.03 with warmup, but not at eta = 0.1 without warmup (Gate A, two seeds: `gate_a.md`).
- **low** (eta 0.01): +2.09, +3.46, +4.62 and +6.27 pp at 49.1, 24.2, 12.0 and 6.0% remaining (two-seed means, SE ≤ 0.28 pp). 4/4 winning.
- **warm03** (eta 0.03, 10k-step warmup): +1.97, +3.72, +4.83 and +6.43 pp (SE ≤ 1.0 pp). 4/4 winning on the means.
- **high** (eta 0.1, no warmup): +0.26, +0.10, +0.53 and +0.20 pp, with tickets 0.8–7.2 pp below dense (89.69%). 0/4 winning.

The seeds agree to within 0.6 pp on every advantage except warm03 at 6.0% (+7.43 vs +5.43 pp). Adding warmup at eta = 0.1 (**high_warm**, seed 0) brings the advantage back:
+1.47, +3.45, +5.11 and +5.93 pp, winning at 3/4 (at 6.0% the ticket is 1.1 pp below dense).

## H2 (stability ratio): **not supported**

- **No run reaches S ≈ 1 during training.** Over 63 tickets and 42 baselines, max S(25–3k) is at most 0.54. The failing high tickets peak at 0.16–0.54.
- **A condition-level association holds.** The high tickets' range (0.16–0.54) does not overlap the winning conditions' (≤ 0.15). Over all tickets, ρ(max S, gap to dense) = −0.59 (n = 56).
  The difference is a transient: in 15 of 18 high tickets the maximum falls at step 25.
- **S does not order the tickets within a condition.** ρ is +0.26 for high, −0.38 for low and +0.19 for warm03. S(0) tracks mask density, not failure.
- **The causal test (H4) goes against the transient.** warm03's masks trained at eta = 0.1 show the same transient (max S 0.24–0.34) and still win.
  SAM lowered S at every mask and left the advantage unchanged (Δ ≤ 0.5 pp). In reverse, warmup removed the transient from high's own masks (F2) and they still failed.
- **The threshold itself is uncertain.** Under momentum the step-0 threshold is 2/lambda (Kalra & Barkeshli 2024), and against that dense high starts above it (1.27 / 2.03).
  The minibatch threshold is lower than (2 + 2·beta)/lambda and was not measured.

## H3 (warmup mechanism): **partly supported**

- **The coincidence holds.** At eta = 0.1, warmup brings the ticket advantage back (high_warm 3/4 winning vs high 0/4) and removes the step-25 transient (S ≤ 0.003 vs 0.13–0.54).
- **"Keeps S below 1" does not discriminate.** high stays below 1 as well, and from step ~1,000 both runs sit at the same plateau (S 0.03–0.04).
- **Warmup is not needed to *train* a good mask.** H4(a) shows this.
- **Warmup matters for *finding* the mask during IMP.** high_warm finds winning masks at eta = 0.1 and high does not.
  The reverse swap (F2) tests this: warmup applied only when high's (bad) masks are trained does not help them (≤ +0.67 pp vs shuffle).
- **The Phase A coincidence is confounded:** warmup also keeps more weight correlation (dense R_0.2 0.33 vs 0.25) and halves the mean LR over the first 10k steps.

## H4 (causal test, masks held fixed): **the premise fails; the failure comes from the mask, not from training at eta = 0.1** (confirmed by two follow-ups)

Rescue rule (D13, fixed before the data): ticket − shuffle ≥ +2.0 pp at ≥ 2 of 3 masks, and the intervention must move its target variable.
Masks are warm03 seed 0 rounds 3, 6 and 8 (34.5, 12.0 and 6.0% remaining); one seed.

| | 34.5% | 12.0% | 6.0% | Rule |
|---|---|---|---|---|
| (a) plain, eta 0.1 | +0.89 | **+3.59** | **+4.80** | 2/3 → masks still win |
| (b) SAM (rho 0.05) | +1.15 | **+3.22** | **+5.26** | 2/3, S lowered at 3/3 |
| (c) L2 anchor (lam 0.01) | **+7.63** | **+10.83** | **+12.32** | 3/3, R_0.2 raised at 3/3 |

- **(a) There is no failure to rescue.** At eta = 0.1 with no warmup, warm03's masks give 88.91% and 87.24% at 12.0% and 6.0%.
  The high chain's own masks give 85.52% and 82.72% under identical training. The baselines are equal (82.4% vs 82.2% at 6.0%).
  This happens with high's early sharpness transient present (max S 0.24–0.34) and R_0.2 near chance (0.18–0.20).
  So, by the rule fixed in advance (`H4.md` §4), high's failure lies in **the masks IMP finds at eta = 0.1**: neither sharpness during training nor the loss of weight correlation explains it.
- **(b) SAM changes nothing** that matters (Δ advantage ≤ 0.5 pp, tickets within 0.3 pp of (a)).
- **(c) The anchor's large advantage is not a better ticket.** The anchored tickets lose 2.7–3.6 pp against (a) and the shuffles lose 10.0–10.4 pp.
  So pulling toward theta_0 helps the original pairing of mask and theta_0 relative to a shuffle, but it does not improve the ticket.
- **Follow-ups (pre-registered in `H4.md` §8, results in §9):**
  - **F1, (a) on seed 1: replicates.** +0.93 / **+3.54** / **+4.40** pp (2/3 ≥ +2.0), within 0.4 pp of seed 0 at every mask.
  - **F2, high's own masks trained *with* warmup (same theta_0, eta = 0.1): no rescue.** −0.02 / +0.67 / +0.31 pp (0/3).
    Warmup removed the early transient (max S 0.09–0.12 vs 0.18–0.29) and the tickets still did not improve (82.04% vs 82.72% at 6.0%).
  - So, with theta_0 and eta held fixed, a good mask wins without warmup and a bad mask loses with it.
- **Limits.**
  - One seed for (b), (c) and F2; (a) now has two. The (a)/(b) advantages of +3.2 to +5.3 pp are well above the seed-to-seed spread of advantages in low / high / warm03 (≤ 0.6 pp, and 2.0 pp once: warm03 at 6.0%).
    The (a) verdict needs both passing masks. Each clears +2.0 pp by more than the largest spread seen there: 12.0% by 1.6 pp (spread ≤ 0.3), 6.0% by 2.8 pp (spread up to 2.0).
  - Under the strict winning-ticket definition, (a)'s tickets are still 0.56, 0.92 and 2.59 pp below dense high (89.83%), so some of eta = 0.1's cost survives a good mask.
  - Only warm03's masks were tested.

**Overall.** Tickets fail at eta = 0.1 without warmup (H1), but the cause is not the stability ratio reaching 1 (H2), and warmup's benefit does not come from training a given mask more stably (H3, H4).
A good mask trains fine at eta = 0.1 (two seeds), and a bad mask stays bad with warmup. What the high LR breaks is IMP's choice of mask.

## Compute

137 trainings (incl. 2 anchor-selection runs; 12 of them the follow-ups), 26.1 h of the 30 h Colab runtime budget (`compute_log.csv`; `src/utils.py` re-calibrated against the Colab usage page after each VM loss).
