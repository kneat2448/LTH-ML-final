# NOTES: session log and handoff

Running log for continuing this project across sessions and machines. The spec is
`src/CLAUDE_1.md` (read it first). This file records what exists, what was decided, what was
measured and what is still open. **Update it at the end of every work session.**

---

## 1. Current status (2026-09-29, session 4, Colab T4)

- **SESSION 6 (2026-10-01, from 03:32 UTC): recovered from the VM loss (≈ 10:00 UTC 09-30) and relaunched everything; H4 is queued.**
  Status at 03:32: Phase A 55/60 (high_warm: r6 reinit, r7 ticket, r8 ticket/reinit/shuffle left), seed 1 12/45 (low 6, high 6, warm03 0), H4 0/18.
  Running under MPS: high_warm s0, low s1, high s1; `resume_phaseB.sh warm03` waits for high_warm. **H4 queue (`colab/queue_h4.sh`)**: slot A waits
  for low s1, then `h4_high_anchor --select-lam` → `h4_high_anchor` → `h4_high`; slot B waits for high s1, then `h4_high_sam`.
  Forecast: high_warm ≈ 05:40, low/high s1 ≈ 07:15, warm03 s1 ≈ 11:40, H4 ≈ 12:00–12:30 UTC → **all runs done ≈ 12:30 UTC (≈ 9 h wall)**.
  **Restart after a VM reset:** `bash colab/resume_phaseA.sh high_warm; bash colab/resume_phaseB.sh low high` (or `warm03` once high_warm is done),
  then the two `queue_h4.sh` lines from the section 9 log (they start at once if the seed-1 chain is not running).
  **Budget (user, session 6): 14 h of Colab runtime left at 03:37 UTC** → `RUNTIME_USED_BEFORE_H = 16.0`, `RUNTIME_SINCE = 2026-10-01T03:37:00`.
  Plan ≈ 9 h → ≈ 5 h reserve. (Chains started before this print the old figures in COST ESTIMATE.)
- **SESSION 5 (2026-09-30, from 04:38 UTC): Phase A resumed; 52/60 done at 07:25 UTC (only high_warm left), Gate A final: PASSED; seed 1 running.** high, warm03 and high_warm run under MPS
  with the watcher. Left: warm03 1 (r8 shuffle), high_warm 12. high is complete. **Phase B seed 1 (low, high) starts when warm03 exits (≈ 06:20), warm03 seed 1 when
  high_warm exits**; high_warm then ends ≈ 10:45 UTC, seed 1 ≈ 13:30–14:00 UTC. Restart after a VM reset: `bash colab/resume_phaseA.sh high_warm; bash colab/resume_phaseB.sh`.
  **Budget (user, session 5): 20 h of Colab runtime left** → `src/utils.py`: `RUNTIME_USED_BEFORE_H = 10.0`, `RUNTIME_SINCE = 2026-09-30T05:02:00`.
  (Chains started before that still print the old figures in their COST ESTIMATE lines.) Remaining plan ≈ 3 h Phase A + ≈ 9 h Phase B
  ≈ 12 h, leaving ≈ 8 h reserve. See the section 9 log for progress. The session-4 notes below are kept for reference.

- **END OF SESSION 4 (09:13 UTC, user shut the runtime down). Phase A 37/60 trainings done (62%); whole study ≈ 37/123 (30%).**
  | Chain | Done | Left |
  |---|---|---|
  | low | **15/15, complete** | — |
  | high | 11/15 | r7 ticket, r8 ticket + reinit + shuffle |
  | warm03 | 11/15 | r7 ticket, r8 ticket + reinit + shuffle |
  | high_warm | 0/15 | all (it had just started r0 at 09:04; that run is lost) |
  The trainings in progress at shutdown (high r7, warm03 r7, high_warm r0) are lost; they restart from scratch on resume.
  **Gate A (seed 0): PASSED**: low 4/4, warm03 3/3, high 0/3 winning tickets (D12). high/warm03 r8 and high_warm are still to add.
- **NEXT SESSION, step by step:**
  1. Mount Drive, `cd /content/drive/MyDrive/final_project`, check `git status` / `git log -1` (everything was pushed; HEAD = the
     "Session 4 end" commit).
  2. **Re-calibrate the budget:** read the remaining Colab hours from the usage page. In `src/utils.py`, set `RUNTIME_USED_BEFORE_H = 30 − remaining`
     and `RUNTIME_SINCE` = now (UTC, ISO format). Commit. (Logged at shutdown: 7.0 h, plus unlogged idle time.)
  3. `bash colab/resume_phaseA.sh`: copies CIFAR, starts MPS, and launches high, warm03 and high_warm (low exits at once: all done) + the watcher
     (commits and pushes each result, 15-min autosave).
  4. **Forecast:** 23 trainings left. high/warm03 ≈ 1.6 h in parallel; high_warm ≈ 4 trainings alongside them, then 11 alone at ~9 min →
     **Phase A done ≈ 3.3 h after the resume**. Then `python -m analysis.plots`, read `results/gate_a.md`, and write up Phase A.
  5. Phase B (Gate A passed): `--seeds 1` for low/high/warm03 (45 trainings, 3 chains under MPS ≈ 6 h), then H4 (`--select-lam`
     first; ≈ 18 trainings incl. SAM ≈ 3 h). **Total study ≈ 7 + 3.3 + 9 ≈ 19–20 h of the 30 h runtime budget.**
- **After a VM restart, run `bash colab/resume_phaseA.sh`** (copies CIFAR, starts MPS, relaunches any chain that is not running,
  starts the watcher). Finished trainings are skipped; only the training in progress is lost (~25 min per chain).
- **Watcher, now in the repo (`colab/watcher.sh`):** commits and pushes each finished result, and **every 15 min autosaves**
  (regenerates `analysis.plots` + the Methodology table, commits changes in `results/`, `Methodology.md` and `NOTES.md`, pushes, and
  retries failed pushes). Output: `/content/watcher.out`. Claude also adds a NOTES entry every ~15 min / per batch.
- Derived analysis outputs (`results/summary.csv`, `figures/`, `gate_a.md`, …) **are committed since session 4** (user request),
  and the autosave keeps them current.
- **Session 4 (05:10 UTC):** the Colab VM had restarted, so no chains were running. The trainings in progress were lost
  (low r4, high r1, warm03 r1); every finished result was already committed. The three chains were relaunched under MPS with the
  same commands, and finished trainings were skipped. A new watcher (in the session scratchpad) commits each result and **pushes it**
  (token read from Drive `.secrets`; the token is never printed). Remaining: low 10, high 14, warm03 14 trainings (~25 min each in parallel)
  → low done ≈ 09:30, high/warm03 ≈ 11:00 UTC, if the VM survives. Then `high_warm`.
- **Budget redefined (session 4, user):** the budget is **30 h of Colab runtime (wall-clock usage), not GPU-hours**. At 05:10 UTC
  on 2026-09-29, 27 h were left. `src/utils.py`: `RUNTIME_USED_BEFORE_H = 3.0` at `RUNTIME_SINCE`, plus the **union** of the
  training intervals logged after that time, so parallel chains count once. Idle runtime (setup, analysis, gaps between runs) is not logged, so
  this underestimates. **Re-calibrate both constants from the Colab usage page at each session start.** `LTH_BUDGET_H` is gone.
  The laptop keeps its 18 GPU-h. The chains running now use the old code in memory; their `COST ESTIMATE` lines still show GPU-h.
- **H2 metric changed (session 4, user-approved):** the H2 statistic is now **max S over steps 25–3,000** (`max_S_train`), with
  **S(0)** reported separately (`src/metrics.py::stability_summary`, used by `analysis/plots.py` and `analysis/methodology_table.py`).
  It is derived from the saved trajectories, so no re-run was needed; run JSONs keep the old `max_S_3k`. Listed as a deviation in `Methodology.md` section 9.
- **Phase A started (session 3):** `resnet20_low` seed 0 is running (15 trainings, ~9.5 min each on the T4).
  Each finished training is committed and pushed as it lands, by a watcher script in the session's scratchpad (not in the repo).
  Before each commit it runs `python -m analysis.methodology_table`. One commit per result file, so `git log`
  shows how far it got. **The watcher dies with the Colab session.** For later chains, either commit by hand after each
  round (`python -m analysis.methodology_table && git add results/<cond> results/compute_log.csv Methodology.md`) or ask
  Claude to start the watcher again. **If the session died mid-chain, re-run
  `python -m src.imp configs/resnet20_low.yaml`; finished trainings are skipped.** Then `resnet20_high` → `warm03` → `high_warm`.
- **Parallel chains under MPS (session 3, user request):** `low`, `high` and `warm03` run at the same time, as 3 processes under
  NVIDIA MPS. Start the daemon before any CUDA process:
  `export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log; mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY; nvidia-cuda-mps-control -d`,
  then launch each chain with the same env vars (`nohup python -m src.imp configs/resnet20_<c>.yaml > /content/phaseA_<c>.out 2>&1 &`).
  The gain is only ~20% in total throughput (section 6), and each chain runs ~3x slower. **Wall times, `it_per_s`, sharpness
  overhead and `compute_log.csv` GPU-hours of parallel runs are inflated (~2.5x per run)**, so do not use them for timing or budget.
  Real GPU use ≈ wall time of the session.
- **Git:** everything is pushed (session 2's commits included). The token is in `/content/drive/MyDrive/.secrets/github_token`
  (outside the repo, never committed); push command in section 8, step 1.
- Earlier status (end of session 2): section 7 step 1 (benchmark) **done** on a **Colab T4** with the new O1 schedule (section 6):
  ~9 min per 15k training, ~19 T4-h for Phase A+B.
- **O1 decided** (D11): 10 Lanczos steps, points at 0, 25, 50, 100, 200, every 500 to 3k, then every 3k, final.
- **Runtime:** T4, so fp16 + GradScaler (D2). Budget is the spec's 30 T4-hours (built into `src/utils.py`,
  no `LTH_BUDGET_H` needed on a T4). O4 (G4 budget) only matters if you switch back to a G4.
- **CIFAR-10:** `data/cifar-10-python.tar.gz` is on Drive (uploaded from the laptop, MD5 c58f3010… verified).
  **Do not download it from cs.toronto.edu on Colab** (see O7). Copy it to local disk at session start:
  `mkdir -p /content/cifar_raw && cp data/cifar-10-python.tar.gz /content/cifar_raw/` and set `LTH_RAW_DATA=/content/cifar_raw`.
- Commits are authored as `Nitai <nitaikoundinye@gmail.com>` (set in the repo's local git config on Colab).
- **Pilot re-analysis (session 2):** `pilot_results/PILOT_FINDINGS.md`. It changes how section 10 of the spec may be cited:
  P1 reversed (dense eta = 0.1 collapses in 2 of 3 runs), P3 weakened (high-LR ticket deficit is seed-dependent),
  P2 unverified (pilot lambda values look like the D1 power-iteration failure), and a new signal: an early loss spike > 8x
  predicted every failure, all before step 200 (the reason for D11's early points).
- **Forecast (T4, ~9 min per training):** Gate A after Phase A ≈ 9 GPU-h (≈ 2 days of Colab sessions);
  full H1–H4 findings after Phase B ≈ 19 GPU-h total (≈ 1 week including analysis and write-up).
- **Next:** Phase A in the order of section 7 (see section 8).

## 2. Quick start

### Local (Windows laptop)
```
conda activate lth           # C:\Users\Nitai\anaconda3\envs\lth\python.exe
python -m pytest             # 21 tests, ~40 s
python -m src.imp configs/smoke.yaml
```

### Colab (preferred from now on; session 2 used a T4)
1. `MyDrive/final_project` is a git clone of the repo (session 2) with `data/cifar-10-python.tar.gz` added.
2. Open `colab/run_on_colab.ipynb` in Colab and pick a GPU runtime (T4 works: ~9 min per 15k training).
3. **Before any cell that loads CIFAR-10**, copy the Drive archive to local disk; the notebook's own download
   from cs.toronto.edu is blocked/slow on Colab (O7):
   `!mkdir -p /content/cifar_raw && cp /content/drive/MyDrive/final_project/data/cifar-10-python.tar.gz /content/cifar_raw/`
   (`LTH_RAW_DATA=/content/cifar_raw` is set in cell 2). torchvision then finds the file, checks the MD5 and skips the download.
4. The budget (30 h of Colab runtime) is built in. Run cells 1–3 (mount, env, tests); cells 4–5 (smoke,
   benchmark) are already done for the T4 (section 6).
4. Phase A/B cells follow. **Run one training cell at a time.**
5. If the session dies: re-run cells 1–2, then the same command. Finished trainings are skipped.

### Commands (same everywhere)
```
python -m src.imp configs/<cfg>.yaml [--seeds 0 1] [--fresh]         # IMP chain, resumable
python -m src.fixed_mask configs/h4_high_anchor.yaml --select-lam     # H4 anchor lam, before the H4 anchor run
python -m src.fixed_mask configs/h4_<x>.yaml                          # H4 fixed-mask runs
python -m analysis.plots                                              # summary.csv, figs, tables, gate_a.md
```
- Outputs: `results/<name>/seed<k>/round<r>_<variant>.json` plus `log.txt`, `config.json`, `results/compute_log.csv`.
- Checkpoints (theta_0, masks, final weights): `checkpoints/<name>/seed<k>/`. Not in git; on Drive they persist.
- If you edit a config after a run exists, the next run goes to a timestamped `results/<name>_<ts>/` (never overwritten).
  **Consequence:** code changes do NOT trigger a new directory. After a bug fix, archive old results
  by hand (`results/_archive/`), as was done for the smoke run.

## 3. Moving the folder to Google Drive

- `final_project_drive.zip` (in `ML_lab/`) is ready to upload. It contains:
  `src/ configs/ analysis/ tests/ colab/ pilot/ pilot_results/ results/ data/*.json NOTES.md
  requirements.txt pytest.ini .gitignore`.
- **Left out on purpose:**
  - `data/cifar-10-python.tar.gz`: Colab downloads it in ~1 min into `LTH_RAW_DATA=/content/cifar_raw`.
  - `checkpoints/`: nothing useful yet.
  - `research paper/` and the `.pptx`: not needed to run. Upload them separately if wanted.
- `data/splits.json`, `stats.json` and `hessian_idx.json` ARE included, so Colab uses exactly the
  same 45k/5k split, normalization and Hessian batch as the laptop. They are deterministic (seed 0) anyway.
- **Do not `pip install -r requirements.txt` on Colab.** It pins Windows `+cu128` wheels. Colab's
  preinstalled torch/numpy/pandas/sklearn/matplotlib/pyyaml/pytest are enough. The notebook prints the versions;
  record them in section 6 on first use.
- Paths are all relative to the project root (`pathlib`), so nothing needs editing on Linux.
- Environment variables used by the code:
  - `LTH_RAW_DATA`: where the raw CIFAR-10 archive goes. Default `data/`.
  - (`LTH_BUDGET_H` was removed in session 4: on Colab the budget is 30 h of runtime, see section 1.)

## 4. What exists

| Path | What |
|---|---|
| `src/data.py` | CIFAR-10 / Fashion-MNIST on GPU as uint8; GPU crop+flip; cached splits/stats/Hessian idx |
| `src/models.py` | `resnet20` (269,722 params, option-A shortcuts, BN), `conv4` (pilot), `init_model(name, seed, save_path)` |
| `src/prune.py` | global magnitude pruning (first conv excluded, output layer at half rate), masks on weights + momentum |
| `src/baselines.py` | `reinit` (seed+1000), `rescaled_reinit`, `shuffle` (layerwise, surviving positions) |
| `src/sharpness.py` | lambda_max by **Lanczos** (see D1), fp32/no TF32, BN stats restored, masked HVPs, micro-batches of 512 |
| `src/metrics.py` | acc, macro P/R/F1, OvR AUC, confusion (from scratch, tested vs sklearn); R_p; loss spike |
| `src/train.py` | `Trainer`: plain / warmup / SAM / anchor; config validation (no hidden defaults); AMP bf16/fp16 |
| `src/imp.py` | IMP driver, resumable, divergence rule, cost estimate, compute log |
| `src/fixed_mask.py` | H4 fixed-mask runs + anchor lam selection |
| `src/utils.py` | seeding, run metadata (git hash, versions, GPU), Tee, compute log, budgets (not in the spec's layout) |
| `analysis/plots.py` | summary.csv, fig1–5, metrics_table.{md,csv}, ticket_advantage.csv, gate_a.md |
| `analysis/pilot_summary.py` | session 2: re-analysis of all pilot JSONs → `pilot_results/{pilot_summary.csv, pilot_tables.md, fig_pilot_acc.png, fig_pilot_spike.png}` |
| `Methodology.md` | session 3: the method as actually run, deviations to state in the report, study steps with status, results table (section 10) |
| `analysis/methodology_table.py` | session 3: rewrites the results table in `Methodology.md` from `results/*/seed*/round*.json` |
| `pilot_results/PILOT_FINDINGS.md` | session 2: pilot findings F1–F7 vs spec section 10 (2 seeds of the 4-epoch design: CPU + GPU re-run) |
| `configs/` | smoke, resnet20_{low,high,warm03,high_warm}, conv4_fmnist_high, h4_high, h4_high_sam, h4_high_anchor |
| `colab/run_on_colab.ipynb` | Colab runner (mount, env, tests, smoke, benchmark summary, Phase A/B cells) |
| `colab/watcher.sh` | session 4: commits + pushes each result, 15-min autosave of results/Methodology/NOTES |
| `colab/resume_phaseA.sh` | session 4: one-command resume after a VM restart (CIFAR copy, MPS, chains, watcher); default = all four Phase A chains |
| `colab/resume_phaseB.sh` | session 5: start/resume the seed-1 chains (default low high warm03) under MPS; `WAIT_FOR=<cond>` waits for that seed-0 chain to exit |
| `colab/start_after.sh` | session 4: start one chain when another finishes (used for low → high_warm) |
| `tests/` | 23 tests (section 12 list plus schedules incl. the D11 schedule, NaN-mask guard, negative-dominant spectrum) |
| `data/cifar-10-python.tar.gz` | official archive on Drive (not in git), uploaded from the laptop in session 2 |
| `results/smoke/seed0/` | T4 smoke run with the D11 schedule (6 trainings); benchmark reference |
| `results/_archive/smoke_laptop_20260928_old_lambda/` | first smoke run, before the Lanczos fix; timing reference only |

## 5. Decisions and deviations from the spec

- **D1. lambda_max uses Lanczos (20 steps, full reorthogonalization), not power iteration + shift.**
  This is bug O2, now fixed. At dense ResNet-20 theta_0 on the real Hessian batch, plain power iteration
  converged to lambda_min ≈ −152. The shifted re-run then returned 0.04 (20 iterations) or 2.6 (60 iterations),
  because after the shift nearly the whole spectrum sits near +152 and power iteration hardly moves.
  Lanczos on the same network: 25.46 (10 steps), **25.60 (20)**, 25.60 (40). The cost per step is the
  same (1 HVP). Cite Lanczos 1950 / Ghorbani et al. 2019 / Cohen et al. 2021 in the report, and
  mention it as a methods note (it strengthens pilot finding P6). `power_iteration` / `top_eigenvalue`
  are kept in the code for reference only.
- **D2. Mixed precision is chosen per GPU:** bf16 autocast if compute capability ≥ 8 (4060, G4, A100, L4),
  else fp16 + GradScaler (T4, as section 4 requires). The dtype is recorded as `amp_dtype` in each run JSON.
  The fp16 path was checked on the GPU in all 4 modes (forced).
- **D3. The fp32 Hessian runs in NCHW.** It switches from channels_last and back inside `sharpness.py`.
  Measured: HVP 422 ms (NCHW) vs 720 ms (channels_last), while bf16 training showed no difference, so training stays channels_last.
- **D4.** "Ended at chance" (divergence) = validation accuracy < 0.15 (`divergence_val_acc`). The test set is not used for choices.
- **D5.** The anchor and R_p use each run's own initialization. For the shuffle baseline that is the shuffled init.
- **D6.** Anchor lam selection: a 3k-step run per lam at the round-3 mask, compared with the `low` ticket's R_0.2
  at step 3000. R_0.2 is logged at every sharpness step for this purpose.
- **D7.** If a ticket diverges, that round's baselines still run on its (good) mask. The chain then stops (`chain_stopped.json`).
- **D8.** HVP micro-batches of 512 each use their own BN batch statistics. BN buffers are restored afterwards.
- **D9.** SAM's second forward pass does not update BN running stats. Its perturbation direction is scale-invariant, so it works with GradScaler.
- **D11. O1 resolved (session 2, user-approved): sharpness schedule.** `sharpness: {batch: 2048, iters: 10,
  micro_batch: 512, every: 500, dense_until: 3000, then_every: 3000, early: [25, 50, 100, 200]}` in every main
  config (smoke: `every: 100, early: [25, 50]`; conv4: `every: 50, early: [10, 25]`). New config key `early`
  = extra measurement steps. They were added because the pilot's collapses happen before step 200
  (`pilot_results/PILOT_FINDINGS.md`, F7). 15 points × 10 HVPs instead of 21 × 20 (≈ 2.8x cheaper).
  Test: `tests/test_train.py::test_sharpness_schedule_o1`. State this deviation in the report.
- **D12. Gate A `high` criterion (session 4, user decision, 08:40 UTC):** for `high`, Gate A counts only *winning tickets*
  in the section-5 sense (advantage over the best trained baseline > 0 **and** ticket acc ≥ dense − 0.5 pp; with ≥ 2 seeds, mean > 2 SE),
  not any positive advantage. Reason: high had two tiny positive advantages (+0.36 pp at 49.1%, +0.13 pp at 24.2%) while its tickets
  were 0.62 and 2.55 pp below dense. The low/warm03 criterion is unchanged (advantage > 0 at ≥ 3 sparsities). `analysis/plots.py::gate_a`
  now also lists high's positive-but-not-winning points in `gate_a.md`. State this in the report.
- **D13. H4 rescue rule (session 6, user decision, 06:00 UTC, before any H4 result):** ticket − shuffle advantage ≥ +2.0 pp at
  ≥ 2 of the 3 warm03 masks, plus a manipulation check (SAM lowers max S(25–3k) vs plain; anchor raises R_0.2 at 3k vs plain), else
  "inconclusive". Calibration: high chain ≤ +0.7 pp everywhere; warm03 smallest win +1.91 pp. `results/H4.md` §5.
- **D10.** Conv-4 / Fashion-MNIST chain (`conv4_fmnist_high.yaml`): 3k iterations, 60%/round, 5 rounds (pilot rates),
  55k/5k train/val split, no augmentation.

## 6. Measurements

### 2026-09-28: laptop RTX 4060, smoke run (before the Lanczos fix; timing only)
- Training step ~35 ms (≈ 9 min per 15k run; the spec assumed 6 min).
- One lambda_max measurement (20 × 4 micro-batch HVPs, fp32, channels_last at the time): 50–117 s.
- Peak VRAM 2.45 GB (limit 6 GB, OK).
- **The GPU was heavily throttled:** 285–630 MHz under load (max 3105), "SW Power Cap" and "SW Thermal Slowdown"
  active, 55 W limit, Windows plan "Balanced". Laptop numbers are not representative until that is fixed
  (Best performance mode, vendor performance mode, airflow).
- After the NCHW fix, still throttled: 20-step Lanczos ≈ 67 s; 10 steps ≈ 22 s.

### 2026-09-28: Colab Tesla T4, smoke run with the O1 schedule (session 2)
- torch 2.11.0+cu128, CUDA 12.8, fp16 autocast + GradScaler (D2). No throttling (1590/1590 MHz, 66 °C, 56 W).
- First training (dense, 300 iters, 6 sharpness points): wall 137.9 s, of which sharpness 130.0 s.
  - Training: 38 it/s in the first (cold) run, **~72 it/s ≈ 14 ms/step** in the 5 later runs → one 15k run ≈ **3.5 min** without sharpness.
  - **One lambda_max (10 Lanczos steps, 2,048 images, fp32) ≈ 21.7 s.**
  - Peak VRAM **2.46 GB** (limit 6 GB, OK).
- **Per 15k training with the O1 schedule (15 points): 3.5 + 15 × 0.36 ≈ 9 min** (sharpness overhead ≈ 150%,
  still far above the spec's 15% rule; that rule cannot be met with a 2,048-image, 10-step Lanczos on any GPU).
  Under the spec's 12-min threshold, so `iters` stays at 15k.
- Dense smoke ticket (lr 0.1, 300 iters): acc 42.3%, R_0.2 0.80, max S(3k) 0.67, not diverged.
- The full smoke chain finished (6 trainings: r0 ticket; r1 ticket + reinit; r2 ticket + reinit + shuffle), none diverged,
  peak VRAM 2.46 GB in all. `python -m analysis.plots` runs end to end on it (its Gate A "NOT PASSED" is expected:
  there are no low/warm03 chains yet). Derived files (summary.csv, figures, gate_a.md) are regenerated, so not committed.

### 2026-09-29: parallel training on the T4 (session 3)
Synthetic ResNet-20 training loop (fp16 + GradScaler, batch 128, 1,000 steps, no sharpness), total it/s over all processes:

| Setup | Total it/s | vs 1 process |
|---|---|---|
| 1 process | 70.6 | 1.00x |
| 3 processes, no MPS (time-slicing) | 79.4 | 1.12x |
| 2 processes, MPS | 84.0 | 1.19x |
| 3 processes, MPS | 85.0 | 1.20x |
| 4 processes, MPS | 82.0 | 1.16x |

- The T4 is **compute-bound** with one ResNet-20 run (99% util). VRAM (~3 GB per process of 15 GB) and RAM (~2 GB per process
  of 12 GB) are not the limit, and the VM has only 2 CPU cores. More processes cannot use the spare memory to go faster.
- Choice: 3 chains under MPS (+20%). This cuts Phase A from ~9 to ~7.5 T4-h of wall time.

### Re-derived budget (section 9 of the spec), T4, O1 schedule
| Phase | Trainings | T4 GPU-h |
|---|---|---|
| Benchmark + tests | smoke | ~0.3 |
| A: low, high, warm03, high_warm (seed 0) | ~60 | ~9 |
| B: seed 1 of low/high/warm03 (~45) + H4 (18, SAM ≈ 2x training) | ~63 (+6 SAM-equivalents) | ~10 |
| Reserve | | ~10.7 |
| **Total** | | **~30** |
- About 19 T4-h for Phase A+B, leaving a large reserve. The Conv-4 chain is not included (O8). If trainings run slower than 12 min in practice, cut `iters` to 10k
  (milestones 6.7k/8.3k, warmup 6.7k) before cutting conditions or seeds (spec section 7).

## 7. Open issues

- **O1. RESOLVED in session 2 → D11.** (Original text kept below for the record.)
  Sharpness cost vs the 15%-overhead rule.
  One measurement = 20 HVPs on 2,048 images ≈ 1,300 training steps of compute. The spec's 21 points
  per run ≈ 190% overhead on **any** GPU; a faster GPU does not change the ratio. Options (they combine):
  - (a) **fewer Lanczos steps**: 10 steps were within 0.5% of the converged value on dense theta_0 (≈ 2x cheaper);
  - (b) warm-start from the previous top eigenvector (≈ 2–4x);
  - (c) Hessian batch 512 instead of 2,048 (≈ 4x, noisier);
  - (d) fewer points: every 500 steps up to 3k, then every 3k (21 → ~11 points, ≈ 2x).
  - Or accept the overhead on the G4 and budget for it.

  All of these are config/code changes that the spec says need approval. After the decision, update
  `sharpness:` in every config (the smoke config too) and re-run the smoke benchmark.
- **O3. Resolved (2026-09-28); pushed in session 3.** The token pasted in chat in session 2 was never used; revoke it.
  A token needs write access: fine-grained → Contents: Read and write on this repo; classic → `repo` scope
  (a read-only token gives `403 Permission ... denied`, as happened once in session 3).
  Original entry: Git repo initialised, remote `origin` = https://github.com/kneat2448/LTH-ML-final.git
  (branch `main`). Not committed on purpose: `data/`, `checkpoints/`, `*.pt`, `pilot/data/` (Fashion-MNIST .gz;
  get it from the Drive zip or zalandoresearch/fashion-mnist), `research paper/` (third-party PDFs), `*.zip`.
  On Colab, `git clone` also works instead of the Drive zip, but then copy `pilot/data/*.gz` in by hand.
  Uploading `.git` with the Drive copy lets runs record the commit hash.
- **O4. Moot since session 4**: the budget is 30 h of Colab runtime on any GPU (section 1). Old text: **Budget for the G4. Moot while on a T4** (the built-in 30 T4-h budget applies). Only needed if runs move to a G4. Sections 4 and 9 only cover the RTX 4060 (18 h) and a T4 (30 h). Agree on a
  G4 budget (compute units), set `LTH_BUDGET_H`, and note it in the spec's section 9.
- **O5.** Laptop throttling (section 6). Irrelevant on Colab, but fix it before using the laptop again.
- **O7. cs.toronto.edu blocks this Colab IP (session 2).** The single-connection download ran at ~55 kB/s. A
  16-connection aria2c retry then triggered TLS handshake failures (probably rate limiting). Always use the Drive
  copy in `data/` instead.
- **O8. `pilot/data/` (Fashion-MNIST .gz) is not on Drive**, so `configs/conv4_fmnist_high.yaml` (the first
  Phase A step) cannot run on Colab yet. Upload the four .gz files to `pilot/data/`, or skip the Conv-4 chain
  (its purpose, de-risking H2, is partly covered by the pilot re-analysis).
- **O6.** `results/compute_log.csv` contains 4 laptop rows from the archived smoke run (0.47 GPU-h, real time spent).
  They count toward the 4060 budget only (budgets are tracked per exact GPU name).

## 8. Handoff: what to do next (in order)

1. **Push the unpushed commits.** The token lives in `/content/drive/MyDrive/.secrets/github_token` (on Drive,
   **outside the repo**, so it is never committed; never write the token itself into this file, because NOTES.md is pushed
   and GitHub would revoke it). Create it once from a Colab cell:
   `!mkdir -p /content/drive/MyDrive/.secrets && echo '<TOKEN>' > /content/drive/MyDrive/.secrets/github_token && chmod 600 /content/drive/MyDrive/.secrets/github_token`
   then push: `!git -C /content/drive/MyDrive/final_project push https://$(cat /content/drive/MyDrive/.secrets/github_token)@github.com/kneat2448/LTH-ML-final.git main`
2. **Session start on Colab (T4):** mount Drive, `cd /content/drive/MyDrive/final_project`,
   `mkdir -p /content/cifar_raw && cp data/cifar-10-python.tar.gz /content/cifar_raw/`, `export LTH_RAW_DATA=/content/cifar_raw`.
3. ~~Smoke run~~: done on the T4 (section 6).
4. **Phase A**, one chain at a time, each as its own long cell (~2.3 h per chain on the T4):
   `python -m src.imp configs/resnet20_low.yaml` → `resnet20_high` → `resnet20_warm03` → `resnet20_high_warm`.
   Conv-4 chain: only if `pilot/data/*.gz` is uploaded (O8). Then `python -m analysis.plots` and read `results/gate_a.md`.
   If Gate A fails, stop and write up why.
5. **Phase B** only if Gate A passes: `--seeds 1` for low/high/warm03, then H4 (`--select-lam` before the anchor run).

Reminders:
- Do not `pip install -r requirements.txt` on Colab (Windows cu128 pins). Colab's preinstalled packages suffice.
- Never run two training cells at once *without MPS* (time-slicing gains only 12%). With MPS, up to 3 chains (section 6). After any code bug fix, archive affected results by hand
  (resume only compares configs, not code).
- Do not download CIFAR-10 from cs.toronto.edu on Colab (O7); use the Drive copy.
- The laptop GPU was throttled (O5). Fix the power settings before using it again.

## 9. Log

- **2026-09-28, session 1.**
  - Built the repo from the spec and created the conda env `lth` (torch 2.11 + cu128). Downloaded CIFAR-10 (MD5 verified).
  - Integration check on Fashion-MNIST: IMP, baselines, divergence stop, resume, summary, plots all work.
  - Bugs fixed:
    - GPU/CPU device mix when building masks from checkpoints (`map_location="cpu"`);
    - lambda_max shift failure → Lanczos (D1);
    - `plots.py` syntax error;
    - read-only numpy warning.
  - Added: NCHW Hessian (D3), AMP bf16/fp16 selection (D2), per-GPU budgets + `LTH_BUDGET_H`, `LTH_RAW_DATA`,
    Colab notebook, Drive zip.
  - The smoke run on the laptop showed throttling and the O1 cost problem. It was archived.
  - The user is moving runs to Colab G4. Open decisions: O1, O3, O4.
  - End of session: the user asked only for NOTES.md to be finalized. No further code changes.
    Session 2 starts at section 8, step 1.
  - Initialised git, first commit of the code, configs, notebook, notes and pilot (remote: kneat2448/LTH-ML-final).
  - Added README.md and pushed `main` to GitHub.
- **2026-09-28, session 2 (Colab T4).**
  - Cloned the repo into `MyDrive/final_project`. `pytest`: 20/20 pass on the T4 (66 s, torch 2.11 + cu128).
  - No main-study run yet (still Gate 0; O1 and O4 still open).
  - Re-analysed all pilot JSONs, including the `res_gpu_*` set, which the spec does not describe.
    New script `analysis/pilot_summary.py` → `pilot_results/{pilot_summary.csv, pilot_tables.md, fig_pilot_acc.png, fig_pilot_spike.png}`.
    Write-up: `pilot_results/PILOT_FINDINGS.md`.
  - Key changes vs spec section 10:
    - dense eta=0.1 collapse reproduced on GPU (P1 reversed);
    - the high-LR ticket deficit is seed-dependent (P3 weakened);
    - pilot lambda values look like the D1 power-iteration failure (P2 unverified);
    - an early loss spike > 8x predicts failure, and it happens before step 200 (relevant to O1).
- **2026-09-28, session 2 (continued).**
  - O1 decided by the user → D11 (10 Lanczos steps, 15 points incl. early steps 25/50/100/200). Code: new `early`
    key in `sharpness_steps`; configs, conftest and a new schedule test updated (21 tests pass).
  - CIFAR-10: the Toronto download was too slow, then blocked (O7). The user uploaded the laptop copy to `data/` (MD5 OK).
  - Smoke benchmark on the T4 (all 6 trainings done): 14 ms/step warm, 21.5 s per lambda_max, 2.46 GB peak
    → ~9 min per 15k training. Budget re-derived (section 6): Phase A+B ≈ 19 T4-h of the 30 h budget.
  - A GitHub push with a pasted token was blocked by the auto-mode permission check; commits are local only (step 1 of section 8).
  - Gave the user a time forecast (section 1): ≈ 2 days to Gate A, ≈ 1 week to full findings on the T4.
  - Set the repo-local git identity on Colab to `Nitai <nitaikoundinye@gmail.com>` (matches earlier commits).
  - Proposals from the pilot re-analysis, not yet acted on: report a per-condition collapse rate in the main
    summary; cite P1 as "collapses in 2 of 3 short runs"; drop P2 or re-measure it with Lanczos.
  - Next: Phase A (section 8, step 4).
- **2026-09-29, session 3 (Colab T4, ~2 h).**
  - The user saved a GitHub token in `/content/drive/MyDrive/.secrets/github_token` (not in NOTES.md: this file is pushed,
    and GitHub revokes tokens it finds in pushes). The first push got 403 (token without write access); after the user fixed
    the permission, all session-2 commits were pushed.
  - Started Phase A: `python -m src.imp configs/resnet20_low.yaml` (seed 0) in the background, with a watcher that commits and
    pushes each new `results/low/seed0/round*.json`.
  - **low r0 (dense, eta 0.01):** test acc **86.66%** (F1 0.867, AUC 0.990), 9.3 min (sharpness overhead 151%, as predicted).
    lambda_max 25.5 at init, peak 30.9 at step 25, then down to ~8 by step 6k and 8.6 at the end. S = eta*lambda/(2(1+beta))
    peaks at 0.081 (step 25): the low-LR run stays far from the stability edge (S = 1), as H1 assumes. R_0.2 ends at 0.640.
  - **low r1 (70.1%):** test acc **87.43%** (+0.77 pp over dense), max S(3k) **0.148** (about 1.8x dense at the same eta),
    R_0.2 0.584. (First read as "S rises with sparsity"; **retracted after r2**: the 0.148 is one point at step 500,
    and the rest of r1's trajectory is 0.04–0.08.)
  - Added `Methodology.md` (method as run, deviations, study steps and status, results table) and
    `analysis/methodology_table.py`, which regenerates that table after every finished training.
  - **low r2 (49.1%):** test acc **87.85%** (+1.19 pp over dense), max S(3k) 0.085, loss spike 1.0 (no early spike in r0–r2).
    At eta = 0.01, S stays in 0.03–0.09 at every sparsity so far, apart from single-point jumps. max S over one
    trajectory is **sensitive to single-point noise**, so fig. 3 should also show a robust statistic
    (e.g. the median of the early points). This is a proposal, not implemented.
  - **low r2 reinit:** 85.78% vs ticket 87.85% → **ticket advantage +2.07 pp at 49.1%**, the first H1 data point, in the expected direction.
  - The user asked to use the whole GPU. Benchmark (section 6): T4 compute-bound, 3 processes under MPS give +20% total throughput.
    Stopped `low` (its r3 training restarted from scratch, ~2 min lost), started the MPS daemon, relaunched `low`, `high` and
    `warm03` in parallel. The watcher now commits all conditions.
  - **Results by 04:44 (parallel, MPS):** low r3 (34.5%) ticket 87.87%. **high r0 (dense, eta 0.1) 89.83%**, no collapse
    (spike 1.19x), R_0.2 0.247. **warm03 r0 87.48%**, R_0.2 0.558. Details and interpretation are in `Methodology.md` section 10.
  - **Found: max S(≤3k) is dominated by step 0 in runs without warmup.** high dense: S = 0.67 at step 0 (eta · lambda_max at init),
    then ≤ 0.23, because lambda_max falls from 25.5 to 6.2 within 25 steps. H2 needs max S over steps ≥ 25, with S(0) reported
    separately. **Proposal, needs approval** (it changes the H2 metric in `analysis/plots.py`; the raw trajectories are already
    saved, so no re-run is needed).
  - warm03 shows progressive sharpening during warmup (lambda_max 25 → 53 at step 2k) with S ≤ 0.08, the pattern Kalra & Barkeshli describe.
- **2026-09-29, session 4 (Colab T4).**
  - The VM had restarted: low r4, high r1 and warm03 r1 were lost, and every finished result was already committed. Relaunched
    low/high/warm03 under MPS. New watcher (scratchpad) commits locally only; the GitHub token could not be read, so push by hand.
  - User: the budget is 30 h of Colab runtime, not GPU-h, and 27 h are left → runtime accounting in `src/utils.py` / `src/imp.py`
    (union of training intervals plus a calibrated offset), `LTH_BUDGET_H` removed (notebook too), test `tests/test_utils.py`.
  - User approved the H2 fix → `stability_summary` (S0, max_S_train over steps 25–3k), fig. 3 (S(0) as hollow markers), fig. 5,
    the Methodology table and the deviation list. Test in `tests/test_metrics.py`. 23 tests pass.
  - **First results after the relaunch (05:34–05:36 UTC, pushed by the watcher):** low r4 (24.2%) ticket 87.92% (still above dense);
    high r1 (70.1%) 89.39% (−0.44 pp vs dense), max S(25–3k) **0.536** vs 0.226 dense, but only at step 25 (lambda_max had not
    dropped yet: 0.536 → 0.10 by step 50), with no loss spike (1.16x); warm03 r1 (70.1%) 88.22% (+0.74 pp vs dense), max S 0.051.
    Parallel wall ≈ 23–25 min per training; runtime used 3.41 h of 30.
  - **06:01 UTC batch:** low r4 reinit 84.64% vs ticket 87.92% → **ticket advantage +3.28 pp at 24.2%** (2nd H1 point for low).
    high r2 (49.1%) 89.21% (−0.62 pp vs dense), max S(25–3k) back to 0.24 (the r1 0.536 did not repeat). warm03 r2 (49.1%) 88.57%
    (+1.09 pp vs dense), max S 0.045. No divergence, no spikes. Runtime 3.81 h.
  - **06:25 UTC batch (baselines):** low r4 shuffle 83.48% → low advantage at 24.2% stays **+3.28 pp** (best baseline = reinit).
    **high r2 reinit 88.85% vs ticket 89.21% → +0.36 pp**. It is small and positive; H1 expects no advantage at high LR, so this
    point counts against Gate A's `high` criterion. It is one seed, well within seed noise, so watch r4/r6/r8.
    The reinit had S(0) 0.817 and max S(25–3k) 0.334 (ticket 0.659 / 0.240).
    **warm03 r2 reinit 86.66% vs ticket 88.57% → +1.91 pp** (1st H1 point for warm03). No divergence.
  - **06:50 UTC batch (tickets only; no baselines at these rounds):** low r5 (17.0%) 87.84% (+1.18 pp vs dense), max S(25–3k) 0.109.
    **high r3 (34.5%) 88.36%: −1.47 pp vs dense**, the first high ticket clearly below dense (r1 −0.44, r2 −0.62, so the gap is
    widening with sparsity); max S 0.244, S(0) 0.551. warm03 r3 (34.5%) 88.52% (+1.04 pp vs dense), max S 0.043.
    No divergence, no spikes. Runtime 4.62 h. Next: r4 baselines for high/warm03, which decide the 24.2% H1 points.
  - **07:13 UTC batch (tickets):** low r6 (12.0%) 87.59% (+0.93 pp vs dense), max S(25–3k) 0.077. **high r4 (24.2%) 87.28%:
    −2.55 pp vs dense** (gap −0.44 → −0.62 → −1.47 → −2.55 pp with sparsity), max S 0.306, S(0) 0.438. **warm03 r4 (24.2%) 89.11%:
    +1.63 pp vs dense**, its best so far, max S 0.048. At 24.2% the warm03 ticket beats the high ticket by 1.8 pp. No divergence,
    no spikes. Runtime 5.02 h. Next: reinit baselines at low r6 / high r4 / warm03 r4.
  - **07:30 UTC:** the user asked to save progress every 15 min so a runtime error loses nothing. The watcher moved from the scratchpad into
    the repo (`colab/watcher.sh`) with a 15-min autosave and push retries. Added `colab/resume_phaseA.sh`. high r4 reinit landed at 07:25
    (reported with the next batch).
  - **07:37 UTC batch (reinit baselines):** low r6 reinit 83.05% vs ticket 87.59% → **+4.54 pp at 12.0%** (low: +2.07 → +3.28 → +4.54,
    growing with sparsity). **warm03 r4 reinit 85.24% vs ticket 89.11% → +3.87 pp at 24.2%**. **high r4 reinit 87.15% vs ticket 87.28%
    → +0.13 pp**: tiny, but the second small positive high point (r2 +0.36), so the strict Gate A 'high: no wins' criterion keeps
    failing even though the high advantage is an order of magnitude smaller than low/warm03. The r4 shuffles (high, warm03) are still to come.
    Runtime 5.42 h.
  - **07:53 UTC check:** high r4 shuffle 87.11% (07:47), so the high advantage at 24.2% stays +0.13 pp (best baseline = reinit 87.15%). The first 15-min autosave ran at 07:47 and pushed. Running: low r7 ticket (8.4%),
    high r5 ticket (17.0%), warm03 r4 shuffle. Runtime 5.59 h.
  - **08:08 UTC check:** low r7 (8.4%) ticket 87.17% (+0.51 pp vs dense, slowly falling toward dense). **high r5 (17.0%) ticket 86.13%:
    −3.70 pp vs dense** (the gap keeps widening), max S(25–3k) 0.248. warm03 r4 shuffle 84.53%, so the warm03 advantage at 24.2% stays +3.87 pp
    (best baseline = reinit). Running: low r8 ticket (5.97%, the last low round), warm03 r5 ticket, the next high training. Runtime 5.96 h.
  - **08:25 UTC:** low r8 (6.0%) ticket 86.13%, the first low ticket below dense (−0.53 pp). warm03 r5 (17.0%) 88.65% (+1.17 pp), max S 0.030.
    Updated `Methodology.md` (step statuses, accuracy-vs-dense table, advantages, Gate A reading, H2 observation). Added
    `colab/start_after.sh` and started it: high_warm launches when low exits. The `resume_phaseA.sh` default now covers all four chains.
    Gate A (strict script): low PASS 3/3, warm03 2/2 (needs a 3rd), high FAIL (2 tiny wins). Runtime 6.21 h.
  - **08:40 UTC:** user decision D12 (Gate A `high` = no *winning* tickets). `analysis/plots.py::gate_a` changed, and `gate_a.md` regenerated:
    low PASS 3/3, high PASS 0/2 winning (2 positive-but-not-winning points listed), warm03 2/2, still needs a 3rd win.
  - **08:42 UTC batch:** **low r8 reinit 80.13% vs ticket 86.13% → +6.00 pp at 6.0%** (low: +2.07 → +3.28 → +4.54 → +6.00, growing
    monotonically with sparsity). The low r8 ticket is 0.53 pp below dense, just outside the 0.5 pp tolerance, so it is not a *winning*
    ticket in the section-5 sense even though it beats the baselines by 6 pp. **high r6 (12.0%) ticket 85.52%: −4.31 pp vs dense**, max S(25–3k)
    0.183. Running: low r8 shuffle (last low training), high r6 reinit, warm03 r6 ticket. Runtime 6.50 h.
    (Monitoring note: count chains with `pgrep -f "^python3 -m src.imp"`; a looser `grep src.imp` also matches helper shells.)
  - **08:58 UTC check:** high r6 reinit 85.13% vs ticket 85.52% → +0.39 pp at 12.0%, but the ticket is 4.31 pp below dense, so it is **not a
    winning ticket** (D12). high is now 0/3 winning, with 3 tiny positive advantages (+0.36, +0.13, +0.39). **warm03 r6 (12.0%) ticket 88.65%
    (+1.17 pp vs dense)**, max S 0.036. The warm03 r6 reinit (running) will likely give warm03's 3rd Gate A win. low r8 shuffle is still running
    (low's last training); high_warm not yet started. Runtime 6.70 h.
  - **09:13 UTC, session end (user shut the runtime down):** low r8 shuffle 78.30%, so **low is complete** (advantage +6.00 pp at 6.0%;
    4/4 Gate A wins). **warm03 r6 reinit 83.76% vs ticket 88.65% → +4.89 pp at 12.0%**, its 3rd win → **Gate A PASSED on seed 0**
    (low 4/4, warm03 3/3, high 0/3 winning). high_warm started automatically at 09:04 (`start_after.sh`); its r0 was lost at shutdown,
    as were high r7 and warm03 r7. The watcher was stopped before the final commit. Everything, including analysis outputs, is committed and pushed.
    Logged runtime 7.0 h (plus unlogged idle time).
- **2026-09-30, session 5 (Colab T4).**
  - **04:38 UTC:** The high r7 ticket had finished at 09:14 on 09-29, after the handoff was written (it was listed as lost): **high r7 (8.4%) 84.25%,
    −5.58 pp vs dense**, max S(25–3k) 0.177, S(0) 0.151. Committed. `git config core.fileMode false` set (Drive drops exec bits,
    so the `colab/*.sh` files showed as modified).
  - Resumed with `bash colab/resume_phaseA.sh high warm03 high_warm`. high_warm crashed at once ("Dataset not found or corrupted"):
    all three chains extracted the CIFAR archive at the same time. Relaunched it; all 3 chains are running under MPS (high r8 ticket, warm03 r7
    ticket, high_warm r0). **Fix:** `resume_phaseA.sh` now extracts the archive once before starting any chain.
  - Left: high 3 (r8 ticket + reinit + shuffle), warm03 4, high_warm 15 = 22 trainings. Budget not re-calibrated yet (needs the remaining
    hours from the Colab usage page).
  - **05:02 UTC:** user: about 20 h of compute left → budget re-calibrated (10 h used at 05:02 UTC; see section 1). `test_utils` passes.
  - **05:04 UTC batch (first results of session 5):**
    - **high r8 (6.0%) ticket 82.72%: −7.11 pp vs dense** (gap −4.31 → −5.58 → −7.11 pp from 12% to 6%). max S(25–3k) 0.286 (at step 50),
      S(0) 0.134. At this sparsity lambda_max at init is only 5.1 (dense 25.5), rises to 10.9 at step 50, then falls to ~2. No spike.
    - **warm03 r7 (8.4%) ticket 87.94%: +0.46 pp vs dense** (still above dense, but the margin is shrinking: +1.17 at 12%). max S 0.045, R_0.2 0.229.
    - **high_warm r0 (dense, eta 0.1 + 10k warmup) 89.46%**, R_0.2 0.331 (between high 0.247 and warm03 0.558), max S(25–3k) 0.092, no spike.
      Sharpening during warmup stops earlier than in warm03: lambda_max 25.5 → 36.6 at step 500, then falls as the LR passes ~0.01
      (11.3 at 2k, 6.1 at 3k). warm03 sharpened to 52.7 at 2k. In both runs S peaks at ≈ 0.08–0.09 when lambda_max turns down.
    - `Methodology.md` updated (step statuses, table, accuracy-vs-dense table, high_warm and sparse-high sharpness notes).
  - **05:54 UTC check (6 more results):**
    - **high is complete (15/15).** r8 (6.0%): ticket 82.72%, reinit 82.75%, shuffle 82.15% → **advantage −0.03 pp**; high ends with 0/4 winning
      tickets. At 6% the high ticket is no better than a random reinit.
    - **warm03 r8 (6.0%) ticket 87.86% (+0.38 pp vs dense), reinit 80.43% → +7.43 pp**, its 4th win. warm03 tickets beat dense at every sparsity.
      warm03 r8 shuffle is running (warm03's last training).
    - **Gate A final for seed 0: PASSED** (low 4/4, warm03 4/4, high 0/4 winning). `results/gate_a.md` regenerated.
    - **high_warm r1 (70.1%) 90.10% (+0.64 pp vs its dense 89.46%)**, r2 (49.1%) 89.75% (+0.29 pp), max S 0.069 / 0.072. The best accuracy
      in the study so far. Unlike high, the warmup tickets at eta = 0.1 are above dense (H3 direction). r2 reinit running.
    - Runtime used 10.87 h (calibrated). **Forecast:** warm03 ends ≈ 06:20; high_warm then runs alone, 11 trainings × ~9 min → **Phase A done ≈ 08:00 UTC,
      ≈ 12.9 h used**. Phase B (seed 1 low/high/warm03 ≈ 6.3 h under MPS + H4 ≈ 3 h) → ≈ 22 h used, ≈ 8 h reserve.
  - **05:57 UTC: Phase B seed 1 queued (user approved).** New `colab/resume_phaseB.sh` (seed-1 chains under MPS, same setup as Phase A; also
    the restart command after a VM reset: `bash colab/resume_phaseB.sh`). Two waiting launchers:
    `WAIT_FOR=warm03 … resume_phaseB.sh low high` (starts when warm03 seed 0 exits, ≈ 06:20) and `WAIT_FOR=high_warm … resume_phaseB.sh warm03`
    (starts when high_warm exits). Output in `/content/phaseB_launch.out` and `/content/phaseB_<c>_s1.out`; results go to `results/<c>/seed1/`.
    The pgrep patterns in `resume_phaseA.sh` and `start_after.sh` are now anchored (`…yaml$`) so they do not match the seed-1 processes.
    **Correction of the 05:54 estimate:** the saving is ≈ 15 min, not ≈ 1 h. Measured: 9.3 min per training alone vs ≈ 24 min per training
    with 3 in parallel → 6.5 vs 7.5 trainings/h (+16%). **Cost: high_warm (Phase A) now finishes ≈ 10:45 UTC instead of ≈ 08:00**, because it
    shares the GPU. Remaining work: 12 high_warm + 45 seed-1 trainings ≈ 7.6 h → Phase A + seed 1 done ≈ 13:30–14:00 UTC (≈ 18.5 h used), then H4 ≈ 3 h.
  - **06:12 UTC:** **warm03 seed 0 complete (15/15)**; last training r8 shuffle: `acc=0.8026 f1=0.8013 R02=0.299 maxS3k=0.065 diverged=False wall=17.4min sharp_overhead=97.2% peakVRAM=2460MB`. low seed 1 started by hand at 06:11 (3rd GPU slot),
    high seed 1 by the launcher at 06:12. Now running under MPS: high_warm s0 (r3), low s1 (r0), high s1 (r0). warm03 s1 waits for high_warm.
    Phase A 49/60. Runtime 11.1 h. Each IMP chain is sequential (round r needs round r−1's mask), but the chains are independent; the T4 is
    compute-bound, so 3 processes under MPS is the useful maximum (4 is slower, section 6). **H4 is unblocked** (its masks come from warm03 seed 0
    r3/r6/r8, now done); use it to fill GPU slots as chains finish.
  - **06:44 UTC check:** 3 chains running, GPU 100%. Runtime 11.6 h. Phase A 50/60, seed 1 2/45.
    - **high_warm r2 reinit 88.28% vs ticket 89.75% → +1.47 pp at 49.1%**: the first winning ticket at eta = 0.1 (ticket above its dense).
      **high_warm r3 (34.5%) 90.30%** (+0.84 pp vs dense), best in the study; max S 0.051.
    - **low s1 r0 (dense) 86.48%** (seed 0 86.66%), R_0.2 0.641. **high s1 r0 (dense) 89.55%** (seed 0 89.83%), no spike, but **S(0) = 1.07**:
      seed 1's theta_0 has lambda_max ≈ 40.6 (seed 0: 25.5), so it starts above S = 1 and still trains normally (max S(25–3k) 0.495).
    - Forecast unchanged: high_warm ≈ 10:30, low/high s1 ≈ 12:15, then warm03 s1 + H4 → all done ≈ 17:00 UTC, ≈ 22 h used.
  - **07:25 UTC check:** 3 chains running, GPU 100%. Runtime 12.4 h. Phase A 52/60, seed 1 5/45.
    - **high_warm r4 (24.2%) ticket 90.35% (+0.89 pp vs dense), reinit 86.67% → +3.68 pp**: 2nd winning ticket at eta = 0.1. r4 shuffle next.
    - **Seed 1 replicates (ticket − dense):** low r1 87.11% (+0.63 pp; seed 0 +0.77); high r1 89.23% (−0.32; seed 0 −0.44), high r2 88.53% (−1.02; seed 0 −0.62).
      high s1 S(0) 0.53 / 0.87 on the sparse masks, no divergence, no spikes.
    - `colab/watcher.sh`: commit messages now say `Phase B (seed1)` for seed-1 results (the ones before 07:30 say "Phase A"). Watcher restarted
      (a `pkill -f colab/watcher.sh` also killed the calling shell; use `pkill -f "^bash colab/watcher.sh"`).
  - **09:28 UTC: VM reset (≈ 09:16) found and recovered.** Nothing was running; the last watcher commit was 08:14 + the high s1 r3 ticket.
    Lost in progress: high_warm r6 ticket, low s1 r4 ticket, high s1 r4 ticket. The uncommitted low s1 r3 ticket (88.12%, +1.64 pp vs its dense 86.48%;
    seed 0 r3 gap for comparison in `summary.csv`) was committed by hand. Relaunched with `bash colab/resume_phaseA.sh high_warm`,
    `bash colab/resume_phaseB.sh low high`, and a waiting `WAIT_FOR=high_warm … resume_phaseB.sh warm03` (the plain `resume_phaseB.sh` would start
    warm03 as a 4th chain). GPU 100%, watcher running. Status: Phase A 54/60 (high_warm 6 left), seed 1 10/45 (low 5, high 5; warm03 0).
    Logged runtime 13.41 h, **excluding the idle time around the reset, so re-calibrate from the Colab usage page.**
    Forecast: 41 trainings left at ≈ 6.5/h → seed 1 done ≈ 16:00 UTC (≈ 20 h logged), then H4 ≈ 3 h.
  - **09:30 UTC: budget re-calibrated (user): 15 h of Colab runtime left** → `src/utils.py`: `RUNTIME_USED_BEFORE_H = 15.0`,
    `RUNTIME_SINCE = 2026-09-30T09:30:30`. The reset cost ≈ 1.6 h beyond the logged 13.41 h. The chains running now use the old constants in
    their COST ESTIMATE lines. Plan: seed 1 ≈ 6.5 h wall (done ≈ 16:00) + H4 ≈ 3 h → ≈ 9.5 h, leaving ≈ 5.5 h reserve. Start H4 in any GPU slot
    that frees up (low/high s1 end ≈ 13:30, while warm03 s1 still runs) so the GPU never runs fewer than 3 chains.
- **2026-10-01, session 6 (Colab T4).**
  - **03:32 UTC:** new VM, nothing running. The 09-30 VM died after the 09:59 autosave (git was clean; every finished result was committed).
    Lost in progress: high_warm r6 reinit, low s1 r4 reinit, high s1 r4 reinit. Relaunched: `bash colab/resume_phaseA.sh high_warm`,
    `bash colab/resume_phaseB.sh low high`, `WAIT_FOR=high_warm nohup bash colab/resume_phaseB.sh warm03 >> /content/phaseB_launch.out 2>&1 &`.
    GPU 3 chains, 7.6 GB.
  - **H4 queued** with the new `colab/queue_h4.sh` (fills a GPU slot as soon as a seed-1 chain exits):
    `nohup bash colab/queue_h4.sh low "configs/h4_high_anchor.yaml --select-lam" configs/h4_high_anchor.yaml configs/h4_high.yaml >> /content/h4_queue_A.out 2>&1 &`
    and `nohup bash colab/queue_h4.sh high configs/h4_high_sam.yaml >> /content/h4_queue_B.out 2>&1 &`. Masks/theta_0 from `checkpoints/warm03/seed0` (present).
    `colab/watcher.sh` now labels H4 commits `H4` (was "Phase A" for any seed0 dir); watcher restarted.
  - Remaining: 5 + 33 + 18 H4 (SAM ≈ 2x) ≈ 62 training-equivalents at ≈ 6.5/h → done ≈ 12:30 UTC, ≈ 9 h runtime.
  - **03:37 UTC: budget re-calibrated (user): 14 h left** → `RUNTIME_USED_BEFORE_H = 16.0`, `RUNTIME_SINCE = 2026-10-01T03:37:00`. Remaining plan ≈ 9 h, reserve ≈ 5 h.
  - **05:13 UTC: papers fetched from arXiv** into `research paper/` on Drive (gitignored, as on the laptop): Frankle & Carbin 2019,
    Liu 2021, Kalra & Barkeshli 2024, Cohen 2021 (EoS), Foret 2021 (SAM), Frankle 2020 (LMC), Paul 2023, Sakamoto & Sato 2022,
    Lange & Sprekeler 2023, McDermott & Parhi 2025. IDs in `results/PHASE_A.md` (References). Draft Phase A write-up: `results/PHASE_A.md`.
    Checked facts: Frankle's ResNet warmup (eta 0.03, k = 20k of 30k) also ends at the first decay; the eta 0.1 + warmup failure is one
    sentence with no figure; they use Glorot init (ours: Kaiming). **Kalra & Barkeshli §4.2: with momentum, the threshold is 2/lambda at init
    and only later (2+2·beta)/lambda, and the late SGD-M threshold is "much smaller" at finite batch.** So S(0) is understated 1.9x
    (dense high 1.27 vs 2/lambda) and S = 1 is not the minibatch edge. Both caveats are in PHASE_A.md §4.

  - **05:23 UTC hourly check:** 3 chains + watcher + H4 queues alive, GPU 100%. Phase A 59/60 (high_warm r8 reinit/shuffle left), seed 1 20/45 (low 10, high 10).
    Seed 1 r6 (12.0%): low ticket 87.39% (+0.91 pp vs dense; seed 0 +0.93), high 85.63% (−3.92; seed 0 −4.31). Runtime 17.6 h of 30. Forecast unchanged (≈ 12:30 UTC).
  - **05:55 UTC: Phase A complete (60/60).** high_warm r8 (6.0%): ticket 88.35%, reinit 82.42%, shuffle 82.34% → +5.93 pp, but ticket −1.11 pp vs dense → not winning; high_warm wins 3/4. warm03 s1 started 05:34 (`resume_phaseB.sh` waiter). `PHASE_A.md` pending cells filled.
    Seed 1 r6 advantages: low +4.70 pp (ticket 87.39 vs reinit 82.69; seed 0 +4.54), high +0.67 (85.63 vs 84.96; seed 0 +0.39). Runtime 18.0 h.
  - **05:55 UTC: H4 write-up drafted before any H4 result** (`results/H4.md`): design, predictions, a *proposed* rescue rule
    (ticket − shuffle ≥ +2.0 pp at ≥ 2 of 3 masks, plus a manipulation check), **awaiting user approval before results land (≈ 07:25 start)**.
    Methodology §7 D6 wording fixed to match the code (smallest lam reaching the target, else max R_0.2). Anchor target R_0.2 = 0.661 (low r3, step 3k).
  - **06:00 UTC: D13 (user): H4 rescue rule approved** (≥ +2.0 pp ticket − shuffle at ≥ 2/3 masks + manipulation check). In H4.md §5 and Methodology §9.
  - **06:24 UTC hourly check:** all processes alive, GPU 100%. Seed 1 25/45 (low 12, high 12, warm03 1). Runtime 18.4 h.
    r7 (8.4%) tickets vs dense: low 87.16% (+0.68; seed 0 +0.51), high 84.38% (−5.17; seed 0 −5.58). **warm03 s1 dense 87.81%** (seed 0 87.48%), max S 0.053.
    Forecast unchanged: low/high s1 ≈ 07:25, H4 then starts; all done ≈ 12:30 UTC.
  - **06:45 UTC: H2/H3 write-up drafted** (`results/H2_H3.md`), s0+s1 data. H2: not supported as stated (no run > 0.54; within-condition ρ ≈ 0; S(0) anti-predictive in high, ρ +0.86, confounded with sparsity);
    condition-level separation holds (high 0.16–0.54 vs ≤ 0.15 elsewhere, max at step 25 in 15/18 high tickets). H3: partly supported. Fixed a wrong PHASE_A.md claim: early losses are logged per step (first 200), and dense high s0 r0/r1 show a step-4 catapult (1.44x/1.51x the running min).
  - **07:17 UTC: low and high seed 1 complete (15/15 each); H4 started.** Seed 1 r8 (6.0%): low ticket 86.22% (−0.26 vs dense, adv +6.55 → winning), high 82.36% (−7.19, adv +0.43).
    Two-seed Gate A still PASSED. H4: queue A running `h4_high_anchor --select-lam` (07:12), queue B `h4_high_sam` (07:17); warm03 s1 running. H2_H3.md and PHASE_A.md updated with the final s1 low/high numbers.
  - **07:24 UTC hourly check:** warm03 s1 4/15, H4 lam selection (lam 1e-3 run) and SAM r3 ticket running, GPU 100%, runtime 19.8 h. Forecast ≈ 12:30 UTC.
- **2026-10-02, session 7 (Colab T4).**
  - **11:41 UTC:** new VM, nothing running. The 10-01 VM died after the 08:11 autosave (git clean). Lost in progress: warm03 s1 r4 reinit,
    h4_high_anchor r3 shuffle, h4_high_sam r3 shuffle. Done: Phase A 60/60, seed 1 37/45 (warm03 6/15), H4 2/18 + lam selection (lam = 0.01, R_0.2 0.757 ≥ 0.661).
    Relaunched: `bash colab/resume_phaseB.sh warm03`; `queue_h4.sh low configs/h4_high_anchor.yaml configs/h4_high.yaml` (no `--select-lam`: already done);
    `queue_h4.sh high configs/h4_high_sam.yaml`. GPU 3 procs, 9.2 GB. Logged runtime 20.4 h, **excluding the idle time before the VM died; re-calibrate from the usage page.**
  - Left: warm03 s1 9, anchor 5, SAM 5 (≈ 2x), h4_high 6 → ≈ 30 training-equivalents. Slots: A anchor→high ≈ 3.9 h, B SAM ≈ 3.5 h, C warm03 ≈ 3.3 h
    → **all runs done ≈ 15:30–16:00 UTC**, then H4.md / Phase B write-up.
  - **11:44 UTC: budget re-calibrated (user): 9.6 h left** → `RUNTIME_USED_BEFORE_H = 20.4`, `RUNTIME_SINCE = 2026-10-02T11:44:31`. Plan ≈ 5 h → reserve ≈ 4.5 h. Hourly checks.
  - **12:04 UTC check:** 3 procs + 2 queues + watcher alive, GPU 100%. No new results yet (first after relaunch due ≈ 12:05–12:25: warm03 s1 r4 reinit, anchor/SAM r3 shuffle). warm03 s1 6/15, H4 anchor 1/6, SAM 1/6, high 0/6. Runtime ≈ 20.7 h of 30. Forecast unchanged: all runs ≈ 15:30–16:00 UTC.
  - **12:15 UTC check:** all alive, GPU 100%. warm03 s1 7/15, anchor 2/6, SAM 1/6, high 0/6. **warm03 s1 r4: ticket 89.08% vs reinit 85.52% → +3.56 pp** (seed 0 +3.58), two-seed mean +3.72 pp. **H4 anchor r3 (34.5%): ticket 85.54% vs shuffle 78.01% → +7.53 pp** (rescue threshold +2.0; 1 of 3 masks so far). Manipulation check holds: R_0.2 0.851 (ticket) and 0.863 (shuffle), both ≥ 0.661. The anchor lowers absolute accuracy, though: ticket 85.5% vs 88.5% for the warm03 ticket and 88.4% for the high chain at this sparsity. Runtime ≈ 20.9 h. Forecast unchanged (≈ 15:30–16:00).
