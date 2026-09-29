# NOTES: session log and handoff

Running log for continuing this project across sessions and machines. The spec is
`src/CLAUDE_1.md` (read it first). This file records what exists, what was decided, what was
measured and what is still open. **Update it at the end of every work session.**

---

## 1. Current status (2026-09-28, end of session 2, Colab T4)

- **Phase:** section 7 step 1 (benchmark) **done** on a **Colab T4** with the new O1 schedule (section 6):
  ~9 min per 15k training, ~19 T4-h for Phase A+B. No Phase A run yet.
- **O1 decided** (D11): 10 Lanczos steps, points at 0, 25, 50, 100, 200, every 500 to 3k, then every 3k, final.
- **Runtime:** T4, so fp16 + GradScaler (D2). Budget is the spec's 30 T4-hours (built into `src/utils.py`,
  no `LTH_BUDGET_H` needed on a T4). O4 (G4 budget) only matters if you switch back to a G4.
- **CIFAR-10:** `data/cifar-10-python.tar.gz` is on Drive (uploaded from the laptop, MD5 c58f3010… verified).
  **Do not download it from cs.toronto.edu on Colab** (see O7). Copy it to local disk at session start:
  `mkdir -p /content/cifar_raw && cp data/cifar-10-python.tar.gz /content/cifar_raw/` and set `LTH_RAW_DATA=/content/cifar_raw`.
- **Git:** every session-2 commit after `2cfb0de` is **local only** (list them with `git log origin/main..main`):
  pilot re-analysis, O1 schedule, smoke results and NOTES updates. The push needs a GitHub token; run it yourself (section 8, step 1). Commits are authored as
  `Nitai <nitaikoundinye@gmail.com>` (set in the repo's local git config on Colab).
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
4. On a T4 no `LTH_BUDGET_H` is needed (30 h built in). Run cells 1–3 (mount, env, tests); cells 4–5 (smoke,
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
  - `LTH_BUDGET_H`: GPU-hour budget for GPUs other than the RTX 4060 (18 h) or T4 (30 h).
    Without it there is a warning and no budget alarms.

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
| `pilot_results/PILOT_FINDINGS.md` | session 2: pilot findings F1–F7 vs spec section 10 (2 seeds of the 4-epoch design: CPU + GPU re-run) |
| `configs/` | smoke, resnet20_{low,high,warm03,high_warm}, conv4_fmnist_high, h4_high, h4_high_sam, h4_high_anchor |
| `colab/run_on_colab.ipynb` | Colab runner (mount, env, tests, smoke, benchmark summary, Phase A/B cells) |
| `tests/` | 21 tests (section 12 list plus schedules incl. the D11 schedule, NaN-mask guard, negative-dominant spectrum) |
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
- **O3. Resolved (2026-09-28); push pending since session 2.** Session-2 commits are local only (section 1).
  A token pasted in chat in session 2 was not used (the push was blocked by the auto-mode permission check); **revoke it**.
  Original entry: Git repo initialised, remote `origin` = https://github.com/kneat2448/LTH-ML-final.git
  (branch `main`). Not committed on purpose: `data/`, `checkpoints/`, `*.pt`, `pilot/data/` (Fashion-MNIST .gz;
  get it from the Drive zip or zalandoresearch/fashion-mnist), `research paper/` (third-party PDFs), `*.zip`.
  On Colab, `git clone` also works instead of the Drive zip, but then copy `pilot/data/*.gz` in by hand.
  Uploading `.git` with the Drive copy lets runs record the commit hash.
- **O4. Budget for the G4. Moot while on a T4** (the built-in 30 T4-h budget applies). Only needed if runs move to a G4. Sections 4 and 9 only cover the RTX 4060 (18 h) and a T4 (30 h). Agree on a
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
- Never run two training cells at once. After any code bug fix, archive affected results by hand
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
