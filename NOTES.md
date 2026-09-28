# NOTES: session log and handoff

Running log for continuing this project across sessions and machines. The spec is
`src/CLAUDE_1.md` (read it first). This file records what exists, what was decided, what was
measured and what is still open. **Update it at the end of every work session.**

---

## 1. Current status (2026-09-28, end of session 1)

- **Phase:** section 7 step 1 (benchmark). All code is written and tested (20 tests pass),
  and the full pipeline has been checked end to end. No real experiment has run yet.
- **Where to run:** moving to **Google Colab (G4)**. The folder is prepared for Drive (section 3).
- **Before any Phase A run (Gate 0):**
  1. **Decide the sharpness cost (O1).** Nothing else blocks Phase A.
  2. Run the smoke benchmark on the Colab GPU (notebook cell 4–5), record the numbers in section 6,
     and re-derive the section 9 budget table of the spec.
  3. Set the GPU-hour budget for the G4 (O4) in the notebook's `LTH_BUDGET_H`.
- **Then:** Phase A in the order of section 7: Conv-4 chain, low, high, warm03, high_warm → Gate A.

## 2. Quick start

### Local (Windows laptop)
```
conda activate lth           # C:\Users\Nitai\anaconda3\envs\lth\python.exe
python -m pytest             # 20 tests, ~40 s
python -m src.imp configs/smoke.yaml
```

### Colab (preferred from now on)
1. Upload the whole `final_project` folder to `MyDrive/final_project`
   (or unzip `final_project_drive.zip` there; see section 3).
2. Open `colab/run_on_colab.ipynb` in Colab and pick a GPU runtime (G4).
3. Edit `LTH_BUDGET_H` in cell 2, then run cells 1–5 (mount, env, tests, smoke, benchmark summary).
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
| `configs/` | smoke, resnet20_{low,high,warm03,high_warm}, conv4_fmnist_high, h4_high, h4_high_sam, h4_high_anchor |
| `colab/run_on_colab.ipynb` | Colab runner (mount, env, tests, smoke, benchmark summary, Phase A/B cells) |
| `tests/` | 20 tests (section 12 list plus schedules, NaN-mask guard, negative-dominant spectrum) |
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

### Colab G4: not measured yet (fill in from notebook cell 5)
- GPU name / torch / CUDA: …
- ms per step, s per lambda, min per 15k run, peak VRAM: …

## 7. Open issues

- **O1. Sharpness cost vs the 15%-overhead rule. USER DECISION NEEDED before Phase A.**
  One measurement = 20 HVPs on 2,048 images ≈ 1,300 training steps of compute. The spec's 21 points
  per run ≈ 190% overhead on **any** GPU; a faster GPU does not change the ratio. Options (they combine):
  - (a) **fewer Lanczos steps**: 10 steps were within 0.5% of the converged value on dense theta_0 (≈ 2x cheaper);
  - (b) warm-start from the previous top eigenvector (≈ 2–4x);
  - (c) Hessian batch 512 instead of 2,048 (≈ 4x, noisier);
  - (d) fewer points: every 500 steps up to 3k, then every 3k (21 → ~11 points, ≈ 2x).
  - Or accept the overhead on the G4 and budget for it.

  All of these are config/code changes that the spec says need approval. After the decision, update
  `sharpness:` in every config (the smoke config too) and re-run the smoke benchmark.
- **O3. Resolved (2026-09-28).** Git repo initialised, remote `origin` = https://github.com/kneat2448/LTH-ML-final.git
  (branch `main`). Not committed on purpose: `data/`, `checkpoints/`, `*.pt`, `pilot/data/` (Fashion-MNIST .gz;
  get it from the Drive zip or zalandoresearch/fashion-mnist), `research paper/` (third-party PDFs), `*.zip`.
  On Colab, `git clone` also works instead of the Drive zip, but then copy `pilot/data/*.gz` in by hand.
  Uploading `.git` with the Drive copy lets runs record the commit hash.
- **O4. Budget for the G4.** Sections 4 and 9 only cover the RTX 4060 (18 h) and a T4 (30 h). Agree on a
  G4 budget (compute units), set `LTH_BUDGET_H`, and note it in the spec's section 9.
- **O5.** Laptop throttling (section 6). Irrelevant on Colab, but fix it before using the laptop again.
- **O6.** `results/compute_log.csv` contains 4 laptop rows from the archived smoke run (0.47 GPU-h, real time spent).
  They count toward the 4060 budget only (budgets are tracked per exact GPU name).

## 8. Handoff: what to do next (in order)

1. **Upload:** unzip `ML_lab/final_project_drive.zip` into `MyDrive/final_project`
   (or upload the folder without `checkpoints/`, `data/*.tar.gz`, papers, pptx).
2. **Decide O1 (sharpness cost).** Recommendation from session 1: **10 Lanczos steps + fewer points**
   (every 500 steps up to 3k, then every 3k, plus the final step). That's ≈ 4x cheaper and keeps the
   2,048-image batch. Apply it to the `sharpness:` line of every config, smoke included:
   `sharpness: {batch: 2048, iters: 10, micro_batch: 512, every: 500, dense_until: 3000, then_every: 3000}`.
   Then check that `tests/test_train.py::test_sharpness_schedule` still reflects the chosen schedule
   (it tests the spec's schedule via its own config, so it needs no change unless the defaults are meant to be tested).
3. **Decide O4:** agree on a G4 GPU-hour budget, put it in notebook cell 2 (`LTH_BUDGET_H`), and add it to section 9 of the spec.
4. ~~O3 git init~~: done (see section 7).
5. **On Colab:** open `colab/run_on_colab.ipynb` with a G4 runtime and run cells 1–5 (mount, env, tests, smoke, benchmark summary).
   Record GPU name, torch/CUDA versions, ms/step, s/lambda, min per 15k run and peak VRAM in section 6.
   If one 15k training takes > 12 min, cut `iters` to 10k (milestones 6.7k/8.3k, warmup 6.7k) before cutting conditions or seeds.
   Re-derive the section 9 budget table of the spec.
6. **Phase A** (notebook cells, one at a time): conv4_fmnist_high → low → high → warm03 → high_warm, then
   `python -m analysis.plots` and read `results/gate_a.md`. If Gate A fails, stop and write up why.
7. **Phase B** only if Gate A passes: seed 1 of low/high/warm03, then H4 (`--select-lam` before the anchor run).

Reminders:
- Do not `pip install -r requirements.txt` on Colab (Windows cu128 pins). Colab's preinstalled packages suffice.
- Never run two training cells at once. After any code bug fix, archive affected results by hand
  (resume only compares configs, not code).
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
