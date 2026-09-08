# Anxolotl

Machine learning pipeline for **stress/anxiety detection from wearable
physiological signals**, with *within-dataset* evaluation (LOSO —
Leave-One-Subject-Out) and *cross-dataset* evaluation (train on one dataset,
test on another).

## Goal

Train and evaluate binary classification models (rest vs. stress) from
signals collected by wearables (heart rate, IBI, accelerometer, and — WESAD
only — EDA/temperature/chest ECG), across three datasets:

- **WESAD** (Schmidt et al., 2018) — public dataset, lab protocol with a
  wrist sensor (Empatica E4) + a chest sensor (RespiBAN).
- **WDD** ("Wearable Device dataset during stress and exercise") — public
  dataset, wrist sensor only.
- **Anxolotl** — own dataset, collected with an Apple Watch + a
  stress-induction protocol (PASAT), with reference-grade validation via
  Biopac ECG on a subsample of sessions.

The central goal is to characterise how well a model trained on a lab
dataset (WESAD/WDD) generalises to the naturalistic own dataset (Anxolotl)
— **zero-shot cross-dataset evaluation** — and what performance "ceiling" is
reachable within the Anxolotl dataset itself (LOSO).

## End-to-end pipeline

```
raw data (per dataset)
      │
      ▼
┌─────────────────────┐
│ 1. Preprocessing      │  signal filters, phase/marker detection,
│                       │  1 Hz resampling, outlier removal
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│ 2. Feature extraction │  sliding windows (60s, 30s or 1s stride),
│    tabular features   │  8 or 11 stats/channel depending on schema
│    (or sequences      │  (see DATA.md) — mean, std, min, max, p25, p75,
│    for DL)            │  skew, kurtosis (+ slope, lf_hf, delta_mean, new
│                       │  schema only) → processed_data/{dataset}/*.pkl
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│ 3. BL-norm            │  per-subject z-score normalisation, anchored on
│  (baseline-relative   │  the "leading run" of baseline-labelled windows —
│   normalization)      │  applied per fold, never sees the true label
│                       │  beyond that initial window (see §BL-norm below)
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│ 4. Model training     │  classical, deep learning, ensembles, AutoML
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│ 5. Evaluation         │  LOSO (within-dataset) + cross-dataset (zero-shot)
│                       │  + SHAP/XAI + statistical tests (Friedman-Nemenyi,
│                       │  Wilcoxon)
└──────────┬───────────┘
           ▼
   results/*.json, *.txt   +   figures/*.png   +   results_master.csv
```

### 1. Preprocessing

Each dataset has its own preprocessing script, but they share the same
logic: signal filtering, protocol structure detection (baseline → stress →
baseline), 1 Hz resampling, and production of a per-second labelled
DataFrame.

- **Anxolotl** — [`3_anxolotl/anxolotl_preprocessing.py`](3_anxolotl/anxolotl_preprocessing.py):
  reads the Apple Watch CSVs (`sessions/session_XXXX.csv`), detects the 8
  protocol markers via a wrist gesture (arm raise) on the accelerometer's
  z-axis, builds the labels (`0`=rest, `1`=stress, `-1`=discard), and
  computes HR/IBI/RMSSD/SDNN at 1 Hz by interpolation/sliding window over
  the watch's sparse readings.
- **WESAD** — [`1_wesad/wesad_preprocessing.py`](1_wesad/wesad_preprocessing.py):
  reads `wesad.zip` directly (no extraction), processes the 7 wrist
  channels + 4 chest channels from the `S*.pkl`/`S*_E4_Data/` files.
- **WDD** — [`2_cross_dataset/wearable-device-dataset.ipynb`](2_cross_dataset/wearable-device-dataset.ipynb):
  this notebook **is** the WDD preprocessing script (equivalent of the two
  `.py` scripts above, just as a notebook) — processes the data from a
  Kaggle path (see `DATA.md`); there is no `.py` equivalent, only the
  notebook.

### 2. Feature extraction

Sliding windows over the 1 Hz signal. The number of statistics per channel
is **schema-dependent, not universal** — see the table below and DATA.md's
"WESAD feature schemas" section: the **current** schema (WDD, Anxolotl, and
WESAD's regenerated/new schema) uses **11** — mean, std, min, max, p25,
p75, skew, kurtosis, slope, lf_hf via Welch, delta_mean; the **old** WESAD
schema (137 feat, thesis Sections 4.2/4.4) uses only the first **8** (no
slope/lf_hf/delta_mean). A window is discarded if it crosses a phase
boundary (start label ≠ end label). *Sequence* versions (no stats, channels
× time) are also generated for the deep learning models.

| Dataset | Channels | Nº features | Window / stride | Function |
|---|---|---|---|---|
| Anxolotl (canonical) | acc_wrist, hr, ibi, rmssd, sdnn (5) | 55 | 60s / 30s | `build_features_tabular()` in `anxolotl_preprocessing.py` |
| WESAD full (thesis, Section 4.2) | 16 channels (wrist+chest) + 9 demographic | **137** | 60s / 1s | old schema, frozen snapshot — see DATA.md |
| WESAD wrist (thesis, Section 4.3) | 5 wrist channels (acc_wrist, hr, ibi, rmssd, sdnn) | **55** | 60s / 1s | subset of current `wesad_preprocessing.py` output — see DATA.md |
| WESAD full (current preprocessing output, not used in thesis results) | 11 channels (wrist+chest) + 9 demographic | 130 | 60s / 1s | `wesad_preprocessing.py` |
| WESAD wrist, generic (current preprocessing output, not used in thesis results) | 7 wrist channels | 77 | 60s / 1s | `wrist_cols()` over `wesad_preprocessing.py` output |
| WDD (current) | 7 wrist channels + 6 demographic | 83 | — | `wearable-device-dataset.ipynb` |

NOTE: WESAD has two feature schemas, used deliberately in different thesis
sections (4.2 = old/137 feat, 4.3/4.13 = new/55 feat, 4.4 = both — old for
the cross-dataset training space, new for Table 4.12's evaluation target)
— see [`DATA.md`](DATA.md) for the full per-section mapping. Anxolotl uses 55
features. `results_master.csv` documents, row by row (columns
`feature_set`/`n_features`/`source_file`), which configuration each result
uses — don't assume two files with the same model name share a feature
count.

### 3. BL-norm (baseline-relative normalization)

Per-subject z-score normalisation, computed from the **leading run** of that
subject's `baseline`-labelled windows (the initial ~5 min rest period,
always first by protocol design) — never from `y==0` at any point in the
timeline.

Equivalent implementations: `baseline_relative_scale()` in
[`3_anxolotl/demographics_analysis.py:157`](3_anxolotl/demographics_analysis.py#L157)
(canonical version, used by the whole `*_et1000_leakfixed.py` family via
[`3_anxolotl/_canonical_sensitivity_common.py`](3_anxolotl/_canonical_sensitivity_common.py))
and `bl_norm_labels()` in [`scripts/common.py:54`](scripts/common.py#L54).

**Important history:** this function was fixed on **2026-07-19** for a
*data leak*. The previous version used every `y==0` window of a subject —
including, inside each test fold, the very true label the model was
supposed to predict — which made the reference windows collapse to
~N(0,1) regardless of whether there was real signal (confirmed by label
permutation: Macro F1 stayed at ~0.92–0.93 even with shuffled labels). The
fixed version was validated the same way: re-running the identical
label-permutation test against the corrected function gives Macro F1 ≈
**0.48** (mean of 10 seeds, `results/label_permutation_test.csv`) — right at
chance for a balanced binary task, confirming the fix actually removed the
leak rather than just changing its shape. The fixed version only looks at
the initial baseline run, never past the point
where a window's label would be needed — it cannot leak a fold's true label
into its own features. **Any result generated before 2026-07-19 using this
normalisation should be treated as unreliable** — check the date in
`results_master.csv` (`run_date` column) before citing any result that
depends on BL-norm.

### 4. Trained models

| Category | Models | Where |
|---|---|---|
| Classical | LogReg, SVM, RF, ExtraTrees, XGBoost, LightGBM, HistGradientBoosting, CatBoost | `1_wesad/full/` (Table 4.9's RF/XGBoost/LightGBM), `1_wesad/full/table_4_9_macro_f1_recompute.py` + `svm_no_smote_recompute.py` (Table 4.9's LogReg/SVM), `1_wesad/wrist_only/run_wesad_wrist55_blnorm.py` + `run_ensemble_et_rf_hgb_blnorm.py` (Table 4.10, all 8 rows), `1_wesad/classical/wesad_classical_rerun.py` (Table 4.5), `scripts/wdd_classical.py`, `scripts/anxolotl_loso.py` |
| Ensembles | Voting (RF+ET / XGB+RF+LGBM / RF+ET+HGB), stacking | `scripts/wesad_ensemble.py`, `2_cross_dataset/wdd_vs_anxolotl_et_vs_ensemble_wilcoxon.py` (Table 4.12) |
| Deep learning | 1D-CNN, CNN-GRU, RNN/BiGRU, Transformer, Mamba (state-space) | `scripts/wesad_{1dcnn,cnngru,rnn,transformer,mamba}.py` (via `scripts/_dl_loso.py`) |
| Domain adaptation | DANN (tabular and sequence) | `scripts/wesad_dann.py`, `scripts/_dann_tabular.py` |
| AutoML | AutoGluon TabularPredictor | `1_wesad/classical/wesad_autogluon_rerun_fixed.py` (Table 4.9), `1_wesad/full/autogluon-wesad.ipynb` |

The canonical model for the Anxolotl chapter is
`ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
class_weight='balanced', random_state=42)` with median imputation — see
`3_anxolotl/_canonical_sensitivity_common.py`.

### 5. Evaluation

- **LOSO (Leave-One-Subject-Out)** within each dataset — `run_loso*()` in
  `scripts/common.py` / `_canonical_sensitivity_common.py`.
- **Zero-shot cross-dataset** — train on WESAD and/or WDD, test on Anxolotl
  (never seen in training) and vice versa —
  `2_cross_dataset/wdd_vs_anxolotl_et_vs_ensemble_wilcoxon.py` (Table 4.12,
  686-window leak-fixed target).
- **XAI/SHAP** — `scripts/xai.py` (TreeExplainer/LinearExplainer/
  GradientExplainer/DeepExplainer depending on the model) → `figures/shap/`.
- **Statistical tests** — Friedman-Nemenyi between models
  (`scripts/friedman_nemenyi.py`, `3_anxolotl/friedman_nemenyi_anxolotl.py`),
  Wilcoxon between datasets (`2_cross_dataset/wdd_vs_anxolotl_et_vs_ensemble_wilcoxon.py`).
- Reported metrics: Macro F1, F1@optimal-threshold (Youden), AUROC,
  Balanced Accuracy, MCC, confusion matrix — `report_metrics()` in
  `scripts/common.py`.

All reconciled numeric results live in `results_master.csv` (one row = one
experimental configuration, with provenance columns `source_file`/
`run_date`); `MISSING` cells explicitly mark metrics that were never
computed, instead of being invented.

## Repository structure — mapping to the thesis

Each top-level folder corresponds to a specific thesis section. The
pipeline/XAI infrastructure (Ch. 3.5–3.6) is **shared** across all phases —
it doesn't live in a phase-specific folder, but in `scripts/` + each
dataset's `*_preprocessing.py` module.

| Folder | Thesis section | What it produces |
|---|---|---|
| `scripts/` | **Ch. 3.5** (Machine Learning Pipeline) + **Ch. 3.6** (XAI) | Shared infrastructure: loaders, BL-norm, `report_metrics()`, LOSO training, SHAP (`xai.py`) — called by the three phases below, not a phase in itself |
| `1_wesad/` | **Ch. 4.2** (WESAD-Full) + **Ch. 4.3** (WESAD Wrist-Only) | Preprocessing and models trained on the public WESAD dataset — see feature-count caveat below |
| `2_cross_dataset/` | **Ch. 4.4** (Cross-Dataset) | Zero-shot transfer WESAD/WDD → Anxolotl — canonical source: `wdd_vs_anxolotl_et_vs_ensemble_wilcoxon.py` (686-window, leak-fixed target, **Table 4.12**) |
| `3_anxolotl/` | **Ch. 4.5** (Anxolotl) | Main contribution: LOSO, ablations, SHAP on the own dataset |

### WESAD-Full vs. WESAD Wrist-Only (Ch. 4.2 vs. 4.3) — physically separated

`1_wesad/` is split into three subfolders by the notebooks' actual content
(what data they read, what columns they use), not by file name:

| Subfolder | Section | Classification criterion |
|---|---|---|
| `1_wesad/full/` | **Ch. 4.2** — WESAD-Full | Uses all WESAD columns (no channel filter) |
| `1_wesad/wrist_only/` | **Ch. 4.3** — WESAD Wrist-Only (Anxolotl-compatible) | DL models (`1d-cnn/mamba/recurrent-nn/transformer.ipynb`) restricted to `acc_wrist, hr, ibi, rmssd, sdnn` via `selected_idx=[0,5,6,7,8]` (same `SEQ_WRIST_IDX` as `scripts/common.py:20`); classical models (`run_wesad_wrist55_blnorm.py` + `run_ensemble_et_rf_hgb_blnorm.py`, `assert len(reduced_cols)==55`) are the canonical source for **Table 4.10, all 8 rows** |
| `1_wesad/smote_ablation/` | superseded — no longer sources Table 4.5 | see `1_wesad/classical/wesad_classical_rerun.py` for the current source |

`1_wesad/wesad_preprocessing.py` stays at the root — it generates
`processed_data/wesad/processed_data.pkl`, shared by `full/` (current
schema) and `wrist_only/` (tabular); the older `full/*` notebooks read from
a Kaggle path (`/kaggle/input/...`), which corresponds to the old schema
(137 feat) — see the config table below before assuming every file in
`full/` uses the same schema.

NOTE: `1_wesad/full/*.ipynb` were executed on Kaggle and do not run locally
as-is — they load data via a Kaggle-side path that doesn't exist on this
machine or any local clone. They are kept **as-executed, with their
original outputs**, so the committed notebook stays traceable to the
thesis numbers it produced — don't rewrite the Kaggle path in the `.ipynb`
files themselves. To reproduce one locally instead: copy its code into a
`.py` script and point the load at
`processed_data/wesad_archive/processed_data_137feat_snapshot.pkl` (see
"WESAD feature schemas" in `DATA.md`) — `1_wesad/classical/
wesad_classical_rerun.py`, `wesad_autogluon_rerun_fixed.py`, and
`2_cross_dataset/wdd_vs_anxolotl_et_vs_ensemble_wilcoxon.py` are examples
of scripts that already do this.

`1_wesad/classical/wesad_classical_rerun.py` is Table 4.5's canonical
source for 4 of its 5 models (LogReg, RF, XGBoost, LightGBM). SVM uses a
separate source shared with Table 4.9 —
`1_wesad/full/svm_no_smote_recompute.py` +
`table_4_9_macro_f1_recompute.py`. `1_wesad/classical/` also holds
`wesad_autogluon_rerun_fixed.py` (Table 4.9's AutoGluon row) and
`roc_data/` (outputs of an archived script).

Table 4.10's canonical code lives in `1_wesad/wrist_only/`
(`run_wesad_wrist55_blnorm.py` + `run_ensemble_et_rf_hgb_blnorm.py`,
`assert len(reduced_cols)==55`) — copied from
`results/section1_2_wesad_wrist55_blnorm_2026-07-26/`, which stays
untouched as the original run cited by `source_file` in
`results_master.csv`.

**`full/svm-wesad-gridsearch-subsample.ipynb` and `full/svm-wesad-full.ipynb`
are two sequential stages of the same methodology** (thesis Section 4.2.1):
the SVM's polynomial kernel scales poorly, so the *grid search* only runs
on a 40% stratified subsample; the best hyperparameters found there are
then reused by `1_wesad/full/svm_no_smote_recompute.py` +
`table_4_9_macro_f1_recompute.py` (Table 4.5/4.9's current SVM source) at
100% of the data.

NOTE: Feature counts differ across WESAD schemas — the thesis itself uses
137/40/55, depending on config and section (56/77 are a generic 7-channel
wrist byproduct that exists in the code but is NOT used by the thesis) —
see [`DATA.md`](DATA.md) for the authoritative table (which config each
thesis section uses, which file/snapshot generates it, and which one to
point a script at to reproduce a given number).
`results_master.csv` documents, result by result, which config backs each
number (columns `feature_set`, `n_features`, `source_file`) — don't assume
two files with the same model name use the same feature count without
checking there.

## Directory structure

```
anxolotl/
├── 1_wesad/                    Phase 1 — Ch. 4.2 (WESAD-Full) + Ch. 4.3 (WESAD Wrist-Only)
│   ├── wesad_preprocessing.py  reads wesad.zip → processed_data/wesad/*.pkl (current schema, 130 feat)
│   ├── full/                   Ch. 4.2 — ensemble/logreg/rf/svm/xgboost/autogluon/cnn-gru,
│   │                            all columns (no channel filter); Table 4.9 source
│   ├── wrist_only/              Ch. 4.3 — 1d-cnn/mamba/rnn/transformer (DL) +
│   │                            run_wesad_wrist55_blnorm.py/run_ensemble_et_rf_hgb_blnorm.py
│   │                            (classical, canonical for Table 4.10),
│   │                            restricted to acc_wrist/hr/ibi/rmssd/sdnn (Anxolotl-compatible)
│   ├── smote_ablation/          superseded — see classical/wesad_classical_rerun.py instead
│   └── classical/               wesad_autogluon_rerun_fixed.py (Table 4.9) +
│                                wesad_classical_rerun.py (Table 4.5)
│
├── 2_cross_dataset/             Phase 2 — Ch. 4.4 (Cross-Dataset), transfer between datasets
│   ├── wdd_vs_anxolotl_et_vs_ensemble_wilcoxon.py   canonical, Table 4.12
│   │                             (RF+ET, 686-window leak-fixed target + Wilcoxon)
│   ├── wearable-device-dataset.ipynb   WDD ingestion/preprocessing (via Kaggle)
│   └── cross-dataset.ipynb      DANN/AutoGluon exploration
│
├── 3_anxolotl/                  Phase 3 — Ch. 4.5 (Anxolotl), main contribution of the thesis
│   ├── anxolotl_preprocessing.py        Anxolotl preprocessing pipeline
│   ├── demographics_analysis.py         canonical BL-norm (leak-fixed)
│   ├── _canonical_sensitivity_common.py helpers shared by the ablation family
│   ├── *_et1000_leakfixed.py            ablations (window, stride, overlap,
│   │                                     calibration, normalisation, ECG, marker...)
│   ├── shap_canonical_et1000_leakfixed.py  SHAP for the canonical model
│   ├── leakage_check.py                 data leakage audit
│   ├── friedman_nemenyi_anxolotl.py, robust_statistics.py  statistical tests
│   ├── results/, figures/, hrv_ablation/  outputs specific to this phase
│   └── *.ipynb                          exploratory/historical notebooks
│
├── scripts/                     Ch. 3.5/3.6 — shared pipeline and XAI (called by run_all.sh)
│   ├── common.py                 dataset loaders, BL-norm, report_metrics()
│   ├── wesad_*.py, wdd_classical.py, anxolotl_loso.py   one model per script
│   ├── _dl_loso.py, _dann_tabular.py     shared training loop (DL/DANN)
│   ├── xai.py                    SHAP for all models/datasets
│   └── plot_results.py           figure generation from results/*.json
│
├── utils/                        notebook maintenance utilities (not part of the pipeline)
├── archive/                      superseded/exploratory scripts, not imported by anything active
│
├── processed_data/               [generated, git-ignored] extracted features (.pkl)
├── models_data/                  [generated, git-ignored] trained model checkpoints
├── results/                      outputs of each run (results/*.json, *.txt)
├── figures/                      generated figures (incl. figures/shap/)
├── catboost_info/                [generated, git-ignored] automatic CatBoost logs
│
├── sessions/                     [raw data, git-ignored] Anxolotl — Apple Watch CSVs
├── dataset.zip                   [raw data, git-ignored] Anxolotl — Biopac + PASAT
├── wesad.zip                     [raw data, git-ignored] WESAD raw
│
├── eads21_analysis/               exploratory EADS-21 vs. per-subject F1 correlation (Section 4.6.3)
├── hr_reactivity_analysis/        same, HR reactivity vs. anxiety
├── hrv_reactivity_analysis/       same, HRV reactivity vs. anxiety
│
├── results_master.csv             reconciled master table of all results
│
├── run_all.sh, run_remaining.sh   full pipeline orchestration
├── Dockerfile, docker-compose.yml  containerised environment (GPU)
├── requirements.txt               pinned Python dependencies
├── DATA.md                        how to obtain each dataset
└── README.md                      this file
```

## Running each step

### Environment

```bash
pip install -r requirements.txt
# Deep learning (GPU): the Dockerfile uses torch==2.1.2+cu121; locally, the
# development environment (conda env "ml") uses torch==2.9.1 — see the
# "Reproducibility risks" section below on this divergence.
```

Or via Docker (GPU required for the DL phase):
```bash
docker compose up --build
```

### Obtaining the data

See [`DATA.md`](DATA.md) — no raw dataset is in the repository.

### Full pipeline (all phases)

```bash
./run_all.sh            # WESAD LOSO → WDD LOSO → cross-dataset → Anxolotl LOSO → figures → SHAP
# or, if the GPU phase fails partway through (e.g. missing NVIDIA driver):
./run_remaining.sh
```

Results in `results/*.json`/`*.txt`, figures in `figures/`.

### Individual steps

```bash
# Preprocessing
python 1_wesad/wesad_preprocessing.py
python 3_anxolotl/anxolotl_preprocessing.py sessions/ processed_data/anxolotl/

# One classical WESAD model (Ensemble; the other classical models are notebooks, see above)
python scripts/wesad_ensemble.py

# Anxolotl LOSO (all classical models + BL-norm)
python scripts/anxolotl_loso.py

# Cross-dataset (WDD → Anxolotl, canonical, Table 4.12)
python 2_cross_dataset/wdd_vs_anxolotl_et_vs_ensemble_wilcoxon.py

# One canonical leak-fixed ablation (e.g. window-size sensitivity)
python 3_anxolotl/window_size_ablation_et1000_leakfixed.py

# SHAP / XAI (all models, all datasets)
python scripts/xai.py

# Figures from the already-generated results/*.json
python scripts/plot_results.py
```

## Reproducibility — random seeds

`random_state=42` is used **very consistently** across all classical code
(sklearn/XGBoost/LightGBM/CatBoost), including the whole
`3_anxolotl/*_et1000_leakfixed.py` family (same seed in every script,
deliberately, to isolate each ablation's effect from seed variance). Known
exceptions and risks:

| Component | Seed status |
|---|---|
| Classical models (`scripts/`, `1_wesad/`, `2_cross_dataset/`, `3_anxolotl/*_et1000_leakfixed.py`) | consistent `random_state=42` |
| **AutoGluon** (`1_wesad/classical/wesad_autogluon_rerun_fixed.py`, `wdd_classical.py`) | no seed control — `TabularPredictor.fit(...)` doesn't fix a seed; runs are not bit-exact reproducible |
| **WESAD deep learning scripts** (`wesad_1dcnn.py`, `wesad_cnngru.py`, `wesad_rnn.py`, `wesad_transformer.py`, `wesad_mamba.py`, `wesad_dann.py`, via `scripts/_dl_loso.py`) | no seed at all (neither `torch.manual_seed` nor `np.random.seed`) — weight initialisation and `DataLoader` sampling are not run-to-run reproducible |
| `scripts/_dann_tabular.py` (used by `wdd_classical.py`'s DANN and `dann_tabular_variants.py`) | no seed |
| `scripts/dl_variance_analysis.py` | explicitly seeded (`torch.manual_seed`, `np.random.seed`) — written **specifically to quantify** the variance introduced by the missing seeds above; see `results/dl_variance_summary.txt` |
| Some `3_anxolotl/all_models_*.py` scripts, `demographics_analysis.py` | minor inconsistency: baseline models (shallow DecisionTree/RF/AdaBoost/GradientBoosting) use `random_state=0` while the main models in the same file use `random_state=42` |
| `anxolotl_preprocessing.py`, `wesad_preprocessing.py` | N/A — deterministic signal processing, no RNG |

**If you need bit-exact reproducible DL results, this is not guaranteed by
the current pipeline** — the run-to-run variance of WESAD's deep learning
models is characterised (not eliminated) in
`results/dl_variance_summary.txt`/`dl_variance_raw.csv` (5 seeds ×
{1D-CNN, CNN-GRU, BiGRU, Transformer} × {wrist, full}).

For the full history of corrections, provenance investigations, and
superseded configurations behind the numbers above, see the internal audit
log (not part of this public repository).

## License

Code under the MIT license — see [`LICENSE`](LICENSE). The datasets are
**not** covered by this license — see [`DATA.md`](DATA.md).

## Citation

See [`CITATION.cff`](CITATION.cff).
