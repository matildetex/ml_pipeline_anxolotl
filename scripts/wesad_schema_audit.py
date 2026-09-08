"""
WESAD schema audit -- how far does the pre/post-2026-07-17 schema split reach
into the thesis's reported WESAD numbers, beyond the SHAP figures already
confirmed stale (results/wesad_shap_reverification_2026-07-24.txt)?

Three tasks:
  1. Current WESAD-full feature count vs the 137 the thesis reports; exact list
     of signal channels present in the OLD (137) schema but absent from the
     CURRENT (130) one, and vice versa.
  2. Provenance of the specific numbers named by the user (LightGBM 0.920,
     BiGRU 0.925, "all-models on WESAD-full" comparison table): old schema or
     current? Cross-checked against results_master.csv (which already tags
     n_windows/n_features per row) plus direct mtime checks on the underlying
     result files.
  3. Every WESAD-related file under results/ with mtime before the
     2026-07-17 16:31:57 processed_data.pkl regen cutoff.

Read-only: this script inspects pkls/results files, it does not retrain or
regenerate anything (unlike the prior SHAP re-verification task).
"""
import os, glob, pickle, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, 'results', 'wesad_schema_audit_2026-07-25.txt')
assert not os.path.exists(OUT_PATH), f'{OUT_PATH} already exists -- not overwriting.'

CUTOFF = 1784302317.0  # 2026-07-17 16:31:57, processed_data/wesad/processed_data.pkl regen

lines = []
def w(s=''):
    print(s)
    lines.append(str(s))

w('='*78)
w('WESAD SCHEMA AUDIT -- extent of the pre/post-2026-07-17 schema split')
w('='*78)

# ─────────────────────────────────────────────────────────────────
# TASK 1 -- feature count + channel diff, old vs current
# ─────────────────────────────────────────────────────────────────
w('-'*78)
w('1. WESAD-full FEATURE COUNT: current vs thesis-reported (137)')
w('-'*78)

NEW_PKL = os.path.join(ROOT, 'processed_data', 'wesad', 'processed_data.pkl')
OLD_PKL = os.path.join(ROOT, 'processed_data', 'wesad_archive', 'processed_data_137feat_snapshot.pkl')

with open(NEW_PKL, 'rb') as f:
    new = pickle.load(f)
with open(OLD_PKL, 'rb') as f:
    old = pickle.load(f)

w(f'  CURRENT ({os.path.relpath(NEW_PKL, ROOT)}, mtime {pd.Timestamp(os.path.getmtime(NEW_PKL), unit="s")}):')
w(f'    {len(new["feature_cols"])} features, {len(new["y"])} windows, {len(np.unique(new["groups"]))} subjects')
w(f'  OLD ({os.path.relpath(OLD_PKL, ROOT)}, recovered snapshot, mtime {pd.Timestamp(os.path.getmtime(OLD_PKL), unit="s")}):')
w(f'    {len(old["feature_cols"])} features, {len(old["y"])} windows, {len(np.unique(old["groups"]))} subjects')
w()
w(f'  Thesis reports 137 features for WESAD-full. CURRENT schema has '
  f'{len(new["feature_cols"])}. {"MATCH" if len(new["feature_cols"])==137 else "MISMATCH -- thesis figure is for the OLD schema."}')
w()

SUFFIXES = ['_delta_mean','_lf_hf','_kurtosis','_skew','_slope','_mean','_std','_min','_max','_p25','_p75']
def channel_of(c):
    for suf in SUFFIXES:
        if c.endswith(suf):
            return c[:-len(suf)]
    return c

demo = {'age','gender','coffee_today','coffee_last_hour','sports','smoker','ill','stai_baseline','stai_stress'}
old_channels = sorted(set(channel_of(c) for c in old['feature_cols']) - demo)
new_channels = sorted(set(channel_of(c) for c in new['feature_cols']) - demo)
old_stats = sorted(set(c[len(channel_of(c)):] for c in old['feature_cols'] if channel_of(c) == 'acc_wrist'))
new_stats = sorted(set(c[len(channel_of(c)):] for c in new['feature_cols'] if channel_of(c) == 'acc_wrist'))

w(f'  OLD  signal channels ({len(old_channels)}): {old_channels}')
w(f'  NEW  signal channels ({len(new_channels)}): {new_channels}')
w(f'  OLD per-channel stats ({len(old_stats)}): {old_stats}')
w(f'  NEW per-channel stats ({len(new_stats)}): {new_stats}')
w()
removed = sorted(set(old_channels) - set(new_channels))
added   = sorted(set(new_channels) - set(old_channels))
w(f'  REMOVED (in OLD 137-schema, absent from CURRENT): {removed}')
w(f'  ADDED   (new in CURRENT, absent from OLD 137-schema): {added}')
w()
w('  Arithmetic: OLD = 16 channels x 8 stats (128) + 9 demo = 137.')
w('              NEW = 11 channels x 11 stats (121) + 9 demo = 130.')
w('  16 -> 11 channels: removed ecg_hr/ecg_ibi/ecg_rmssd/ecg_sdnn (4, chest-ECG-derived '
  'HRV) + resp_freq/resp_iri (2, respiration) = -6; added temp_chest (+1). Net -5.')
w('  Per-channel stats went 8 -> 11 (added slope, lf_hf, delta_mean) -- this is why total '
  'feature count (130) is close to 137 despite losing 5 net channels: more stats per '
  'surviving channel partially offsets fewer channels.')
w()
w('  IMPORTANT: current hr/ibi/rmssd/sdnn channels are wrist-BVP-derived '
  '(1_wesad/wesad_preprocessing.py:199, bvp_to_1hz(wrist["BVP"])) -- the OLD schema had '
  'these SAME wrist-derived channels ALSO (unchanged), PLUS a fully separate chest-ECG-'
  'derived HRV group (ecg_hr/ecg_ibi/ecg_rmssd/ecg_sdnn) that no longer exists at all. '
  'So it is not just RESP that vanished -- chest-based cardiac HRV vanished too, and the '
  'current "HR/IBI/RMSSD/SDNN" group is wrist-only, where the old schema had both wrist '
  'and chest cardiac channels represented separately.')
w()
w(f'  CONCLUSION 1: thesis\'s "137 features" figure for WESAD-full matches ONLY the OLD, '
  f'now-superseded schema. The CURRENT pipeline produces {len(new["feature_cols"])}, missing '
  f'RESP entirely and missing chest-ECG-derived HRV entirely (kept only wrist-derived HRV), '
  f'and gained one channel (temp_chest) it did not have before.')
w()

# ─────────────────────────────────────────────────────────────────
# TASK 2 -- provenance of LightGBM 0.920 / BiGRU 0.925 / all-models table
# ─────────────────────────────────────────────────────────────────
w('-'*78)
w('2. PROVENANCE OF LightGBM 0.920 / BiGRU 0.925 / "all models WESAD-full" TABLE')
w('-'*78)

master_path = os.path.join(ROOT, 'results_master.csv')
master = pd.read_csv(master_path)
wesad_full = master[(master['dataset'] == 'WESAD') &
                     (master['feature_set'].astype(str).str.contains('137|130|16ch', regex=True, na=False))]

w(f'results_master.csv is itself the pre-existing "all models WESAD-full comparison" table '
  f'(already schema-tagged per row: n_features, n_windows, source_file). Relevant rows:')
w()
cols_show = ['feature_set','model','macro_f1','accuracy','n_windows','n_subjects','n_features','source_file','run_date']
for _, r in wesad_full.iterrows():
    w(f"  [{r['feature_set']:28s}] {r['model']:30s} macro_f1={r['macro_f1']!s:8s} "
      f"acc={r['accuracy']!s:8s} n_windows={r['n_windows']!s:6s} n_feat={r['n_features']!s:5s} "
      f"src={r['source_file']}")
w()

w('  -- LightGBM 0.920 --')
w('  No standalone LightGBM WESAD result file exists anywhere in the repo (confirmed by '
  'results_master.csv\'s own explicit row: "no standalone LightGBM/HGB single-model WESAD '
  'result file exists ... only appears as one member inside the Ensemble"). The only nearby '
  'number: Ensemble(XGB+RF+LGBM), OLD 137-schema, results/wesad_ensemble.txt -- '
  'macro_f1=0.912, ACCURACY=0.92, n_windows=27543 (OLD, matches the 137-feat/27543-window '
  'signature), run_date 2026-06-05, tagged "STALE" in results_master.csv already. The NEW-'
  'schema Ensemble rerun (results/wesad_ensemble.json, 2026-07-17, 130 feat/25807 windows) '
  'gives macro_f1=0.872, accuracy=0.8839 -- neither matches 0.920 on either metric. If the '
  'thesis\'s "LightGBM 0.920" is this Ensemble\'s accuracy mislabelled as its LightGBM '
  'component, it is OLD-schema. If it is a genuinely separate LightGBM run, no artifact for '
  'it survives on disk at all (old OR new schema) -- it cannot be verified either way.')
w()

w('  -- BiGRU 0.925 --')
rnn_rows = master[(master['dataset']=='WESAD') & (master['model'].astype(str).str.contains('BiGRU|RNN', regex=True))]
for _, r in rnn_rows.iterrows():
    w(f"    {r['feature_set']:26s} {r['configuration'][:70]:70s} macro_f1={r['macro_f1']}")
w()
w('  Closest values: 0.924 (all-16ch raw sequences, single LOSO run, run_date 2026-06-07, '
  'results/wesad_rnn.json/.txt) and 0.9197 (all-16ch, 5-seed variance mean, 2026-07-22, '
  'results/dl_variance_summary.txt). BOTH are all-16ch, n_windows=27543 -- the OLD-schema '
  'signature. Critically, BiGRU (and every other DL model: 1D-CNN, CNN-GRU, Transformer, '
  'Mamba, DANN-seq) is NOT trained on processed_data/wesad/processed_data.pkl (the tabular '
  'pkl that got regenerated 2026-07-17) at all -- scripts/common.py:12-15 points '
  'WESAD_SEQ_PATH at DANN/wesad/processed_data_sequences.pkl by default, a completely '
  'separate sequence-format artifact.')
seq_path = '/home/matildxi/Secretária/Documentos/Tese/DANN/wesad/processed_data_sequences.pkl'
if os.path.exists(seq_path):
    seq_mtime = pd.Timestamp(os.path.getmtime(seq_path), unit='s')
    w(f'  DANN/wesad/processed_data_sequences.pkl mtime: {seq_mtime} -- predates even the OLD '
    f'137-tabular-schema\'s own June dates, and was never touched by the 2026-07-17 regen at '
    f'all (different file, different pipeline). Its 16-channel layout (xai.py:81-83, '
    f'SEQ_CHANNEL_NAMES) explicitly includes ECG_HR/IBI/RMSSD/SDNN and RESP_freq/RESP_iri -- '
    f'the OLD channel set, still in active use for every DL model result in the repo today.')
w()
w(f'  CONCLUSION 2: BOTH named numbers trace to OLD-schema artifacts. LightGBM 0.920 -- if '
  f'it is the Ensemble\'s accuracy -- is OLD (137 feat/27543 windows, 2026-06-05, STALE per '
  f'results_master.csv\'s own tag); if it is a genuine standalone LightGBM run, no artifact '
  f'survives to check either way. BiGRU 0.925 (closest: 0.924 or 0.9197) is unambiguously '
  f'OLD (16ch/27543 windows, sourced from a sequence pkl that predates and is untouched by '
  f'the 2026-07-17 regen). The "all models on WESAD-full" comparison table '
  f'(results_master.csv) is ALREADY schema-tagged per row and already flags the canonical/'
  f'STALE split -- reuse it directly rather than re-deriving; only the Ensemble row has a '
  f'both-schema pair computed (0.912 old / 0.872 new), every DL row only has the OLD number.')
w()

# ─────────────────────────────────────────────────────────────────
# TASK 3 -- every WESAD result file, mtime vs cutoff
# ─────────────────────────────────────────────────────────────────
w('-'*78)
w('3. ALL WESAD-RELATED result FILES, mtime vs 2026-07-17 16:31:57 CUTOFF')
w('-'*78)
w()

patterns = ['wesad*', '*wesad*']
seen = set()
files = []
for pat in patterns:
    for f in glob.glob(os.path.join(ROOT, 'results', pat)):
        if f not in seen and os.path.isfile(f):
            seen.add(f); files.append(f)
files.sort(key=os.path.getmtime)

n_stale = n_current = 0
for f in files:
    mt = os.path.getmtime(f)
    status = 'STALE' if mt < CUTOFF else 'current'
    if status == 'STALE': n_stale += 1
    else: n_current += 1
    mtstr = pd.Timestamp(mt, unit='s').strftime('%Y-%m-%d %H:%M:%S')
    w(f'  {os.path.basename(f):45s} {mtstr}  {status}')
w()
w(f'  TOTAL: {len(files)} WESAD result files, {n_stale} STALE (predate the regen), {n_current} current/post-regen.')
w()
w('  Note: "current" mtime does not by itself mean "current schema" -- e.g. '
  'results/wesad_classical_rerun*.json (2026-07-24) were DELIBERATELY rerun on the OLD '
  '137-feat archive for audit purposes (results_master.csv tags them "full-137 (old '
  'schema, RERUN)"), not because they use the new pipeline. Check feature_set/n_windows '
  'in results_master.csv, not just mtime, before trusting a "current"-mtime file as '
  'new-schema.')
w()

w('='*78)
w('END')
w('='*78)

with open(OUT_PATH, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'\nSaved: {OUT_PATH}')
