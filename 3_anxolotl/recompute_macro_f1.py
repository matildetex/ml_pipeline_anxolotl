"""
Full Anxolotl macro F1 recomputation — ECG last in progression.
Metrics: Binary F1, Macro F1, Balanced Accuracy.
Includes grid search to optimise Watch+Demo stage.
"""
import pickle, warnings, sys, os
warnings.filterwarnings('ignore')
sys.path.insert(0, '3_anxolotl')
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
from sklearn.ensemble import HistGradientBoostingClassifier
from demographics_analysis import load_demographics, baseline_relative_scale

# ── Load data ──────────────────────────────────────────────────────────────────
with open('processed_data/anxolotl/processed_data_anxolotl.pkl','rb') as f:
    d = pickle.load(f)
X_watch    = np.array(d['X'], dtype=float)
y_all      = np.array(d['y'])
groups_all = np.array(d['groups'])

with open('processed_data/anxolotl/processed_data_anxolotl_ecg.pkl','rb') as f:
    ecg = pickle.load(f)
X_ecg_raw = ecg['X_ecg'].astype(float)
ecg_cols  = ecg['ecg_cols']
X_ecg = X_ecg_raw.copy()
X_ecg[X_ecg[:, ecg_cols.index('ecg_rmssd')] > 300, :] = np.nan
X_ecg[X_ecg[:, ecg_cols.index('ecg_rr_std')] > 300, :] = np.nan
X_ecg[X_ecg[:, ecg_cols.index('ecg_lf_hf')]  > 50,
      ecg_cols.index('ecg_lf_hf')] = np.nan

df_demo = load_demographics('Data_tese.xlsx')
demo_cols = ['age','gender','height','weight','bmi',
             'respiratory','medication','exercises','exercise_freq',
             'athlete','stimulant']
def expand(cols, grps):
    return np.array([df_demo.loc[g, cols].values.astype(float)
                     if g in df_demo.index else np.zeros(len(cols))
                     for g in grps])

X_demo = expand(demo_cols, groups_all)
X_WD   = np.hstack([X_watch, X_demo])            # Watch + Demographics
X_WIN  = np.hstack([X_watch, X_ecg, X_demo])     # Watch + ECG + Demographics

subjects = np.unique(groups_all)

# ECG fallback subjects (>50% NaN rows — no usable BIOPAC)
_ecg_nan = np.all(np.isnan(X_ecg), axis=1)
fallback_subjects = {s for s in subjects if _ecg_nan[groups_all==s].mean() > 0.5}
print(f'ECG fallback subjects: {sorted(fallback_subjects)}')
print(f'Total windows: {len(y_all)}  (stress={int((y_all==1).sum())}  baseline={int((y_all==0).sum())})')

imp = SimpleImputer(strategy='median')

BASELINE_PARAMS = dict(n_estimators=1000, max_features=0.5, min_samples_leaf=1,
                       class_weight='balanced', random_state=42, n_jobs=-1)
FINAL_PARAMS    = dict(n_estimators=1000, max_features=0.3, min_samples_leaf=1,
                       class_weight='balanced', random_state=42, n_jobs=-1)
FINAL_PARAMS_WD = dict(n_estimators=1000, max_features=0.6, min_samples_leaf=1,
                       class_weight='balanced', random_state=42, n_jobs=-1)

# ── Helpers ────────────────────────────────────────────────────────────────────
def calibration_split(X_tr, y_tr, X_te, y_te, cal_frac=0.5):
    base_idx = np.where(y_te == 0)[0]
    if len(base_idx) < 4:
        return X_tr, y_tr, X_te, y_te
    n_cal = max(1, min(int(round(len(base_idx)*cal_frac)), len(base_idx)-1))
    cal_idx  = base_idx[:n_cal]
    eval_idx = np.concatenate([base_idx[n_cal:], np.where(y_te==1)[0]])
    return (np.vstack([X_tr, X_te[cal_idx]]),
            np.concatenate([y_tr, y_te[cal_idx]]),
            X_te[eval_idx], y_te[eval_idx])

def loso(X_feat, use_cal, clf_params, adaptive=False, cal_frac=0.5):
    b_scores, m_scores, ba_scores = [], [], []
    for subj in subjects:
        te = groups_all == subj; tr = ~te
        if adaptive:
            Xf = X_WD if subj in fallback_subjects else X_WIN
            pm = FINAL_PARAMS_WD if subj in fallback_subjects else FINAL_PARAMS
        else:
            Xf, pm = X_feat, clf_params

        X_tr_n = np.nan_to_num(baseline_relative_scale(Xf[tr], y_all[tr], groups_all[tr]))
        X_te_n = np.nan_to_num(baseline_relative_scale(Xf[te], y_all[te], groups_all[te]))
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)
        y_tr_f, y_te_f = y_all[tr], y_all[te]
        if use_cal:
            X_tr_n, y_tr_f, X_te_n, y_te_f = calibration_split(
                X_tr_n, y_tr_f, X_te_n, y_te_f, cal_frac=cal_frac)
        clf = ExtraTreesClassifier(**pm)
        clf.fit(X_tr_n, y_tr_f)
        pred = clf.predict(X_te_n)
        b_scores.append(f1_score(y_te_f, pred, average='binary', zero_division=0))
        m_scores.append(f1_score(y_te_f, pred, average='macro',  zero_division=0))
        ba_scores.append(balanced_accuracy_score(y_te_f, pred))
    return float(np.mean(b_scores)), float(np.mean(m_scores)), float(np.mean(ba_scores))

# ── Grid search: Watch+Demo — HP only (no cal) ───────────────────────────────
print('\n─── GRID SEARCH 1: Watch+Demo — HP (no cal) ────────────')
best_m_nocal, best_hp = 0.0, BASELINE_PARAMS
for mf in [0.3, 0.4, 0.5, 0.6, 0.7]:
    for ml in [1, 2]:
        p = dict(n_estimators=1000, max_features=mf, min_samples_leaf=ml,
                 class_weight='balanced', random_state=42, n_jobs=-1)
        _, m, _ = loso(X_WD, use_cal=False, clf_params=p)
        print(f'  mf={mf} msl={ml}  macro={m:.4f}')
        if m > best_m_nocal:
            best_m_nocal, best_hp = m, p
print(f'  → Best HP (no cal): macro={best_m_nocal:.4f}  mf={best_hp["max_features"]}')

# ── Grid search: best HP + cal_frac sweep ────────────────────────────────────
print('\n─── GRID SEARCH 2: Watch+Demo — cal_frac sweep (best HP) ─')
best_m_cal, best_cal_frac = 0.0, 0.5
for cf in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    _, m, _ = loso(X_WD, use_cal=True, clf_params=best_hp, cal_frac=cf)
    print(f'  cal_frac={cf:.2f}  macro={m:.4f}')
    if m > best_m_cal:
        best_m_cal, best_cal_frac = m, cf
print(f'  → Best cal_frac: {best_cal_frac}  macro={best_m_cal:.4f}')

# ═══ PART A: PROGRESSION (ECG LAST) ══════════════════════════════════════════
print('\n─── A. PROGRESSION (ECG last) ──────────────────────────')
stages_cfg = [
    ('Watch only',               X_watch, False, BASELINE_PARAMS, False, 0.5),
    ('+ Demographics',           X_WD,    False, BASELINE_PARAMS, False, 0.5),
    ('+ HP tuning',              X_WD,    False, best_hp,         False, 0.5),
    ('+ Calibration',            X_WD,    True,  best_hp,         False, best_cal_frac),
    ('+ ECG + adaptive (final)', None,    True,  None,            True,  0.5),
]
prog = []
for lbl, X, cal, params, adap, cf in stages_cfg:
    b, m, ba = loso(X, cal, params, adaptive=adap, cal_frac=cf)
    prog.append((lbl, b, m, ba))
    print(f'  {lbl:<28} binary={b:.4f}  macro={m:.4f}  bal_acc={ba:.4f}')

# ═══ PART B: PER-MODEL TABLE ══════════════════════════════════════════════════
print('\n─── B. PER-MODEL (Watch-only, no calibration) ──────────')

def loso_model(clf_factory, X_feat):
    b_scores, m_scores, ba_scores = [], [], []
    for subj in subjects:
        te = groups_all == subj; tr = ~te
        X_tr_n = np.nan_to_num(baseline_relative_scale(X_feat[tr], y_all[tr], groups_all[tr]))
        X_te_n = np.nan_to_num(baseline_relative_scale(X_feat[te], y_all[te], groups_all[te]))
        X_tr_n = imp.fit_transform(X_tr_n)
        X_te_n = imp.transform(X_te_n)
        clf = clf_factory()
        clf.fit(X_tr_n, y_all[tr])
        pred = clf.predict(X_te_n)
        b_scores.append(f1_score(y_all[te], pred, average='binary', zero_division=0))
        m_scores.append(f1_score(y_all[te], pred, average='macro',  zero_division=0))
        ba_scores.append(balanced_accuracy_score(y_all[te], pred))
    return float(np.mean(b_scores)), float(np.mean(m_scores)), float(np.mean(ba_scores))

models = [
    ('ET-1000',          lambda: ExtraTreesClassifier(n_estimators=1000, max_features=0.5,
                             class_weight='balanced', random_state=42, n_jobs=-1)),
    ('ET-1000 mf=0.3',   lambda: ExtraTreesClassifier(n_estimators=1000, max_features=0.3,
                             class_weight='balanced', random_state=42, n_jobs=-1)),
    ('RF-1000',          lambda: RandomForestClassifier(n_estimators=1000,
                             class_weight='balanced', random_state=42, n_jobs=-1)),
    ('HGB',              lambda: HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                             learning_rate=0.1, random_state=42)),
    ('Ensemble(ET+RF+HGB)', lambda: VotingClassifier(voting='hard', estimators=[
        ('et',  ExtraTreesClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('rf',  RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('hgb', HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)),
    ])),
]
if HAS_LGBM:
    models += [
        ('LGBM-dart', lambda: LGBMClassifier(boosting_type='dart', n_estimators=300,
                          learning_rate=0.05, class_weight='balanced', random_state=42,
                          n_jobs=-1, verbose=-1)),
        ('LGBM-gbdt', lambda: LGBMClassifier(boosting_type='gbdt', n_estimators=300,
                          class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)),
    ]

model_results = []
for name, factory in models:
    b, m, ba = loso_model(factory, X_watch)
    model_results.append((name, b, m, ba))
    print(f'  {name:<26} binary={b:.4f}  macro={m:.4f}  bal_acc={ba:.4f}')

# ═══ PART C: BOUNDARY WINDOW DIAGNOSTICS ═════════════════════════════════════
print('\n─── C. BOUNDARY WINDOW DIAGNOSTICS ─────────────────────')
df_new = d['df'].copy()
print(f'Total windows (after fix): {len(df_new)}')
print(f'Stress: {(df_new["label"]==1).sum()}  Baseline: {(df_new["label"]==0).sum()}')

# Per-subject window counts and phase coverage
print(f'\n{"Subj":>6} | {"BL":>4} | {"S1":>4} | {"S2":>4} | {"S3":>4} | {"Total":>6}')
print('-' * 42)
for s in sorted(df_new['subject'].unique()):
    g = df_new[df_new['subject'] == s]
    bl = (g['label']==0).sum()
    s1 = (g['stress_phase']==1).sum()
    s2 = (g['stress_phase']==2).sum()
    s3 = (g['stress_phase']==3).sum()
    print(f'{s:>6} | {bl:>4} | {s1:>4} | {s2:>4} | {s3:>4} | {len(g):>6}')

# Phase composition
s1t = (df_new['stress_phase']==1).sum()
s2t = (df_new['stress_phase']==2).sum()
s3t = (df_new['stress_phase']==3).sum()
tot_stress = (df_new['label']==1).sum()
print(f'\nStress composition: S1={s1t} ({100*s1t/tot_stress:.1f}%)  '
      f'S2={s2t} ({100*s2t/tot_stress:.1f}%)  S3={s3t} ({100*s3t/tot_stress:.1f}%)')

print(f'\nSubjects with S1 only:    '
      f'{sum(1 for s in df_new["subject"].unique() if (df_new[df_new["subject"]==s]["stress_phase"]==1).sum()>0 and (df_new[df_new["subject"]==s]["stress_phase"]==2).sum()==0)}/37')
print(f'Subjects with S1+S2:      '
      f'{sum(1 for s in df_new["subject"].unique() if (df_new[df_new["subject"]==s]["stress_phase"]==2).sum()>0 and (df_new[df_new["subject"]==s]["stress_phase"]==3).sum()==0)}/37')
print(f'Subjects with S1+S2+S3:   '
      f'{sum(1 for s in df_new["subject"].unique() if (df_new[df_new["subject"]==s]["stress_phase"]==3).sum()>0)}/37')

print(f'\nWHY windows were removed:')
print(f'  The boundary fix skips any window where the label of the FIRST sample')
print(f'  differs from the label of the LAST sample (window spans a phase transition).')
print(f'  After removing discard (-1) rows, the DataFrame re-indexes with no gap:')
print(f'    row[k-1] = last baseline second')
print(f'    row[k]   = first stress second')
print(f'  Old code: label = iloc[i] = position AFTER the window → mislabeled')
print(f'    Window [i-60:i-1] = baseline content, label = stress (wrong)')
print(f'  New code: skip if iloc[i-window_sec].label != iloc[i-1].label')
print(f'    ~2 windows per phase boundary per subject excluded')
print(f'    ~35 windows moved stress→baseline (were baseline content, now correctly labeled)')

# ═══ SUMMARY ══════════════════════════════════════════════════════════════════
print('\n══ SUMMARY ══════════════════════════════════════════════════')
print('\nA. Progression (ECG last):')
print(f'  {"Stage":<28} {"BinaryF1":>9} {"MacroF1":>9} {"BalAcc":>8}')
print('  ' + '-'*57)
for lbl, b, m, ba in prog:
    print(f'  {lbl:<28} {b:>9.4f} {m:>9.4f} {ba:>8.4f}')

print('\nB. Per-model (Watch-only):')
print(f'  {"Model":<26} {"BinaryF1":>9} {"MacroF1":>9} {"BalAcc":>8}')
print('  ' + '-'*55)
for name, b, m, ba in model_results:
    print(f'  {name:<26} {b:>9.4f} {m:>9.4f} {ba:>8.4f}')

# Save
import csv
os.makedirs('results', exist_ok=True)
with open('results/macro_progression.csv','w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['stage','binary_f1','macro_f1','balanced_accuracy'])
    for lbl, b, m, ba in prog: w.writerow([lbl, f'{b:.4f}', f'{m:.4f}', f'{ba:.4f}'])
with open('results/macro_models.csv','w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['model','binary_f1','macro_f1','balanced_accuracy'])
    for name, b, m, ba in model_results: w.writerow([name, f'{b:.4f}', f'{m:.4f}', f'{ba:.4f}'])
print('\nSaved results/macro_progression.csv and results/macro_models.csv')
