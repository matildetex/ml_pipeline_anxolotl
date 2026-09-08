"""
Stress detection with demographic and EADS-21 features added to wearable data.

Compares three feature sets via LOSO-CV + BL-norm:
  1. watch_only     — 60 wearable features (baseline)
  2. watch_demo     — watch + demographics (age, BMI, gender, exercise, stimulant, etc.)
  3. watch_eads     — watch + EADS-21 subscales + VAS stress score

Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

warnings.filterwarnings('ignore')
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11})

# ── 1. Session → form_ID mapping (verified by timestamp matching) ─────────────

SESSION_TO_FORM = {
    'p6': 20,   # excluded from processed_data
    'p7': 21,  'p8': 22,  'p9': 23,  'p10': 24,
    'p11': 27, 'p12': 28, 'p13': 29, 'p14': 30,
    'p15': 31, 'p16': 32, 'p17': 33, 'p18': 34,
    'p19': 35, 'p20': 36, 'p21': 37, 'p22': 38,
    'p23': 39,
    'p24': 40,  # excluded from processed_data
    'p25': 41, 'p26': 42, 'p27': 43, 'p28': 44,
    'p29': 45, 'p30': 46, 'p31': 47, 'p32': 48,
    'p33': 49, 'p34': 50, 'p35': 51, 'p36': 52,
    'p37': 53, 'p38': 54, 'p39': 55, 'p40': 56,
    'p41': 14,  # form_ID=14 has wrong date, confirmed by elimination
    'p42': 57, 'p43': 58, 'p44': 59,
}

# ── 2. EADS-21 item → subscale mapping (DASS-21 structure) ───────────────────
# Scores multiplied by 2 to give standard DASS-21 scale (0-42 per subscale)
EADS_COLS = [
    ' [Tive dificuldades em me acalmar]',                           # 1  Stress
    ' [Senti a minha boca seca]',                                   # 2  Anxiety
    ' [Não consegui sentir nenhum sentimento positivo]',            # 3  Depression
    ' [Senti dificuldades em respirar]',                            # 4  Anxiety
    ' [Tive dificuldade em tomar iniciativa para fazer coisas]',    # 5  Depression
    ' [Tive tendência a reagir em demasia em determinadas situações]', # 6 Stress
    ' [Senti tremores (por ex., nas mãos)]',                        # 7  Anxiety
    ' [Senti que estava a utilizar muita energia nervosa pânico e fazer figura ridícula]', # 8 Stress
    ' [Preocupei-me com situações em que podia entrar em pânico e fazer figura ridícula]', # 9 Anxiety
    ' [Senti que não tinha nada a esperar do futuro]',              # 10 Depression
    ' [Dei por mim a ficar agitado]',                               # 11 Stress
    ' [Senti dificuldade em me relaxar]',                           # 12 Stress
    ' [Senti-me desanimado e melancólico]',                         # 13 Depression
    ' [Estive intolerante em relação a qualquer coisa que me impedisse de terminar aquilo que estava a fazer]', # 14 Stress
    ' [Senti-me quase a entrar em pânico]',                         # 15 Anxiety
    ' [Não fui capaz de ter entusiasmo por nada]',                  # 16 Depression
    ' [Senti que não tinha muito valor como pessoa]',               # 17 Depression
    ' [Senti que por vezes estava sensível]',                       # 18 Stress
    ' [Senti alterações no meu coração sem fazer exercício físico]',# 19 Anxiety
    ' [Senti-me assustado sem ter tido uma boa razão para isso]',   # 20 Anxiety
    ' [Senti que a vida não tinha sentido]',                        # 21 Depression
]
DEPRESSION_IDX = [2, 4, 9, 12, 15, 16, 20]   # 0-indexed from above list
ANXIETY_IDX    = [1, 3, 6, 8, 14, 18, 19]
STRESS_IDX     = [0, 5, 7, 10, 11, 13, 17]

# ── 3. Load demographic + EADS + VAS data ─────────────────────────────────────

def load_demographics(xlsx_path):
    xl = pd.ExcelFile(xlsx_path)
    q    = xl.parse('Questionário')
    eads = xl.parse('EADS-21')
    vas  = xl.parse('VAS')

    id_q = 'ID do Participante (a ser atribuído pelo investigador): '
    id_e = 'ID (dado pelo investigador)'
    id_v = 'ID do Participante (a ser atribuído pelo investigador):'
    vas_col = 'De 0 a 10, quão stressado se sentiu durante este teste'

    # Demographics encoding
    q = q.set_index(id_q)

    def encode_freq(v):
        mapping = {'Nenhuma': 0, '1-2 vezes': 1, '3-4 vezes': 2, '5 ou mais vezes': 3}
        return mapping.get(str(v).strip(), 0)

    demo_rows = {}
    for form_id, row in q.iterrows():
        gender = 1 if str(row.get('Gênero', '')).strip() == 'Feminino' else 0
        def parse_num(v, default):
            try: return float(str(v).replace('cm','').replace('kg','').strip())
            except: return default
        height = parse_num(row.get('Altura', 170), 170) if pd.notna(row.get('Altura')) else 170
        weight = parse_num(row.get('Peso', 70), 70) if pd.notna(row.get('Peso')) else 70
        bmi    = weight / ((height / 100) ** 2)
        resp   = 1 if str(row.get('Possui alguma condição respiratória ou cardíaca?', 'Não')).startswith('Sim') else 0
        meds   = 1 if str(row.get('Toma atualmente medicação que possa influenciar a frequência cardíaca ou respiratória?', 'Não')).startswith('Sim') else 0
        exer   = 1 if str(row.get('Pratica exercícios físicos regularmente?', 'Não')).startswith('Sim') else 0
        freq   = encode_freq(row.get('Frequência semanal de exercícios:', 'Nenhuma'))
        athlt  = 1 if str(row.get('É atleta de alta competição?', 'Não')).startswith('Sim') else 0
        stim   = 1 if str(row.get('Ingeriu algum estimulante (ex.: café, energéticos, tabaco) nas últimas 2 horas?', 'Não')).startswith('Sim') else 0
        age    = float(row.get('Idade:', 22)) if pd.notna(row.get('Idade:')) else 22
        demo_rows[form_id] = {
            'age': age, 'gender': gender, 'height': height, 'weight': weight,
            'bmi': bmi, 'respiratory': resp, 'medication': meds,
            'exercises': exer, 'exercise_freq': freq,
            'athlete': athlt, 'stimulant': stim,
        }

    # EADS-21 subscales
    eads = eads.set_index(id_e)
    eads_rows = {}
    for form_id, row in eads.iterrows():
        items = [float(row.get(c, 0)) if pd.notna(row.get(c)) else 0
                 for c in EADS_COLS]
        dep = sum(items[i] for i in DEPRESSION_IDX) * 2
        anx = sum(items[i] for i in ANXIETY_IDX) * 2
        str_= sum(items[i] for i in STRESS_IDX) * 2
        tot = sum(items) * 2
        eads_rows[form_id] = {
            'eads_depression': dep, 'eads_anxiety': anx,
            'eads_stress': str_, 'eads_total': tot,
        }

    # VAS
    vas = vas.set_index(id_v)
    vas_rows = {form_id: {'vas_stress': float(row[vas_col])}
                for form_id, row in vas.iterrows()}

    # Merge all into subject-level dict
    subject_features = {}
    for subj, form_id in SESSION_TO_FORM.items():
        if subj in ('p6', 'p24'):
            continue
        d = {}
        d.update(demo_rows.get(form_id, {}))
        d.update(eads_rows.get(form_id, {}))
        d.update(vas_rows.get(form_id, {}))
        subject_features[subj] = d

    df_subj = pd.DataFrame(subject_features).T
    df_subj.index.name = 'subject'
    return df_subj

# ── 4. BL-norm ────────────────────────────────────────────────────────────────

def baseline_relative_scale(X, y, groups):
    """
    Fixed 2026-07-19: the reference set is now the subject's LEADING run of
    baseline-labelled windows (the initial ~5min rest period, always first
    by protocol design) instead of every y==0 window anywhere in the
    timeline. The previous version used the true label of the fold being
    normalised (including the test fold's own ground truth) to pick its own
    normalisation reference, which self-referentially made reference rows
    collapse to ~N(0,1) regardless of any real signal — confirmed via label
    permutation (macro F1 stayed ~0.92-0.93 with shuffled labels) and a
    pure-Gaussian-noise control (macro F1 ~0.95 with zero real signal).
    Using only the leading run never looks past the point a window's own
    label would first be needed, so it can't leak a fold's own ground truth
    into its own features, train or test.
    """
    X = np.array(X, dtype=float).copy()
    y = np.array(y); groups = np.array(groups)
    for subj in np.unique(groups):
        subj_idx = np.where(groups == subj)[0]
        n_baseline = 0
        for v in y[subj_idx]:
            if v == 0:
                n_baseline += 1
            else:
                break
        baseline_idx = subj_idx[:n_baseline]
        if len(baseline_idx) > 1:
            for j in range(X.shape[1]):
                col_bl = X[baseline_idx, j]
                valid = col_bl[~np.isnan(col_bl)]
                if len(valid) > 1:
                    mu, sd = valid.mean(), valid.std()
                    X[subj_idx, j] = (X[subj_idx, j] - mu) / (sd + 1e-8)
    return X

# ── 5. Classifiers ────────────────────────────────────────────────────────────

def make_classifiers():
    return {
        'ExtraTrees': ExtraTreesClassifier(n_estimators=500, min_samples_leaf=1,
                                           class_weight='balanced', random_state=42),
        'DT':  DecisionTreeClassifier(random_state=0),
        'RF':  RandomForestClassifier(max_depth=4, random_state=0),
        'AB':  AdaBoostClassifier(random_state=0),
        'KN':  KNeighborsClassifier(n_neighbors=9),
        'LDA': LinearDiscriminantAnalysis(),
        'SVM': svm.SVC(probability=True),
        'GB':  GradientBoostingClassifier(random_state=0),
    }

# ── 6. LOSO-CV ────────────────────────────────────────────────────────────────

def run_loso(X, y, groups, label=''):
    classifiers = make_classifiers()
    subjects = np.unique(groups)
    results = {name: {'f1': [], 'auc': []} for name in classifiers}

    for subj in subjects:
        test_mask  = groups == subj
        train_mask = ~test_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        X_train_n = np.nan_to_num(baseline_relative_scale(X_train, y_train, groups[train_mask]))
        X_test_n  = np.nan_to_num(baseline_relative_scale(X_test,  y_test,  groups[test_mask]))

        for name, clf in classifiers.items():
            clf.fit(X_train_n, y_train)
            y_pred = clf.predict(X_test_n)
            try:
                y_prob = clf.predict_proba(X_test_n)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            except:
                auc = float('nan')
            results[name]['f1'].append(f1_score(y_test, y_pred, average='binary', zero_division=0))
            results[name]['auc'].append(auc)

    summary = {name: {'f1': np.mean(v['f1']), 'auc': np.nanmean(v['auc'])}
               for name, v in results.items()}
    print(f'\n{label}')
    for name, m in summary.items():
        print(f'  {name:<12}: F1={m["f1"]:.3f}  AUC={m["auc"]:.3f}')
    return summary

# ── 7. Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    XLSX = 'Data_tese.xlsx'

    # Load processed wearable data
    print('Loading wearable data...')
    with open('processed_data/anxolotl/processed_data_anxolotl.pkl', 'rb') as f:
        meta = pickle.load(f)
    X_all      = np.array(meta['X'], dtype=float)
    y_all      = np.array(meta['y'])
    groups_all = np.array(meta['groups'])
    feat_cols  = meta['feature_cols']
    print(f'  {X_all.shape[0]} windows, {X_all.shape[1]} features, {len(np.unique(groups_all))} subjects')

    # Load demographics
    print('\nLoading demographics from Excel...')
    df_demo = load_demographics(XLSX)
    demo_cols = ['age', 'gender', 'height', 'weight', 'bmi',
                 'respiratory', 'medication', 'exercises', 'exercise_freq',
                 'athlete', 'stimulant']
    eads_cols_feat = ['eads_depression', 'eads_anxiety', 'eads_stress', 'eads_total', 'vas_stress']
    print(df_demo[demo_cols + eads_cols_feat].round(1).to_string())

    # Build per-window demographic features (constant within subject)
    def expand_subj_features(df_demo, groups, cols):
        rows = []
        for g in groups:
            if g in df_demo.index:
                rows.append(df_demo.loc[g, cols].values.astype(float))
            else:
                rows.append(np.zeros(len(cols)))
        return np.array(rows)

    X_demo  = expand_subj_features(df_demo, groups_all, demo_cols)
    X_eads  = expand_subj_features(df_demo, groups_all, eads_cols_feat)

    X_watch      = X_all
    X_watch_demo = np.hstack([X_all, X_demo])
    X_watch_eads = np.hstack([X_all, X_eads])

    print(f'\nFeature set sizes:')
    print(f'  watch_only     : {X_watch.shape[1]} features')
    print(f'  watch_demo     : {X_watch_demo.shape[1]} features (+{X_demo.shape[1]} demographic)')
    print(f'  watch_eads     : {X_watch_eads.shape[1]} features (+{X_eads.shape[1]} EADS+VAS)')

    # Run LOSO-CV for each feature set
    print('\n' + '='*60)
    print('LOSO-CV — BL-norm — Binary F1 (stress class)')
    print('='*60)
    res_watch = run_loso(X_watch,      y_all, groups_all, '── Watch only ──')
    res_demo  = run_loso(X_watch_demo, y_all, groups_all, '── Watch + Demographics ──')
    res_eads  = run_loso(X_watch_eads, y_all, groups_all, '── Watch + EADS-21 + VAS ──')

    # Save results
    rows = []
    for name in make_classifiers():
        rows.append({
            'classifier': name,
            'F1_watch':      res_watch[name]['f1'],
            'F1_watch_demo': res_demo[name]['f1'],
            'F1_watch_eads': res_eads[name]['f1'],
            'AUC_watch':      res_watch[name]['auc'],
            'AUC_watch_demo': res_demo[name]['auc'],
            'AUC_watch_eads': res_eads[name]['auc'],
        })
    df_res = pd.DataFrame(rows).set_index('classifier')
    df_res.to_csv('results/demographics_f1_comparison.csv')
    print(f'\nSaved results/demographics_f1_comparison.csv')

    # ── Figures ───────────────────────────────────────────────────────────────
    clf_names = list(make_classifiers().keys())
    x = np.arange(len(clf_names))
    w = 0.25

    C_WATCH = '#4C72B0'
    C_DEMO  = '#55A868'
    C_EADS  = '#DD8452'

    # Figure A — grouped bar per classifier
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle('Effect of adding demographic and EADS-21 features to wearable data\n'
                 '(Anxolotl LOSO-CV, BL-norm, all 8 classifiers)', fontsize=12)

    ax = axes[0]
    f1_watch = [res_watch[n]['f1'] for n in clf_names]
    f1_demo  = [res_demo[n]['f1']  for n in clf_names]
    f1_eads  = [res_eads[n]['f1']  for n in clf_names]
    b1 = ax.bar(x - w, f1_watch, w, label='Watch only (60 feat)',     color=C_WATCH, alpha=0.88)
    b2 = ax.bar(x,     f1_demo,  w, label='Watch + Demographics (+11)', color=C_DEMO,  alpha=0.88)
    b3 = ax.bar(x + w, f1_eads,  w, label='Watch + EADS-21/VAS (+5)',  color=C_EADS,  alpha=0.88)
    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(clf_names)
    ax.set_ylabel('Binary F1 (stress class)')
    ax.set_ylim(0, 1.1)
    ax.set_title('F1 by classifier and feature set')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Figure B — delta bars
    ax = axes[1]
    delta_demo = [d - w_ for d, w_ in zip(f1_demo, f1_watch)]
    delta_eads = [e - w_ for e, w_ in zip(f1_eads, f1_watch)]
    ax.bar(x - 0.2, delta_demo, 0.35, label='Δ(Demo − Watch)',    color=C_DEMO, alpha=0.88)
    ax.bar(x + 0.2, delta_eads, 0.35, label='Δ(EADS+VAS − Watch)', color=C_EADS, alpha=0.88)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(clf_names)
    ax.set_ylabel('ΔF1 vs watch-only')
    ax.set_title('Gain/loss from adding extra features')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig8_demographics_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/fig8_demographics_effect.png')

    # Figure C — summary (best per feature set = ExtraTrees)
    fig, ax = plt.subplots(figsize=(10, 5))
    et_watch = res_watch['ExtraTrees']['f1']
    et_demo  = res_demo['ExtraTrees']['f1']
    et_eads  = res_eads['ExtraTrees']['f1']
    vals  = [et_watch, et_demo, et_eads]
    lbls  = ['Watch only\n(60 features)', 'Watch +\nDemographics\n(71 features)',
             'Watch +\nEADS-21/VAS\n(65 features)']
    cols  = [C_WATCH, C_DEMO, C_EADS]
    bars  = ax.bar(range(3), vals, 0.5, color=cols, alpha=0.88)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{v:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_xticks(range(3)); ax.set_xticklabels(lbls, fontsize=10)
    ax.set_ylabel('Binary F1 (stress class)')
    ax.set_ylim(0, 1.12)
    ax.set_title('ExtraTrees 500 + BL-norm — feature set comparison\n(Anxolotl LOSO-CV)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/fig9_extratrees_feature_sets.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/fig9_extratrees_feature_sets.png')

    # Figure D — demographic feature importance (ExtraTrees, watch+demo)
    print('\nComputing demographic feature importances (ExtraTrees, watch+demo)...')
    from sklearn.ensemble import ExtraTreesClassifier as ETC
    X_n = np.nan_to_num(baseline_relative_scale(X_watch_demo, y_all, groups_all))
    et_full = ETC(n_estimators=500, min_samples_leaf=1, class_weight='balanced', random_state=42)
    et_full.fit(X_n, y_all)
    all_feat = feat_cols + demo_cols
    imp = pd.Series(et_full.feature_importances_, index=all_feat)
    demo_imp = imp[demo_cols].sort_values(ascending=True)
    top_watch = imp[feat_cols].nlargest(10).sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('ExtraTrees feature importance (watch + demographics model)', fontsize=12)

    ax = axes[0]
    ax.barh(demo_imp.index, demo_imp.values, color=C_DEMO, alpha=0.88)
    ax.set_xlabel('Feature importance')
    ax.set_title('Demographic features')
    ax.grid(axis='x', alpha=0.3)

    ax = axes[1]
    ax.barh(top_watch.index, top_watch.values, color=C_WATCH, alpha=0.88)
    ax.set_xlabel('Feature importance')
    ax.set_title('Top 10 wearable features')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig10_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/fig10_feature_importance.png')

    # Final summary
    print('\n' + '='*60)
    print('SUMMARY — ExtraTrees 500 + BL-norm')
    print('='*60)
    print(f'  Watch only                : F1={et_watch:.3f}')
    print(f'  Watch + Demographics      : F1={et_demo:.3f}  (Δ={et_demo-et_watch:+.3f})')
    print(f'  Watch + EADS-21/VAS       : F1={et_eads:.3f}  (Δ={et_eads-et_watch:+.3f})')
    print()
    print(f'  Mean F1 across all classifiers:')
    mean_w = np.mean([res_watch[n]['f1'] for n in clf_names])
    mean_d = np.mean([res_demo[n]['f1']  for n in clf_names])
    mean_e = np.mean([res_eads[n]['f1']  for n in clf_names])
    print(f'    Watch only           : {mean_w:.3f}')
    print(f'    Watch + Demographics  : {mean_d:.3f}  (Δ={mean_d-mean_w:+.3f})')
    print(f'    Watch + EADS-21/VAS  : {mean_e:.3f}  (Δ={mean_e-mean_w:+.3f})')
