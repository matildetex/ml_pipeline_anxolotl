"""WESAD LOSO — AutoGluon (wrist + all features)

NÃO USAR — AutoGluon com bug NVML/Ray: LightGBM/CatBoost/NeuralNet falhavam
silenciosamente dentro do TabularPredictor, só RF/ET treinavam de facto.
O número que este script produziu (results/wesad_autogluon.json, 2026-06-05)
é o que está citado na Tabela 4.9 da tese (Macro F1=0.88, mean-per-fold),
mas descreve essa execução com bug, não o AutoGluon a funcionar. Ver
1_wesad/classical/wesad_autogluon_rerun_fixed.py para a versão corrigida
(CUDA_VISIBLE_DEVICES="" antes do import) e RECONCILIATION.md §G7.
Arquivado em 2026-08-31, não corrigir para o schema 137 — não é fonte válida
para nenhum número futuro.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from autogluon.tabular import TabularPredictor
from tqdm import tqdm
from common import load_wesad_tabular, wrist_cols, report_metrics, save_results


def run_loso(X_df, y, groups, feature_cols, label):
    logo = LeaveOneGroupOut()
    all_true, all_prob, fold_f1s = [], [], []

    _splits = list(logo.split(X_df.values, y, groups))
    _pbar = tqdm(_splits, desc="AutoGluon WESAD", ncols=90,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folds [{elapsed}<{remaining}]", file=sys.stderr)
    for fold, (tr, te) in enumerate(_pbar):
        subj = groups[te[0]]
        print(f'  fold {fold+1:2d} | subject: {subj}')

        train_df = X_df.iloc[tr][feature_cols].copy()
        train_df['label'] = y[tr]
        test_df  = X_df.iloc[te][feature_cols]
        y_te = y[te]

        pred = TabularPredictor(label='label', eval_metric='f1_macro', verbosity=0
                               ).fit(train_df, time_limit=300, presets='good_quality')

        preds = pred.predict(test_df).values
        proba = pred.predict_proba(test_df)[1].values
        f1 = f1_score(y_te, preds, average='macro', zero_division=0)
        fold_f1s.append(f1)
        all_true.extend(y_te); all_prob.extend(proba)
        print(f'           F1={f1:.4f}')

    return report_metrics(np.array(all_true), np.array(all_prob), label, fold_f1s)


if __name__ == '__main__':
    print('Loading WESAD...')
    X_df, y, groups, feat_cols = load_wesad_tabular()
    all_results = []

    print('\n' + '='*65)
    print('  WESAD LOSO — AutoGluon — WRIST features')
    print('='*65)
    wcols = wrist_cols(feat_cols)
    print(f'  Features: {len(wcols)} wrist')
    r = run_loso(X_df, y, groups, wcols, f'AutoGluon WESAD — wrist ({len(wcols)} feat)')
    all_results.append(r)

    print('\n' + '='*65)
    print('  WESAD LOSO — AutoGluon — ALL features')
    print('='*65)
    print(f'  Features: {len(feat_cols)} total')
    r = run_loso(X_df, y, groups, feat_cols, f'AutoGluon WESAD — all ({len(feat_cols)} feat)')
    all_results.append(r)

    save_results(all_results, os.path.join(os.path.dirname(__file__), '..', 'results', 'wesad_autogluon.json'))
