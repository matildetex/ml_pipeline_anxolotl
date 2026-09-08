import pickle, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, roc_auc_score, balanced_accuracy_score,
    matthews_corrcoef, roc_curve, classification_report,
    confusion_matrix, precision_recall_curve, average_precision_score,
    precision_score, recall_score
)

WESAD_TABULAR = os.path.join(os.path.dirname(__file__), '..', 'processed_data', 'wesad', 'processed_data.pkl')
WESAD_SEQ     = os.environ.get('WESAD_SEQ_PATH')  # obrigatório para load_wesad_seq() — ver DATA.md
ANXOLOTL      = os.path.join(os.path.dirname(__file__), '..', 'processed_data', 'anxolotl', 'processed_data_anxolotl.pkl')
WDD           = os.path.join(os.path.dirname(__file__), '..', 'processed_data', 'wdd', 'processed_data.pkl')

WRIST_PREFIXES = ('acc_wrist', 'eda_wrist', 'temp_wrist', 'hr_', 'ibi_', 'rmssd_', 'sdnn_')
SEQ_WRIST_IDX  = [0, 5, 6, 7, 8]  # wrist channels in the 16-channel WESAD sequence


def wrist_cols(feature_cols):
    return [c for c in feature_cols if c.startswith(WRIST_PREFIXES)]


def load_wesad_tabular():
    with open(WESAD_TABULAR, 'rb') as f:
        d = pickle.load(f)
    X = d['X'] if isinstance(d['X'], pd.DataFrame) else pd.DataFrame(d['X'], columns=d['feature_cols'])
    return X, np.array(d['y']), np.array(d['groups']), list(d['feature_cols'])


def load_wesad_seq():
    if not WESAD_SEQ:
        raise RuntimeError('WESAD_SEQ_PATH não definido — ver DATA.md')
    with open(WESAD_SEQ, 'rb') as f:
        d = pickle.load(f)
    return d['X_seq'], d['y_seq'], d['groups_seq']


def load_anxolotl():
    with open(ANXOLOTL, 'rb') as f:
        d = pickle.load(f)
    X = d['X'] if isinstance(d['X'], pd.DataFrame) else pd.DataFrame(d['X'], columns=d['feature_cols'])
    return X, np.array(d['y']), np.array(d['groups']), list(d['feature_cols'])


def load_wdd():
    with open(WDD, 'rb') as f:
        d = pickle.load(f)
    X = d['X'] if isinstance(d['X'], pd.DataFrame) else pd.DataFrame(d['X'], columns=d['feature_cols'])
    return X, np.array(d['y']), np.array(d['groups']), list(d['feature_cols'])


def bl_norm_labels(X, y, groups):
    """Fixed 2026-07-19: uses only the leading run of baseline-labelled
    windows per subject (protocol-known initial rest period), not every
    y==0 row anywhere in the timeline — see baseline_relative_scale in
    3_anxolotl/demographics_analysis.py for the full leak writeup."""
    X = np.array(X, dtype=float).copy()
    y = np.array(y); groups = np.array(groups)
    for s in np.unique(groups):
        idx = np.where(groups == s)[0]
        n_baseline = 0
        for v in y[idx]:
            if v == 0:
                n_baseline += 1
            else:
                break
        bl = idx[:n_baseline]
        if len(bl) > 1:
            for j in range(X.shape[1]):
                v = X[bl, j][~np.isnan(X[bl, j])]
                if len(v) > 1:
                    X[idx, j] = (X[idx, j] - v.mean()) / (v.std() + 1e-8)
    return X


def bl_norm_protocol(X, groups, n=8):
    X = np.array(X, dtype=float).copy()
    groups = np.array(groups)
    for s in np.unique(groups):
        idx = np.where(groups == s)[0]; bl = idx[:n]
        for j in range(X.shape[1]):
            v = X[bl, j][~np.isnan(X[bl, j])]
            if len(v) > 1:
                X[idx, j] = (X[idx, j] - v.mean()) / (v.std() + 1e-8)
    return X


def report_metrics(y_true, y_prob, label, fold_f1s=None):
    y_true = np.array(y_true); y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    ba  = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    fpr_arr = tpr_arr = pre_arr = rec_arr = None
    ap = float('nan')
    try:
        auc = roc_auc_score(y_true, y_prob)
        fpr_arr, tpr_arr, thr = roc_curve(y_true, y_prob)
        t_opt = float(thr[np.argmax(tpr_arr - fpr_arr)])
        f1opt = f1_score(y_true, (y_prob >= t_opt).astype(int), average='macro', zero_division=0)
        pre_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
        ap = float(average_precision_score(y_true, y_prob))
    except Exception:
        auc = float('nan'); f1opt = float('nan'); t_opt = 0.5

    print(f'\n{"="*65}')
    print(f'  {label}')
    print(f'{"="*65}')
    print(classification_report(y_true, y_pred, target_names=['baseline', 'stress'], zero_division=0))
    print(f'Macro F1   : {f1:.3f}')
    print(f'AUC        : {auc:.3f}')
    print(f'F1@opt     : {f1opt:.3f}  (t={t_opt:.3f})')
    print(f'Bal. Acc   : {ba:.3f}')
    print(f'MCC        : {mcc:.3f}')
    if fold_f1s is not None:
        print(f'F1 std     : {np.std(fold_f1s):.3f}  (mean per fold: {np.mean(fold_f1s):.3f})')
    cm = confusion_matrix(y_true, y_pred)
    print(f'Confusion  : TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}')
    return {
        'label': label, 'f1': f1, 'auc': auc, 'f1opt': f1opt, 'ba': ba, 'mcc': mcc,
        'f1_std': float(np.std(fold_f1s)) if fold_f1s else None,
        'fold_f1s': [float(v) for v in fold_f1s] if fold_f1s else None,
        'cm': cm.tolist(),
        'fpr': fpr_arr.tolist() if fpr_arr is not None else None,
        'tpr': tpr_arr.tolist() if tpr_arr is not None else None,
        'precision': pre_arr.tolist() if pre_arr is not None else None,
        'recall':    rec_arr.tolist() if rec_arr is not None else None,
        'ap': ap,
        'precision_1': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'recall_1':    float(recall_score(y_true, y_pred,    pos_label=1, zero_division=0)),
        'precision_0': float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'recall_0':    float(recall_score(y_true, y_pred,    pos_label=0, zero_division=0)),
    }


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if obj != obj:  # NaN
        return None
    return str(obj)


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f'\nResultados guardados em: {path}')
