"""
Fix broken data-loading paths in all notebooks.
Replaces Kaggle paths and old DANN paths with the current local paths.
Run from the project root: python3 utils/fix_notebook_paths.py
"""
import json, os, re, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Path mapping ──────────────────────────────────────────────────────────────
WESAD_TAB  = os.path.join(ROOT, 'processed_data', 'wesad', 'processed_data.pkl')
WESAD_SEQ  = '/home/matildxi/Secretária/Documentos/Tese/DANN/wesad/processed_data_sequences.pkl'
WDD_TAB    = os.path.join(ROOT, 'processed_data', 'wdd', 'processed_data.pkl')
ANX_TAB    = os.path.join(ROOT, 'processed_data', 'anxolotl', 'processed_data_anxolotl.pkl')
MODELS_DIR = os.path.join(ROOT, 'models_data')

REPLACEMENTS = [
    # Kaggle WESAD tabular
    (r'/kaggle/input/notebooks/matildeteixeira/wesad-preprocess/processed_data\.pkl', WESAD_TAB),
    # Kaggle WESAD sequences
    (r'/kaggle/input/notebooks/matildeteixeira/wesad-preprocess/processed_data_sequences\.pkl', WESAD_SEQ),
    # Old DANN WESAD tabular
    (r'/home/matildxi/Secretária/Documentos/Tese/DANN/wesad/processed_data\.pkl', WESAD_TAB),
    # Old DANN sequences (wrong path — missing /wesad/)
    (r'/home/matildxi/Secretária/Documentos/Tese/DANN/processed_data_sequences\.pkl', WESAD_SEQ),
    # Old DANN models
    (r'/home/matildxi/Secretária/Documentos/Tese/DANN/lgbm_wrist_final\.pkl',
     os.path.join(MODELS_DIR, 'lgbm_wrist_final.pkl')),
    (r'/home/matildxi/Secretária/Documentos/Tese/DANN/ensemble_wrist_final\.pkl',
     os.path.join(MODELS_DIR, 'ensemble_wrist_final.pkl')),
    # Kaggle WDD processed pkl (originally needed for rf-smote-version-v2.ipynb;
    # that notebook is now archive/rf-smote-version-v2_WDD_EXPLORATION.ipynb,
    # 2026-09-07, so this glob's 'archive' not in fp exclusion means it's no
    # longer actually applied there -- kept in case another notebook needs it)
    (r'/kaggle/input/notebooks/matildeteixeira/wearable-device-dataset/processed_data\.pkl', WDD_TAB),
    # Kaggle save paths
    (r'/kaggle/working/rnn_wrist_final\.pt',
     os.path.join(MODELS_DIR, 'rnn_wrist_final.pt')),
    (r'/kaggle/working/roc_data', os.path.join(ROOT, 'results', 'roc_data')),
    # Old anxolotl output path
    (r'/home/matildxi/Secretária/Documentos/Tese/anxolotl/output/processed_data_anxolotl\.pkl', ANX_TAB),
    # Old models_data path
    (r'/home/matildxi/Secretária/Documentos/Tese/anxolotl/models_data/',
     MODELS_DIR + '/'),
    # AutoGluon model path
    (r"ag_path\s*=\s*['\"][^'\"]+autogluon_wrist_final['\"]",
     f"ag_path = '{os.path.join(MODELS_DIR, 'autogluon_wrist_final')}'"),
]


def fix_source(src: str) -> str:
    for pattern, replacement in REPLACEMENTS:
        src = re.sub(pattern, replacement.replace('\\', '/'), src)
    return src


def fix_notebook(path: str) -> bool:
    with open(path) as f:
        nb = json.load(f)

    changed = False
    for cell in nb['cells']:
        src = cell.get('source', [])
        if isinstance(src, str):
            fixed = fix_source(src)
            if fixed != src:
                cell['source'] = fixed; changed = True
        else:
            fixed = [fix_source(line) for line in src]
            if fixed != src:
                cell['source'] = fixed; changed = True

    if changed:
        with open(path, 'w') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f'  FIXED  {os.path.relpath(path, ROOT)}')
    else:
        print(f'  ok     {os.path.relpath(path, ROOT)}')

    return changed


if __name__ == '__main__':
    notebooks = sorted(
        fp for fp in glob.glob(os.path.join(ROOT, '**', '*.ipynb'), recursive=True)
        if 'archive' not in fp and '.ipynb_checkpoints' not in fp
    )
    print(f'Checking {len(notebooks)} notebooks…\n')
    fixed_count = sum(fix_notebook(nb) for nb in notebooks)
    print(f'\n{fixed_count} notebook(s) updated.')
