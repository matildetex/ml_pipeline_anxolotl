import json, copy

with open('cross-dataset.ipynb') as f:
    nb = json.load(f)

def make_code_cell(source_lines, cell_id):
    if isinstance(source_lines, str):
        source_lines = [source_lines]
    return {
        'cell_type': 'code',
        'execution_count': None,
        'id': cell_id,
        'metadata': {},
        'outputs': [],
        'source': source_lines,
    }

# ── 1. cross-dataset.ipynb — keep only cells 0-21 ─────────────────────────
nb_cross = copy.deepcopy(nb)
nb_cross['cells'] = nb_cross['cells'][:22]

with open('cross-dataset.ipynb', 'w') as f:
    json.dump(nb_cross, f, indent=1)
print(f"cross-dataset.ipynb: {len(nb_cross['cells'])} cells")

# ── 2. anxolotl-loso.ipynb — self-contained LOSO-CV notebook ──────────────
imports_cell = make_code_cell([
    "import pickle\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.metrics import (\n",
    "    f1_score, roc_auc_score, balanced_accuracy_score,\n",
    "    matthews_corrcoef, roc_curve, auc as sklearn_auc,\n",
    ")\n",
    "from sklearn.model_selection import LeaveOneGroupOut\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier\n",
    "import lightgbm as lgb\n",
], 'loso_imports')

data_cell = make_code_cell([
    "with open('output/processed_data_anxolotl.pkl', 'rb') as f:\n",
    "    data_anx = pickle.load(f)\n",
    "\n",
    "print(f\"Windows: {len(data_anx['y'])}\")\n",
    "print(f\"Subjects: {len(set(data_anx['groups']))}\")\n",
    "print(f\"Features: {len(data_anx['feature_cols'])}\")\n",
    "print(f\"Stress: {sum(data_anx['y'])}  Baseline: {sum(1 for v in data_anx['y'] if v==0)}\")\n",
], 'loso_data')

utils_cell = make_code_cell([
    "def find_threshold(y_true, y_prob):\n",
    "    fpr, tpr, thresholds = roc_curve(y_true, y_prob)\n",
    "    best = np.argmax(tpr - fpr)\n",
    "    return float(thresholds[best])\n",
    "\n",
    "def metrics_at_threshold(y_true, y_prob, threshold):\n",
    "    y_pred = (y_prob >= threshold).astype(int)\n",
    "    ba = balanced_accuracy_score(y_true, y_pred)\n",
    "    return {\n",
    "        'f1':      f1_score(y_true, y_pred, average='macro', zero_division=0),\n",
    "        'ba':      ba,\n",
    "        'bal_acc': ba,\n",
    "        'mcc':     matthews_corrcoef(y_true, y_pred),\n",
    "    }\n",
], 'loso_utils')

loso_cells = [imports_cell, data_cell, utils_cell] + [
    copy.deepcopy(nb['cells'][i]) for i in [22, 23, 24, 25]
]

nb_loso = copy.deepcopy(nb)
nb_loso['cells'] = loso_cells

with open('anxolotl-loso.ipynb', 'w') as f:
    json.dump(nb_loso, f, indent=1)
print(f"anxolotl-loso.ipynb: {len(nb_loso['cells'])} cells")
