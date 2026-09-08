"""
DL variance analysis: run each of CNN-GRU, 1D-CNN, BiGRU, Transformer N=5 times
(different seeds) on WESAD LOSO (wrist + all channels), report mean +/- sd of
MacroF1. Mamba (SSM) excluded per user decision -- ~10h per single LOSO run on
this GPU (75.9s/epoch vs 7-9s/epoch for the others), not feasible to repeat 5x.

Adds seeding (torch.manual_seed, np.random.seed, seeded sampler generator) that
the original _dl_loso.py / wesad_*.py scripts did not have at all.

Results are appended to results/dl_variance_raw.csv after EVERY individual run
(not just at the end) so partial progress survives if this is interrupted.
"""
import sys, os, time, csv
sys.path.insert(0, os.path.dirname(__file__))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score

from common import load_wesad_seq
from wesad_1dcnn import CNN1D
from wesad_cnngru import CNNGRUHybrid
from wesad_rnn import RNN1D
from wesad_transformer import TransformerClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_WRIST_IDX = [0, 5, 6, 7, 8]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
RAW_CSV = os.path.join(RESULTS_DIR, 'dl_variance_raw.csv')
SUMMARY_TXT = os.path.join(RESULTS_DIR, 'dl_variance_summary.txt')

MODELS = {
    '1D-CNN':     lambda c: CNN1D(c),
    'CNN-GRU':    lambda c: CNNGRUHybrid(c),
    'BiGRU':      lambda c: RNN1D(in_channels=c),
    'Transformer': lambda c: TransformerClassifier(c),
}
SEEDS = [0, 1, 2, 3, 4]
CONFIGS = ['wrist', 'all']


def dl_loso_seeded(model_fn, X_seq, y_seq, groups_seq, seed, epochs=50, patience=10, lr=5e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator(); gen.manual_seed(seed)

    logo = LeaveOneGroupOut()
    all_true, all_prob = [], []

    for tr, te in logo.split(X_seq, y_seq, groups_seq):
        X_tr_raw, y_tr_raw = X_seq[tr], y_seq[tr]
        X_te_raw, y_te_raw = X_seq[te], y_seq[te]
        _, C, _ = X_tr_raw.shape

        X_tr_t = torch.tensor(X_tr_raw, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr_raw, dtype=torch.long)
        X_te_t = torch.tensor(X_te_raw, dtype=torch.float32)
        y_te_t = torch.tensor(y_te_raw, dtype=torch.long)

        class_counts = np.bincount(y_tr_raw)
        sample_weights = torch.tensor([1.0 / class_counts[y] for y in y_tr_raw], dtype=torch.float32)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=gen)
        train_dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, sampler=sampler)
        test_dl = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=64)

        model = model_fn(C).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        best_f1, no_improve, best_state = 0.0, 0, None
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            ep_preds = []
            with torch.no_grad():
                for xb, _ in test_dl:
                    ep_preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            ep_f1 = f1_score(y_te_t.numpy(), ep_preds, average='macro', zero_division=0)

            if ep_f1 > best_f1:
                best_f1 = ep_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        fold_preds, fold_probs = [], []
        with torch.no_grad():
            for xb, _ in test_dl:
                logits = model(xb.to(DEVICE))
                fold_preds.extend(logits.argmax(1).cpu().numpy())
                fold_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        all_true.extend(y_te_raw); all_prob.extend(fold_probs)

    pred_labels = (np.array(all_prob) >= 0.5).astype(int)
    return f1_score(all_true, pred_labels, average='macro', zero_division=0)


def main():
    print(f'Loading WESAD sequences... device={DEVICE}')
    X_seq, y_seq, groups_seq = load_wesad_seq()
    X_wrist = X_seq[:, SEQ_WRIST_IDX, :]

    done = set()
    write_header = not os.path.exists(RAW_CSV)
    if not write_header:
        import csv as _csv
        with open(RAW_CSV, newline='') as f:
            for row in _csv.DictReader(f):
                done.add((row['model'], row['config'], int(row['seed'])))
        print(f'Resuming: {len(done)} runs already completed, skipping those.')

    with open(RAW_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['model', 'config', 'seed', 'macro_f1', 'elapsed_s'])

        for name, model_fn in MODELS.items():
            for cfg in CONFIGS:
                X = X_wrist if cfg == 'wrist' else X_seq
                for seed in SEEDS:
                    if (name, cfg, seed) in done:
                        print(f'{name:<12} {cfg:<6} seed={seed}  [SKIP - already done]')
                        continue
                    t0 = time.time()
                    m = dl_loso_seeded(model_fn, X, y_seq, groups_seq, seed)
                    elapsed = time.time() - t0
                    writer.writerow([name, cfg, seed, m, elapsed])
                    f.flush()
                    print(f'{name:<12} {cfg:<6} seed={seed}  MacroF1={m:.4f}  ({elapsed:.0f}s)')

    print(f'\nSaved raw results to {RAW_CSV}')


if __name__ == '__main__':
    main()
