"""Shared LOSO training loop for DL models on WESAD sequences."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from tqdm import tqdm
from common import load_wesad_seq, report_metrics, save_results

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_WRIST_IDX = [0, 5, 6, 7, 8]


def dl_loso(model_fn, X_seq, y_seq, groups_seq, label, epochs=50, patience=10, lr=5e-4):
    """
    model_fn(n_channels) -> nn.Module
    X_seq: (N, C, T)  — channels first
    """
    logo = LeaveOneGroupOut()
    all_true, all_prob, fold_f1s = [], [], []

    splits = list(logo.split(X_seq, y_seq, groups_seq))
    pbar = tqdm(enumerate(splits), total=len(splits), desc=label[:35], ncols=90,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} folds [{elapsed}<{remaining}]', file=sys.stderr)
    for fold, (tr, te) in pbar:
        subj = groups_seq[te[0]]

        X_tr_raw, y_tr_raw = X_seq[tr], y_seq[tr]
        X_te_raw, y_te_raw = X_seq[te], y_seq[te]
        _, C, _ = X_tr_raw.shape

        X_tr_t = torch.tensor(X_tr_raw, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr_raw, dtype=torch.long)
        X_te_t = torch.tensor(X_te_raw, dtype=torch.float32)
        y_te_t = torch.tensor(y_te_raw, dtype=torch.long)

        class_counts = np.bincount(y_tr_raw)
        sample_weights = torch.tensor([1.0 / class_counts[y] for y in y_tr_raw], dtype=torch.float32)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, sampler=sampler)
        test_dl  = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=64)

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
                    print(f'           early stop epoch {epoch+1}')
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

        f1 = f1_score(y_te_raw, fold_preds, average='macro', zero_division=0)
        fold_f1s.append(f1)
        all_true.extend(y_te_raw)
        all_prob.extend(fold_probs)
        pbar.set_postfix(F1=f'{f1:.3f}', subj=str(subj))

    return report_metrics(np.array(all_true), np.array(all_prob), label, fold_f1s)


def run_dl_model(model_fn_wrist, model_fn_all, name, out_name):
    print(f'Loading WESAD sequences... device={DEVICE}')
    X_seq, y_seq, groups_seq = load_wesad_seq()

    all_results = []

    print(f'\n{"="*65}')
    print(f'  WESAD LOSO — {name} — WRIST (5 channels: {SEQ_WRIST_IDX})')
    print('='*65)
    X_wrist = X_seq[:, SEQ_WRIST_IDX, :]
    r = dl_loso(model_fn_wrist, X_wrist, y_seq, groups_seq,
                f'{name} WESAD — wrist (5 ch)')
    all_results.append(r)

    print(f'\n{"="*65}')
    print(f'  WESAD LOSO — {name} — ALL (16 channels)')
    print('='*65)
    r = dl_loso(model_fn_all, X_seq, y_seq, groups_seq,
                f'{name} WESAD — all (16 ch)')
    all_results.append(r)

    save_results(all_results, os.path.join(os.path.dirname(__file__), '..', 'results', f'{out_name}.json'))
