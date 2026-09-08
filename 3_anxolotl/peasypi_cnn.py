"""
Adaptation of peasypi/Stress-Detection-From-Wearables CNN to Anxolotl.

Original: Gil-Martin et al. 2022 — Human Stress Detection With Wearable Sensors
Using Convolutional Neural Networks.
https://github.com/peasypi/Stress-Detection-From-Wearables

peasypi pipeline on WESAD:
    raw wrist signals (BVP@64Hz, EDA@4Hz, TEMP@4Hz, ACC@32Hz)
    → short subwindow FFT → average across subwindows
    → 2D CNN (signals × FFT_bins)
    → LOSO-CV, binary (baseline vs stress)

Anxolotl adaptation:
    5 pre-processed 1Hz channels (acc_wrist, hr, ibi, rmssd, sdnn)
    → FFT of full 60s window → 30 frequency bins (0..0.5 Hz)
    → same 2D CNN architecture
    → LOSO-CV, binary (baseline vs stress)

The 1Hz resolution captures physiologically relevant HRV frequencies:
  LF band: 0.04–0.15 Hz, HF band: 0.15–0.40 Hz (within Nyquist @ 0.5 Hz).

Published WESAD results (their os_scores_f1.csv, cannot retrain — disk full):
    E4    (BVP+EDA+TEMP+ACC): F1 = 0.878
    Fitbit (ACC+TEMP+EDA):    F1 = 0.894
    WearOS (ACC+TEMP):        F1 = 0.801
    Tizen  (ACC+TEMP+BVP):   F1 = 0.730

Run from: /home/matildxi/Secretária/Documentos/Tese/anxolotl/
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import scipy.fft
from sklearn.metrics import f1_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

SEQ_PATH    = 'processed_data/anxolotl/processed_data_anxolotl_seq.pkl'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Published peasypi WESAD results (cannot retrain — disk full at time of writing)
WESAD_PUBLISHED = {
    'E4':            0.878,   # BVP + EDA + TEMP + ACC (all sensors)
    'Fitbit':        0.894,   # ACC + TEMP + EDA
    'WearOS_watchOS': 0.801,  # ACC + TEMP (closest to our Anxolotl setup)
    'Tizen':         0.730,   # ACC + TEMP + BVP
}

# ── FFT feature extraction ────────────────────────────────────────────────────

def compute_fft_features(X_seq):
    """
    Apply FFT to each channel of each window.

    X_seq : (N, C, T)  — N windows, C channels, T timesteps
    Returns: (N, C, T//2)  — one-sided FFT magnitude per channel
    """
    N, C, T = X_seq.shape
    n_bins   = T // 2
    X_fft    = np.zeros((N, C, n_bins), dtype=np.float32)
    for i in range(N):
        for c in range(C):
            yf = scipy.fft.fft(X_seq[i, c])
            X_fft[i, c] = np.abs(yf[:n_bins])
    return X_fft  # (N, C, n_bins)

# ── CNN model (mirrors peasypi architecture) ──────────────────────────────────

def build_cnn(n_channels, n_bins):
    """
    2D CNN that treats [channels × frequency_bins] as an image.
    Kernel (1,3) convolves only along the frequency axis per channel.
    Same depth/width as peasypi repo.
    """
    inp = keras.Input(shape=(n_channels, n_bins, 1))
    x = keras.layers.Conv2D(64, (1, 3), activation='relu', padding='same')(inp)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv2D(64, (1, 3), activation='relu', padding='same')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv2D(64, (1, 3), activation='relu', padding='same')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    out = keras.layers.Dense(2, activation='softmax')(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

# ── LOSO-CV ───────────────────────────────────────────────────────────────────

def run_loso(X_fft, y, groups, verbose=True):
    """
    Leave-One-Subject-Out CV with the peasypi CNN.
    Returns per-subject F1 and AUC.
    """
    subjects = sorted(set(groups))
    n_channels, n_bins = X_fft.shape[1], X_fft.shape[2]

    rows = []
    for subj in subjects:
        test_mask  = groups == subj
        train_mask = ~test_mask

        if test_mask.sum() < 4 or len(set(y[test_mask])) < 2:
            if verbose:
                print(f'  {subj}: skipped (too few samples or single class)')
            continue

        X_tr = X_fft[train_mask][..., np.newaxis]
        y_tr = y[train_mask]
        X_te = X_fft[test_mask][..., np.newaxis]
        y_te = y[test_mask]

        model = build_cnn(n_channels, n_bins)
        model.fit(
            X_tr, y_tr,
            epochs=100, batch_size=64,
            validation_split=0.1, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True, monitor='val_loss'
            )],
        )

        y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
        y_prob = model.predict(X_te, verbose=0)[:, 1]
        f1  = f1_score(y_te, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_te, y_prob)
        except Exception:
            auc = float('nan')

        rows.append({'subject': subj, 'f1': f1, 'auc': auc,
                     'n_test': int(test_mask.sum()),
                     'n_stress': int((y_te == 1).sum()),
                     'n_base': int((y_te == 0).sum())})

        if verbose:
            print(f'  {subj}: F1={f1:.3f}  AUC={auc:.3f}'
                  f'  (test={test_mask.sum()}, stress={y_te.sum()})')

        del model
        tf.keras.backend.clear_session()

    return pd.DataFrame(rows)

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 1. Load pre-built sequence data
    print('Loading sequence data...')
    with open(SEQ_PATH, 'rb') as f:
        seq = pickle.load(f)
    X_seq  = seq['X_seq'].astype(np.float32)   # (N, 5, 60)
    y      = seq['y_seq'].astype(np.int64)
    groups = np.array([str(g) for g in seq['groups_seq']])

    print(f'  X_seq: {X_seq.shape}  (windows × channels × timesteps)')
    print(f'  Subjects: {len(set(groups))}  |  stress={y.sum()}  baseline={(y==0).sum()}')

    channels = ['acc_wrist', 'hr', 'ibi', 'rmssd', 'sdnn']

    # 2. FFT features
    print('\nComputing FFT features...')
    X_fft = compute_fft_features(X_seq)   # (N, 5, 30)
    print(f'  X_fft: {X_fft.shape}  (windows × channels × freq_bins)')
    print(f'  Frequency resolution: {1/60:.4f} Hz/bin  '
          f'(Nyquist: 0.5 Hz → LF/HF HRV bands captured)')

    # 3. LOSO-CV
    print(f'\nRunning LOSO-CV (peasypi CNN, {len(set(groups))} subjects)...')
    results_df = run_loso(X_fft, y, groups)

    mean_f1  = results_df['f1'].mean()
    mean_auc = results_df['auc'].mean()

    # 4. Summary
    print('\n' + '='*65)
    print('RESULTS — peasypi CNN adapted to Anxolotl')
    print('='*65)
    print(f'  Input shape  : {X_fft.shape[1]} channels × {X_fft.shape[2]} FFT bins')
    print(f'  Channels     : {channels}')
    print(f'  Window       : 60 s @ 1 Hz → 30 frequency bins (0–0.5 Hz)')
    print(f'  Subjects     : {len(results_df)}')
    print(f'\n  MEAN F1  = {mean_f1:.3f}')
    print(f'  MEAN AUC = {mean_auc:.3f}')
    print(f'\n  Per-subject F1:')
    for _, row in results_df.iterrows():
        print(f'    {row["subject"]}: {row["f1"]:.3f}')

    print('\n' + '='*65)
    print('COMPARISON — peasypi published WESAD vs Anxolotl (this work)')
    print('='*65)
    print(f'  {"Config":<30} {"F1":>8}  {"Dataset":>12}')
    print(f'  {"-"*52}')
    for cfg, f1 in WESAD_PUBLISHED.items():
        note = '← closest' if cfg == 'WearOS_watchOS' else ''
        print(f'  peasypi {cfg:<22} {f1:>8.3f}  WESAD    {note}')
    print(f'  {"-"*52}')
    print(f'  This work (5ch FFT-CNN, BL-norm)       {mean_f1:>8.3f}  Anxolotl')
    print(f'  Heo et al. best (ExtraTrees+BL-norm)   {"0.922":>8}  Anxolotl')
    print()
    print('  Note: peasypi uses raw 25–64 Hz signals + subwindow FFT;')
    print('  this work uses 1Hz pre-processed signals + single-window FFT.')
    print('  Direct comparison requires same signals and dataset.')

    # 5. Save results
    out_path = os.path.join(RESULTS_DIR, 'peasypi_cnn_anxolotl.csv')
    results_df.to_csv(out_path, index=False)
    print(f'\nPer-subject results saved → {out_path}')
