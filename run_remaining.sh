#!/bin/bash
# Corre apenas os modelos que faltam (transformer, mamba, dann, wdd, cross-dataset, anxolotl, plots, xai)
# Usar depois de instalar o driver NVIDIA e reiniciar.

_DEFAULT_PY=/home/matildxi/anaconda3/envs/ml/bin/python3
PYTHON="${PYTHON:-$_DEFAULT_PY}"
command -v "$PYTHON" &>/dev/null || PYTHON=python3

ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$ROOT/scripts"
RESULTS="$ROOT/results"
CROSS="$ROOT/2_cross_dataset"
LOG="$RESULTS/run_remaining_log.txt"

NCPUS=$(nproc 2>/dev/null || echo 4)
CPU_ENV="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$NCPUS MKL_NUM_THREADS=$NCPUS OPENBLAS_NUM_THREADS=$NCPUS"
GPU_ENV="env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$RESULTS" "$ROOT/figures"
exec > >(tee -a "$LOG") 2>&1

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DISABLE_IMPORT_WARNING=1
export PYTHONWARNINGS="ignore::FutureWarning:ray"

START_TIME=$SECONDS
TOTAL_STEPS=10
DONE_STEPS=0

progress() {
    local elapsed=$(( SECONDS - START_TIME ))
    local pct=$(( DONE_STEPS * 100 / TOTAL_STEPS ))
    local filled=$(( pct * 28 / 100 ))
    local empty=$(( 28 - filled ))
    local bar=""; for i in $(seq 1 $filled 2>/dev/null); do bar="${bar}█"; done
                  for i in $(seq 1 $empty  2>/dev/null); do bar="${bar}░"; done
    local el_h=$(( elapsed / 3600 )); local el_m=$(( (elapsed % 3600) / 60 ))
    if [ "$DONE_STEPS" -gt 0 ]; then
        local rem=$(( elapsed * (TOTAL_STEPS - DONE_STEPS) / DONE_STEPS ))
        local re_h=$(( rem / 3600 )); local re_m=$(( (rem % 3600) / 60 ))
        printf "\n  [%s] %d/%d  %d%%  | elapsed %dh%02dm | restante ~%dh%02dm\n" \
            "$bar" "$DONE_STEPS" "$TOTAL_STEPS" "$pct" "$el_h" "$el_m" "$re_h" "$re_m"
    else
        printf "\n  [%s] %d/%d  0%%  | elapsed %dh%02dm\n" \
            "$bar" "$DONE_STEPS" "$TOTAL_STEPS" "$el_h" "$el_m"
    fi
}

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  run_remaining.sh — $(date)"
echo "  Python: $($PYTHON --version 2>&1)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'não detectada')"
echo "════════════════════════════════════════════════════════════════"

run_model() {
    local name="$1" script="$2" device_env="${3:-}"
    local out="$RESULTS/$name.txt"
    local t0=$SECONDS
    echo ""; echo "  ── $name ──"
    ${device_env} $PYTHON "$script" > "$out"
    local status=$? elapsed_model=$(( SECONDS - t0 ))
    if [ $status -eq 0 ]; then
        echo "  OK  (${elapsed_model}s)  →  $out"
        grep -E "Macro F1|AUC|F1@opt|Bal" "$out" | tail -6 || true
    else
        echo "  ERRO (${elapsed_model}s)  — ver $out"
        tail -10 "$out"
    fi
    DONE_STEPS=$(( DONE_STEPS + 1 )); progress
}

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  FASE 1 (continuação) — WESAD DL  (GPU)"
echo "════════════════════════════════════════════════════════════════"

run_model "wesad_transformer" "$SCRIPTS/wesad_transformer.py"  "$GPU_ENV"
run_model "wesad_mamba"       "$SCRIPTS/wesad_mamba.py"        "$GPU_ENV"
run_model "wesad_dann"        "$SCRIPTS/wesad_dann.py"         "$GPU_ENV"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  FASE 2 — WDD LOSO"
echo "════════════════════════════════════════════════════════════════"

run_model "wdd_classical" "$SCRIPTS/wdd_classical.py" "$CPU_ENV"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  FASE 3 — Cross-dataset"
echo "════════════════════════════════════════════════════════════════"

echo ""; echo "  ── WDD → Anxolotl  (RF+ET, 60 feat) ──"
(cd "$CROSS" && $PYTHON cross_dataset_v4.py) > "$RESULTS/cross_dataset_wdd.txt" 2>&1
[ $? -eq 0 ] && echo "  OK" && grep -E "F1=|melhor" "$RESULTS/cross_dataset_wdd.txt" | tail -5 || echo "  ERRO — ver $RESULTS/cross_dataset_wdd.txt"
DONE_STEPS=$(( DONE_STEPS + 1 )); progress

echo ""; echo "  ── WESAD → Anxolotl  (ET BL-norm, 40 feat) ──"
(cd "$CROSS" && $PYTHON cross_dataset_blnorm.py) > "$RESULTS/cross_dataset_wesad.txt" 2>&1
[ $? -eq 0 ] && echo "  OK" && grep -E "Macro F1|ROC-AUC" "$RESULTS/cross_dataset_wesad.txt" | tail -4 || echo "  ERRO — ver $RESULTS/cross_dataset_wesad.txt"
DONE_STEPS=$(( DONE_STEPS + 1 )); progress

echo ""; echo "  ── DANN  (WDD/WESAD → Anxolotl) ──"
(cd "$CROSS" && $PYTHON cross_dataset_dann.py) > "$RESULTS/cross_dataset_dann.txt" 2>&1
[ $? -eq 0 ] && echo "  OK" && grep -E "F1=" "$RESULTS/cross_dataset_dann.txt" | tail -6 || echo "  ERRO — ver $RESULTS/cross_dataset_dann.txt"
DONE_STEPS=$(( DONE_STEPS + 1 )); progress

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  FASE 4 — Anxolotl LOSO"
echo "════════════════════════════════════════════════════════════════"

run_model "anxolotl_loso" "$SCRIPTS/anxolotl_loso.py" "$CPU_ENV"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  FASE 5 — Gráficos"
echo "════════════════════════════════════════════════════════════════"

$CPU_ENV $PYTHON "$SCRIPTS/plot_results.py" 2>&1 | tee "$RESULTS/plots_log.txt"
DONE_STEPS=$(( DONE_STEPS + 1 )); progress

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  FASE 6 — XAI / SHAP"
echo "════════════════════════════════════════════════════════════════"

$GPU_ENV $PYTHON "$SCRIPTS/xai.py" 2>&1 | tee "$RESULTS/xai_log.txt"
[ $? -eq 0 ] && echo "  OK — figuras em $ROOT/figures/shap/" || echo "  ERRO — ver $RESULTS/xai_log.txt"
DONE_STEPS=$(( DONE_STEPS + 1 )); progress

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  SUMÁRIO FINAL"
echo "════════════════════════════════════════════════════════════════"

$PYTHON - "$RESULTS" <<'PYEOF'
import json, os, glob, sys
results_dir = sys.argv[1]
rows = []
for jf in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
    try:
        with open(jf) as f: data = json.load(f)
        if not isinstance(data, list): data = [data]
        for r in data:
            if r is None: continue
            rows.append(r)
    except Exception: pass

if not rows:
    print("  Nenhum resultado JSON encontrado.")
else:
    print(f"\n{'Modelo':<55} {'F1':>6}  {'AUC':>6}  {'F1@opt':>7}  {'BalAcc':>7}")
    print('─' * 85)
    for r in rows:
        lbl   = str(r.get('label', '?'))[:55]
        f1    = r.get('f1');    f1    = f'{f1:.3f}'    if f1    is not None else '  —  '
        auc   = r.get('auc');   auc   = f'{auc:.3f}'   if auc   is not None else '  —  '
        f1opt = r.get('f1opt'); f1opt = f'{f1opt:.3f}' if f1opt is not None else '  —  '
        ba    = r.get('ba');    ba    = f'{ba:.3f}'     if ba    is not None else '  —  '
        print(f'{lbl:<55} {f1:>6}  {auc:>6}  {f1opt:>7}  {ba:>7}')
PYEOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Concluído: $(date)"
echo "════════════════════════════════════════════════════════════════"
