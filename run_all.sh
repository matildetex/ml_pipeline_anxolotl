#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# run_all.sh — Pipeline completo de detecção de stress/ansiedade
#
# FLUXO:
#  1. WESAD LOSO        — todos os modelos, wrist (56 feat) + all (137 feat)
#  2. WDD LOSO          — modelos clássicos, 60 feat comuns + 90 feat total
#  3. Cross-dataset     — treino WDD/WESAD → teste Anxolotl (zero-shot)
#  4. Anxolotl LOSO     — todos os modelos clássicos + BL-norm (40 + 60 feat)
#  5. Gráficos          — comparações, curvas ROC, heatmaps, etc.
#
# Resultados: results/*.txt  results/*.json
# Figuras:    figures/*.png
#
# Nunca pára em erro — falhas são registadas e a execução continua.
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_PY=/home/matildxi/anaconda3/envs/ml/bin/python3
PYTHON="${PYTHON:-$_DEFAULT_PY}"
command -v "$PYTHON" &>/dev/null || PYTHON=python3

ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$ROOT/scripts"
RESULTS="$ROOT/results"
CROSS="$ROOT/2_cross_dataset"
LOG="$RESULTS/run_log.txt"

# ── CPU / GPU configuration ───────────────────────────────────────────────────
NCPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# CPU models: hide GPU so no CUDA initialisation overhead, use all cores
CPU_ENV="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$NCPUS MKL_NUM_THREADS=$NCPUS OPENBLAS_NUM_THREADS=$NCPUS"

# GPU models: expose all GPUs (or override with CUDA_VISIBLE_DEVICES=0 if needed)
GPU_ENV="env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$RESULTS" "$ROOT/figures"
exec > >(tee -a "$LOG") 2>&1

# ── Progress tracking ─────────────────────────────────────────────────────────
START_TIME=$SECONDS
TOTAL_STEPS=19   # 6 CPU-WESAD + 6 GPU-WESAD + 1 WDD + 3 cross-dataset + 1 Anxolotl + 1 XAI + 1 plots
DONE_STEPS=0

# Suppress Ray internal metrics/telemetry noise (does not affect results)
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DISABLE_IMPORT_WARNING=1
export PYTHONWARNINGS="ignore::FutureWarning:ray"

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
echo "  run_all.sh — $(date)"
echo "  Python: $($PYTHON --version 2>&1)"
echo "  CPUs: $NCPUS  |  Device CPU: $CPU_ENV"
echo "  Device GPU: $GPU_ENV"
echo "════════════════════════════════════════════════════════════════"

run_model() {
    local name="$1"
    local script="$2"
    local device_env="${3:-}"   # CPU_ENV or GPU_ENV
    local out="$RESULTS/$name.txt"
    local t0=$SECONDS
    echo ""
    echo "  ── $name ──"
    # stdout → ficheiro de resultados; stderr (tqdm) → terminal visível
    ${device_env} $PYTHON "$script" > "$out"
    local status=$? elapsed_model=$(( SECONDS - t0 ))
    if [ $status -eq 0 ]; then
        echo "  OK  (${elapsed_model}s)  →  $out"
        grep -E "Macro F1|AUC|F1@opt|Bal" "$out" | tail -6 || true
    else
        echo "  ERRO (${elapsed_model}s)  — ver $out"
        tail -10 "$out"
    fi
    DONE_STEPS=$(( DONE_STEPS + 1 ))
    progress
}

# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — WESAD LOSO
# Benchmark: todos os modelos, features wrist (56) vs todas (137)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  FASE 1 — WESAD LOSO  (27 543 janelas, 15 sujeitos, lab stress)"
echo "════════════════════════════════════════════════════════════════"

# ── CPU: modelos clássicos ────────────────────────────────────────────────────
# wesad_{logreg,svm,rf,et}.py arquivados (archive/wesad_{logreg,svm,rf,et}_EXPLORATION.py) --
# confirmado 2026-08-31 que não são fonte da Tabela 4.9 (a fonte real é
# 1_wesad/full/*.ipynb, grid search/Optuna). Correr os notebooks manualmente
# para reproduzir/atualizar essas linhas.
run_model "wesad_ensemble"    "$SCRIPTS/wesad_ensemble.py"     "$CPU_ENV"
# wesad_autogluon.py arquivado (archive/wesad_autogluon_BUGGED_nvml_ray.py) --
# tinha o bug NVML/Ray (só RF/ET treinavam). Usar 1_wesad/classical/wesad_autogluon_rerun_fixed.py
# para reproduzir/atualizar o AutoGluon corrigido.

# ── GPU: modelos deep learning ────────────────────────────────────────────────
run_model "wesad_1dcnn"       "$SCRIPTS/wesad_1dcnn.py"        "$GPU_ENV"
run_model "wesad_cnngru"      "$SCRIPTS/wesad_cnngru.py"       "$GPU_ENV"
run_model "wesad_rnn"         "$SCRIPTS/wesad_rnn.py"          "$GPU_ENV"
run_model "wesad_transformer" "$SCRIPTS/wesad_transformer.py"  "$GPU_ENV"
run_model "wesad_mamba"       "$SCRIPTS/wesad_mamba.py"        "$GPU_ENV"
run_model "wesad_dann"        "$SCRIPTS/wesad_dann.py"         "$GPU_ENV"

# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — WDD LOSO
# Validação intermédia: modelos clássicos num dataset diferente (apenas pulso)
# 60 feat comuns com Anxolotl vs 90 feat totais WDD
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  FASE 2 — WDD LOSO  (5 595 janelas, 16 sujeitos, lab stress)"
echo "════════════════════════════════════════════════════════════════"

run_model "wdd_classical" "$SCRIPTS/wdd_classical.py" "$CPU_ENV"

# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — Cross-dataset (treino WDD/WESAD → teste Anxolotl)
# Zero-shot: modelo nunca viu dados Anxolotl durante o treino
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  FASE 3 — Cross-dataset  (treino lab → teste naturalistic)"
echo "════════════════════════════════════════════════════════════════"

echo ""
echo "  ── WDD → Anxolotl  (RF+ET, 60 feat) ──"
(cd "$CROSS" && $PYTHON cross_dataset_v4.py) > "$RESULTS/cross_dataset_wdd.txt" 2>&1
[ $? -eq 0 ] && echo "  OK" && grep -E "F1=|melhor" "$RESULTS/cross_dataset_wdd.txt" | tail -5 || echo "  ERRO — ver $RESULTS/cross_dataset_wdd.txt"
DONE_STEPS=$(( DONE_STEPS + 1 )); progress

echo ""
echo "  ── WESAD → Anxolotl  (ET BL-norm, 40 feat) ──"
(cd "$CROSS" && $PYTHON cross_dataset_blnorm.py) > "$RESULTS/cross_dataset_wesad.txt" 2>&1
[ $? -eq 0 ] && echo "  OK" && grep -E "Macro F1|ROC-AUC" "$RESULTS/cross_dataset_wesad.txt" | tail -4 || echo "  ERRO — ver $RESULTS/cross_dataset_wesad.txt"
DONE_STEPS=$(( DONE_STEPS + 1 )); progress

echo ""
echo "  ── DANN  (WDD/WESAD → Anxolotl, domain adaptation) ──"
(cd "$CROSS" && $PYTHON cross_dataset_dann.py) > "$RESULTS/cross_dataset_dann.txt" 2>&1
[ $? -eq 0 ] && echo "  OK" && grep -E "F1=" "$RESULTS/cross_dataset_dann.txt" | tail -6 || echo "  ERRO — ver $RESULTS/cross_dataset_dann.txt"
DONE_STEPS=$(( DONE_STEPS + 1 )); progress

# ══════════════════════════════════════════════════════════════════════════════
# FASE 4 — Anxolotl LOSO
# Tecto in-domain: todos os modelos clássicos com BL-norm
# 40 feat (comuns WESAD) vs 60 feat (todas Anxolotl)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  FASE 4 — Anxolotl LOSO  (712 janelas, 37 sujeitos, ansiedade)"
echo "════════════════════════════════════════════════════════════════"

run_model "anxolotl_loso" "$SCRIPTS/anxolotl_loso.py" "$CPU_ENV"

# ══════════════════════════════════════════════════════════════════════════════
# FASE 5 — Gráficos
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  FASE 5 — Geração de gráficos"
echo "════════════════════════════════════════════════════════════════"

$CPU_ENV $PYTHON "$SCRIPTS/plot_results.py" 2>&1 | tee "$RESULTS/plots_log.txt"
DONE_STEPS=$(( DONE_STEPS + 1 )); progress

# ══════════════════════════════════════════════════════════════════════════════
# FASE 6 — XAI / SHAP
# WESAD wrist+all | WDD 60+90 | Anxolotl 40+60 | Cross-dataset comparison
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  FASE 6 — XAI / SHAP  (RF + ET em todos os datasets)"
echo "════════════════════════════════════════════════════════════════"

$GPU_ENV $PYTHON "$SCRIPTS/xai.py" 2>&1 | tee "$RESULTS/xai_log.txt"
[ $? -eq 0 ] && echo "  OK — figuras em $ROOT/figures/shap/" \
             || echo "  ERRO — ver $RESULTS/xai_log.txt"
DONE_STEPS=$(( DONE_STEPS + 1 )); progress

# ══════════════════════════════════════════════════════════════════════════════
# SUMÁRIO FINAL
# ══════════════════════════════════════════════════════════════════════════════
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
        with open(jf) as f:
            data = json.load(f)
        if not isinstance(data, list): data = [data]
        for r in data:
            if r is None: continue
            rows.append(r)
    except Exception:
        pass

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
echo "  Resultados: $RESULTS"
echo "  Figuras:    $ROOT/figures"
echo "════════════════════════════════════════════════════════════════"
