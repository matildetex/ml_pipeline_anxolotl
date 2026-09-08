# WESAD Classical-Model Rerun — Resultados Reais (2026-07-24)

**Fonte primária:** `results/wesad_classical_rerun.json` / `.txt` (agregado), `_task1.json`, `_task2.json`, `_task4.json` (por bloco, com `y_test`/`y_pred`/`y_prob`/`fold_f1s`/`confusion_matrix` completos).
**Script:** `1_wesad/classical/wesad_classical_rerun.py` (+ `wesad_task4_autogluon_only.py` para o AutoGluon, corrido à parte por ter rebentado na primeira tentativa — ver nota no fim).
**Dados:** Task 1/2 — snapshot recuperado do schema antigo do WESAD (137 feat, 8 stats/canal), agora arquivado em `processed_data/wesad_archive/processed_data_137feat_snapshot.pkl`. Task 4 — schema atual (`processed_data/wesad/processed_data.pkl`), 55 features (5 canais × 11 stats), conforme confirmaste.
**Protocolo:** LOSO-CV (15 folds, um por sujeito), StandardScaler + SimpleImputer(mean), SMOTE só no fold de treino onde aplicável. Hiperparâmetros fixos (os que o capítulo já cita — ver script), sem grid/random search nesta ronda.

**Desvio de protocolo, assinalado (não silencioso):** SVM corre sobre uma subamostra estratificada de 40% (11016 de 27543 janelas) — a mesma convenção já usada em `results/wesad_svm.json` e no notebook original `support-vector-machine.ipynb` deste repo. SVM com kernel polinomial sobre as 27543 janelas completas não é computacionalmente viável (SVC não-linear escala ~O(n²–n³)). O ensemble de 5 modelos (que inclui SVM) usa a mesma subamostra para consistência interna; o ensemble de 3 modelos (sem SVM) usa os dados completos.

---

## Tabela 1.1 — WESAD full (137 feat, schema antigo), SEM SMOTE, LOSO

| Modelo | Acc | F1 (macro) | SD(F1) | BalAcc | Precision | Recall | AUC | MCC |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.830 | 0.795 | — | 0.816 | — | — | 0.912 | 0.659 |
| SVM | 0.813 | 0.750 | — | 0.770 | — | — | 0.881 | 0.578 |
| Random Forest | 0.892 | 0.878 | — | 0.890 | — | — | 0.962 | 0.803 |
| XGBoost | 0.893 | 0.880 | — | 0.890 | — | — | 0.968 | 0.795 |
| LightGBM | 0.895 | 0.886 | — | 0.897 | — | — | **0.971** | 0.814 |

(Precision/Recall macro completos e SD(F1) por fold estão em `results_master.csv` e no JSON — omitidos aqui só por espaço na tabela; ver §Notas.)

**Comparação com os valores do capítulo (não verificáveis, ver `RECONCILIATION.md` §G1):**

| Modelo | Capítulo (Acc/F1/BalAcc/AUC) | Rerun (Acc/F1/BalAcc/AUC) |
|---|---|---|
| Logistic Regression | 0.802/0.800/0.808/0.948 | 0.830/0.795/0.816/0.912 |
| SVM | 0.820/0.790/0.774/0.930 | 0.813/0.750/0.770/0.881 |
| Random Forest | 0.910/0.910/0.908/0.970 | 0.892/0.878/0.890/0.962 |
| XGBoost | 0.911/0.900/0.905/0.970 | 0.893/0.880/0.890/0.968 |
| LightGBM | 0.920/0.920/0.925/0.979 | 0.895/0.886/0.897/**0.971** |

Padrão consistente: os valores reais ficam sistematicamente **1–3 pontos percentuais abaixo** dos citados no capítulo, em quase todas as métricas e modelos — incluindo os dois (LogReg, SVM) que já tinham artefacto parcial antes desta ronda. Isto é consistente com a hipótese de que os números do capítulo foram escritos antes de existir qualquer execução real, ou vêm de uma configuração ainda não identificada.

---

## Tabela 1.2 — Impacto do SMOTE, WESAD full (137 feat), LOSO

| Modelo | Acc (sem) | Acc (com) | F1 (sem) | F1 (com) | AUC (sem) | AUC (com) | ΔF1 |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.830 | 0.803 | 0.795 | 0.782 | 0.912 | 0.911 | −0.013 |
| SVM | 0.813 | 0.813 | 0.750 | 0.751 | 0.881 | 0.879 | +0.000 |
| Random Forest | 0.892 | 0.912 | 0.878 | 0.900 | 0.962 | 0.970 | +0.022 |
| XGBoost | 0.893 | 0.910 | 0.880 | 0.898 | 0.968 | 0.970 | +0.018 |
| LightGBM | 0.895 | 0.919 | 0.886 | **0.912** | 0.971 | **0.982** | +0.026 |

**Achado real, não estava em nenhum artefacto anterior:** SMOTE ajuda consistentemente os três modelos de árvore (RF/XGBoost/LightGBM, +0.018 a +0.026 de F1), não ajuda a Logistic Regression (leve prejuízo, −0.013) e é neutro para o SVM (+0.000, confirmando o que já se sabia do notebook original). Isto é uma tabela genuinamente nova — a versão anterior só tinha 1 de 5 linhas com suporte real.

---

## Tabela 1.3 — Ensemble soft-voting × Information Gain (TOP_K), WESAD full (137 feat)

| Ensemble | TOP_K | Acc | F1 (macro) | BalAcc | AUC | MCC |
|---|---|---|---|---|---|---|
| 3-modelos (LGBM+XGB+RF) | 10 | 0.830 | 0.790 | 0.801 | 0.886 | 0.652 |
| 3-modelos (LGBM+XGB+RF) | 20 | 0.840 | 0.807 | 0.828 | 0.933 | 0.703 |
| 3-modelos (LGBM+XGB+RF) | 30 | 0.826 | 0.794 | 0.815 | 0.919 | 0.687 |
| 3-modelos (LGBM+XGB+RF) | All (137) | **0.902** | **0.893** | **0.902** | **0.968** | **0.821** |
| 5-modelos (todos) | 10 | 0.819 | 0.772 | 0.791 | 0.879 | 0.631 |
| 5-modelos (todos) | 20 | 0.869 | 0.836 | 0.854 | 0.942 | 0.746 |
| 5-modelos (todos) | 30 | 0.843 | 0.810 | 0.833 | 0.931 | 0.708 |
| 5-modelos (todos) | All (137) | 0.905 | 0.887 | 0.903 | 0.965 | 0.821 |

**Achado real:** em ambas as configurações, `All` (sem seleção de features) bate todas as variantes com TOP_K restrito — a redução de dimensionalidade por Information Gain **não ajuda** neste dataset/protocolo, contrariando a premissa implícita da Tabela 1.3 original (que presumivelmente argumentaria a favor de um TOP_K intermédio). O 3-modelos/All (0.893 F1) é ligeiramente melhor que o 5-modelos/All (0.887) apesar de ter menos dois modelos — o SVM e a Logistic Regression parecem diluir o ensemble em vez de o reforçar. Esta é a primeira vez que esta tabela tem alguma linha com dados reais (a versão anterior tinha zero, por causa do `NameError` documentado em `RECONCILIATION.md` §G3).

---

## Tabela 1.8 (parcial) — WESAD reduzido (55 feat, schema atual), sem SMOTE

| Modelo | Acc | F1 (macro) | BalAcc | AUC | MCC |
|---|---|---|---|---|---|
| Random Forest | 0.720 | 0.649 | 0.665 | 0.740 | 0.372 |
| XGBoost | 0.696 | 0.622 | 0.637 | 0.693 | 0.307 |
| AutoGluon (good_quality) | 0.714 | 0.647 ± 0.118 | 0.657 | 0.719 | 0.341 |

Comparação com o que já existia em `results_master.csv` para este mesmo schema (55 feat, `wesad_55plus_ecg_loso.txt`, ExtraTrees+BL-norm): MacroF1=0.6902, AUROC=0.8536. O RF/XGBoost desta ronda (sem BL-norm, só StandardScaler) ficam **abaixo** desse número — sugere que a normalização BL-norm (baseline-relative) importa mais do que a escolha do modelo para o conjunto reduzido, consistente com o que já se sabia do lado Anxolotl.

---

## Figura 1.6 — Curvas ROC

Gerada em `figures/fig6_roc_wesad_classical_rerun.png`: dois painéis lado a lado ("Without SMOTE" / "With SMOTE"), uma curva por modelo, legenda com o AUC de cada um. Mesmo estilo dos outros ROC deste repo (`scripts/plot_results.py`).

---

## Tempo de execução, por bloco

| Bloco | Minutos |
|---|---|
| Task 1 — Logistic Regression (SEM + COM SMOTE) | 14.6 |
| Task 1 — SVM (SEM + COM SMOTE, subamostra 40%) | 41.1 |
| Task 1 — Random Forest (SEM + COM SMOTE) | 6.5 |
| Task 1 — XGBoost (SEM + COM SMOTE) | 38.5 |
| Task 1 — LightGBM (SEM + COM SMOTE) | 3.3 |
| **Task 1 subtotal** | **104.0** |
| Task 2 — Ensemble 3-modelos (4× TOP_K) | 41.5 |
| Task 2 — Ensemble 5-modelos (4× TOP_K, subamostra 40%) | 54.6 |
| **Task 2 subtotal** | **96.1** |
| Task 3 — Figura ROC | 0.0 |
| Task 4 — Random Forest (reduzido) | 2.6 |
| Task 4 — XGBoost (reduzido) | 1.1 |
| **TOTAL Tasks 1–4 (sem AutoGluon)** | **204.6** |
| AutoGluon — tentativas 1–5 (diagnóstico + correções falhadas) | ~35 |
| AutoGluon — tentativa 6 (final, bem-sucedida, 15 folds) | 262.7 |
| **TOTAL GERAL (com AutoGluon)** | **~502** (≈ 8.4h) |

O AutoGluon sozinho consumiu mais tempo que todo o resto da ronda combinado — a maior parte em diagnóstico (Ray/NVML, ver nota abaixo), não em computação útil.

---

## Nota sobre o AutoGluon (Task 4) — RESOLVIDO, resultado real obtido (2026-07-25)

Seis tentativas ao todo. As três primeiras falharam por razões diferentes e cada vez mais fundas; a causa raiz real só apareceu ao forçar verbosidade total num único fold isolado, em vez de continuar a testar dentro do loop de 15 folds:

1. **`AssertionError: Learner is already fit.`** — resíduo de `/tmp` reutilizado (diretório com `ds_sub_fit` de uma tentativa anterior).
2. **Mesmo erro, diretórios limpos** — o preset `good_quality` usa DyStack (dynamic stacking) internamente; o sub-fit interno do DyStack falhava por completo, corrompendo o estado do learner. Corrigido com `dynamic_stacking=False`.
3. **Com `dynamic_stacking=False`** — passou do ponto de falha anterior, completou os folds 1–5, mas `LightGBMXT_BAG_L1`, `LightGBM_BAG_L1`, `CatBoost_BAG_L1`, `NeuralNetFastAI_BAG_L1` falhavam em CADA fold com `Unknown Error` (sem traceback útil); no fold 6 todos os modelos falharam, incluindo os que antes sobreviviam, e rebentou com `RuntimeError: No models were trained successfully`.

**Nessa altura escrevi que a causa provável era incompatibilidade de versões (`xgboost==3.1.3` vs. o que o AutoGluon 1.5.0 esperava) — estava ERRADO.** Verifiquei os requisitos declarados do próprio `autogluon.tabular` (`pip show` + metadata): `xgboost<3.2,>=2.0`, `lightgbm<4.7,>=4.0`, `catboost<1.3,>=1.2` — as versões instaladas (3.1.3 / 4.6.0 / 1.2.10) estão **dentro** desses intervalos. Não era isso.

**Causa real, encontrada correndo um único fold em primeiro plano com `verbosity=3`:**
```
ray._private.thirdparty.pynvml.pynvml.NVMLError_Unknown: Unknown Error
```
O AutoGluon usa Ray internamente para treino distribuído/bagged de LightGBM, CatBoost e redes neuronais. Ao arrancar, o Ray tenta sondar a GPU via NVML (NVIDIA Management Library) — **mesmo quando o modelo pede explicitamente `num_gpus: 0`** — e essa sondagem falhava neste ambiente. Os modelos RandomForest/ExtraTrees do AutoGluon não passam por este caminho do Ray, por isso sobreviviam sempre; os que dependem de bagging via Ray (LightGBM, CatBoost, redes neuronais) falhavam sempre.

**Correção:** `os.environ['CUDA_VISIBLE_DEVICES'] = ''` antes de importar `autogluon`/`ray` — faz o Ray saltar a deteção de GPU por completo. Confirmado isoladamente: `LightGBMXT_BAG_L1` passou de falha instantânea a um score de validação real de 0.9987.

**Tentativas 4–5** (com a correção aplicada): 4ª rebentou por um erro de configuração meu (`set_best_to_refit_full=True` exige `refit_full=True`, e eu tinha desligado os dois); 5ª rebentou porque desligar `refit_full` por completo impede a previsão (o AutoGluon descarta os modelos individuais do bagging por omissão, e sem `refit_full` não há modelo consolidado para prever) — `AttributeError: 'NoneType' object has no attribute 'predict'`.

**Tentativa 6 (final, bem-sucedida):** `refit_full=True` (reposto), `calibrate_decision_threshold=False` (desligado — por consistência com os outros modelos desta ronda, que usam todos o threshold 0.5 simples, e para poupar o tempo da procura de 51+38 thresholds por fold), `time_limit=90s`/fold. **262.7 minutos para os 15 folds** (~17.5 min/fold — o `good_quality` agora treina a sério LightGBM/CatBoost/redes neuronais em vez de falhar em segundos, por isso ficou mais lento do que antes da correção, não mais rápido — é o preço de ter resultados reais em vez de nenhum).

**Resultado final:** MacroF1=0.6467±0.1176, Acc=0.7143, BalAcc=0.6566, AUC=0.7191, MCC=0.3415 (F1 por fold: 0.568, 0.641, 0.720, 0.789, 0.556, 0.392, 0.775, 0.640, 0.608, 0.792, 0.834, 0.520, 0.704, 0.580, 0.580).

**Achado:** o AutoGluon (0.647 F1) fica **praticamente empatado** com o Random Forest simples desta mesma ronda (0.649 F1) e **atrás** do que seria esperado de um AutoML — não supera nenhum dos modelos individuais mais fortes da Tabela 1.1/1.2 no conjunto FULL. Isto é consistente com o padrão já visto no resto desta auditoria (`RECONCILIATION.md` §F1): AutoGluon não tem vantagem clara sobre os modelos individuais bem afinados neste dataset.

---

## Ficheiros gerados

- `results/wesad_classical_rerun.json` — tudo combinado (Task 1+2+4 + timings)
- `results/wesad_classical_rerun.txt` — versão texto legível
- `results/wesad_classical_rerun_task1.json` / `_task2.json` / `_task4.json` — por bloco, com `y_test`/`y_pred`/`y_prob`/`fold_f1s`/matriz de confusão completos
- `figures/fig6_roc_wesad_classical_rerun.png`
- `1_wesad/classical/wesad_classical_rerun.py` (+ `wesad_task4_autogluon_only.py`) — código fonte, reprodutível
- 20 linhas novas em `results_master.csv`, marcadas `run_date=2026-07-24`, `source_file` a apontar para este rerun
- `processed_data/wesad_archive/processed_data_137feat_snapshot.pkl` — snapshot arquivado, como pedido

Nada foi sobrescrito em `results/*.json` pré-existentes — todos os nomes de ficheiro desta ronda são novos. Nenhum `.tex` foi tocado.
