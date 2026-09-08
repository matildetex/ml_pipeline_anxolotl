# Delta HR (baseline→stress) vs EADS-21 Ansiedade — análise exploratória

Análise descritiva isolada, sem impacto na dissertação ou nos resultados existentes.
**Nenhum ficheiro fora desta pasta foi modificado.** Os dados brutos e `Data_tese.xlsx`
foram só lidos (read-only). O pipeline de treino não foi corrido.

## Fonte do HR

`processed_data/anxolotl/processed_data_anxolotl.pkl` → `df` (699 janelas, 37 sujeitos),
colunas `subject`, `stress_phase` (-1=baseline, 1=stress1, 2=stress2, 3=stress3) e `hr_mean`
(HR médio por janela).

**Confirmação de que é bruto (bpm), não normalizado:**
- `hr_mean` varia entre ~59 e ~134 bpm (média 86.0) — intervalo fisiológico plausível em bpm,
  não uma escala normalizada.
- O BL-norm (z-score por sujeito relativo à baseline, `_bl_train`/`_bl_test` em
  `scripts/anxolotl_loso.py`) é aplicado **em tempo de treino**, sobre a matriz `X` carregada
  a partir deste pkl — não está pré-aplicado no ficheiro. Este pkl contém os valores
  pré-normalização.

## Definição do delta HR

Por sujeito:
- `baseline_hr` = média de `hr_mean` sobre as janelas com `stress_phase == -1` (label 0).
- `stress_hr` = média de `hr_mean` sobre as janelas com `stress_phase in {1, 2}` — **exclui
  explicitamente stress_phase 3** (13 janelas no total), consistente com o resto da tese
  (`canonical_numbers_for_thesis.txt`: só stress1+stress2 = 352 janelas de stress).
- `delta_hr = stress_hr − baseline_hr` (bpm).

37 sujeitos canónicos (`p7`...`p44`, excluindo `p6`/`p24`) — todos com janelas de baseline e
de stress1/stress2 disponíveis, sem missing.

## Junção com EADS-21 Ansiedade

Reutilizada a subescala de Ansiedade já calculada e validada em
`eads21_analysis/eads21_vs_performance.csv` (soma bruta dos itens 2,4,7,9,15,19,20, mapeamento
`subject`↔`questionnaire_id` já auditado) — **não recalculada** aqui. Essa tabela foi
entretanto atualizada para incluir `p41` (ver nota em `eads21_analysis/README.md`: uma
submissão adicional, `questionnaire_id=60`, apareceu em `Data_tese.xlsx` — ficheiro editado ao
vivo durante a sessão de trabalho — e foi associada a `p41` por alinhamento de timestamp).
Cobertura agora: **37/37**.

## Resultado — Correlação de Spearman (n=37)

| Comparação | rho | p |
|---|---|---|
| **EADS-21 Ansiedade vs Delta HR (bpm)** | **0.1308** | **0.4405** |

Delta HR: média=1.79 bpm, sd=5.20, min=-8.05, max=16.52.

Ver `hr_reactivity_scatter.png`.

## Interpretação (neutra, para a dissertação integrar depois)

A correlação entre a subescala de Ansiedade da EADS-21 e a reatividade cardíaca objetiva
(delta HR baseline→stress) é **fraca e não estatisticamente significativa** (rho=0.13,
p=0.44, n=37) — mais fraca ainda do que a já observada entre EADS-21 Ansiedade e o Macro-F1
do modelo (rho=0.26, p=0.12, `eads21_analysis/`). Este é outro resultado nulo, mas
informativo: não há evidência, nesta amostra, de que sujeitos com ansiedade autorreportada
mais elevada na semana anterior tenham tido maior (ou menor) reatividade cardíaca objetiva
durante o teste de stress. É consistente com a literatura de psicofisiologia sobre a
dissociação entre ansiedade-traço autorreportada e reatividade autonómica aguda, e reforça
que, nesta amostra, as várias medidas de autorrelato (EADS-21, VAS) não se alinham fortemente
nem com o desempenho do modelo nem com a resposta fisiológica objetiva medida diretamente.

## Ficheiros nesta pasta

- `hr_reactivity_per_subject.csv` — subject, baseline_hr, stress_hr, delta_hr (37 sujeitos).
- `hr_reactivity_vs_eads21.csv` — o mesmo, com a coluna `eads21_anxiety` juntada.
- `hr_reactivity_scatter.png` — scatter Ansiedade EADS-21 (x) vs Delta HR (y).
- `_stats.txt` — valores numéricos brutos usados neste README.
