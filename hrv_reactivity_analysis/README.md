# Delta HRV (baseline→stress) vs EADS-21 Ansiedade — análise exploratória

Análise descritiva isolada, sem impacto na dissertação ou nos resultados existentes.
**Nenhum ficheiro fora desta pasta foi modificado.** Os dados brutos e `Data_tese.xlsx`
foram só lidos (read-only). O pipeline de treino não foi corrido.

## Nota metodológica (desvio da tarefa original)

A tarefa pedia HRV "em bpm bruto" — mas bpm não é uma unidade de HRV (aplica-se a HR).
Além disso não existe uma única coluna "HRV" no pkl: há três métricas candidatas
(`rmssd_mean`, `sdnn_mean`, `ibi_mean`), todas armazenadas em **segundos**. Reportado ao
utilizador antes de avançar; decisão confirmada:
- Calcular as **3 métricas** (RMSSD, SDNN, IBI), em vez de escolher só uma.
- Reportar os deltas em **milissegundos** (convenção da literatura de HRV), não em bpm nem
  segundos brutos.

## Fonte do HRV

`processed_data/anxolotl/processed_data_anxolotl.pkl` → `df` (699 janelas, 37 sujeitos),
colunas `subject`, `stress_phase` (-1=baseline, 1=stress1, 2=stress2, 3=stress3),
`rmssd_mean`, `sdnn_mean`, `ibi_mean` (em segundos no ficheiro; convertidos ×1000 para ms
nesta análise).

**Confirmação de que é bruto, não normalizado:**
- Ordens de grandeza plausíveis para dados fisiológicos brutos: RMSSD ≈ 27ms, SDNN ≈ 35ms
  (aproximadamente), IBI ≈ 711ms (equivalente a ≈84 bpm, consistente com o `hr_mean` já usado
  em `hr_reactivity_analysis/`).
- O BL-norm (`_bl_train`/`_bl_test` em `scripts/anxolotl_loso.py`) é aplicado em tempo de
  treino sobre a matriz `X` carregada a partir deste pkl, genericamente a todas as colunas —
  não está pré-aplicado no ficheiro. Este pkl contém os valores pré-normalização, para RMSSD,
  SDNN e IBI tal como para HR.

## Definição do delta HRV

Por sujeito e por métrica (RMSSD, SDNN, IBI):
- `baseline_X_ms` = média da métrica sobre as janelas com `stress_phase == -1` (label 0), ×1000.
- `stress_X_ms` = média da métrica sobre as janelas com `stress_phase in {1, 2}` — **exclui
  explicitamente stress_phase 3** (13 janelas), consistente com o resto da tese.
- `delta_X_ms = stress_X_ms − baseline_X_ms`.

37 sujeitos canónicos (`p7`...`p44`, excluindo `p6`/`p24`) — todos com janelas de baseline e
de stress1/stress2 disponíveis, sem missing.

## Junção com EADS-21 Ansiedade

Reutilizada a subescala de Ansiedade já calculada e validada em
`eads21_analysis/eads21_vs_performance.csv` (soma bruta dos itens 2,4,7,9,15,19,20, mapeamento
`subject`↔`questionnaire_id` já auditado) — **não recalculada** aqui. Essa tabela foi
entretanto atualizada para incluir `p41` (ver nota em `eads21_analysis/README.md`: submissão
adicional `questionnaire_id=60` apareceu em `Data_tese.xlsx`, editado ao vivo durante a sessão
de trabalho, associada a `p41` por alinhamento de timestamp). Cobertura agora: **37/37**.

## Resultado — Correlação de Spearman (n=37)

| Métrica | rho | p |
|---|---|---|
| Delta RMSSD (ms) | 0.2357 | 0.1602 |
| Delta SDNN (ms) | 0.0874 | 0.6071 |
| Delta IBI (ms) | -0.1659 | 0.3264 |

- Delta RMSSD: média=0.59ms, sd=6.97, min=-13.49, max=16.05.
- Delta SDNN: média=1.71ms, sd=8.90, min=-12.50, max=29.70.
- Delta IBI: média=-11.21ms, sd=35.62, min=-93.03, max=53.49.

Ver `hrv_reactivity_scatter.png` (3 painéis, um por métrica).

## Interpretação (neutra, para a dissertação integrar depois)

Nenhuma das três métricas de reatividade de HRV (RMSSD, SDNN, IBI) correlaciona de forma
estatisticamente significativa com a subescala de Ansiedade da EADS-21 (todos os p > 0.16,
n=37). A correlação de RMSSD é a mais próxima de um efeito fraco-moderado (rho=0.24, ainda
assim não significativo), enquanto SDNN e IBI mostram correlações próximas de zero (ou
negativas, no caso do IBI). Este padrão é consistente com os resultados nulos já obtidos para
Delta HR (`hr_reactivity_analysis/`, rho=0.13, p=0.44) e para Macro-F1 (`eads21_analysis/`,
rho=0.26, p=0.12): globalmente, a ansiedade autorreportada (traço, semana anterior) não
mostra associação forte nem consistente com nenhuma medida fisiológica objetiva de
reatividade ao stress nesta amostra. Não é evidência de que a reatividade fisiológica não
exista — só de que, ao nível do sujeito e com este n, ela não é explicada pela ansiedade
autorreportada na EADS-21.

## Ficheiros nesta pasta

- `hrv_reactivity_per_subject.csv` — subject, baseline/stress/delta para RMSSD, SDNN e IBI
  (ms), 37 sujeitos.
- `hrv_reactivity_vs_eads21.csv` — o mesmo, com a coluna `eads21_anxiety` juntada.
- `hrv_reactivity_scatter.png` — 3 scatters (RMSSD, SDNN, IBI) vs Ansiedade EADS-21.
- `_stats.txt` — valores numéricos brutos usados neste README.
