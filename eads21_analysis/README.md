# EADS-21 (subescala Ansiedade) vs desempenho do modelo — análise exploratória

Análise descritiva isolada, sem impacto na dissertação ou nos resultados existentes.
**Nenhum ficheiro fora desta pasta foi modificado.** `Data_tese.xlsx` foi só lido (read-only).

## Dados usados

- **`Data_tese.xlsx`, sheet `EADS-21`**: 40 linhas (39 originais + 1 submissão adicionada
  posteriormente — ver nota abaixo), 21 colunas de itens (0-3 cada) + coluna
  `ID (dado pelo investigador)` (aqui referida como `questionnaire_id`). O Excel só tem os
  itens individuais — as subescalas não estavam somadas, foram calculadas nesta análise.
- **`results/self_report_vs_loso_f1.csv`**: fornece o mapeamento já validado (sessão de
  2026-07-24, por alinhamento de timestamp) entre `questionnaire_id` (Excel) e `subject`
  (código de sujeito do LOSO, `p7`...`p44`), e a coluna `f1_et1000_wristonly` — Macro-F1 por
  sujeito do modelo canónico ET-1000 (mf=0.5, wrist-only, LOSO). Este mapeamento **não foi
  refeito** para os 36 sujeitos já cobertos por ele — foi reutilizado conforme confirmado com
  o utilizador.

## Nota — atualização com o sujeito `p41`

Na primeira versão desta análise, `p41` não tinha `questionnaire_id` (n=36). Investigação
posterior encontrou que `Data_tese.xlsx` estava aberto e foi editado ao vivo durante a sessão
de trabalho: uma nova submissão (`questionnaire_id=60`) apareceu nas três sheets
(`Questionário`, `VAS`, `EADS-21`), com timestamp **2026-05-20 15:33:20** — apenas 13s depois
do fim da sessão `p41` (`sessions/session_0041.csv`, fim 15:33:07 local), usando o mesmo
critério de alinhamento por timestamp da mapeamento original. Este par `p41 ↔ 60` foi
adicionado manualmente no script desta análise (não em `results/self_report_vs_loso_f1.csv`,
que continua intocado/read-only). Cobertura agora: **37/37** sujeitos canónicos.

## Cálculo da subescala de Ansiedade

Chave de subscoring padrão do DASS-21 / EADS-21 (Pais-Ribeiro, Honrado & Leal, 2004), pela
posição do item na ordem canónica (que corresponde à ordem das colunas na sheet):

- **Ansiedade** = soma bruta dos itens 2, 4, 7, 9, 15, 19, 20 (boca seca, dificuldade em
  respirar, tremores, medo de pânico/fazer figura ridícula, quase a entrar em pânico,
  alterações cardíacas sem exercício, assustado sem razão). Intervalo possível: 0-21.
- **Stress** (coluna extra) = soma bruta dos itens 1, 6, 8, 11, 12, 14, 18. Intervalo
  possível: 0-21.
- Sanity check: a soma dos 21 itens recalculada por nós coincide exatamente com a coluna
  `eads21_total` já existente em `results/self_report_vs_loso_f1.csv` para todas as linhas
  pré-existentes com match — confirma que o método de soma (bruto, sem inversão de itens,
  sem multiplicador ×2) está correto.

## Cobertura

37 sujeitos canónicos LOSO; **37 têm dados de EADS-21** (incluindo `p41`, ver nota acima).

## Resultado — Correlação de Spearman (n=37)

| Comparação | rho | p |
|---|---|---|
| **EADS-21 Ansiedade vs Macro-F1 (ET-1000)** | **0.2584** | **0.1225** |
| EADS-21 Stress vs Macro-F1 (ET-1000) — secundário | 0.1684 | 0.3190 |

EADS-21 Ansiedade: média=3.62, sd=3.09, min=0, max=12 (escala 0-21).
Macro-F1 (ET-1000): média=0.9169, sd=0.0895.

Ver `eads21_scatter.png`.

## Interpretação (neutra, para a dissertação integrar depois)

A correlação entre a subescala de Ansiedade da EADS-21 (traço/estado da semana anterior à
recolha) e a dificuldade de classificação por sujeito do modelo (Macro-F1 LOSO) é **fraca e
não estatisticamente significativa** (rho=0.26, p=0.12, n=37). Este é um resultado nulo, mas
informativo: não há evidência, nesta amostra, de que sujeitos com níveis de ansiedade
autorreportados mais elevados na semana anterior tenham sido sistematicamente mais fáceis ou
mais difíceis de classificar pelo modelo. É consistente com o resultado já obtido para o VAS
de stress momentâneo (rho=0.17, p=0.32, `results/self_report_vs_loso_f1_summary.txt`), o que
reforça a leitura de que a deteção fisiológica de stress e as medidas de autorrelato
(momentâneas ou de traço) capturam construtos pelo menos parcialmente dissociados a este
nível de sujeito. Como a EADS-21 mede a semana anterior (não o momento do teste), este
resultado deve ser lido como puramente exploratório/descritivo, não como validação (nem
invalidação) externa do modelo.

## Ficheiros nesta pasta

- `eads21_vs_performance.csv` — tabela por sujeito (subject, questionnaire_id,
  eads21_anxiety, eads21_stress, eads21_total, vas_stress_0_10, macro_f1_et1000).
- `eads21_scatter.png` — scatter Ansiedade EADS-21 (x) vs Macro-F1 (y).
- `_stats.txt` — valores numéricos brutos usados neste README.
