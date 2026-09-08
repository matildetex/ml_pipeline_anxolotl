"""
EADS-21 (subescala Ansiedade) vs desempenho do modelo (Macro-F1, ET-1000 wrist-only, LOSO)
— Correlacao de Spearman. Reproduz os numeros citados na tese, secao 4.6.3.

ORIGEM DOS DADOS (ambos read-only, nao modificados por este script):
  - Data_tese.xlsx, sheet "EADS-21": respostas brutas ao questionario (40 linhas = 39
    submissoes originais + 1 adicionada durante a sessao de trabalho de 2026-08-17, ver
    nota "p41" abaixo). Colunas = 21 itens (0-3 cada), na ordem canonica do
    DASS-21/EADS-21 (Pais-Ribeiro, Honrado & Leal, 2004). O ficheiro so tem os itens
    individuais; as subescalas (Ansiedade, Stress) sao calculadas aqui por soma bruta,
    sem inversao de itens nem multiplicador x2.
  - results/self_report_vs_loso_f1.csv: mapeamento ja validado (sessao de 2026-07-24,
    por alinhamento de timestamp) entre questionnaire_id (Excel) e subject (codigo de
    sujeito do LOSO, p7...p44), e a coluna f1_et1000_wristonly = Macro-F1 por sujeito
    do modelo canonico ET-1000 (mf=0.5, wrist-only, LOSO). Esse mapeamento em si NAO e
    recalculado aqui, apenas reutilizado.

NOTA - sujeito p41:
  results/self_report_vs_loso_f1.csv tem uma linha para p41 com f1_et1000_wristonly
  preenchido mas questionnaire_id/vas/eads21_total em branco (p41 nao tinha resposta
  EADS-21 mapeada quando esse CSV foi gerado). Investigacao posterior (2026-08-17)
  encontrou que Data_tese.xlsx estava aberto e foi editado ao vivo durante essa sessao:
  uma submissao nova (questionnaire_id=60) apareceu com timestamp 2026-05-20 15:33:20,
  13s depois do fim da sessao p41 (sessions/session_0041.csv, fim 15:33:07 local) —
  mesmo criterio de alinhamento por timestamp do mapeamento original. Esse par
  p41 <-> questionnaire_id 60 e adicionado manualmente abaixo (MANUAL_ID_OVERRIDES),
  documentado aqui em vez de re-escrever o CSV de mapeamento original.

VALIDACAO: os valores (rho, p, n) produzidos por este script devem coincidir com
eads21_analysis/_stats.txt e com a Tabela/texto da tese em 4.6.3
(rho~=0.2584, p~=0.1225, n=37 para Ansiedade vs Macro-F1).
"""

import pandas as pd
from scipy.stats import spearmanr

DATA_XLSX = "../Data_tese.xlsx"
SELF_REPORT_CSV = "../results/self_report_vs_loso_f1.csv"

OUT_CSV = "eads21_vs_performance.csv"
OUT_STATS = "_stats.txt"

# subject -> questionnaire_id pairs missing from SELF_REPORT_CSV, added manually.
# See docstring note "p41" above for the justification (timestamp alignment).
MANUAL_ID_OVERRIDES = {
    "p41": 60,
}

# Item numbers (1-indexed, canonical EADS-21/DASS-21 order) per subscale.
ANXIETY_ITEMS = [2, 4, 7, 9, 15, 19, 20]
STRESS_ITEMS = [1, 6, 8, 11, 12, 14, 18]


def load_eads21_items():
    df = pd.read_excel(DATA_XLSX, sheet_name="EADS-21")
    id_col = "ID (dado pelo investigador)"
    # Item columns start right after timestamp/id/date (3 leading non-item columns),
    # in canonical item order 1..21.
    item_cols = df.columns[3:24]
    assert len(item_cols) == 21, f"esperados 21 itens, encontrados {len(item_cols)}"

    out = pd.DataFrame({"questionnaire_id": df[id_col]})
    out["eads21_anxiety"] = df[[item_cols[i - 1] for i in ANXIETY_ITEMS]].sum(axis=1)
    out["eads21_stress"] = df[[item_cols[i - 1] for i in STRESS_ITEMS]].sum(axis=1)
    out["eads21_total_recomputed"] = df[item_cols].sum(axis=1)
    return out


def load_vas_for_override(questionnaire_id):
    # The mapping CSV has no VAS value for a manually-overridden subject (it predates
    # the override), so pull it directly from the VAS sheet for that questionnaire_id.
    vas = pd.read_excel(DATA_XLSX, sheet_name="VAS")
    id_col = [c for c in vas.columns if "ID" in c][0]
    val_col = "De 0 a 10, quão stressado se sentiu durante este teste"
    row = vas[vas[id_col] == questionnaire_id]
    assert len(row) == 1, f"questionnaire_id={questionnaire_id}: esperada 1 linha na sheet VAS"
    return float(row[val_col].iloc[0])


def main():
    eads21 = load_eads21_items()

    mapping = pd.read_csv(SELF_REPORT_CSV)  # subject, questionnaire_id, vas_stress_0_10, eads21_total, f1_et1000_wristonly

    # Split off p41 (no questionnaire_id in the mapping file) and patch it in manually.
    has_id = mapping["questionnaire_id"].notna()
    mapped, unmapped = mapping[has_id].copy(), mapping[~has_id].copy()
    for subj, qid in MANUAL_ID_OVERRIDES.items():
        rows = unmapped[unmapped["subject"] == subj]
        assert len(rows) == 1, f"{subj}: esperada 1 linha sem questionnaire_id, encontradas {len(rows)}"
        unmapped.loc[unmapped["subject"] == subj, "questionnaire_id"] = qid
        unmapped.loc[unmapped["subject"] == subj, "vas_stress_0_10"] = load_vas_for_override(qid)
    mapping = pd.concat([mapped, unmapped], ignore_index=True)
    mapping["questionnaire_id"] = mapping["questionnaire_id"].astype(int)

    merged = mapping.merge(eads21, on="questionnaire_id", how="left")
    assert merged["eads21_anxiety"].notna().all(), "sujeito(s) sem EADS-21 apos o merge"

    # Sanity check: soma recalculada dos 21 itens == eads21_total ja existente no CSV de
    # mapeamento, para as linhas que ja tinham eads21_total (exclui p41, que nao tinha).
    has_total = merged["eads21_total"].notna()
    mismatch = merged.loc[has_total & (merged["eads21_total_recomputed"] != merged["eads21_total"])]
    assert mismatch.empty, f"eads21_total nao bate para: {mismatch['subject'].tolist()}"

    result = merged[[
        "subject", "questionnaire_id", "eads21_anxiety", "eads21_stress",
        "eads21_total_recomputed", "vas_stress_0_10", "f1_et1000_wristonly",
    ]].rename(columns={
        "eads21_total_recomputed": "eads21_total",
        "f1_et1000_wristonly": "macro_f1_et1000",
    })
    result["_subj_num"] = result["subject"].str.lstrip("p").astype(int)
    result = result.sort_values("_subj_num").drop(columns="_subj_num").reset_index(drop=True)

    result.to_csv(OUT_CSV, index=False)

    rho, p = spearmanr(result["eads21_anxiety"], result["macro_f1_et1000"])
    rho_stress, p_stress = spearmanr(result["eads21_stress"], result["macro_f1_et1000"])
    n = len(result)

    with open(OUT_STATS, "w") as f:
        f.write(f"n={n}\n")
        f.write(f"rho={rho}\n")
        f.write(f"p={p}\n")
        f.write(f"rho_stress={rho_stress}\n")
        f.write(f"p_stress={p_stress}\n")
        f.write(f"anxiety_mean={result['eads21_anxiety'].mean()}\n")
        f.write(f"anxiety_sd={result['eads21_anxiety'].std()}\n")
        f.write(f"anxiety_min={result['eads21_anxiety'].min()}\n")
        f.write(f"anxiety_max={result['eads21_anxiety'].max()}\n")
        f.write(f"f1_mean={result['macro_f1_et1000'].mean()}\n")
        f.write(f"f1_sd={result['macro_f1_et1000'].std()}\n")

    print(f"n={n}  rho={rho:.4f} p={p:.4f}  rho_stress={rho_stress:.4f} p_stress={p_stress:.4f}")


if __name__ == "__main__":
    main()
