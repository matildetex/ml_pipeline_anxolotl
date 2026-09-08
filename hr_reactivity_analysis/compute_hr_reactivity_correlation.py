"""
Delta HR (baseline -> stress) vs EADS-21 Ansiedade — Correlacao de Spearman.
Reproduz os numeros citados na tese, secao 4.6.3.

ORIGEM DOS DADOS (ambos read-only, nao modificados por este script):
  - processed_data/anxolotl/processed_data_anxolotl.pkl, chave "df" (699 janelas,
    37 sujeitos). Colunas usadas: subject, stress_phase (-1=baseline, 1=stress1,
    2=stress2, 3=stress3), hr_mean (HR medio por janela, em bpm — bruto, PRE
    baseline-normalization; o BL-norm e aplicado em tempo de treino em
    scripts/anxolotl_loso.py, nao esta pre-aplicado neste pkl).
  - eads21_analysis/eads21_vs_performance.csv (gerado por
    eads21_analysis/compute_eads21_correlations.py): fornece eads21_anxiety por
    subject, ja validado (37/37 sujeitos, incluindo p41 — ver nota no script da
    pasta eads21_analysis).

DEFINICAO DO DELTA:
  Por sujeito: baseline_hr = media(hr_mean) sobre stress_phase == -1 (label 0);
  stress_hr = media(hr_mean) sobre stress_phase in {1, 2} — exclui explicitamente
  stress_phase 3, consistente com o resto da tese (canonical_numbers_for_thesis.txt:
  so stress1+stress2 = 352 janelas de "stress"). delta_hr = stress_hr - baseline_hr.

VALIDACAO: os valores (rho, p, n) produzidos por este script devem coincidir com
hr_reactivity_analysis/_stats.txt e com a tese em 4.6.3 (rho~=0.1308, p~=0.4405, n=37).
"""

import pandas as pd
from scipy.stats import spearmanr

PKL_PATH = "../processed_data/anxolotl/processed_data_anxolotl.pkl"
EADS21_CSV = "../eads21_analysis/eads21_vs_performance.csv"

OUT_PER_SUBJECT_CSV = "hr_reactivity_per_subject.csv"
OUT_MERGED_CSV = "hr_reactivity_vs_eads21.csv"
OUT_STATS = "_stats.txt"

STRESS_PHASES = [1, 2]  # excludes stress_phase 3, see docstring


def compute_reactivity():
    d = pd.read_pickle(PKL_PATH)
    df = d["df"]

    baseline = df[df["stress_phase"] == -1].groupby("subject")["hr_mean"].mean()
    stress = df[df["stress_phase"].isin(STRESS_PHASES)].groupby("subject")["hr_mean"].mean()

    out = pd.DataFrame({"baseline_hr": baseline, "stress_hr": stress}).dropna()
    out["delta_hr"] = out["stress_hr"] - out["baseline_hr"]
    out = out.reset_index()

    out["_subj_num"] = out["subject"].str.lstrip("p").astype(int)
    out = out.sort_values("_subj_num").drop(columns="_subj_num").reset_index(drop=True)
    return out


def main():
    reactivity = compute_reactivity()
    reactivity.to_csv(OUT_PER_SUBJECT_CSV, index=False)

    eads21 = pd.read_csv(EADS21_CSV)[["subject", "eads21_anxiety"]]
    merged = reactivity.merge(eads21, on="subject", how="inner")
    assert len(merged) == len(reactivity), "sujeito(s) perdido(s) no merge com EADS-21"
    merged.to_csv(OUT_MERGED_CSV, index=False)

    rho, p = spearmanr(merged["eads21_anxiety"], merged["delta_hr"])
    n = len(merged)

    with open(OUT_STATS, "w") as f:
        f.write(f"n={n}\n")
        f.write(f"rho={rho}\n")
        f.write(f"p={p}\n")
        f.write(f"delta_hr_mean={merged['delta_hr'].mean()}\n")
        f.write(f"delta_hr_sd={merged['delta_hr'].std()}\n")
        f.write(f"delta_hr_min={merged['delta_hr'].min()}\n")
        f.write(f"delta_hr_max={merged['delta_hr'].max()}\n")

    print(f"n={n}  rho={rho:.4f} p={p:.4f}")


if __name__ == "__main__":
    main()
