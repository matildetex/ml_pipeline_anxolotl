"""
Delta HRV (baseline -> stress; RMSSD, SDNN, IBI) vs EADS-21 Ansiedade —
Correlacoes de Spearman. Reproduz os numeros citados na tese, secao 4.6.3.

ORIGEM DOS DADOS (ambos read-only, nao modificados por este script):
  - processed_data/anxolotl/processed_data_anxolotl.pkl, chave "df" (699 janelas,
    37 sujeitos). Colunas usadas: subject, stress_phase (-1=baseline, 1=stress1,
    2=stress2, 3=stress3), rmssd_mean, sdnn_mean, ibi_mean — armazenadas em
    SEGUNDOS no pkl (bruto, PRE baseline-normalization; convertidas para ms aqui,
    convencao da literatura de HRV).
  - eads21_analysis/eads21_vs_performance.csv (gerado por
    eads21_analysis/compute_eads21_correlations.py): fornece eads21_anxiety por
    subject, ja validado (37/37 sujeitos, incluindo p41).

DEFINICAO DO DELTA (por sujeito, por metrica):
  baseline_X_ms = media(X_mean) sobre stress_phase == -1, x1000.
  stress_X_ms = media(X_mean) sobre stress_phase in {1, 2} — exclui explicitamente
  stress_phase 3, consistente com o resto da tese. delta_X_ms = stress_X_ms - baseline_X_ms.

VALIDACAO: os valores (rho, p, n) produzidos por este script devem coincidir com
hrv_reactivity_analysis/_stats.txt e com a tese em 4.6.3 (todos p>0.15, n=37;
RMSSD rho~=0.2357, SDNN rho~=0.0874, IBI rho~=-0.1659).
"""

import pandas as pd
from scipy.stats import spearmanr

PKL_PATH = "../processed_data/anxolotl/processed_data_anxolotl.pkl"
EADS21_CSV = "../eads21_analysis/eads21_vs_performance.csv"

OUT_PER_SUBJECT_CSV = "hrv_reactivity_per_subject.csv"
OUT_MERGED_CSV = "hrv_reactivity_vs_eads21.csv"
OUT_STATS = "_stats.txt"

STRESS_PHASES = [1, 2]  # excludes stress_phase 3, see docstring
METRICS = ["rmssd", "sdnn", "ibi"]  # source columns: <metric>_mean, in seconds


def compute_reactivity():
    d = pd.read_pickle(PKL_PATH)
    df = d["df"]

    out = None
    for metric in METRICS:
        col = f"{metric}_mean"
        baseline = df[df["stress_phase"] == -1].groupby("subject")[col].mean() * 1000
        stress = df[df["stress_phase"].isin(STRESS_PHASES)].groupby("subject")[col].mean() * 1000
        m = pd.DataFrame({
            f"baseline_{metric}_ms": baseline,
            f"stress_{metric}_ms": stress,
        }).dropna()
        m[f"delta_{metric}_ms"] = m[f"stress_{metric}_ms"] - m[f"baseline_{metric}_ms"]
        out = m if out is None else out.join(m, how="inner")

    out = out.reset_index()
    out["_subj_num"] = out["subject"].str.lstrip("p").astype(int)
    out = out.sort_values("_subj_num").drop(columns="_subj_num").reset_index(drop=True)

    ordered_cols = ["subject"] + [
        f"{stage}_{metric}_ms" for metric in METRICS for stage in ("baseline", "stress", "delta")
    ]
    return out[ordered_cols]


def main():
    reactivity = compute_reactivity()
    reactivity.to_csv(OUT_PER_SUBJECT_CSV, index=False)

    eads21 = pd.read_csv(EADS21_CSV)[["subject", "eads21_anxiety"]]
    merged = reactivity.merge(eads21, on="subject", how="inner")
    assert len(merged) == len(reactivity), "sujeito(s) perdido(s) no merge com EADS-21"
    merged.to_csv(OUT_MERGED_CSV, index=False)

    n = len(merged)
    lines = []
    for metric in METRICS:
        delta_col = f"delta_{metric}_ms"
        rho, p = spearmanr(merged["eads21_anxiety"], merged[delta_col])
        lines.append(f"metric={metric}\n")
        lines.append(f"n={n}\n")
        lines.append(f"rho={rho}\n")
        lines.append(f"p={p}\n\n")
        lines.append(f"delta_{metric}_mean_ms={merged[delta_col].mean()}\n")
        lines.append(f"delta_{metric}_sd_ms={merged[delta_col].std()}\n\n")
        lines.append(f"delta_{metric}_min_ms={merged[delta_col].min()}\n")
        lines.append(f"delta_{metric}_max_ms={merged[delta_col].max()}\n\n")
        print(f"{metric}: n={n}  rho={rho:.4f} p={p:.4f}")

    with open(OUT_STATS, "w") as f:
        f.write("".join(lines).rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
