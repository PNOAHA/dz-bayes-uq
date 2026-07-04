# -*- coding: utf-8 -*-
r"""Fig Y — supervised-NMF (first paper) vs Bayesian posterior mean (this paper)
on the four real NCC north-margin K1 samples.

IMPORTANT framing (audited 2026-07-04): the first paper's case study
(dz-unmix-cg/case_study_demo.py) did NOT run on these four real samples — it
used four *literature-grounded synthetic basin* proxies. The valid comparison
is therefore to APPLY the first paper's supervised-NMF method (identical
end-members S1 130/15, S2 280/30, S3 1900/100) to the SAME real grain-age
spectra the Bayesian model consumed. The first-paper functions are imported
read-only (its code is DOI-reserved; not modified).

Only four samples overlap: the eleven Puetz/OneDZ Songliao samples were never
run through the first-paper pipeline, so the comparison is limited to the NCC
north-margin group and must be labelled as such.

Panel: supervised-NMF point estimate (x) vs Bayesian posterior mean +- scan std
(y), 12 points (4 samples x 3 end-members), coloured by end-member, with the
1:1 line. Agreement on the hard craton weight, larger Bayesian error bars on
the soft arc/basement weights.

Env: local Python 3.12 + numpy/pandas/pyarrow/matplotlib/scikit-learn.
Out: FigY_nmf_comparison.{png,pdf} (300 dpi, GJ single/1.5-col).
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
PARQUET = HERE / "grains_bayes_gpu.parquet"
CG = HERE.parents[1] / "dz-unmix-cg"          # first paper (read-only import)
AGES_DIR = HERE.parents[1] / "bayes_gpu" / "realdata_k1_samples"

sys.path.insert(0, str(CG))
from dz_unmixing_experiment import (           # noqa: E402  (read-only reuse)
    kde_density, build_source_endmembers, supervised_unmix,
)

# NCC north-margin samples common to both studies (short name -> csv stem)
NCC = {
    "05FW007":  "NCC_north_margin__onedz__05FW007",
    "05FW008":  "NCC_north_margin__onedz__05FW008",
    "100821-1": "NCC_north_margin__onedz__100821-1",
    "100821-3": "NCC_north_margin__onedz__100821-3",
}
WK = ["S1", "S2", "S3"]
EM_C = {"S1": "#E69F00", "S2": "#009E73", "S3": "#0072B2"}   # match fig_bayes
EM_LAB = {"S1": "S1  Arc (130 Ma)", "S2": "S2  Basement (280 Ma)", "S3": "S3  Craton (1900 Ma)"}


def load_ages(stem):
    p = AGES_DIR / f"{stem}.csv"
    return pd.read_csv(p, comment="#")["age_Ma"].to_numpy(float)


def supervised_point_estimates():
    """First-paper supervised NMF on the four real NCC age spectra."""
    H = build_source_endmembers()
    out = {}
    for short, stem in NCC.items():
        ages = load_ages(stem)
        x = kde_density(ages).reshape(1, -1)
        out[short] = supervised_unmix(x, H)[0]        # (w_S1, w_S2, w_S3)
    return out


def bayesian_stats():
    """Posterior mean +- scan std for the four NCC samples, from the parquet."""
    cols = ["sample_name"] + [f"w_{k}_mean" for k in WK]
    df = pd.read_parquet(PARQUET, columns=cols)
    df["short"] = df["sample_name"].str.split("__").str[-1]
    df = df[df["short"].isin(NCC)]
    mean = df.groupby("short")[[f"w_{k}_mean" for k in WK]].mean()
    std = df.groupby("short")[[f"w_{k}_mean" for k in WK]].std()
    return mean, std


def main():
    nmf = supervised_point_estimates()
    bmean, bstd = bayesian_stats()

    # assemble comparison table
    rows = []
    for short in NCC:
        for j, k in enumerate(WK):
            rows.append(dict(
                sample=short, em=k,
                nmf=float(nmf[short][j]),
                bayes=float(bmean.loc[short, f"w_{k}_mean"]),
                bstd=float(bstd.loc[short, f"w_{k}_mean"]),
            ))
    tab = pd.DataFrame(rows)
    tab["absdiff"] = (tab["nmf"] - tab["bayes"]).abs()
    tab.to_csv(HERE / "FigY_nmf_comparison_data.csv", index=False)

    mae = tab["absdiff"].mean()
    mae_hard = tab.loc[tab.em == "S3", "absdiff"].mean()
    mae_soft = tab.loc[tab.em != "S3", "absdiff"].mean()

    print("=== supervised-NMF (first paper) vs Bayesian mean (this paper) ===")
    with pd.option_context("display.width", 160):
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nMAE overall {mae:.3f} | craton w_S3 {mae_hard:.3f} | "
          f"arc/basement w_S1,w_S2 {mae_soft:.3f}")

    # ---- figure: NMF (x) vs Bayesian mean +- std (y), 1:1 line ----
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    })
    fig, ax = plt.subplots(figsize=(4.3, 4.2))
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(0.97, 0.99, "1:1", color="0.5", fontsize=7.5, ha="right", va="top")

    for k in WK:
        sub = tab[tab.em == k]
        ax.errorbar(sub["nmf"], sub["bayes"], yerr=sub["bstd"], fmt="o", ms=6,
                    color=EM_C[k], ecolor=EM_C[k], elinewidth=0.9, capsize=2.2,
                    capthick=0.7, mec="black", mew=0.5, zorder=4, label=EM_LAB[k])

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel("Supervised NMF point estimate  (first paper)")
    ax.set_ylabel("Bayesian posterior mean $\\pm$ scan std  (this paper)")
    ax.grid(color="0.92", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5, handletextpad=0.4)
    ax.text(0.03, 0.985,
            f"4 NCC north-margin samples (n=4)\nMAE = {mae:.02f}"
            f"   (w$_{{S3}}$ {mae_hard:.02f},  w$_{{S1}}$/w$_{{S2}}$ {mae_soft:.02f})",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.0, color="0.2")

    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"FigY_nmf_comparison.{ext}", dpi=300, bbox_inches="tight")
    print(f"\n[ok] saved FigY_nmf_comparison.png / .pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
