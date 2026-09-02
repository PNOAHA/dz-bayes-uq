# -*- coding: utf-8 -*-
r"""Endmember shape-parameter figure: prior vs posterior for mu (peak age) and
sigma (peak width) of the three end-members — "is the prior dominating the
posterior, or does the data update the end-member shapes?"

For each end-member k and each shape parameter, the prior is Gaussian centred
on the reference value (mu: 130/280/1900 Ma, sigma: 15/30/100 Ma) with relative
std +-10% (mu) / +-20% (sigma), i.e. prior std = 13/28/190 (mu), 3/6/20 (sigma).
The posterior mu_Sk_*/sig_Sk_* are read from the parquet at the CENTRAL prior
config (mean_shift = sigma_shift = 0), where the prior centre equals the
reference — the clean baseline for "did the data move / narrow the belief".

Each of 6 panels overlays: prior Gaussian (grey), a representative single-fit
posterior Gaussian (endmember colour; median-over-samples of the posterior
mean and within-fit std), the 15 per-sample posterior means as a rug, and the
reference value. The posterior narrows to 0.3-0.5x (mu) / 0.75-0.9x (sigma) of
the prior and shifts by up to ~1.6 prior-sigma -> the shapes are data-informed,
not prior-dominated. (Consistent with the weights being prior-insensitive, §0.3:
the data pins the shapes, so the weights do not depend on the prior.)

Reads grains_bayes_gpu.parquet. Out: m3_mu_sigma_posterior.{png,pdf} (300 dpi).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
PARQUET = HERE.parent / "data" / "grains_bayes_gpu.parquet"

REF_MU = {"S1": 130.0, "S2": 280.0, "S3": 1900.0}
REF_SIG = {"S1": 15.0, "S2": 30.0, "S3": 100.0}
REL_MU, REL_SIG = 0.10, 0.20            # prior relative std
EM_C = {"S1": "#E69F00", "S2": "#009E73", "S3": "#0072B2"}
EM_NAME = {"S1": "arc", "S2": "basement", "S3": "craton"}
PRIOR_C = "#9E9E9E"


def gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def main():
    df = pd.read_parquet(PARQUET)
    c = df[(df.mean_shift == 0.0) & (df.sigma_shift == 0.0)]      # central prior
    g = c.groupby("sample_name").mean(numeric_only=True)          # per-sample (avg over 100 boot)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 9, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(2, 3, figsize=(6.85, 4.5))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.10,
                        hspace=0.42, wspace=0.22)

    rows = [("mu", "$\\mu$  (peak age, Ma)", REF_MU, REL_MU),
            ("sig", "$\\sigma$  (peak width, Myr)", REF_SIG, REL_SIG)]

    for ri, (par, row_lab, ref, rel) in enumerate(rows):
        for ci, k in enumerate(["S1", "S2", "S3"]):
            ax = axes[ri, ci]
            r = ref[k]
            pri_s = rel * r                                   # prior std
            col = f"{par}_{k}"
            post_mean = float(g[f"{col}_mean"].median())      # representative posterior mean
            post_std = float(g[f"{col}_std"].median())        # representative within-fit std
            sample_means = g[f"{col}_mean"].to_numpy()        # 15 per-sample means (rug)

            span = 3.4 * pri_s
            x = np.linspace(r - span, r + span, 400)
            # prior (grey fill) + representative posterior (colour)
            ax.fill_between(x, 0, gauss(x, r, pri_s), color=PRIOR_C, alpha=0.30,
                            lw=0, zorder=1)
            ax.plot(x, gauss(x, r, pri_s), color=PRIOR_C, lw=0.9, zorder=2)
            ax.plot(x, gauss(x, post_mean, post_std), color=EM_C[k], lw=1.7, zorder=4)
            ax.fill_between(x, 0, gauss(x, post_mean, post_std), color=EM_C[k],
                            alpha=0.16, lw=0, zorder=3)
            ax.axvline(r, color="0.35", lw=0.7, ls=(0, (3, 2)), zorder=2)
            # rug of 15 per-sample posterior means
            ymax = gauss(post_mean, post_mean, post_std)
            ax.plot(sample_means, np.full_like(sample_means, -0.045 * ymax),
                    marker="|", ls="none", color=EM_C[k], ms=5, mew=0.8, zorder=5,
                    clip_on=True)

            narrow = post_std / pri_s
            shift = (post_mean - r) / pri_s
            ax.set_title(f"{k}  {EM_NAME[k]}", fontsize=8.5, color=EM_C[k], weight="bold")
            ax.text(0.035, 0.95,
                    f"post σ = {narrow:.2f}× prior\nshift {shift:+.1f} prior-σ",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
                    color="0.2")
            ax.set_xlim(r - span, r + span)
            ax.set_ylim(-0.09 * ymax, 1.45 * ymax)
            ax.set_yticks([])
            ax.set_xlabel(row_lab if ci == 1 else "", fontsize=8)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(axis="x", labelsize=7)

    axes[0, 0].set_ylabel("density", fontsize=8)
    axes[1, 0].set_ylabel("density", fontsize=8)

    handles = [
        Line2D([0], [0], color=PRIOR_C, lw=2.2, label="Prior  N(ref, ±10%μ/±20%σ)"),
        Line2D([0], [0], color="#444444", lw=2.2, label="Posterior (representative single fit)"),
        Line2D([0], [0], color="0.35", lw=0.9, ls=(0, (3, 2)), label="Reference value"),
        Line2D([0], [0], color="#444444", marker="|", ls="none", ms=7, mew=1.0,
               label="Per-sample posterior mean (n=15)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, frameon=False, fontsize=7.0, columnspacing=1.4, handlelength=1.6)

    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"m3_mu_sigma_posterior.{ext}", dpi=300, bbox_inches="tight")
    print("[ok] saved m3_mu_sigma_posterior.png / .pdf")

    # numeric summary for caption
    print("\n[caption numbers] central prior config, median over 15 samples")
    for par, ref, rel in [("mu", REF_MU, REL_MU), ("sig", REF_SIG, REL_SIG)]:
        for k in ["S1", "S2", "S3"]:
            pri_s = rel * ref[k]
            pm = g[f"{par}_{k}_mean"].median(); ps = g[f"{par}_{k}_std"].median()
            print(f"  {par}_{k}: ref {ref[k]:.0f}  post {pm:.1f}±{ps:.1f}  "
                  f"(prior σ {pri_s:.0f}; narrow {ps/pri_s:.2f}×; shift {(pm-ref[k])/pri_s:+.2f}σ)")


if __name__ == "__main__":
    main()
