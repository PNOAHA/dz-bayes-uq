# -*- coding: utf-8 -*-
r"""M3 companion figure: variance-source decomposition of the posterior weight
uncertainty, per sample.

Visualises the law-of-total-variance split from m3_within_vs_between.py. For
each sample the pooled posterior weight covariance (over the 121x100 prior x
bootstrap grid) has total variance tr(Sigma) = sum_k Var(w_k), which splits
exactly into three additive sources:

    tr(Sigma) = WITHIN (mean single-fit credible variance)
              + BOOTSTRAP (between-fit scatter from data resampling)
              + PRIOR     (between-fit scatter from +-10%/+-20% prior shifts)

(a) absolute stacked variance (bar height = total posterior spread; total
    spread sqrt(tr Sigma) annotated on top) -> which samples are uncertain and
    from what source.
(b) the same, normalised to 100% -> the WITHIN share is ~half everywhere, the
    PRIOR sliver is negligible except for the two shape-inflated samples
    (05FW007, R31370-S12652).

Reads m3_within_vs_between_results.csv (this dir). Renders headless (Agg).
Out: m3_variance_sources.{png,pdf} (300 dpi, GJ double-column).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "data" / "results" / "m3_within_vs_between_results.csv"
OUT = HERE / "m3_variance_sources"

# ---- Geological Journal (Wiley) rcParams (mirror fig_bayes_uncertainty.py) ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
})
# source colours: deliberately AVOID the endmember trio (orange/green/blue) used
# in fig_bayes_uncertainty. within = calm sky-blue; bootstrap = neutral grey
# (the baseline data noise); prior = vermillion accent so its rare appearance pops.
SRC = [
    ("within", "Within-fit (credible width)", "#56B4E9"),
    ("bootstrap", "Bootstrap (data resampling)", "#808080"),
    ("prior", "Prior sensitivity", "#D55E00"),
]
HIGHLIGHT = {"05FW007", "R31370-S12652"}   # the two shape-inflated samples


def is_ncc(group):
    return str(group).startswith("NCC")


def main():
    df = pd.read_csv(RESULTS)
    # per sample: sum of per-weight variances -> tr(Sigma) source components
    g = df.assign(
        within_var=df.within_rms ** 2,
        boot_var=df.boot_sd ** 2,
        prior_var=df.prior_sd ** 2,
    ).groupby(["sample", "group", "n"], as_index=False).agg(
        within=("within_var", "sum"),
        bootstrap=("boot_var", "sum"),
        prior=("prior_var", "sum"),
    )
    g["total"] = g.within + g.bootstrap + g.prior
    # order: NCC group first, then Songliao, alphabetical within group
    g["_ncc"] = ~g.group.map(is_ncc)          # False (NCC) sorts first
    g = g.sort_values(["_ncc", "sample"]).reset_index(drop=True)

    x = np.arange(len(g))
    n_ncc = int((~g["_ncc"]).sum())

    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(6.85, 6.4), sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 0.85], hspace=0.14,
                         left=0.095, right=0.97, top=0.88, bottom=0.20))

    # ---------- (a) absolute stacked variance ----------
    bottom = np.zeros(len(g))
    for key, _lab, col in SRC:
        axa.bar(x, g[key], bottom=bottom, width=0.72, color=col,
                edgecolor="white", linewidth=0.4, zorder=3)
        bottom += g[key].to_numpy()
    # total spread sqrt(tr Sigma) annotated on top
    for xi, tot in zip(x, g.total):
        axa.text(xi, tot + g.total.max() * 0.015, f"{np.sqrt(tot):.2f}",
                 ha="center", va="bottom", fontsize=5.6, color="0.25", rotation=90)
    axa.set_ylabel("Posterior variance  tr($\\Sigma$)  (weight$^2$)")
    axa.set_ylim(0, g.total.max() * 1.16)
    axa.grid(axis="y", color="0.9", lw=0.4); axa.set_axisbelow(True)
    for sp in ("top", "right"):
        axa.spines[sp].set_visible(False)
    axa.text(-0.082, 1.10, "(a)", transform=axa.transAxes, fontsize=11,
             weight="bold", va="top")
    axa.text(0.99, 0.93, "top label = total spread $\\sqrt{\\mathrm{tr}\\,\\Sigma}$",
             transform=axa.transAxes, ha="right", va="top", fontsize=6.5, color="0.35")

    # ---------- (b) normalised 100% stacked ----------
    frac = g[["within", "bootstrap", "prior"]].div(g.total, axis=0) * 100.0
    bottom = np.zeros(len(g))
    for key, _lab, col in SRC:
        axb.bar(x, frac[key], bottom=bottom, width=0.72, color=col,
                edgecolor="white", linewidth=0.4, zorder=3)
        bottom += frac[key].to_numpy()
    med_within = frac["within"].median()
    axb.axhline(med_within, color="0.15", lw=0.7, ls=(0, (4, 2)), zorder=4)
    axb.text(len(g) - 0.4, med_within + 1.5,
             f"median within = {med_within:.0f}%", ha="right", va="bottom",
             fontsize=6.5, color="0.15")
    # annotate the visible prior segments in-place (the two shape-inflated samples)
    for xi, samp, pr in zip(x, g["sample"], frac["prior"]):
        if samp in HIGHLIGHT:
            axb.text(xi, 100 - pr / 2, f"prior\n{pr:.0f}%", ha="center",
                     va="center", fontsize=6.0, weight="bold", color="white")
    axb.set_ylabel("Variance share (%)")
    axb.set_ylim(0, 100)
    axb.set_yticks([0, 25, 50, 75, 100])
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    axb.text(-0.082, 1.12, "(b)", transform=axb.transAxes, fontsize=11,
             weight="bold", va="top")

    # ---------- shared x, group divider ----------
    labels = [f"{s}  (n={int(nn)})" for s, nn in zip(g["sample"], g["n"])]
    axb.set_xticks(x)
    axb.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.3)
    # bold the two highlighted sample ticks
    for tick, samp in zip(axb.get_xticklabels(), g["sample"]):
        if samp in HIGHLIGHT:
            tick.set_color("#D55E00"); tick.set_weight("bold")
    axb.set_xlim(-0.7, len(g) - 0.3)
    for ax in (axa, axb):
        ax.axvline(n_ncc - 0.5, color="0.6", lw=0.6, ls=(0, (4, 3)), zorder=2)
    axa.text((n_ncc - 1) / 2, axa.get_ylim()[1] * 0.995, "NCC north margin",
             ha="center", va="top", fontsize=7, color="0.35", weight="bold")
    axa.text((n_ncc + len(g) - 1) / 2, axa.get_ylim()[1] * 0.995, "Songliao",
             ha="center", va="top", fontsize=7, color="0.35", weight="bold")

    # ---------- legend ----------
    handles = [Patch(facecolor=col, edgecolor="white", label=lab)
               for _k, lab, col in SRC]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.995),
               frameon=False, fontsize=7.5, ncol=3, handlelength=1.3,
               columnspacing=1.8, borderaxespad=0.2)

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=300)
    print(f"[ok] saved {OUT}.png / .pdf")

    # ---- numeric summary for the caption ----
    g2 = g.set_index("sample")
    print("\n[caption numbers]")
    print(f"  within share: median {frac['within'].median():.0f}%  "
          f"range {frac['within'].min():.0f}-{frac['within'].max():.0f}%")
    nonhi = frac["prior"][~g["sample"].isin(HIGHLIGHT).to_numpy()].max()
    print(f"  prior share : median {frac['prior'].median():.1f}%  "
          f"(<={nonhi:.0f}% outside the two highlighted samples)")
    for s in ["05FW007", "R31370-S12652"]:
        fr = frac.loc[g.index[g["sample"] == s][0]]
        print(f"  {s:14s}: within {fr.within:.0f}% / bootstrap {fr.bootstrap:.0f}% "
              f"/ prior {fr.prior:.0f}%   total spread {np.sqrt(g2.loc[s,'total']):.2f}")


if __name__ == "__main__":
    main()
