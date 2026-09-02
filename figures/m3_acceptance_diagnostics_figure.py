# -*- coding: utf-8 -*-
r"""MCMC acceptance diagnostics for the 181,500-fit GPU scan.

The Goodman-Weare affine-invariant ensemble sampler (emcee) is healthy when the
acceptance fraction sits in ~[0.2, 0.5]. This figure shows that it did, across
the whole scan and every sample.

(a) Histogram of the acceptance fraction over all 181,500 fits, with the healthy
    [0.2, 0.5] band shaded and the median marked.
(b) Per-sample box plots (15 samples, NCC north-margin then Songliao), sharing
    the acceptance axis. All samples sit in the band; the only sample with a
    depressed median is R31370-S12652 (the most degenerate / shape-inflated
    sample, §0.2), whose flatter posterior is intrinsically harder to sample —
    still healthy at ~0.28.

Burn-in / autocorrelation: 64 walkers × 5000 steps with a 1000-step burn-in
(Methods §3); per-fit autocorrelation times were not stored, so convergence is
reported here via the acceptance fraction and the emcee-vs-GPU equivalence check
(Methods), not an integrated-autocorrelation panel.

Reads grains_bayes_gpu.parquet. Out: m3_acceptance_diagnostics.{png,pdf} (300 dpi).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
PARQUET = HERE.parent / "data" / "grains_bayes_gpu.parquet"
BAND = (0.2, 0.5)                      # emcee healthy acceptance band
GRP_C = {"NCC": "#CC79A7", "Songliao": "#595959"}
HIGHLIGHT = "R31370-S12652"


def is_ncc(name):
    return name.startswith("NCC")


def main():
    df = pd.read_parquet(PARQUET, columns=["sample_name", "acceptance_fraction",
                                           "n_walkers", "n_steps"])
    a = df["acceptance_fraction"].to_numpy()
    df["short"] = df["sample_name"].str.split("__").str[-1]
    df["grp"] = np.where(df["sample_name"].map(is_ncc), "NCC", "Songliao")
    nw, ns = int(df.n_walkers.iloc[0]), int(df.n_steps.iloc[0])

    frac_band = float(((a >= BAND[0]) & (a <= BAND[1])).mean())
    med = float(np.median(a))

    # sample order: NCC first, then Songliao, alphabetical
    order = (df[["short", "grp"]].drop_duplicates()
             .assign(_n=lambda d: d.grp.ne("NCC"))
             .sort_values(["_n", "short"])["short"].tolist())
    per_sample = [df.loc[df["short"] == s, "acceptance_fraction"].to_numpy() for s in order]
    grp_of = {s: df.loc[df["short"] == s, "grp"].iloc[0] for s in order}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    })
    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(6.85, 5.8), sharex=True,
        gridspec_kw=dict(height_ratios=[0.62, 1.0], hspace=0.13,
                         left=0.16, right=0.975, top=0.955, bottom=0.085))

    # ---- (a) overall histogram ----
    axa.axvspan(*BAND, color="#009E73", alpha=0.10, zorder=0)
    axa.hist(a, bins=np.linspace(0.10, 0.50, 61), color="#56B4E9",
             edgecolor="white", linewidth=0.2, zorder=3)
    axa.axvline(med, color="#0B3D2E", lw=1.0, ls=(0, (4, 2)), zorder=4)
    axa.text(med - 0.004, axa.get_ylim()[1] * 0.92, f"median {med:.2f}",
             ha="right", va="top", fontsize=7.5, color="#0B3D2E")
    axa.text(0.135, axa.get_ylim()[1] * 0.92,
             f"{frac_band*100:.1f}% of all\n181,500 fits\nin [0.2, 0.5]",
             ha="left", va="top", fontsize=7.3, color="0.2")
    axa.set_ylabel("fits (count)")
    # 图级标题移除(2026-07-08):期刊惯例图名归 caption;采样器参数在 §3.4 与图注
    for sp in ("top", "right"):
        axa.spines[sp].set_visible(False)
    axa.text(-0.11, 1.0, "(a)", transform=axa.transAxes, fontsize=11, weight="bold", va="top")

    # ---- (b) per-sample box plots ----
    axb.axvspan(*BAND, color="#009E73", alpha=0.10, zorder=0,
                label="healthy band [0.2, 0.5]")
    ypos = np.arange(len(order))[::-1]          # first sample at top
    bp = axb.boxplot(per_sample, positions=ypos, orientation="horizontal", widths=0.62,
                     patch_artist=True, showfliers=False, zorder=3,
                     medianprops=dict(color="black", lw=1.0),
                     whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7),
                     boxprops=dict(lw=0.5))
    for patch, s in zip(bp["boxes"], order):
        patch.set_facecolor(GRP_C[grp_of[s]])
        patch.set_alpha(0.85)
        if s == HIGHLIGHT:
            patch.set_edgecolor("#D55E00"); patch.set_linewidth(1.3)
    axb.set_yticks(ypos)
    axb.set_yticklabels(order, fontsize=6.8)
    for tick, s in zip(axb.get_yticklabels(), order):
        if s == HIGHLIGHT:
            tick.set_color("#D55E00"); tick.set_weight("bold")
    axb.set_xlim(0.10, 0.50)
    axb.set_xlabel("acceptance fraction")
    axb.grid(axis="x", color="0.9", lw=0.4, zorder=0); axb.set_axisbelow(True)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    axb.text(-0.11, 1.02, "(b)", transform=axb.transAxes, fontsize=11, weight="bold", va="top")
    # group divider between NCC (top 4) and Songliao
    n_ncc = sum(grp_of[s] == "NCC" for s in order)
    axb.axhline(ypos[n_ncc] + 0.5, color="0.6", lw=0.6, ls=(0, (4, 3)))
    axb.text(0.105, ypos[0], "NCC", fontsize=7, color="0.35", va="center", weight="bold")
    axb.text(0.105, ypos[n_ncc + 3], "Songliao", fontsize=7, color="0.35", va="center", weight="bold")

    handles = [Patch(facecolor="#009E73", alpha=0.10, label="healthy band [0.2, 0.5]"),
               Patch(facecolor=GRP_C["NCC"], alpha=0.85, label="NCC north-margin"),
               Patch(facecolor=GRP_C["Songliao"], alpha=0.85, label="Songliao")]
    axa.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.32, 0.96),
               frameon=False, fontsize=7.0)

    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"m3_acceptance_diagnostics.{ext}", dpi=300, bbox_inches="tight")
    print("[ok] saved m3_acceptance_diagnostics.png / .pdf")
    print(f"[caption] median {med:.3f}; {frac_band*100:.1f}% in [0.2,0.5]; "
          f"lowest-median sample = {order[int(np.argmin([np.median(x) for x in per_sample]))]}")


if __name__ == "__main__":
    main()
