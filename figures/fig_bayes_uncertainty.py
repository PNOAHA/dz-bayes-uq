# -*- coding: utf-8 -*-
"""
fig_bayes_uncertainty.py — third-paper core figure: real-data Bayesian
DZ-unmix uncertainty quantification (15 NE-Asia K1 samples, 3 endmembers).

Self-contained: inlines Okabe-Ito + Geological Journal rcParams (the L20
container has no geo-figures helper module). Renders headless (Agg).

(a) ternary mixing diagram — per-sample posterior-mean point + the cloud of
    all 121x100 fits (cloud size = uncertainty); NCC squares vs Songliao circles.
(b) dot-whisker — per-sample w_S1/w_S2/w_S3 mean +/- spread (std over the full
    perturbation x bootstrap grid). Whisker length = how trustworthy each weight
    is. The point: w_S3 (craton present?) whiskers ~0 (hard); w_S1/w_S2 (arc vs
    basement split) whiskers 0.05-0.12 (soft).

Run (L20 container):  python fig_bayes_uncertainty.py
Out:  /private/bayes_gpu/fig_bayes_uncertainty.{png,pdf}  (300 dpi, GJ double-col)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.patheffects as pe

# ---- Geological Journal (Wiley) rcParams (mirror geo-figures apply_journal_rcparams("GJ")) ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
})
OKABE_ITO = {"orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
             "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
             "reddish_purple": "#CC79A7", "black": "#000000"}
# endmember colours: young arc -> warm, old craton -> cool (intuitive + colourblind-safe)
C = {"S1": OKABE_ITO["orange"], "S2": OKABE_ITO["bluish_green"], "S3": OKABE_ITO["blue"]}
EM_LABEL = {"S1": "S1  Arc (130 Ma)", "S2": "S2  Basement (280 Ma)", "S3": "S3  Craton (1900 Ma)"}
# group colours deliberately AVOID the endmember trio (orange/green/blue) so panel (a)
# "by sample group" never collides with panel (b) "by endmember": purple + neutral grey,
# disambiguated further by marker shape (square vs circle).
GRP_C = {"NCC": OKABE_ITO["reddish_purple"], "Songliao": "#595959"}

_HERE = Path(__file__).resolve().parent
# local co-located parquet (was /private/bayes_gpu/out/ on the L20 container)
PARQUET = str(_HERE / "grains_bayes_gpu.parquet")
OUT = str(_HERE / "fig_bayes_uncertainty")
SQRT3_2 = np.sqrt(3) / 2.0


def tern_xy(w1, w2, w3):
    """barycentric -> cartesian. S1=(0,0) bottom-left, S2=(1,0) bottom-right, S3=(0.5,.866) top."""
    s = w1 + w2 + w3
    return 0.5 * (2 * w2 + w3) / s, SQRT3_2 * w3 / s


def short(name):
    return name.split("__")[-1]


def group(name):
    return "NCC" if name.startswith("NCC") else "Songliao"


def main():
    d = pd.read_parquet(PARQUET, columns=["sample_name", "w_S1_mean", "w_S2_mean", "w_S3_mean"])
    wcols = ["w_S1_mean", "w_S2_mean", "w_S3_mean"]
    mean = d.groupby("sample_name")[wcols].mean()
    std = d.groupby("sample_name")[wcols].std()
    # order: NCC group first, then Songliao
    order = sorted(mean.index, key=lambda s: (group(s) != "NCC", short(s)))

    fig = plt.figure(figsize=(6.85, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.18, 1.0], hspace=0.28,
                          left=0.10, right=0.97, top=0.95, bottom=0.07)

    # ===================== (a) ternary =====================
    ax = fig.add_subplot(gs[0]); ax.set_aspect("equal"); ax.axis("off")
    V = {"S1": (0, 0), "S2": (1, 0), "S3": (0.5, SQRT3_2)}
    tri = plt.Polygon([V["S1"], V["S2"], V["S3"]], closed=True, fill=False,
                      ec=OKABE_ITO["black"], lw=0.9, zorder=2)
    ax.add_patch(tri)
    # gridlines every 0.2
    for f in np.arange(0.2, 1.0, 0.2):
        ax.plot(*zip(tern_xy(1 - f, f, 0), tern_xy(0, f, 1 - f)), color="0.85", lw=0.4, zorder=1)
        ax.plot(*zip(tern_xy(f, 1 - f, 0), tern_xy(f, 0, 1 - f)), color="0.85", lw=0.4, zorder=1)
        ax.plot(*zip(tern_xy(0, 1 - f, f), tern_xy(1 - f, 0, f)), color="0.85", lw=0.4, zorder=1)
    for k, (vx, vy) in V.items():
        dy = 0.045 if k == "S3" else -0.055
        ax.text(vx, vy + dy, EM_LABEL[k], ha="center",
                va="bottom" if k == "S3" else "top", fontsize=8.5, weight="bold", color=C[k])

    # label only the well-separated samples (dense Songliao bottom-edge cluster -> see panel b)
    LABEL_SET = {"05FW007", "05FW008", "100821-1", "100821-3", "SL0932", "R31370-S12653"}
    # per-label offset (points); 05FW007 nudged below its marker to clear the
    # nearby 100821-1 label (both NCC samples plot at similar ternary positions)
    LABEL_OFFSET = {"05FW007": (5, -11)}
    for name in order:
        g = group(name)
        sub = d[d.sample_name == name]
        cx, cy = tern_xy(sub["w_S1_mean"].values, sub["w_S2_mean"].values, sub["w_S3_mean"].values)
        # uncertainty as a 2-sigma covariance ellipse (size+orientation = posterior spread)
        if len(cx) >= 5:
            cov = np.cov(cx, cy)
            vals, vecs = np.linalg.eigh(cov)
            vals = np.clip(vals, 0, None)
            o = vals.argsort()[::-1]; vals, vecs = vals[o], vecs[:, o]
            ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            ew, eh = 4.0 * np.sqrt(vals)   # 2-sigma full axes
            ax.add_patch(Ellipse((cx.mean(), cy.mean()), ew, eh, angle=ang,
                                  facecolor=GRP_C[g], alpha=0.20, edgecolor=GRP_C[g],
                                  lw=0.6, zorder=3))
        mx, my = tern_xy(*mean.loc[name, wcols].values)
        ax.scatter(mx, my, s=34, marker=("s" if g == "NCC" else "o"),
                   facecolor=GRP_C[g], edgecolor="black", lw=0.6, zorder=5)
        if short(name) in LABEL_SET:
            dx, dy = LABEL_OFFSET.get(short(name), (4, 4))
            ax.annotate(short(name), (mx, my), textcoords="offset points", xytext=(dx, dy),
                        fontsize=6, color="0.1", zorder=6,
                        path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    ax.text(-0.02, 1.0, "(a)", transform=ax.transAxes, fontsize=11, weight="bold", va="top")
    leg = [Line2D([0], [0], marker="s", color="w", markerfacecolor=GRP_C["NCC"],
                  markeredgecolor="black", markersize=7, label="NCC north margin (n=4)"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=GRP_C["Songliao"],
                  markeredgecolor="black", markersize=7, label="Songliao (n=11)")]
    ax.legend(handles=leg, loc="upper right", frameon=False, fontsize=7.5,
              bbox_to_anchor=(1.0, 1.02))

    # ===================== (b) dot-whisker =====================
    ax2 = fig.add_subplot(gs[1])
    n = len(order)
    ypos = {name: n - 1 - i for i, name in enumerate(order)}
    off = {"S1": 0.24, "S2": 0.0, "S3": -0.24}
    for name in order:
        y0 = ypos[name]
        for em, col in (("S1", "w_S1_mean"), ("S2", "w_S2_mean"), ("S3", "w_S3_mean")):
            m, e = mean.loc[name, col], std.loc[name, col]
            ax2.errorbar(m, y0 + off[em], xerr=e, fmt="o", ms=3.4, color=C[em],
                         ecolor=C[em], elinewidth=0.9, capsize=1.8, capthick=0.7, zorder=4)
    # NCC / Songliao divider
    n_ncc = sum(group(s) == "NCC" for s in order)
    ax2.axhline(n - n_ncc - 0.5, color="0.6", lw=0.6, ls=(0, (4, 3)))
    ax2.text(1.005, n - 0.5, "NCC", fontsize=7.5, color="0.35", va="center", weight="bold")
    ax2.text(1.005, (n - n_ncc) / 2 - 0.5, "Songliao", fontsize=7.5, color="0.35", va="center", weight="bold")
    ax2.set_yticks([ypos[s] for s in order]); ax2.set_yticklabels([short(s) for s in order], fontsize=6.5)
    ax2.set_xlim(-0.03, 1.03); ax2.set_ylim(-0.7, n - 0.3)
    ax2.set_xlabel("posterior mixing weight  (mean $\\pm$ spread over 11$\\times$11 prior shifts $\\times$ 100 bootstrap)")
    ax2.grid(axis="x", color="0.9", lw=0.4); ax2.set_axisbelow(True)
    for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
    leg2 = [Line2D([0], [0], marker="o", color=C[k], lw=2, markersize=5, label=EM_LABEL[k]) for k in ("S1", "S2", "S3")]
    ax2.legend(handles=leg2, loc="lower right", frameon=False, fontsize=7.5, ncol=1)
    ax2.text(-0.13, 1.05, "(b)", transform=ax2.transAxes, fontsize=11, weight="bold", va="top")

    for ext, dpi in (("png", 300), ("pdf", 300)):
        fig.savefig(f"{OUT}.{ext}", dpi=dpi, bbox_inches="tight")
    print(f"[ok] saved {OUT}.png / .pdf")
    # quick numeric caption helper
    print("\n[caption numbers] w_S3 std range %.3f-%.3f (hard) | w_S1 std range %.3f-%.3f (soft)"
          % (std.w_S3_mean.min(), std.w_S3_mean.max(), std.w_S1_mean.min(), std.w_S1_mean.max()))


if __name__ == "__main__":
    main()
