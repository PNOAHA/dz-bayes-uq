#!/usr/bin/env python3
r"""Fig 1 (third / Bayesian-UQ paper) — sample location map of the fifteen
Lower Cretaceous (K1) detrital-zircon samples on a NE-Asia tectonic basemap.

Reuses the tectonic framework of the first paper's Fig 1
(`../../dz-unmix-cg/fig1_NEAsia_basemap.py`): extent, coastline, Cretaceous
rift basins (Songliao / Hailar / Erlian), Tanlu Fault Zone + Mongol-Okhotsk /
Solonker sutures, cratonic + CAOB labels, Palaeo-Pacific plate vector, FGDC
marginalia. The case-study stars are removed; instead the 15 samples are
plotted at their real coordinates (extracted from
dz-global-db/data/curated/ne_china_k1_real.csv, lat/lon columns), coloured by
group to match fig_bayes_uncertainty (NCC north-margin = reddish-purple
squares, Songliao = grey circles).

Coincident / near-coincident samples are merged to sites (rounded to 0.1 deg)
for a single marker + label; the full sample -> coordinate table belongs in
Table 1.

Env: Python 3.12 + matplotlib. geo_primitives is imported from the copy bundled
in this repository's code/ directory (added to sys.path below).

History:
  - v1 (2026-07-04): first cut, adapted from dz-unmix-cg Fig 1 v1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib.lines import Line2D
import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1] / "code"   # bundled geo_primitives.py
sys.path.insert(0, str(CODE_DIR))
from geo_primitives import (        # noqa: E402
    apply_journal_rcparams, geographic_aspect, chaikin_smooth,
    add_polygon_collection, label_polygons,
    plot_fault, draw_scale_bar, draw_north_arrow,
    plate_motion_arrow, save_three_formats,
    OKABE_ITO, BASIN_FAMILY_PALETTE,
)

# ── 1. Map extent (same as first-paper Fig 1) ───────────────────────────────
LON_MIN, LON_MAX = 105.0, 135.0
LAT_MIN, LAT_MAX = 38.0, 55.0
MID_LAT = (LAT_MIN + LAT_MAX) / 2

# ── 2. Coastline (schematic, from first-paper Fig 1) ────────────────────────
COASTLINE = [
    (LON_MIN, LAT_MIN), (LON_MIN, LAT_MAX), (130.0, LAT_MAX), (132.0, 53.0),
    (135.0, 51.0), (135.0, 47.0), (132.0, 44.0), (130.5, 42.5), (129.5, 41.0),
    (130.0, 38.5), (130.0, LAT_MIN), (124.0, LAT_MIN), (122.5, 39.0),
    (121.0, 38.5), (LON_MIN, LAT_MIN),
]

# ── 3. Cretaceous rift basins (regional context) ────────────────────────────
BASINS_WIDE_RIFT = {
    "Songliao": [
        (122.7, 42.0), (123.5, 42.5), (124.5, 43.5), (125.5, 44.5),
        (126.5, 45.5), (127.3, 46.5), (127.5, 47.5), (127.0, 48.7),
        (125.7, 49.2), (124.5, 48.5), (123.7, 47.5), (123.2, 46.5),
        (122.7, 45.5), (122.3, 44.0), (122.5, 43.0),
    ],
    "Hailar": [
        (117.0, 48.5), (117.5, 49.3), (118.3, 50.0), (119.5, 50.5),
        (120.7, 50.5), (121.7, 50.0), (122.0, 49.3), (121.7, 48.7),
        (120.5, 48.3), (118.7, 48.3), (117.5, 48.4),
    ],
    "Erlian": [
        (109.5, 43.2), (110.5, 43.5), (111.7, 43.7), (113.0, 43.7),
        (114.5, 43.8), (116.0, 44.2), (117.2, 44.7), (117.0, 45.4),
        (116.0, 45.2), (114.5, 44.7), (113.0, 44.5), (111.7, 44.3),
        (110.3, 43.8),
    ],
}

# ── 4. Faults / sutures ─────────────────────────────────────────────────────
FAULTS = {
    "Tanlu": [
        (122.0, 38.5), (123.0, 40.0), (124.0, 41.5), (125.5, 43.5),
        (127.5, 45.5), (129.5, 47.5), (131.0, 50.0), (132.0, 52.5),
    ],
    "MOS": [
        (LON_MIN, 51.0), (109.0, 51.5), (113.0, 52.5), (118.0, 53.0),
        (123.0, 53.0), (128.0, 52.5), (132.0, 52.0),
    ],
    "Solonker": [
        (LON_MIN, 42.5), (108.0, 42.5), (112.0, 42.7), (115.0, 42.7),
        (118.0, 42.5), (121.0, 42.5), (123.5, 42.0),
    ],
}
FAULT_KINEMATIC = {"Tanlu": "sinistral", "MOS": "thrust", "Solonker": "thrust"}
FAULT_CERTAINTY = {"Tanlu": "certain", "MOS": "approximate", "Solonker": "approximate"}

REGION_LABELS = [
    (115.0, 54.2, "Siberian Craton", 9),
    (112.0, 47.9, "CAOB", 9),
    (117.0, 39.4, "North China Craton", 9),
]
FAULT_LABELS = [
    (123.5, 41.3, "Tanlu Fault Zone", "italic"),
    (115.0, 53.5, "Mongol-Okhotsk Suture", "italic"),
    (110.5, 41.7, "Solonker Suture", "italic"),
]

# ── 5. The fifteen K1 samples (real coordinates from dz-global-db curated) ───
# name: (lat, lon, group)
SAMPLES = {
    "05FW007": (40.833, 115.833, "NCC"), "05FW008": (40.833, 115.833, "NCC"),
    "100821-1": (40.733, 113.817, "NCC"), "100821-3": (40.733, 113.817, "NCC"),
    "SL0929": (43.090, 124.488, "Songliao"), "SL0930": (43.100, 124.692, "Songliao"),
    "SL0931": (43.100, 124.696, "Songliao"), "SL0932": (43.100, 124.701, "Songliao"),
    "R31354-S12518": (43.290, 122.318, "Songliao"),
    "R31354-S12519": (43.290, 122.318, "Songliao"),
    "R31370-S12650": (45.873, 125.532, "Songliao"),
    "R31370-S12651": (45.873, 125.532, "Songliao"),
    "R31370-S12652": (43.881, 125.779, "Songliao"),
    "R31370-S12653": (46.465, 125.803, "Songliao"),
    "R31370-S12654": (45.637, 125.179, "Songliao"),
}
GRP_C = {"NCC": OKABE_ITO["reddish_purple"], "Songliao": "#595959"}
GRP_MARKER = {"NCC": "s", "Songliao": "o"}


def short(name):
    """Abbreviate Puetz R-series ids for compact site labels."""
    return name.split("-")[-1] if name.startswith("R313") else name


def sites():
    """Merge samples to sites (0.1 deg) -> (lon, lat, group, [names])."""
    buckets = {}
    for name, (lat, lon, grp) in SAMPLES.items():
        key = (round(lat, 1), round(lon, 1), grp)
        buckets.setdefault(key, []).append(name)
    out = []
    for (rlat, rlon, grp), names in buckets.items():
        lats = [SAMPLES[n][0] for n in names]
        lons = [SAMPLES[n][1] for n in names]
        out.append((float(np.mean(lons)), float(np.mean(lats)), grp, names))
    return out


# per-site label text-offset in points (dx, dy, ha) chosen to avoid collisions
LABEL_OFFSET = {
    "05FW007/8":      (7, -10, "left"),
    "100821-1/3":     (-7, -10, "right"),
    "SL0929":         (-4, -11, "right"),
    "SL0930/1/2":     (8, -9, "left"),
    "S12518/9":       (-8, 2, "right"),
    "S12650/1":       (9, -1, "left"),
    "S12652":         (9, -1, "left"),
    "S12653":         (8, 7, "left"),
    "S12654":         (-8, -2, "right"),
}


def site_label(names):
    if names[0].startswith("05FW"):
        return "05FW007/8"
    if names[0].startswith("100821"):
        return "100821-1/3"
    if set(names) == {"SL0930", "SL0931", "SL0932"}:
        return "SL0930/1/2"
    if names[0].startswith("SL0929"):
        return "SL0929"
    shorts = sorted(short(n) for n in names)
    if len(shorts) == 2 and shorts[0][:-1] == shorts[1][:-1]:
        return f"{shorts[0]}/{shorts[1][-1]}"      # S12518/9
    return "/".join(shorts)


def build_figure() -> plt.Figure:
    apply_journal_rcparams("GJ")
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]

    fig = plt.figure(figsize=(6.85, 4.8))
    ax = fig.add_axes([0.055, 0.075, 0.66, 0.90])
    legend_ax = fig.add_axes([0.735, 0.075, 0.255, 0.90])

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect(geographic_aspect(MID_LAT))
    ax.set_facecolor(BASIN_FAMILY_PALETTE["ocean"])

    ax.add_patch(MplPolygon(
        chaikin_smooth(COASTLINE, iterations=4, closed=True), closed=True,
        facecolor=BASIN_FAMILY_PALETTE["land"], edgecolor="#888888", lw=0.6, zorder=1))

    add_polygon_collection(
        ax, BASINS_WIDE_RIFT, facecolor=BASIN_FAMILY_PALETTE["wide_rift"],
        edgecolor="#5E3F87", lw=0.6, zorder=4, alpha=1.0)
    label_polygons(ax, {k: BASINS_WIDE_RIFT[k] for k in ("Hailar", "Erlian")},
                   color="#1A1A1A", fontsize=8, weight="bold")
    # Songliao label placed manually in the northern (sample-free) part of the basin
    ax.text(124.3, 48.3, "Songliao", fontsize=8, weight="bold", color="#1A1A1A",
            ha="center", va="center", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.0))

    for name, xy in FAULTS.items():
        plot_fault(ax, xy, color=BASIN_FAMILY_PALETTE["fault"], lw=1.6,
                   certainty=FAULT_CERTAINTY[name], kinematic=FAULT_KINEMATIC[name], zorder=7)

    plate_motion_arrow(ax, start=(133.5, 41.5), end=(130.0, 44.0))
    ax.text(134.5, 40.0, "Palaeo-Pacific\nsubduction", fontsize=7.5, ha="right", va="top",
            style="italic", color="#2B4F8F", zorder=10,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

    # ── sample markers + site labels ──
    for lon, lat, grp, names in sites():
        ax.scatter(lon, lat, s=52 if grp == "NCC" else 44, marker=GRP_MARKER[grp],
                   facecolor=GRP_C[grp], edgecolor="black", lw=0.7, zorder=9)
        lab = site_label(names)
        dx, dy, ha = LABEL_OFFSET.get(lab, (6, 4, "left"))
        ax.annotate(lab, (lon, lat), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va="center", fontsize=7.0, color="#222222", zorder=11,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6))

    for lon, lat, txt, fsize in REGION_LABELS:
        ax.text(lon, lat, txt, fontsize=fsize, weight="bold", color="#333333",
                ha="center", va="center", zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    for lon, lat, txt, style in FAULT_LABELS:
        ax.text(lon, lat, txt, fontsize=7.5, style=style, color="#1A1A1A",
                ha="center", va="center", zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    ax.set_xticks(np.arange(105, 136, 5))
    ax.set_yticks(np.arange(40, 56, 5))
    ax.set_xticklabels([f"{x}°E" for x in ax.get_xticks()], fontsize=7.5)
    ax.set_yticklabels([f"{y}°N" for y in ax.get_yticks()], fontsize=7.5)
    ax.grid(True, color="#BFBFBF", lw=0.3, ls=":", zorder=0)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)

    draw_scale_bar(ax, anchor_x=106.5, anchor_y=39.0, length_km=400, mid_lat=MID_LAT, fontsize=7)
    draw_north_arrow(ax, x=107.0, y=52.5, size=1.0, fontsize=8)

    # ── side legend ──
    legend_ax.axis("off"); legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1)
    y = 0.93

    def L(s, **kw):
        nonlocal y
        legend_ax.text(0.02, y, s, va="top", ha="left", **kw); y -= 0.045

    def patch_row(label, color, edge):
        nonlocal y
        legend_ax.add_patch(Rectangle((0.02, y - 0.035), 0.10, 0.035, facecolor=color,
                            edgecolor=edge, lw=0.6, transform=legend_ax.transAxes, clip_on=False))
        legend_ax.text(0.16, y - 0.018, label, fontsize=7.8, va="center", ha="left"); y -= 0.055

    def line_row(label, ls):
        nonlocal y
        legend_ax.plot([0.02, 0.13], [y - 0.018, y - 0.018], color="#1A1A1A", lw=1.4, ls=ls,
                       solid_capstyle="round", transform=legend_ax.transAxes, clip_on=False)
        legend_ax.text(0.16, y - 0.018, label, fontsize=7.5, va="center", ha="left"); y -= 0.055

    def marker_row(label, marker, color):
        nonlocal y
        legend_ax.scatter([0.07], [y - 0.018], s=48, marker=marker, facecolor=color,
                          edgecolor="black", lw=0.7, transform=legend_ax.transAxes, clip_on=False)
        legend_ax.text(0.16, y - 0.018, label, fontsize=7.5, va="center", ha="left"); y -= 0.052

    L("Legend", fontsize=10, weight="bold"); y -= 0.012
    L("K1 DZ samples (this study)", fontsize=8.5, weight="bold")
    marker_row("NCC north-margin (n = 4)", "s", GRP_C["NCC"])
    marker_row("Songliao Basin (n = 11)", "o", GRP_C["Songliao"])
    y -= 0.015
    L("Cretaceous basins", fontsize=8.5, weight="bold")
    patch_row("Wide-rift / back-arc", BASIN_FAMILY_PALETTE["wide_rift"], "#5E3F87")
    y -= 0.01
    L("Faults / sutures", fontsize=8.5, weight="bold")
    line_row("Certain (solid)", "-")
    line_row("Approximate (long-dash)", (0, (6, 3)))
    L("Kinematic indicator", fontsize=8.5, weight="bold")
    legend_ax.text(0.05, y - 0.02, "→  ←   sinistral\n▲       thrust / suture",
                   fontsize=7.5, va="top", ha="left", linespacing=1.6); y -= 0.10
    L("Plate motion", fontsize=8.5, weight="bold")
    legend_ax.annotate("", xy=(0.13, y - 0.018), xytext=(0.02, y - 0.018),
                       xycoords=legend_ax.transAxes,
                       arrowprops=dict(arrowstyle="-|>", color="#2B4F8F", lw=2.2, mutation_scale=14))
    legend_ax.text(0.16, y - 0.018, "Palaeo-Pacific\nplate vector", fontsize=7.5,
                   va="center", ha="left", linespacing=1.2); y -= 0.075
    # 制图规范注记移除(2026-07-08):工艺 meta 信息不上图面,归 caption/Methods
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--stem", default="Fig1_sample_locations")
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()

    fig = build_figure()
    if args.audit:
        from layout_check import audit_figure
        issues = audit_figure(fig, journal="GJ")
        for i in issues:
            print("!! ", i)
        print(f"-- {len(issues)} layout issue(s)")
    written = save_three_formats(fig, args.output_dir, args.stem)
    for p in written:
        print(f"wrote {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
