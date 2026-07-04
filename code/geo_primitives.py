"""Geological figure primitives — matplotlib helpers for FGDC-compliant maps.

Extracted from fig01_tectonic_basemap.py (Pan Lujia, 2026-04 to 2026-05).
All public functions are stable, documented, and importable from any
fig0N_*.py figure script:

    from geo_primitives import (
        chaikin_smooth, plot_fault, draw_scale_bar, draw_north_arrow,
        add_polygon_collection, label_polygons, apply_journal_rcparams,
        OKABE_ITO, BASIN_FAMILY_PALETTE,
    )

This module has no project-specific state — basin polygon coordinates,
labels, etc. live in the per-figure script.  Only generic primitives.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon, Rectangle


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

OKABE_ITO = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Basin-family palette tuned for NE-Asia Cretaceous tectonic maps (Pan 2026).
BASIN_FAMILY_PALETTE = {
    "passive_rift": "#F4B678",   # warm tan      (E NCC: Jiaolai, Yishu, Hefei)
    "active_rift":  "#E25A5A",   # carmine       (N NCC: Jehol, Luanping)
    "wide_rift":    "#B284D8",   # lavender      (ECAO: Songliao, Hailar)
    "korea_basin":  "#4DA6A6",   # teal          (Gyeongsang, Eumseong, ...)
    "south_china":  "#D5A35A",   # ochre         (Ganzhou, Jianghan)
    "massif":       "#D9D9D9",   # light grey    (Precambrian)
    "uhp_belt":     "#8AB570",   # olive         (Sulu, Dabie)
    "ocean":        "#EAF3F8",   # very pale blue
    "land":         "#FAF7F0",   # cream
    "fault":        "#1A1A1A",   # near-black for fault traces
}


# ---------------------------------------------------------------------------
# Geometry: Chaikin's corner-cutting algorithm
# ---------------------------------------------------------------------------

def chaikin_smooth(points, iterations: int = 4, closed: bool = True):
    """Round off sharp corners on a polygon or polyline.

    Each iteration replaces every vertex with two new ones at 1/4 and 3/4
    along each edge.  4 iterations → 16x denser, curves look like real
    coastlines / fault traces.

    points: array-like (N, 2) of (x, y) vertices.
    closed: True for polygons (wraps last→first); False for open polylines
            like fault traces (preserves endpoints).
    """
    pts = np.asarray(points, dtype=float)
    if closed:
        for _ in range(iterations):
            n = len(pts)
            new = np.empty((2 * n, 2))
            for i in range(n):
                p0 = pts[i]
                p1 = pts[(i + 1) % n]
                new[2 * i]     = 0.75 * p0 + 0.25 * p1
                new[2 * i + 1] = 0.25 * p0 + 0.75 * p1
            pts = new
    else:
        for _ in range(iterations):
            n = len(pts)
            out = [pts[0]]
            for i in range(n - 1):
                p0, p1 = pts[i], pts[i + 1]
                out.append(0.75 * p0 + 0.25 * p1)
                out.append(0.25 * p0 + 0.75 * p1)
            out.append(pts[-1])
            pts = np.array(out)
    return pts


# ---------------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------------

def add_polygon_collection(
    ax, polys, *, facecolor, edgecolor,
    alpha: float = 0.85, lw: float = 0.6, zorder: int = 3,
    hatch: str | None = None, smooth: bool = True, smooth_iters: int = 4,
):
    """Add a dict of named polygons to an axis as a single PatchCollection.

    polys : dict[name, vertices]  — vertices is array-like (N, 2).
    Returns the PatchCollection (so you can tweak it later).
    """
    patches = []
    for v in polys.values():
        verts = chaikin_smooth(v, iterations=smooth_iters, closed=True) if smooth \
                else np.asarray(v, dtype=float)
        patches.append(MplPolygon(verts, closed=True))
    pc = PatchCollection(
        patches, facecolor=facecolor, edgecolor=edgecolor,
        alpha=alpha, lw=lw, zorder=zorder, hatch=hatch,
    )
    ax.add_collection(pc)
    return pc


def label_polygons(
    ax, polys, *, color="black", fontsize: float = 8,
    style: str = "normal", weight: str = "normal",
    bbox_alpha: float = 0.55,
):
    """Label each polygon at its centroid (vertex mean — adequate for schematics)."""
    for name, verts in polys.items():
        arr = np.asarray(verts)
        ax.text(
            arr[:, 0].mean(), arr[:, 1].mean(), name,
            color=color, fontsize=fontsize,
            style=style, weight=weight,
            ha="center", va="center", zorder=12,
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=bbox_alpha, pad=0.5),
        )


def polygon_centroid(verts) -> tuple[float, float]:
    """Vertex-mean centroid (sufficient for hand-digitised schematic polygons)."""
    arr = np.asarray(verts)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


# ---------------------------------------------------------------------------
# FGDC fault / contact line styles
# ---------------------------------------------------------------------------

# FGDC TM 11A02 §1 (Contacts) and §2 (Faults) line conventions:
#   certain      — solid line, "location accurate to within map scale"
#   approximate  — long-dashed line (~6× lw on / 3× lw off)
#   inferred     — short-dashed line (~3× lw on / 3× lw off)
#   concealed    — dotted (round caps), "feature continues under cover"
#   queried      — append "?" near the trace where existence is uncertain
FGDC_CERTAINTY_LS = {
    "certain":      "-",
    "approximate":  (0, (6, 3)),     # long-dashed
    "inferred":     (0, (3, 3)),     # short-dashed
    "concealed":    (0, (1, 3)),     # dotted
}


def fault_certainty_linestyle(certainty: str):
    """Return matplotlib linestyle for an FGDC certainty category."""
    if certainty not in FGDC_CERTAINTY_LS:
        raise ValueError(
            f"unknown certainty {certainty!r}; "
            f"expected one of {list(FGDC_CERTAINTY_LS)}"
        )
    return FGDC_CERTAINTY_LS[certainty]


def plot_fault(
    ax, xy, *,
    color: str = "#1A1A1A",
    lw: float = 1.6,
    certainty: str = "certain",
    kinematic: str | None = None,
    smooth: bool = True,
    smooth_iters: int = 4,
    zorder: int = 7,
    queried: bool = False,
):
    """Plot a fault trace following FGDC TM 11A02 §2 conventions.

    xy        : array-like (N, 2) of fault-trace vertices (in plot coords).
    certainty : "certain" | "approximate" | "inferred" | "concealed"
                → solid / long-dash / short-dash / dotted (FGDC §2.1).
    kinematic : None | "sinistral" | "dextral" | "normal" | "reverse" | "thrust"
                None → no kinematic indicator.  Each kinematic adds the
                appropriate FGDC marginal symbol at trace midpoint.
    queried   : True appends a "?" label at the trace midpoint
                (FGDC TM 11A02 §1.3 — existence/identity queried).
    """
    xy = np.asarray(xy, dtype=float)
    trace = chaikin_smooth(xy, iterations=smooth_iters, closed=False) if smooth else xy
    ls = fault_certainty_linestyle(certainty)
    cap = "round" if certainty == "certain" else "butt"
    ax.plot(
        trace[:, 0], trace[:, 1], color=color, lw=lw, ls=ls, zorder=zorder,
        solid_capstyle=cap, dash_capstyle="butt",
    )
    if kinematic is not None:
        _add_kinematic_indicator(ax, xy, kinematic, color, zorder + 1)
    if queried:
        mid = xy[max(1, len(xy) // 2)]
        ax.text(
            mid[0], mid[1], "?", color=color, fontsize=10, weight="bold",
            ha="center", va="center", zorder=zorder + 2,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.0),
        )


def _add_kinematic_indicator(ax, xy, kinematic: str, color: str, zorder: int):
    """Draw FGDC kinematic indicator at fault midpoint.

    sinistral : pair of half-arrows (→ above trace, ← below)
    dextral   : pair of half-arrows (← above, → below)
    normal    : tick marks + ball on hanging-wall side (down-thrown)
    reverse   : triangular barbs on hanging-wall side (sense same as thrust
                but typically drawn smaller; both use barbs in FGDC §2)
    thrust    : triangular barbs (filled triangles) on hanging-wall side
    """
    xy = np.asarray(xy, dtype=float)
    mid_idx = max(1, len(xy) // 2)
    p0, p1 = xy[mid_idx - 1], xy[mid_idx]
    dx, dy = p1 - p0
    norm = (dx ** 2 + dy ** 2) ** 0.5
    if norm == 0:
        return
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux                    # perpendicular (left-of-trace)
    mx, my = (p0 + p1) / 2

    if kinematic in ("sinistral", "dextral"):
        off = 0.65
        arr_len = 1.20
        half = arr_len / 2
        # Forward arrow on left side (sinistral) or right side (dextral)
        side1 = +1 if kinematic == "sinistral" else -1
        side2 = -side1
        for side, sign in ((side1, +1), (side2, -1)):
            ax.annotate(
                "",
                xy=(mx + side * px * off + sign * ux * half,
                    my + side * py * off + sign * uy * half),
                xytext=(mx + side * px * off - sign * ux * half,
                        my + side * py * off - sign * uy * half),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=0.9, mutation_scale=8),
                zorder=zorder,
            )
        return

    if kinematic == "normal":
        # Ball-and-bar on hanging-wall (left) side
        bar_off = 0.4
        ball_off = 0.7
        ax.plot([mx, mx + px * bar_off], [my, my + py * bar_off],
                color=color, lw=1.0, zorder=zorder)
        ax.scatter([mx + px * ball_off], [my + py * ball_off],
                   s=18, facecolor=color, edgecolor=color, zorder=zorder)
        return

    if kinematic in ("thrust", "reverse"):
        # Triangular barb (filled) on hanging-wall (left) side
        barb_size = 0.5
        # Triangle: tip at mid + px*barb_size, base on the trace
        tip = (mx + px * barb_size, my + py * barb_size)
        base1 = (mx - ux * barb_size * 0.6, my - uy * barb_size * 0.6)
        base2 = (mx + ux * barb_size * 0.6, my + uy * barb_size * 0.6)
        tri = MplPolygon([tip, base1, base2], closed=True,
                          facecolor=color, edgecolor=color, lw=0.5,
                          zorder=zorder)
        ax.add_patch(tri)
        return

    raise ValueError(f"unknown kinematic {kinematic!r}")


# ---------------------------------------------------------------------------
# FGDC marginalia: scale bar, north arrow
# ---------------------------------------------------------------------------

def draw_scale_bar(
    ax, anchor_x: float, anchor_y: float, length_km: float = 200, *,
    mid_lat: float | None = None,
    bar_height: float = 0.45,
    segments: int = 2,
    label_format: str | None = None,
    fontsize: float = 6,
    zorder: int = 15,
):
    """Draw an alternating black/white bar scale (FGDC §35).

    If `mid_lat` is given, the bar's length is converted from km to degrees
    longitude using cos(mid_lat).  Otherwise length_km is interpreted as
    plot-coordinate units directly (use this for non-geographic plots).

    `segments` is the number of alternating black/white segments (FGDC
    examples typically use 2 or 4).
    """
    if mid_lat is not None:
        deg_per_km = 1.0 / (111.0 * np.cos(np.deg2rad(mid_lat)))
        full = length_km * deg_per_km
    else:
        full = length_km
    seg_len = full / segments
    for i in range(segments):
        face = "#1a1a1a" if i % 2 == 0 else "white"
        ax.add_patch(Rectangle(
            (anchor_x + i * seg_len, anchor_y), seg_len, bar_height,
            facecolor=face, edgecolor="#1a1a1a", lw=0.6, zorder=zorder,
        ))
    label = label_format or f"{int(length_km)} km"
    ax.text(anchor_x, anchor_y - bar_height * 1.2, "0",
            fontsize=fontsize, ha="center", va="top", zorder=zorder + 1)
    ax.text(anchor_x + full, anchor_y - bar_height * 1.2, label,
            fontsize=fontsize, ha="center", va="top", zorder=zorder + 1)


def draw_north_arrow(
    ax, x: float, y: float, *,
    size: float = 2.0,
    fontsize: float = 8,
    zorder: int = 15,
    declination_deg: float | None = None,
):
    """Draw a triangular north arrow with 'N' label (FGDC §36).

    `size` is arrow height in plot-coordinate units (typically degrees of
    latitude on a geographic map).

    `declination_deg` (optional, FGDC §36): if given, also draws a thin
    line at the magnetic-declination angle (positive = east of true north),
    with the declination value labeled.  Useful for field-survey maps;
    omit for synthesis / regional schematics.
    """
    tri = np.array([
        (x,                 y + size),                  # tip (N)
        (x - size * 0.30,   y - size * 0.20),           # bottom-left
        (x,                 y - size * 0.05),           # bottom-mid notch
        (x + size * 0.30,   y - size * 0.20),           # bottom-right
    ])
    ax.add_patch(MplPolygon(
        tri, closed=True, facecolor="#1a1a1a", edgecolor="#1a1a1a",
        lw=0.6, zorder=zorder,
    ))
    ax.text(x, y + size + 0.55, "N",
            fontsize=fontsize, weight="bold",
            ha="center", va="bottom", zorder=zorder + 1)
    if declination_deg is not None:
        # Draw a thin line at declination angle from the same tip point
        rad = np.deg2rad(declination_deg)
        dx, dy = size * np.sin(rad), size * np.cos(rad)
        ax.plot([x, x + dx], [y, y + dy],
                color="#666", lw=0.6, zorder=zorder)
        ax.text(x + dx, y + dy + 0.2, f"MN {declination_deg:+.0f}°",
                fontsize=fontsize - 1, ha="center", va="bottom",
                zorder=zorder + 1, color="#666")


# ---------------------------------------------------------------------------
# Plate motion / vector arrows (regional tectonic maps)
# ---------------------------------------------------------------------------

def plate_motion_arrow(
    ax, start, end, *,
    color: str = "#2B4F8F",
    lw: float = 2.4,
    mutation_scale: float = 18,
    zorder: int = 9,
):
    """Draw a thick blue arrow representing a plate-motion vector."""
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=mutation_scale),
        zorder=zorder,
    )


# ---------------------------------------------------------------------------
# Markers for case-study basins / numbered keys
# ---------------------------------------------------------------------------

def case_study_star(
    ax, x: float, y: float, *,
    color: str = "#B32020",
    edge: str = "#400000",
    size: float = 220,
    lw: float = 0.8,
    zorder: int = 14,
):
    """Five-point red star — overlay on case-study basin centroids."""
    ax.scatter([x], [y], marker="*", s=size,
               facecolor=color, edgecolor=edge, lw=lw, zorder=zorder)


def numbered_marker(
    ax, x: float, y: float, label: str | int, *,
    facecolor: str = "#1a1a1a", edgecolor: str = "white",
    text_color: str = "white", marker: str = "o",
    size: float = 70, lw: float = 0.7, fontsize: float = 5.8,
    zorder: int = 15,
):
    """Numbered marker (circle by default) — for legend keys with leader lists."""
    ax.scatter([x], [y], marker=marker, s=size,
               facecolor=facecolor, edgecolor=edgecolor, lw=lw, zorder=zorder)
    ax.text(x, y, str(label), color=text_color,
            fontsize=fontsize, weight="bold",
            ha="center", va="center", zorder=zorder + 1)


# ---------------------------------------------------------------------------
# Journal format setup
# ---------------------------------------------------------------------------

# Geological Journal (Wiley) figure-format constants — verified against
# Wiley author guidelines as of 2026-04 for journal ISSN 1099-1034.
GJ_DOUBLE_COL_MM = 174
GJ_SINGLE_COL_MM = 84
MM_PER_INCH = 25.4
GJ_DOUBLE_COL_IN = GJ_DOUBLE_COL_MM / MM_PER_INCH
GJ_SINGLE_COL_IN = GJ_SINGLE_COL_MM / MM_PER_INCH


def apply_journal_rcparams(journal: str = "GJ") -> None:
    """Apply matplotlib rcParams for a target journal.

    Sets sans-serif font with CJK fallback, sane sizes per journal min-spec,
    and `pdf.fonttype=42` so AI/Inkscape can edit text in the exported PDF.

    Currently supports:
      "GJ"  — Geological Journal (Wiley)
      "ESR" — Earth-Science Reviews (Elsevier; max double-col 190 mm)
      "TEC" — Tectonics (AGU; max double-col 168 mm)
      "GR"  — Gondwana Research (Elsevier; max double-col 190 mm)
    """
    common = {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "DejaVu Sans", "Arial", "Liberation Sans",
            # CJK fallback chain — matplotlib 3.6+ per-glyph fallback picks
            # whichever font is installed on the host:
            "Droid Sans Fallback", "Noto Sans CJK SC", "SimHei",
        ],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,    # TrueType (editable in Illustrator)
        "ps.fonttype":  42,
    }
    if journal in ("GJ", "TEC"):
        sizes = {
            "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
            "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        }
    elif journal in ("ESR", "GR"):
        sizes = {
            "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
            "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
        }
    else:
        raise ValueError(f"unknown journal {journal!r}; expected GJ/TEC/ESR/GR")
    plt.rcParams.update({**common, **sizes})


# ---------------------------------------------------------------------------
# Geographic-aspect helper
# ---------------------------------------------------------------------------

def geographic_aspect(mid_lat: float) -> float:
    """Return matplotlib `set_aspect` value for an unprojected lon/lat plot.

    Without correction, lon/lat axes look stretched at high latitudes.
    Aspect = 1 / cos(mid_lat) keeps shapes visually correct in mid-frame.
    """
    return 1.0 / np.cos(np.deg2rad(mid_lat))


# ---------------------------------------------------------------------------
# Save outputs in three formats (PDF/PNG/SVG)
# ---------------------------------------------------------------------------

def save_three_formats(
    fig, out_dir, stem: str, *,
    dpi_png: int = 300,
    facecolor: str = "white",
    bbox_inches: str = "tight",
    strip_pdf_metadata: bool = True,
):
    """Write a figure to <out_dir>/<stem>.{pdf,png,svg}.

    PDF metadata is stripped by default (CreationDate/ModDate/Producer set
    to None) for reproducible builds.  PNG is rasterized at dpi_png.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_meta = {"CreationDate": None, "ModDate": None, "Producer": None} \
               if strip_pdf_metadata else None
    written = []
    for ext in ("pdf", "png", "svg"):
        save_kw = dict(bbox_inches=bbox_inches, facecolor=facecolor)
        if ext == "png":
            save_kw["dpi"] = dpi_png
        if ext == "pdf" and pdf_meta is not None:
            save_kw["metadata"] = pdf_meta
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, **save_kw)
        written.append(path)
    return written
