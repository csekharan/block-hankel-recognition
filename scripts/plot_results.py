#!/usr/bin/env python3
"""Plot the shared-matrix benchmark in results/time_colab.csv.  Every axis is linear.

Figures A and B reproduce the two charts of notebooks/block_Hankel_benchmarking.ipynb
verbatim (same code, same colours).  Figures C-E are additional views of the same data,
drawn with a colour-blind-safe palette and one small panel per method so that no method
flattens another:

  C  runtime against matrix entries, one panel per method and input kind, each on its own
     linear scale from zero
  D  cost per matrix entry (median time / m^2) per method and input kind; a flat panel is
     linear scaling and its height is the constant factor
  E  median time relative to 2-D hashing on the block-Hankel positives

Usage:  python scripts/plot_results.py [--csv results/time_colab.csv] [--out results/figures] [--reps 3]
Also writes results/summary_medians.csv (median per size and kind, and ns per entry).
"""
import argparse
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
METHOD_KEYS = ["gp_ms", "zpass_ms", "hash_k2_ms", "direct_ms"]

# ------------------------------------------------------------------ notebook figures (verbatim)
NB_METHODS = [("gp_ms", "Galil–Park (deterministic)", "#1f4e79", "o"),
              ("zpass_ms", "Row Z-pass (deterministic)", "#2e8b57", "s"),
              ("hash_k2_ms", "2-D polynomial hashing, k = 2 (Monte Carlo)", "#c0392b", "^"),
              ("direct_ms", "Direct comparison, early exit (baseline)", "#8e44ad", "D")]


def notebook_figures(df, out, reps):
    kinds = [("Yes", "Block Hankel"), ("No", "No block structure")] + \
            ([("Near", "Near block Hankel")] if (df.block_hankel == "Near").any() else [])
    sizes = sorted(df["size"].unique())

    def draw(ax, kind, keep):
        sub = df[df.block_hankel == kind]
        for key, label, color, marker in keep:
            xs, meds = [], []
            for m in sizes:
                ts = list(sub[sub["size"] == m][key])
                if not ts:
                    continue
                ax.scatter([m * m / 1e6] * len(ts), ts, s=18, color=color, marker=marker, alpha=0.4, zorder=3)
                xs.append(m * m / 1e6); meds.append(statistics.median(ts))
            ax.plot(xs, meds, color=color, marker=marker, markersize=6, linewidth=1.8, label=label, zorder=4)
        ax.set_xlabel("matrix size (millions of entries)"); ax.set_ylabel("time (ms)"); ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    figA, axesA = plt.subplots(1, len(kinds), figsize=(5.5 * len(kinds), 4.5))
    for ax, (kind, title) in zip(axesA, kinds):
        draw(ax, kind, NB_METHODS[:1]); ax.set_title(f"Galil–Park witness computation: {title}")
    figA.suptitle(f"Galil–Park (deterministic, exact); single core, best of {reps} runs, 3 matrices per size", fontsize=10.5)
    figA.tight_layout(rect=(0, 0, 1, 0.94)); figA.savefig(out / "fig_A_galil_park_linear.png", dpi=170)

    figB, axesB = plt.subplots(1, len(kinds), figsize=(5.5 * len(kinds), 4.5))
    for ax, (kind, title) in zip(axesB, kinds):
        draw(ax, kind, NB_METHODS[1:]); ax.set_title(title)
    axesB[0].legend(loc="upper left", fontsize=8.5, frameon=False)
    figB.suptitle(f"Row Z-pass, 2-D hashing (k = 2, P = 2^61−1) and direct comparison; single core, best of {reps} runs, 3 matrices per size", fontsize=10.5)
    figB.tight_layout(rect=(0, 0, 1, 0.94)); figB.savefig(out / "fig_B_fast_methods_linear.png", dpi=170)
    plt.close("all")


# ------------------------------------------------------------------ derived figures (linear axes)
# Categorical palette validated for colour-vision deficiency (adjacent pairs, light surface);
# the colour follows the method in every figure below.
METHODS = [("gp_ms", "Galil–Park witness computation", "Galil–Park", "#2a78d6", "o"),
           ("zpass_ms", "Row Z-pass", "Z-pass", "#eb6834", "s"),
           ("hash_k2_ms", "2-D polynomial hashing, k = 2", "hashing", "#1baf7a", "^"),
           ("direct_ms", "Direct comparison, early exit", "direct", "#4a3aa7", "D")]
KINDS = [("Yes", "Block Hankel (3 lattice positives per size)"), ("No", "No block structure (3 random negatives per size)")]
KIND_SHORT = {"Yes": "block Hankel", "No": "random"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

DIRECT_NEG_NOTE = (
    "Not plotted. On random matrices every\n"
    "candidate pair is rejected by its first\n"
    "memcmp: total time ≤ 0.1 ms at every size\n"
    "(the resolution of the CSV), ≈ 40–65 ns per\n"
    "candidate pair, independent of m. Work is\n"
    "Θ(d(m)²), not Θ(m²), so a per-entry figure\n"
    "would only be that floor divided by m².\n"
    "See Figure C.")
ZPASS_NEG_NOTE = (
    "total time still grows (0.1 → 77 ms), but the\n"
    "work is ≈ 2·m·d(m) memcmp calls that end in\n"
    "their first bytes: Θ(m·d(m)), sub-quadratic,\n"
    "so the cost per entry falls")


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS); ax.spines[side].set_linewidth(0.8)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def fmt_ms(v):
    if v >= 1000:
        return f"{v / 1000:.1f} s"
    return f"{v:.1f} ms" if v >= 1 else f"{v:.2f} ms"


def endpoint_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(-4, 7), textcoords="offset points", ha="right", va="bottom",
                fontsize=8, color=INK2)


def series(ax, xs, ys, color, marker, label=None):
    ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=5.5, markeredgecolor=SURFACE,
            markeredgewidth=1, solid_capstyle="round", label=label, zorder=4)


def ordinal_x(ax, sizes, xlabel):
    xs = np.arange(len(sizes))
    ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in sizes], rotation=45, ha="right", fontsize=7.5)
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    if xlabel:
        ax.set_xlabel(xlabel)
    return xs


def medians(df):
    return df.groupby(["block_hankel", "size"])[METHOD_KEYS].median()


def figure_C(df, out, med):
    """Runtime against entries, one linear panel per method and input kind."""
    sizes = sorted(df["size"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.4), facecolor=SURFACE)
    for r, (kind, _) in enumerate(KINDS):
        sub = df[df.block_hankel == kind]
        for c, (key, long, short, color, marker) in enumerate(METHODS):
            ax = axes[r][c]; style(ax)
            pts = sub[["size", key]].to_numpy(dtype=float)
            xs = np.array([m * m / 1e6 for m in sizes]); ys = np.array([med.loc[(kind, m), key] for m in sizes])
            ax.scatter(pts[:, 0] ** 2 / 1e6, pts[:, 1], s=14, color=color, marker=marker, alpha=0.35,
                       edgecolors=SURFACE, linewidths=0.6, zorder=3)
            ax.set_xlim(0, 240)
            ax.set_title(f"{short} · {KIND_SHORT[kind]}", fontsize=9.5, loc="left")
            if key == "direct_ms" and kind == "No":
                # every reading is 0.0 or 0.1 ms: quantisation, not a trend -- markers only, no line
                ax.plot(xs, ys, linestyle="none", marker=marker, markersize=5.5, color=color,
                        markeredgecolor=SURFACE, markeredgewidth=1, zorder=4)
                ax.axhline(0.1, color=AXIS, linewidth=0.8, zorder=2)
                ax.set_ylim(0, 0.34)
                ax.text(0.03, 0.95, "all 30 readings are 0.0 or 0.1 ms: at or below the\n"
                                    "resolution of the CSV (one memcmp per candidate pair,\n"
                                    "Θ(d(m)²) work, ≈ 40–65 ns per pair). No line drawn.",
                        transform=ax.transAxes, fontsize=7.2, color=INK2, va="top", linespacing=1.35)
            else:
                series(ax, xs, ys, color, marker)
                ax.set_ylim(0, float(pts[:, 1].max()) * 1.2)
                endpoint_label(ax, xs[-1], ys[-1], fmt_ms(ys[-1]))
            if c == 0:
                ax.set_ylabel("time (ms)")
            if r == 1:
                ax.set_xlabel("matrix entries (millions)")
    fig.suptitle("Runtime of each recognizer on its own linear scale — faded dots: the 3 matrices per size; "
                 "line: their median; best of 3 runs, single core", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955)); fig.savefig(out / "fig_C_runtime_small_multiples.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def figure_D(df, out, med):
    """Nanoseconds per entry, one linear panel per method and input kind, sizes equally spaced."""
    sizes = sorted(df["size"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 6.4), facecolor=SURFACE)
    for r, (kind, _) in enumerate(KINDS):
        for c, (key, long, short, color, marker) in enumerate(METHODS):
            ax = axes[r][c]; style(ax)
            ax.set_title(f"{short} · {KIND_SHORT[kind]}", fontsize=9.5, loc="left")
            xs = ordinal_x(ax, sizes, "matrix side m (ten sizes, equally spaced)" if r == 1 else None)
            if c == 0:
                ax.set_ylabel("nanoseconds per entry")
            if kind == "No" and key == "direct_ms":
                ax.set_ylim(0, 1); ax.set_yticks([]); ax.grid(False)
                ax.text(0.03, 0.95, DIRECT_NEG_NOTE, transform=ax.transAxes, fontsize=7.2, color=INK2, va="top", linespacing=1.35)
                continue
            ys = np.array([med.loc[(kind, m), key] * 1e6 / (m * m) for m in sizes])
            series(ax, xs, ys, color, marker)
            headroom = 1.8 if (kind == "No" and key == "zpass_ms") else 1.25
            ax.set_ylim(0, ys.max() * headroom)
            endpoint_label(ax, xs[-1], ys[-1], f"{ys[-1]:.1f} ns" if ys[-1] >= 1 else f"{ys[-1]:.2f} ns")
            if kind == "No" and key == "zpass_ms":
                ax.text(0.97, 0.95, ZPASS_NEG_NOTE, transform=ax.transAxes, fontsize=7.2, color=INK2, va="top", ha="right", linespacing=1.35)
    fig.suptitle("Cost per matrix entry (median time ÷ m²) on linear axes — a flat panel is linear scaling; its height is the constant factor",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955)); fig.savefig(out / "fig_D_ns_per_entry.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def figure_E(df, out, med):
    """Block-Hankel positives, median time relative to hashing, linear axes, two panels."""
    sizes = sorted(df["size"].unique())
    base = np.array([med.loc[("Yes", m), "hash_k2_ms"] for m in sizes])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.15, 1]})
    groups = [("Row Z-pass and direct comparison, relative to hashing", ["zpass_ms", "direct_ms"], 4.0),
              ("Galil–Park, relative to hashing", ["gp_ms"], 18.0)]
    for ax, (title, keys, ytop) in zip(axes, groups):
        style(ax)
        xs = ordinal_x(ax, sizes, "matrix side m (ten sizes, equally spaced)")
        ax.axhline(1.0, color=MUTED, linewidth=1, zorder=2)
        ax.text(len(sizes) - 0.6, 1.0, "hashing = 1", fontsize=8, color=MUTED, va="bottom", ha="right")
        for key, long, short, color, marker in METHODS:
            if key not in keys:
                continue
            ratio = np.array([med.loc[("Yes", m), key] for m in sizes]) / base
            series(ax, xs, ratio, color, marker, label=long)
            endpoint_label(ax, xs[-1], ratio[-1], f"{ratio[-1]:.1f}×")
        ax.set_ylim(0, ytop)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_ylabel("median time ÷ median hashing time")
        if len(keys) > 1:
            ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK2)
    fig.suptitle("Block-Hankel positives: time relative to 2-D hashing (linear axes)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out / "fig_E_ratio_to_hashing.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def summary_csv(df, med, path):
    rows = []
    for (kind, m), r in med.iterrows():
        row = {"block_hankel": kind, "size": m, "entries": m * m}
        for k in METHOD_KEYS:
            row[f"median_{k}"] = round(r[k], 1)
            row[f"ns_per_entry_{k.replace('_ms', '')}"] = round(r[k] * 1e6 / (m * m), 3)
        rows.append(row)
    pd.DataFrame(rows).sort_values(["block_hankel", "size"], ascending=[False, True]).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=ROOT / "results" / "time_colab.csv", type=pathlib.Path)
    ap.add_argument("--out", default=ROOT / "results" / "figures", type=pathlib.Path)
    ap.add_argument("--reps", default=3, type=int, help="best-of-REPS shown in the notebook figure titles")
    ap.add_argument("--skip-notebook", action="store_true", help="do not regenerate figures A and B")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv)
    med = medians(df)
    if not a.skip_notebook:
        notebook_figures(df, a.out, a.reps)
    figure_C(df, a.out, med); figure_D(df, a.out, med); figure_E(df, a.out, med)
    summary_csv(df, med, a.csv.parent / "summary_medians.csv")
    print("wrote", sorted(p.name for p in a.out.glob("fig_*.png")), "and summary_medians.csv")


if __name__ == "__main__":
    main()
