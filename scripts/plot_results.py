#!/usr/bin/env python3
"""Plot the shared-matrix benchmark in results/time_colab.csv.

Figures A and B reproduce the two charts of notebooks/block_Hankel_benchmarking.ipynb
verbatim (same code, same colours, linear axes).  Figures C-E are additional views of the
same data, drawn with a colour-blind-safe palette:

  C  runtime on log-log axes, all four methods, one panel per input kind
  D  cost per matrix entry (median time / m^2); a flat line means linear scaling
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


# ------------------------------------------------------------------ derived figures
# Categorical palette validated for colour-vision deficiency (adjacent pairs, light surface);
# the colour follows the method in every figure below.
METHODS = [("gp_ms", "Galil–Park witness computation", "Galil–Park", "#2a78d6", "o"),
           ("zpass_ms", "Row Z-pass", "Z-pass", "#eb6834", "s"),
           ("hash_k2_ms", "2-D polynomial hashing, k = 2", "hashing", "#1baf7a", "^"),
           ("direct_ms", "Direct comparison, early exit", "direct", "#4a3aa7", "D")]
KINDS = [("Yes", "Block Hankel (3 lattice positives per size)"), ("No", "No block structure (3 random negatives per size)")]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FLOOR_MS = 0.05          # a recorded 0.0 ms means "below the 0.1 ms resolution of the CSV"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS); ax.spines[side].set_linewidth(0.8)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, linestyle="-")
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)
    ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def end_labels(ax, items, x, min_sep_frac=0.055):
    """items: (y, text) pairs.  Labels sit right of x; collisions are resolved by nudging
    in log space and drawing a hairline leader from the true y to the label."""
    lo, hi = np.log10(ax.get_ylim())
    sep = min_sep_frac * (hi - lo)
    items = sorted(items, key=lambda t: t[0])
    ys = [np.log10(y) for y, _ in items]
    placed = []
    for v in ys:
        placed.append(v if not placed or v - placed[-1] >= sep else placed[-1] + sep)
    if placed and placed[-1] > hi - 0.02 * (hi - lo):            # keep the stack inside the axes
        shift = placed[-1] - (hi - 0.02 * (hi - lo))
        placed = [p - shift for p in placed]
    for (y, text), p in zip(items, placed):
        yl = 10 ** p
        if abs(np.log10(y) - p) > 1e-9:
            ax.plot([x, x * 1.10], [y, yl], color=AXIS, linewidth=0.7, zorder=2, clip_on=False)
        ax.text(x * 1.13, yl, text, va="center", ha="left", fontsize=8.5, color=INK2, clip_on=False)


def figure_legend(fig, ax, relabel=None):
    """One legend for the whole figure, in the band under the title, so no plot area is covered."""
    handles, labels = ax.get_legend_handles_labels()
    if relabel:
        labels = [relabel.get(l, l) for l in labels]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.012, 0.955), ncol=4, frameon=False,
               fontsize=8.5, labelcolor=INK2, handlelength=2.2, columnspacing=1.8)


def medians(df):
    return df.groupby(["block_hankel", "size"])[METHOD_KEYS].median()


def figure_C(df, out, med):
    sizes = sorted(df["size"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), facecolor=SURFACE)
    for ax, (kind, title) in zip(axes, KINDS):
        style(ax); ax.set_xscale("log"); ax.set_yscale("log")
        sub = df[df.block_hankel == kind]
        labels = []
        for key, long, short, color, marker in METHODS:
            xs = np.array([m * m for m in sizes], dtype=float)
            ys = np.array([med.loc[(kind, m), key] for m in sizes])
            pts = sub[["size", key]].to_numpy(dtype=float)
            zero = pts[:, 1] <= 0
            ax.scatter(pts[~zero, 0] ** 2, pts[~zero, 1], s=14, color=color, marker=marker, alpha=0.35,
                       edgecolors=SURFACE, linewidths=0.6, zorder=3)
            floor = ys <= 0
            ax.plot(xs, np.where(floor, FLOOR_MS, ys), color=color, linewidth=2, marker=marker, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1, solid_capstyle="round", label=long, zorder=4)
            if floor.any():                                         # hollow markers: at or below resolution
                ax.plot(xs[floor], np.full(floor.sum(), FLOOR_MS), linestyle="none", marker=marker, markersize=6,
                        markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=1.4, zorder=5)
            labels.append((max(ys[-1], FLOOR_MS), short))
        ax.set_xlabel("matrix entries N = m² (log scale)"); ax.set_ylabel("time (ms, log scale)")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlim(3e4, 5e8); ax.set_ylim(0.03, 2e5)
        # slope-1 guide: the direction a linear-time method follows (placed in an empty band)
        gx = np.array([3e6, 1.2e8]); gy = (3e-6 if kind == "Yes" else 4e-6) * gx
        ax.plot(gx, gy, color=MUTED, linewidth=1, zorder=2)
        ax.text(gx[1] * 1.15, gy[1], "slope 1\n(linear in N)", fontsize=8, color=MUTED, va="center")
        end_labels(ax, labels, x=sizes[-1] ** 2)
    axes[1].text(0.02, 0.97, "hollow marker: ≤ 0.1 ms, the resolution of the CSV (drawn at 0.05 ms)",
                 transform=axes[1].transAxes, fontsize=8, color=MUTED, va="top")
    figure_legend(fig, axes[0])
    fig.suptitle("Runtime of the four recognizers on the same matrices — log–log axes; "
                 "median of 3 matrices per size, best of 3 runs, single core", fontsize=10.5, color=INK, x=0.01, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 0.985, 0.87)); fig.savefig(out / "fig_C_runtime_loglog.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


DIRECT_NEG_NOTE = (
    "Direct comparison omitted in this panel: on random matrices every candidate\n"
    "pair is rejected by its first memcmp, so its total time is ≤ 0.1 ms at every\n"
    "size (the resolution of the CSV) and does not grow with m². Divided by m² that\n"
    "floor would draw a meaningless 1/m² line. Per candidate pair it costs ≈ 40–65 ns,\n"
    "independent of m (see Figure C, right panel).")


def figure_D(df, out, med):
    sizes = sorted(df["size"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), facecolor=SURFACE)
    for ax, (kind, title) in zip(axes, KINDS):
        style(ax); ax.set_xscale("log"); ax.set_yscale("log")
        labels = []
        for key, long, short, color, marker in METHODS:
            if kind == "No" and key == "direct_ms":
                continue        # Θ(d(m)²) work at the timer floor: a per-entry figure would be an artifact
            ys = np.array([med.loc[(kind, m), key] * 1e6 / (m * m) for m in sizes])   # ns per entry
            ax.plot(sizes, ys, color=color, linewidth=2, marker=marker, markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=1, solid_capstyle="round", label=long, zorder=4)
            labels.append((ys[-1], short))
        ax.set_xlabel("matrix side m (log scale)"); ax.set_ylabel("nanoseconds per entry (log scale)")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes], rotation=45, ha="right", fontsize=8)
        ax.set_xlim(200, 19000); ax.set_ylim(0.02 if kind == "No" else 0.5, 600)
        end_labels(ax, labels, x=sizes[-1])
    axes[1].text(0.02, 0.04, DIRECT_NEG_NOTE, transform=axes[1].transAxes, fontsize=7.8, color=INK2,
                 va="bottom", ha="left", linespacing=1.35)
    figure_legend(fig, axes[0], relabel={"Direct comparison, early exit": "Direct comparison, early exit (left panel only, see note)"})
    fig.suptitle("Cost per matrix entry (median time ÷ m²) — a flat line is linear scaling; the height is the constant factor",
                 fontsize=10.5, color=INK, x=0.01, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 0.985, 0.87)); fig.savefig(out / "fig_D_ns_per_entry.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def figure_E(df, out, med):
    sizes = sorted(df["size"].unique())
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.9), facecolor=SURFACE)
    style(ax); ax.set_xscale("log"); ax.set_yscale("log")
    base = np.array([med.loc[("Yes", m), "hash_k2_ms"] for m in sizes])
    ax.axhline(1.0, color=MUTED, linewidth=1, zorder=2)
    ax.text(sizes[0] * 0.98, 1.0, "2-D hashing = 1", fontsize=8.5, color=MUTED, va="bottom", ha="left")
    labels = []
    for key, long, short, color, marker in METHODS:
        if key == "hash_k2_ms":
            continue
        ratio = np.array([med.loc[("Yes", m), key] for m in sizes]) / base
        ax.plot(sizes, ratio, color=color, linewidth=2, marker=marker, markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1, solid_capstyle="round", label=long, zorder=4)
        labels.append((ratio[-1], f"{short}  {ratio[-1]:.1f}×"))
    ax.set_xlabel("matrix side m (log scale)"); ax.set_ylabel("median time ÷ median hashing time (log scale)")
    ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes], rotation=45, ha="right", fontsize=8)
    ax.set_xlim(200, 19000); ax.set_ylim(0.07, 80)
    ax.set_title("Block Hankel positives: time relative to 2-D hashing", fontsize=11, loc="left")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK2)
    end_labels(ax, labels, x=sizes[-1], min_sep_frac=0.07)
    fig.tight_layout(rect=(0, 0, 0.93, 1)); fig.savefig(out / "fig_E_ratio_to_hashing.png", dpi=170, facecolor=SURFACE)
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
