#!/usr/bin/env python3
"""Emit the thirty-nine-cell graph-arm figure from the assembled strict-contract
cells, the same artifact r37 emits Table 4 from.

Every value is read from assembled_tables.json; none is typed. The verdict rule
is r37's, imported rather than restated, so the figure and the table cannot
disagree. Intervals are drawn at the values the tables print, not smoothed: the
marker sits at delta_vs_M5prime and the bar spans ci[0] to ci[1].

Visual language, shared with r58_emit_ladder_figure.py:
  discipline order   economics, mathematics, neuroscience, physics, chemistry
                     (the order of Table 1, which is early co-authorship)
  accent             crim #9B2226, reserved across every body figure for the
                     thing that crosses a line
  verdict channel    marker shape and fill, not hue, so it survives greyscale
  axis convention    x is the quantity, zero drawn solid

Output: paper/figures/F16_gnn39.pdf and .png
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
TREE = (ROOT / "results" / "revision" / "T2_1_strict_contract" / "T2_1_final"
        / "T2_1_strict_contract")
OUT = ROOT / "paper" / "figures" / "F16_gnn39"

INK = "#242B33"
SLATE = "#6B7280"
CRIM = "#9B2226"
BAND = "#F2F4F5"

FIELDS = [("econ", "economics"), ("math", "mathematics"), ("neuro", "neuroscience"),
          ("physics", "physics"), ("chemistry", "chemistry")]

# row order inside a discipline: the pre-registered four, then the symmetric grid
ORDER = [("Table 12b", "HGT", "strict", "HGT"),
         ("Table 12b", "HGT tuned", "strict", "HGT tuned"),
         ("Table 12b", "RGCN", "strict", "RGCN"),
         ("Table 12b", "GAT", "strict", "GAT"),
         ("Table 13b", "RGCN symmetric", "strict", "RGCN sym"),
         ("Table 13b", "RGCN symmetric", "legacy", "RGCN sym, legacy"),
         ("Table 13b", "GAT symmetric", "strict", "GAT sym"),
         ("Table 13b", "GAT symmetric", "legacy", "GAT sym, legacy")]


def verdict(r):
    """r37's rule, copied under test rather than imported, because r37 is a
    script rather than a module. check_against_table4 asserts the two agree."""
    if r["exceeds"]:
        return "exceeds"
    lo, hi = r["ci"]
    if hi < 0:
        return "below"
    if lo > 0:
        return "null (gate 2)"
    return "null"


STYLE = {
    "exceeds":       dict(marker="o", mfc=CRIM, mec=CRIM, ms=4.4, mew=0.9, color=CRIM, lw=1.15),
    "null (gate 2)": dict(marker="o", mfc="white", mec=INK, ms=4.4, mew=1.2, color=INK, lw=0.9),
    "below":         dict(marker="s", mfc="white", mec=INK, ms=3.8, mew=1.0, color=INK, lw=0.9),
    "null":          dict(marker="o", mfc="white", mec=SLATE, ms=3.0, mew=0.7, color=SLATE, lw=0.7),
}


def load():
    data = json.loads((TREE / "assembled_tables.json").read_text(encoding="utf-8"))
    cells = []
    for key, name in FIELDS:
        for tab, row, cfg, label in ORDER:
            cand = [r for r in data["tables"][tab]["rows"].get(key, [])
                    if r["row"] == row and r["config"] == cfg]
            if not cand:
                continue
            if len(cand) != 1:
                raise SystemExit(f"r57: {key}/{tab}/{row}/{cfg} matched {len(cand)}")
            cells.append((name, label, cand[0]))
    total = sum(data["tables"][t]["n_rows"] for t in data["tables"])
    exceed = sum(data["tables"][t]["n_exceeds"] for t in data["tables"])
    if len(cells) != total:
        raise SystemExit(f"r57: laid out {len(cells)} of {total} cells")
    drawn = sum(1 for _, _, r in cells if r["exceeds"])
    if drawn != exceed:
        raise SystemExit(f"r57: {drawn} exceeding drawn, artifact says {exceed}")
    return cells, total, exceed


def check_against_table4(cells):
    """Every row Table 4 prints must appear in the figure with the same delta,
    the same interval and the same verdict. A figure that disagreed with the
    table it replaces would be worse than no figure."""
    tex = (ROOT / "paper" / "table4_verdicts.tex").read_text(encoding="utf-8")
    n = 0
    for name, label, r in cells:
        lo, hi = r["ci"]
        needle = f"{abs(r['delta_vs_M5prime']):.4f}"
        for line in tex.split("\n"):
            if not line.strip().startswith(("chemistry", "neuro", "physics", "math", "econ")):
                continue
            if needle in line and f"{abs(lo):.4f}" in line and f"{abs(hi):.4f}" in line:
                if verdict(r) not in line:
                    raise SystemExit(f"r57: verdict disagrees with Table 4 on {name} {label}")
                n += 1
    print(f"[r57] {n} of Table 4's rows matched in the figure, verdicts agree")


def main():
    cells, total, exceed = load()
    check_against_table4(cells)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Linux Libertine", "DejaVu Serif"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.5,
    })

    rows = []          # (y, name, label, rec)  y grows downward
    y = 0.0
    bands = []
    for key, name in FIELDS:
        mine = [c for c in cells if c[0] == name]
        top = y
        for _, label, rec in mine:
            rows.append((y, name, label, rec))
            y += 1.0
        bands.append((name, top - 0.5, y - 0.5))
        y += 0.45

    H = 0.0655 * y + 0.56
    fig, ax = plt.subplots(figsize=(3.34, H))
    tr = ax.get_yaxis_transform()          # x in axes fraction, y in data

    for i, (name, y0, y1) in enumerate(bands):
        if i % 2 == 0:
            ax.axhspan(y0, y1, color=BAND, lw=0, zorder=0)
        ax.text(-0.545, (y0 + y1) / 2, name, transform=tr, ha="left",
                va="center", fontsize=5.6, color=INK, style="italic", zorder=3)

    ax.axvline(0.0, color=SLATE, lw=0.7, zorder=1)

    for yy, name, label, rec in rows:
        lo, hi = rec["ci"]
        st = STYLE[verdict(rec)]
        ax.plot([lo, hi], [yy, yy], color=st["color"], lw=st["lw"],
                solid_capstyle="butt", zorder=2)
        ax.plot([rec["delta_vs_M5prime"]], [yy], marker=st["marker"],
                mfc=st["mfc"], mec=st["mec"], ms=st["ms"], mew=st["mew"],
                linestyle="none", zorder=4)
        ax.text(-0.018, yy, label, transform=tr, ha="right", va="center",
                fontsize=4.8, color=INK, zorder=3)

    ax.set_ylim(y - 0.45, -0.75)
    ax.set_xlim(-0.115, 0.072)
    ax.set_yticks([])
    ax.set_xticks([-0.10, -0.05, 0.0, 0.05])
    ax.set_xticklabels(["$-$0.10", "$-$0.05", "0", "$+$0.05"], fontsize=5.4)
    ax.tick_params(axis="x", length=2, width=0.5, pad=1.5, colors=INK)
    ax.set_xlabel("$\\Delta$ against M5$^{\\prime}$, with the student-level interval",
                  fontsize=5.6, color=INK, labelpad=1.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(SLATE)

    handles = [
        Line2D([], [], marker="o", mfc=CRIM, mec=CRIM, ms=4.0, mew=0.9,
               ls="none", label=f"exceeds ({exceed})"),
        Line2D([], [], marker="o", mfc="white", mec=INK, ms=4.0, mew=1.2,
               ls="none", label="null, gate 2 only"),
        Line2D([], [], marker="s", mfc="white", mec=INK, ms=3.6, mew=1.0,
               ls="none", label="below the ceiling"),
        Line2D([], [], marker="o", mfc="white", mec=SLATE, ms=2.8, mew=0.7,
               ls="none", label="null"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.565, 1.003),
              ncol=4, frameon=False, fontsize=4.6, handletextpad=0.25,
              columnspacing=0.75, labelspacing=0.2, borderpad=0.0)

    fig.subplots_adjust(left=0.365, right=0.995, top=1 - 0.19 / H, bottom=0.40 / H)
    fig.savefig(str(OUT) + ".pdf")
    fig.savefig(str(OUT) + ".png", dpi=400)
    print(f"[r57] {total} cells, {exceed} exceeding -> {OUT}.pdf")
    chem = [c for c in cells if c[0] == "chemistry" and c[2]["exceeds"]]
    print(f"[r57] exceeding cells: "
          f"{sorted(set(c[0] for c in cells if c[2]['exceeds']))}, "
          f"{len(chem)} of them chemistry")


if __name__ == "__main__":
    main()
