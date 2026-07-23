#!/usr/bin/env python3
"""Naive versus corrected dumbbell plot for the seven naive crossings of the
graph arm, serif type with no top or right spines and a faint vertical grid.
Colors follow the prior submission's rolling-origin figure: the naive protocol
in gold (the GBDT line color) and the corrected protocol in dark red (the HGT
line color), with a light gray connector joining the two points of each row.
One row per discipline-architecture pair, sorted by the naive gap: the open
gold point is the naive gap to the seed-wise best of M2 and M3 with its
seed-level bootstrap interval (thin line), the filled dark red point is the
corrected gap to the validation-symmetric ceiling M5' with its paired
student-level bootstrap interval (thick line). A dashed vertical line marks
zero. The chemistry relational network, the one corrected interval that
excludes zero, carries an ink outline and a larger marker. Reads
results/results_<field>/e12_hgt_vs_baselines.json (naive) and
results/results_<field>/e12_corrected_vs_m5.json (corrected). Writes a vector
PDF for \\includegraphics plus a PNG preview.
"""
import json
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"


def style(usetex):
    mpl.rcParams.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Libertine", "Times New Roman", "DejaVu Serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


# the seven naive crossings of Table 3 (all other pairs cross under no protocol)
ROWS = [("physics", "hgt", "physics HGT"),
        ("physics", "hgt_tuned", "physics HGT tuned"),
        ("physics", "rgcn", "physics RGCN"),
        ("physics", "gat_cohort_time", "physics GAT"),
        ("chemistry", "rgcn", "chemistry RGCN"),
        ("chemistry", "gat_cohort_time", "chemistry GAT"),
        ("neuro", "rgcn", "neuroscience RGCN")]
SURVIVOR = ("chemistry", "rgcn")
GOLD = "#E1A100"       # naive protocol (prior submission's GBDT/M5 line color)
FIREBRICK = "#B22222"  # corrected protocol (prior submission's HGT line color)
DIMGRAY = "#555555"
INK = "#242B33"
CONNECT = "#c9c9c9"


def load():
    rows = []
    for field, model, label in ROWS:
        nv = json.load(open(RES / f"results_{field}" / "e12_hgt_vs_baselines.json"))["models"][model]
        co = json.load(open(RES / f"results_{field}" / "e12_corrected_vs_m5.json"))["models"][model]
        rows.append({
            "label": label,
            "survivor": (field, model) == SURVIVOR,
            "naive": nv["delta_vs_ceiling_mean"],
            "naive_ci": nv["bootstrap_ci_diff"],
            "corr": co["delta_vs_M5prime"],
            "corr_ci": co["bootstrap"]["vs_M5prime"]["pooled_ci95"],
        })
    rows.sort(key=lambda r: r["naive"], reverse=True)
    return rows


def build():
    import matplotlib.pyplot as plt
    rows = load()
    fig, ax = plt.subplots(figsize=(3.4, 1.95))
    n = len(rows)
    for i, r in enumerate(rows):
        y = n - 1 - i
        # light gray connector joining the naive and corrected points
        ax.plot([r["naive"], r["corr"]], [y + 0.16, y - 0.16], lw=0.8,
                color=CONNECT, zorder=2)
        # naive: thin gold interval, open marker, upper lane of the row
        ax.plot(r["naive_ci"], [y + 0.16] * 2, lw=0.9, color=GOLD, zorder=3)
        ax.plot(r["naive"], y + 0.16, marker="o", ms=3.6, mfc="white",
                mec=GOLD, mew=0.9, ls="none", zorder=4)
        # corrected: thick dark red interval, filled marker, lower lane
        lw = 2.8 if r["survivor"] else 2.2
        ax.plot(r["corr_ci"], [y - 0.16] * 2, lw=lw, color=FIREBRICK,
                solid_capstyle="round", zorder=3)
        if r["survivor"]:
            ax.plot(r["corr"], y - 0.16, marker="o", ms=6.0, mfc=FIREBRICK,
                    mec=INK, mew=1.1, ls="none", zorder=5)
        else:
            ax.plot(r["corr"], y - 0.16, marker="o", ms=4.6, color=FIREBRICK,
                    ls="none", zorder=4)
    ax.axvline(0.0, ls=(0, (4, 3)), lw=0.9, color=DIMGRAY, zorder=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows][::-1])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(r"$\Delta$ test AUC-PR vs tabular ceiling")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=2.5)
    ax.tick_params(axis="y", length=0)
    handles = [
        mpl.lines.Line2D([], [], marker="o", ms=3.6, mfc="white", mec=GOLD,
                         mew=0.9, color=GOLD, lw=0.9,
                         label="naive protocol (gold)"),
        mpl.lines.Line2D([], [], marker="o", ms=4.6, color=FIREBRICK, lw=2.2,
                         label="corrected protocol (dark red)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
               bbox_to_anchor=(0.6, -0.14), handletextpad=0.5)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "F14_audit_dumbbell"
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png")
    plt.close(fig)
    return base


if __name__ == "__main__":
    try:
        style(True)
        base = build()
        print(f"wrote {base}.pdf and .png (usetex)")
    except Exception as e:
        print(f"usetex render failed ({type(e).__name__}); serif fallback")
        style(False)
        base = build()
        print(f"wrote {base}.pdf and .png (serif fallback)")
