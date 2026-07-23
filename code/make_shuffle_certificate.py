#!/usr/bin/env python3
"""Y-scrambling certificate boxplots across the five disciplines, in the paper
house style (serif type, darkblue with a gray neutral, no top or right spines,
faint horizontal grid only). For each discipline two boxes over the 30 per-seed
test AUC-ROC values of the retrained strongest pure-tabular model on permuted
labels: the global shuffle (gray), which destroys every association, and the
within-cohort shuffle (darkblue), which permutes labels inside (split, t0) cells
and so preserves compositional signal by design. A dashed line marks chance at
0.5. Reads the frozen per-seed files results/results_<field>/e9a_global_perseed.jsonl
(global) and results/results_<field>/e9a_perseed.jsonl (within-cohort); no model
is retrained. Writes a vector PDF for \\includegraphics plus a PNG preview.
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


# (field key, short display name), ordered by early co-authorship as in the ladder
PANELS = [("econ", "econ"), ("math", "math"), ("neuro", "neuro"),
          ("physics", "physics"), ("chemistry", "chem")]
DARKBLUE = "#00008B"
DIMGRAY = "#555555"


def rocs(field, fname):
    path = RES / f"results_{field}" / fname
    return [json.loads(line)["metrics"]["auc_roc"] for line in open(path)]


def build():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    for i, (field, name) in enumerate(PANELS):
        for dx, fname, color in [(-0.19, "e9a_global_perseed.jsonl", DIMGRAY),
                                 (+0.19, "e9a_perseed.jsonl", DARKBLUE)]:
            vals = rocs(field, fname)
            assert len(vals) == 30, (field, fname, len(vals))
            bp = ax.boxplot([vals], positions=[i + dx], widths=0.3,
                            patch_artist=True, showfliers=True,
                            flierprops=dict(marker="o", ms=1.8, mfc=color,
                                            mec=color, alpha=0.6),
                            medianprops=dict(color=color, lw=1.3),
                            whiskerprops=dict(color=color, lw=0.9),
                            capprops=dict(color=color, lw=0.9),
                            boxprops=dict(facecolor="white", edgecolor=color,
                                          lw=0.9))
    ax.axhline(0.5, ls=(0, (4, 3)), lw=0.9, color=DIMGRAY, zorder=1)
    ax.text(len(PANELS) + 0.02, 0.506, "chance", ha="right", va="bottom",
            fontsize=7, color=DIMGRAY)
    ax.set_xticks(range(len(PANELS)))
    ax.set_xticklabels([name for _, name in PANELS])
    ax.set_xlim(-0.6, len(PANELS) + 0.05)
    ax.set_ylabel("Placebo test AUC-ROC")
    ax.set_ylim(0.33, 0.67)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=2.5)
    ax.tick_params(axis="x", length=0)
    handles = [
        mpl.patches.Patch(facecolor="white", edgecolor=DIMGRAY, lw=1.1,
                          label="global shuffle"),
        mpl.patches.Patch(facecolor="white", edgecolor=DARKBLUE, lw=1.1,
                          label="within-cohort shuffle"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.56, -0.09), handletextpad=0.5,
               columnspacing=1.2)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "F15_shuffle_certificate"
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
