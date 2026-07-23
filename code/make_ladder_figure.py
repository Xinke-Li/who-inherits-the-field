#!/usr/bin/env python3
"""Five-panel baseline ladder in the paper house style (matches the leakage-free
E9a figure): serif type, a steel-blue tabular family with a single dark-red accent
for the graph rung M5, a restrained gray dashed ceiling with a small inline label,
faint horizontal grid only, no top or right spines. The GNN column, formerly a
hatched pending stub, now carries the measured result: the best graph model per
discipline (10-seed mean test AUC-PR) from the corrected e12 aggregation, judged
against the VAL-symmetric fair ceiling M5' (dashed line). A model that exceeds the
line is drawn exceeding it; whether it also passes the significance and bootstrap
gates is the table's job, not the figure's. Panels are ordered by early-window
co-authorship. Reads results/results_<field>/e1_baselines.json and
results/results_<field>/e12_corrected_vs_m5.json. Writes a vector PDF for
\\includegraphics plus a PNG preview.
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
        "xtick.labelsize": 7.5, "ytick.labelsize": 8,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


# (field key, full display name, early-window co-authorship), ordered by co-authorship
PANELS = [("econ", "economics", 0.58), ("math", "mathematics", 0.68),
          ("neuro", "neuroscience", 0.76), ("physics", "physics", 0.79),
          ("chemistry", "chemistry", 0.79)]
RUNGS = [("M1", "M1_logit_overlap"), ("M2", "M2_logit_tabular"),
         ("M3", "M3_gbdt_tabular"), ("M4", "M4_logit_tfidf"), ("M5", "M5_gbdt_nfa")]
GNN_LABEL = {"hgt": "HGT", "hgt_tuned": "HGT-t", "rgcn": "RGCN",
             "gat_cohort_time": "GAT"}
DARKBLUE = "#00008B"   # tabular rungs M1..M4
DARKRED = "#8B0000"    # graph rung M5, the single accent
DIMGRAY = "#555555"    # ceiling line and label, GNN bar
GNNGRAY = "#666666"


def build():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.15), sharey=True)
    for ax, (field, name, coauth) in zip(axes, PANELS):
        s = json.load(open(RES / f"results_{field}" / "e1_baselines.json"))["summary"]
        corr = json.load(open(RES / f"results_{field}" / "e12_corrected_vs_m5.json"))
        vals = [s[key]["auc_pr"]["mean"] for _, key in RUNGS]
        ceiling = corr["ceilings"]["M5_prime_val_symmetric"]
        best_name, best = max(corr["models"].items(),
                              key=lambda kv: kv[1]["seed_mean_auc_pr"])
        colors = [DARKRED if n == "M5" else DARKBLUE for n, _ in RUNGS]
        x = list(range(len(RUNGS)))
        ax.bar(x, vals, color=colors, width=0.62, linewidth=0.5,
               edgecolor=colors, zorder=3)
        # GNN column: best graph model, measured, set off from M5 by a gap
        gx = len(RUNGS) + 0.4
        ax.bar([gx], [best["seed_mean_auc_pr"]], color=GNNGRAY, width=0.62,
               linewidth=0.5, edgecolor=GNNGRAY, zorder=3)
        ax.text(gx, best["seed_mean_auc_pr"] / 2, GNN_LABEL[best_name],
                rotation=90, ha="center", va="center", fontsize=6.5,
                color="white", zorder=5)
        # fair ceiling M5' spans all columns including the GNN one
        ax.plot([-0.4, gx + 0.4], [ceiling, ceiling],
                ls=(0, (4, 3)), lw=1.0, color=DIMGRAY, zorder=4)
        ax.text(-0.4, ceiling + 0.008, f"M5$'$ {ceiling:.3f}",
                ha="left", va="bottom", fontsize=7.5, color=DIMGRAY)
        ax.set_xlim(-0.6, gx + 0.6)
        ax.set_xticks(x + [gx])
        ax.set_xticklabels([n for n, _ in RUNGS] + ["GNN"])
        ax.set_title(f"{name}\nearly co-authorship {coauth:.2f}", fontsize=8,
                     color="#444444", linespacing=1.4)
        ax.set_ylim(0, 0.72)
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        ax.tick_params(length=2.5)
    axes[0].set_ylabel("Test AUC-PR (10 seeds)")
    fig.tight_layout(w_pad=0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "F11_five_discipline_ladder"
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
