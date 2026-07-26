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
        "xtick.labelsize": 6.0, "ytick.labelsize": 8,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


# (field key, full display name, early-window co-authorship), ordered by co-authorship
PANELS = [("econ", "economics", 0.58), ("math", "mathematics", 0.68),
          ("neuro", "neuroscience", 0.76), ("physics", "physics", 0.79),
          ("chemistry", "chemistry", 0.79)]
RUNGS = [("M0", "M0_prior"), ("M1", "M1_logit_overlap"), ("M2", "M2_logit_tabular"),
         ("M3", "M3_gbdt_tabular"), ("M4", "M4_logit_tfidf"), ("M5", "M5_gbdt_nfa")]

# M4's frozen run fitted its vectoriser on all rows, which read validation and
# test text. T2.3 re-measured it on a train-only fit and the paper's ladder table
# reports the corrected value with the frozen one retained as superseded. The
# figure drew the superseded value until 2026-07-26; it now reads the same
# corrected number the table prints, and rung_values asserts the two agree.
M4_TRAINFIT = ROOT / "results" / "revision" / "T2_3_m4_trainfit" / "summary.json"
GNN_LABEL = {"hgt": "HGT", "hgt_tuned": "HGT-t", "rgcn": "RGCN",
             "gat_cohort_time": "GAT"}

# Four fills separated by luminance, not by hue, so the rung roles survive a
# greyscale print: the prior is an open bar because it is a floor and not a
# model, the tabular family is mid, the ceiling comparator M5 is the darkest and
# carries the accent this paper reserves for the thing a result is judged
# against, and the graph column is lightest. The predecessor drew M1 to M4 in
# #00008B and M5 in #8B0000, which are the same grey.
STEEL = "#6B8CAE"      # tabular rungs M1 to M4
CRIM = "#9B2226"       # graph-aware rung M5, the comparator
GNNGRAY = "#B0B7BD"    # the best graph architecture, measured
SLATE = "#6B7280"      # ceiling line, prior outline, axis furniture


def rung_values(field, s, m4):
    """The six rungs the ladder table prints, with M4 taken from the T2.3 rerun
    rather than from the frozen summary. Refuses on a mismatch against the
    table's own three-decimal value, so the figure cannot drift from it."""
    vals = []
    for name, key in RUNGS:
        if name == "M4":
            v = m4["fields"][field]["trainfit"]["auc_pr"]["mean"]
            legacy = m4["fields"][field]["legacy"]["auc_pr"]["mean"]
            if abs(legacy - s[key]["auc_pr"]["mean"]) > 5e-4:
                raise SystemExit(
                    f"ladder: {field} frozen M4 {s[key]['auc_pr']['mean']:.4f} is not "
                    f"the legacy arm {legacy:.4f}; the two artifacts disagree")
        else:
            v = s[key]["auc_pr"]["mean"]
        vals.append(v)
    return vals


def build():
    import matplotlib.pyplot as plt
    m4 = json.load(open(M4_TRAINFIT))
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.15), sharey=True)
    for ax, (field, name, coauth) in zip(axes, PANELS):
        s = json.load(open(RES / f"results_{field}" / "e1_baselines.json"))["summary"]
        corr = json.load(open(RES / f"results_{field}" / "e12_corrected_vs_m5.json"))
        vals = rung_values(field, s, m4)
        ceiling = corr["ceilings"]["M5_prime_val_symmetric"]
        best_name, best = max(corr["models"].items(),
                              key=lambda kv: kv[1]["seed_mean_auc_pr"])
        faces = ["white" if n == "M0" else CRIM if n == "M5" else STEEL
                 for n, _ in RUNGS]
        edges = [SLATE if n == "M0" else f for (n, _), f in zip(RUNGS, faces)]
        x = list(range(len(RUNGS)))
        ax.bar(x, vals, color=faces, width=0.62, linewidth=0.7,
               edgecolor=edges, zorder=3)
        # GNN column: best graph model, measured, set off from M5 by a gap
        gx = len(RUNGS) + 0.4
        ax.bar([gx], [best["seed_mean_auc_pr"]], color=GNNGRAY, width=0.62,
               linewidth=0.7, edgecolor=GNNGRAY, zorder=3)
        ax.text(gx, best["seed_mean_auc_pr"] / 2, GNN_LABEL[best_name],
                rotation=90, ha="center", va="center", fontsize=6.5,
                color="#242B33", zorder=5)
        # fair ceiling M5' spans all columns including the GNN one
        ax.plot([-0.4, gx + 0.4], [ceiling, ceiling],
                ls=(0, (4, 3)), lw=1.0, color=SLATE, zorder=4)
        ax.text(-0.4, ceiling + 0.008, f"M5$'$ {ceiling:.3f}",
                ha="left", va="bottom", fontsize=7.0, color=SLATE)
        ax.set_xlim(-0.7, gx + 0.6)
        ax.set_xticks(x + [gx])
        ax.set_xticklabels([n for n, _ in RUNGS] + ["GNN"])
        ax.tick_params(axis="x", pad=1.5)
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
