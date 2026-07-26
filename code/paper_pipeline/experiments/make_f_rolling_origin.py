"""F_rolling_origin - rolling-origin temporal view of the model comparison (R2, App D).

Two stacked panels (economics on top, neuroscience below). For each discipline the
x-axis is the four rolling-origin test cohorts in time order; y is test AUC-PR. We
plot, per cohort:
  - HGT             10-seed mean +/- sd  (from the collector-merged "hgt" block)
  - GBDT + NFA (M5) 10-seed mean +/- sd  (the CPU graph-aware rung)
  - tabular ceiling max(M2, M3)          (dashed, the per-cohort reference)

Every number is read from results_<disc>/e12_rolling_origin.json (produced by
e12_rolling_origin.py + e12_collect_hgt_rolling.py). Nothing is recomputed and no
load_dataset is called, so the config econ-path is irrelevant. The extra GNNs
(RGCN / GAT+cohort-time) are evaluated on the frozen MAIN split, not the rolling
origins, so they are deliberately NOT in this figure (they live in the ladder
review copies instead).

Output: paper_pipeline/figures_econ/F_rolling_origin.pdf/.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # ACM camera-ready: embed TrueType
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"

HERE = os.path.dirname(os.path.abspath(__file__))
PP = os.path.normpath(os.path.join(HERE, ".."))
FIGDIR = os.path.join(PP, "figures_econ")

# colorblind-safe (Okabe-Ito); each series distinct in color AND marker.
C_HGT, M_HGT = "#009E73", "D"    # graph-aware teal, diamond
C_M5,  M_M5 = "#D55E00", "o"     # vermillion, circle
C_CEIL = "#555555"               # gray dashed reference


def load(disc):
    d = json.load(open(os.path.join(PP, f"results_{disc}", "e12_rolling_origin.json")))
    labels, ceil, m5m, m5s, hm, hs = [], [], [], [], [], []
    for rec in d["origins"]:
        labels.append(rec["origin"])
        ceil.append(rec["tabular_ceiling"])
        m5m.append(rec["auc_pr_mean"]["M5_gbdt_nfa"])
        m5s.append(rec["auc_pr_std"]["M5_gbdt_nfa"])
        hm.append(rec["hgt"]["auc_pr_mean"])
        hs.append(rec["hgt"]["auc_pr_std"])
    return dict(labels=labels, ceil=np.array(ceil),
                m5m=np.array(m5m), m5s=np.array(m5s),
                hm=np.array(hm), hs=np.array(hs))


def crossings(D):
    """Cohorts where HGT mean exceeds the per-cohort ceiling (honest disclosure)."""
    return [(D["labels"][i], D["hm"][i] - D["ceil"][i])
            for i in range(len(D["labels"])) if D["hm"][i] > D["ceil"][i]]


data = {d: load(d) for d in ("econ", "neuro")}


def draw(ax, D, title):
    x = np.arange(len(D["labels"]))
    # per-cohort tabular ceiling: dashed reference line + small ticks
    ax.plot(x, D["ceil"], color=C_CEIL, lw=1.0, ls=(0, (4, 2)), zorder=2,
            marker="_", ms=9, mew=1.2, label="tabular ceiling max(M2, M3)")
    ax.errorbar(x, D["hm"], yerr=D["hs"], color=C_HGT, marker=M_HGT, ms=5,
                lw=1.3, elinewidth=1.0, capsize=2.5, zorder=4, label="HGT")
    ax.errorbar(x, D["m5m"], yerr=D["m5s"], color=C_M5, marker=M_M5, ms=5,
                lw=1.3, elinewidth=1.0, capsize=2.5, zorder=3, label="GBDT + NFA (M5)")
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(D["labels"], fontsize=7.5)
    ax.set_xlim(-0.35, len(x) - 0.65)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Test AUC-PR", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#EEEEEE", lw=0.5, zorder=0)


fig, axes = plt.subplots(2, 1, figsize=(4.3, 4.9))
fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.20, hspace=0.42)
draw(axes[0], data["econ"], "Economics")
draw(axes[1], data["neuro"], "Neuroscience")
axes[1].set_xlabel("Rolling-origin test cohort (t0 window)", fontsize=8)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7,
           frameon=False, bbox_to_anchor=(0.5, 0.075), columnspacing=1.2,
           handletextpad=0.4)

# honest footnote built from the data (single-cohort within-noise crossings)
cx = {d: crossings(data[d]) for d in ("econ", "neuro")}
def _fmt(d):
    c = cx[d]
    if not c:
        return "%s: HGT at/below ceiling in every cohort" % d
    return "%s: HGT above ceiling only at %s" % (
        d, ", ".join(f"{lab} ({dv:+.3f})" for lab, dv in c))
fig.text(0.5, 0.008,
         "10-seed mean +/- sd. " + "; ".join(_fmt(d) for d in ("econ", "neuro"))
         + " (within sd).",
         ha="center", va="bottom", fontsize=5.8, color="#333333")

os.makedirs(FIGDIR, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGDIR, f"F_rolling_origin.{ext}"),
                dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote F_rolling_origin.pdf/.png")
for d in ("econ", "neuro"):
    D = data[d]
    print(f"[{d}] cohorts {D['labels']}")
    print(f"       ceiling {np.round(D['ceil'],4)}")
    print(f"       HGT     {np.round(D['hm'],4)} +/- {np.round(D['hs'],4)}")
    print(f"       M5      {np.round(D['m5m'],4)} +/- {np.round(D['m5s'],4)}")
    print(f"       HGT-ceiling crossings: {crossings(D)}")
