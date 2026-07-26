"""F11 - econ-vs-neuro baseline ladder as paired-dot small multiples.

Replaces the grouped-bar version. Two panels (AUC-PR, AUC-ROC), models on the
y-axis, economics in blue and neuroscience in orange, 10-seed sd error bars.
Every value is loaded from the result JSONs; nothing is hand-typed.
Output: paper_pipeline/figures_econ/F11_econ_vs_neuro_ladder.pdf/.png
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
# ACM camera-ready: embed TrueType (Type 42), never Type 3 bitmap-glyph fonts.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PP = os.path.join(HERE, "..")
FIGDIR = os.path.join(PP, "figures_econ")

ECON_BLUE = "#1f77b4"
NEURO_ORANGE = "#ff7f0e"

def load_summary(path):
    return json.load(open(path))["summary"]

def hgt_stats(seed_glob):
    fs = sorted(glob.glob(seed_glob))
    pr = [json.load(open(f))["auc_pr"] for f in fs]
    roc = [json.load(open(f))["auc_roc"] for f in fs]
    assert len(fs) == 10, f"expected 10 HGT seeds, got {len(fs)} for {seed_glob}"
    return {"auc_pr": (float(np.mean(pr)), float(np.std(pr, ddof=1))),
            "auc_roc": (float(np.mean(roc)), float(np.std(roc, ddof=1)))}

econ = load_summary(os.path.join(PP, "results_econ", "e1_baselines.json"))
neuro = load_summary(os.path.join(PP, "results_neuro", "e1_baselines.json"))
econ_hgt = hgt_stats(os.path.join(PP, "results_econ", "results_hgt", "hgt_none_seed*.json"))
neuro_hgt = hgt_stats(os.path.join(PP, "results_neuro", "results_hgt", "hgt_none_seed*.json"))

MODELS = [
    ("M0_prior",          "M0 prior"),
    ("M1_logit_overlap",  "M1 logit (overlap)"),
    ("M2_logit_tabular",  "M2 logit (tabular)"),
    ("M3_gbdt_tabular",   "M3 GBDT (tabular)"),
    ("M4_logit_tfidf",    "M4 logit (TF-IDF)"),
    ("M5_gbdt_nfa",       "M5 GBDT + NFA"),
    ("HGT",               "HGT (standard)"),
]

def series(summary, hgt, metric):
    means, sds = [], []
    for key, _ in MODELS:
        if key == "HGT":
            m, s = hgt[metric]
        else:
            m = summary[key][metric]["mean"]
            s = summary[key][metric]["std"]
        means.append(m); sds.append(s)
    return np.array(means), np.array(sds)

fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.45), sharey=True)
plt.rcParams["font.family"] = "Arial"
y = np.arange(len(MODELS))[::-1]  # M0 at top
OFF = 0.17

for ax, metric, ttl in zip(axes, ("auc_pr", "auc_roc"), ("AUC-PR", "AUC-ROC")):
    em, es = series(econ, econ_hgt, metric)
    nm, ns = series(neuro, neuro_hgt, metric)
    ax.errorbar(em, y + OFF, xerr=es, fmt="o", ms=3.4, color=ECON_BLUE,
                ecolor=ECON_BLUE, elinewidth=0.9, capsize=1.6, lw=0, zorder=3)
    ax.errorbar(nm, y - OFF, xerr=ns, fmt="o", ms=3.4, color=NEURO_ORANGE,
                ecolor=NEURO_ORANGE, elinewidth=0.9, capsize=1.6, lw=0, zorder=3)
    for yi in y:
        ax.axhline(yi, color="#DDDDDD", lw=0.4, zorder=1)
    ax.set_title(ttl, fontsize=8, pad=3)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-0.55, len(MODELS) - 0.45)

axes[0].set_yticks(y)
axes[0].set_yticklabels([lbl for _, lbl in MODELS], fontsize=7)
axes[0].tick_params(axis="y", length=0)
axes[0].set_xlim(0.10, 0.50)
axes[1].set_xlim(0.45, 0.85)

# separate the HGT row (comparison model, not a ladder rung)
for ax in axes:
    ax.axhline(0.5, color="#999999", lw=0.6, ls=(0, (2, 2)))

handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=ECON_BLUE, label="Economics"),
           plt.Line2D([], [], marker="o", ls="", ms=4, color=NEURO_ORANGE, label="Neuroscience")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7,
           frameon=False, bbox_to_anchor=(0.56, -0.015), handletextpad=0.2,
           columnspacing=1.0)

fig.tight_layout(rect=(0, 0.05, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGDIR, f"F11_econ_vs_neuro_ladder.{ext}"),
                dpi=300, bbox_inches="tight")
print("wrote F11_econ_vs_neuro_ladder.pdf/.png")
print("econ HGT:", {k: (round(v[0], 4), round(v[1], 4)) for k, v in econ_hgt.items()})
print("neuro HGT:", {k: (round(v[0], 4), round(v[1], 4)) for k, v in neuro_hgt.items()})
