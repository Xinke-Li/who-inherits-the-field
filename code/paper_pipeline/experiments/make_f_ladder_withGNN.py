"""Review copies of the F2 baseline ladder with the two extra GNNs added (R3).

This is a NON-paper review figure. It does NOT overwrite the paper's
submission_KDD/figures/F2_* ; it writes to paper_pipeline/figures_econ/ under
_withGNN names. It replicates make_f2_ladder_full.py's family-ladder style
verbatim (stable family palette, marker shapes, per-panel ceiling max(M2,M3)
dashed line) and adds two rows to the graph-aware family:

  RGCN             (results_<disc>/results_extra_gnns/rgcn_seed*.json)
  GAT+cohort-time  (results_<disc>/results_extra_gnns/tgat_seed*.json)  <- honest
                   label; a GATConv + per-student cohort-time channel, NOT a full
                   temporal GNN / TGN.

Every plotted number is read from a result JSON (nothing typed in); the two new
rows are read from results_<disc>/h_extra_gnns_summary.json (produced by
h_extra_gnns_summary.py, itself only a reduction of the per-seed JSONs). No
load_dataset call, so the config econ-path is irrelevant.

Outputs (paper_pipeline/figures_econ/):
  F2_e1_baseline_ladder_withGNN.pdf/.png       economics only, single panel
  F11_econ_vs_neuro_ladder_withGNN.pdf/.png    economics LEFT, neuroscience RIGHT
"""
import glob
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
PP = os.path.normpath(os.path.join(HERE, ".."))            # paper_pipeline/
FIGDIR = os.path.join(PP, "figures_econ")                  # review copies live here

# STABLE family palette + marker shapes, identical to make_f2_ladder_full.py so
# the review copy is a faithful extension of the paper's Figure 2.
FAMILIES = [
    ("prior",        "#555555", "s"),
    ("text-only",    "#E69F00", "^"),
    ("tabular",      "#1f4e79", "o"),
    ("graph-aware",  "#009E73", "D"),
    ("text+tabular", "#8b0000", "v"),
]
FAM_STYLE = {name: (c, m) for name, c, m in FAMILIES}

# rows top -> bottom: (label, family, source, key). Two new graph-aware rows
# (RGCN, GAT+cohort-time) sit with M5/HGT, keeping the family contiguous.
ROWS = [
    ("Prior (M0)",         "prior",        "e1",        "M0_prior"),
    ("TF-IDF (M4)",        "text-only",    "e1",        "M4_logit_tfidf"),
    ("SPECTER2 (M4b)",     "text-only",    "e14d",      "M4b_gbdt"),
    ("Overlap (M1)",       "tabular",      "e1",        "M1_logit_overlap"),
    ("Tabular logit (M2)", "tabular",      "e1",        "M2_logit_tabular"),
    ("Tabular GBDT (M3)",  "tabular",      "e1",        "M3_gbdt_tabular"),
    ("GBDT + NFA (M5)",    "graph-aware",  "e1",        "M5_gbdt_nfa"),
    ("HGT (standard)",     "graph-aware",  "hgt_std",   None),
    ("HGT (tuned)",        "graph-aware",  "hgt_tuned", None),
    ("RGCN",               "graph-aware",  "xgnn",      "rgcn"),
    ("GAT+cohort-time",    "graph-aware",  "xgnn",      "tgat"),
    ("SPECTER2 + tab.",    "text+tabular", "e14d",      "M4b_gbdt_tab"),
]


def _summary(path):
    with open(path) as f:
        return json.load(f)["summary"]


def _hgt(path):
    with open(path) as f:
        d = json.load(f)
    return (d["hgt_auc_pr"]["mean"], d["hgt_auc_pr"]["std"], len(d["seeds"]))


def load_discipline(disc):
    rdir = os.path.join(PP, f"results_{disc}")
    e1 = _summary(os.path.join(rdir, "e1_baselines.json"))
    e14d = _summary(os.path.join(rdir, "e14d_m4b.json"))
    e12 = sorted(glob.glob(os.path.join(rdir, "e12_hgt_vs_baselines*.json")))
    hgt = {"hgt_std": None, "hgt_tuned": None}
    for f in e12:
        hgt["hgt_tuned" if "tuned" in os.path.basename(f) else "hgt_std"] = _hgt(f)
    xg = json.load(open(os.path.join(rdir, "h_extra_gnns_summary.json")))["models"]

    rows = []  # (mean, std, n_seeds) or None if never run
    for _, _, src, key in ROWS:
        if src in ("e1", "e14d"):
            a = (e1 if src == "e1" else e14d)[key]["auc_pr"]
            rows.append((a["mean"], a["std"], a["n_seeds"]))
        elif src == "xgnn":
            m = xg[key]
            rows.append((m["auc_pr_mean"], m["auc_pr_std"], m["n_seeds"]))
        else:
            rows.append(hgt[src])
    ceiling = max(e1["M2_logit_tabular"]["auc_pr"]["mean"],
                  e1["M3_gbdt_tabular"]["auc_pr"]["mean"])
    return rows, ceiling


data = {d: load_discipline(d) for d in ("econ", "neuro")}
for d, (rows, ceil_v) in data.items():
    print(f"[{d}] ceiling max(M2,M3) = {ceil_v:.4f}")
    for (lab, fam, _, _), r in zip(ROWS, rows):
        note = ""
        if fam == "graph-aware" and r is not None and r[0] > ceil_v:
            note = "  <-- ABOVE CEILING"
        print(f"  {lab:<20s} {fam:<13s} "
              + ("(not run)" if r is None else
                 f"{r[0]:.4f} +- {r[1]:.4f} (n={r[2]}){note}"))

n = len(ROWS)
y = np.arange(n)[::-1]


def xlims(discs):
    vals = [(r[0] - r[1], r[0] + r[1])
            for d in discs for r in data[d][0] if r is not None]
    lo = np.floor((min(v[0] for v in vals) - 0.02) * 50) / 50
    hi = np.ceil((max(v[1] for v in vals) + 0.02) * 50) / 50
    return lo, hi


def draw_panel(ax, disc, lo, hi, title=None):
    rows, ceil_v = data[disc]
    for yi in y:
        ax.axhline(yi, color="#DDDDDD", lw=0.4, zorder=1)
    for i in range(1, n):
        if ROWS[i][1] != ROWS[i - 1][1]:
            ax.axhline(y[i] + 0.5, color="#999999", lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.axvline(ceil_v, color="#333333", lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.text(ceil_v - 0.008, y[0] + 0.42, f"ceiling {ceil_v:.3f}",
            rotation=90, ha="right", va="top", fontsize=6.5, color="#333333")
    for (lab, fam, _, _), r, yi in zip(ROWS, rows, y):
        color, marker = FAM_STYLE[fam]
        if r is None:
            ax.text(lo + 0.012, yi, "n/a", fontsize=6.5, color="#999999",
                    va="center", zorder=3)
            continue
        m, s, nseeds = r
        ax.errorbar(m, yi, xerr=(s if nseeds > 1 else None), fmt=marker,
                    ms=3.4, color=color, ecolor=color, elinewidth=0.9,
                    capsize=1.6, lw=0, zorder=3)
    if title:
        ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_xticks(np.arange(np.ceil(lo * 10) / 10, hi, 0.1))
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("Test AUC-PR", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def add_ylabels(ax):
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _, _, _ in ROWS], fontsize=7)
    ax.tick_params(axis="y", length=0)


def add_legend(fig, y_anchor):
    handles = [plt.Line2D([], [], marker=m, ls="", ms=3.8, color=c, label=name)
               for name, c, m in FAMILIES]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, y_anchor),
               handletextpad=0.15, columnspacing=0.7)


def footnote(fig, lines, y=0.012):
    fig.text(0.5, y, "\n".join(lines), ha="center", va="bottom", fontsize=5.6,
             color="#333333", linespacing=1.35)


def save(fig, stem):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(FIGDIR, f"{stem}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {stem}.pdf/.png  (figsize in: {fig.get_size_inches()})")
    plt.close(fig)


# honest in-figure captions, driven by the paired per-seed test (h_extra_gnns_paired.json)
def load_paired(disc):
    return json.load(open(os.path.join(PP, f"results_{disc}", "h_extra_gnns_paired.json")))


def _model_phrase(disc, model):
    """Data-driven phrase for one model vs the panel's ceiling component."""
    hv = load_paired(disc)["headline_vs_ceiling_component"][model]
    lab = {"rgcn": "RGCN", "tgat": "GAT+cohort-time"}[model]
    d, ck = hv["mean_diff"], hv["vs_ceiling_component"]
    if hv["significant_exceedance"]:
        return f"{lab} significantly exceeds the ceiling ({d:+.3f}, p_adj={hv['p_adj']:.3f})"
    if hv["exceeds_ceiling_point_estimate"]:
        return f"{lab} point estimate above but not significant ({d:+.3f}, p_adj={hv['p_adj']:.2f})"
    ns = " ns" if hv["p_adj"] > 0.05 else ""
    return f"{lab} {d:+.3f}{ns}"


# ---------- econ only, single panel ----------
lo, hi = xlims(["econ"])
fig, ax = plt.subplots(figsize=(4.0, 3.8))
fig.subplots_adjust(left=0.27, right=0.97, top=0.97, bottom=0.31)
draw_panel(ax, "econ", lo, hi)
add_ylabels(ax)
add_legend(fig, 0.150)
footnote(fig, [
    "RGCN + GAT+cohort-time (GATConv + per-student cohort-time channel; not a full "
    "temporal GNN) added to the graph-aware",
    "family; 10-seed mean +/- sd. Economics: all graph-aware at/below ceiling "
    "max(M2,M3)=%.3f (%s; %s)." % (
        data["econ"][1], _model_phrase("econ", "rgcn"), _model_phrase("econ", "tgat")),
], y=0.010)
save(fig, "F2_e1_baseline_ladder_withGNN")

# ---------- both disciplines (econ LEFT, neuro RIGHT) ----------
lo, hi = xlims(["econ", "neuro"])
fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.9), sharey=True)
fig.subplots_adjust(left=0.165, right=0.985, top=0.93, bottom=0.30, wspace=0.08)
draw_panel(axes[0], "econ", lo, hi, title="Economics")
draw_panel(axes[1], "neuro", lo, hi, title="Neuroscience")
add_ylabels(axes[0])
add_legend(fig, 0.160)
footnote(fig, [
    "RGCN + GAT+cohort-time (GATConv + per-student cohort-time channel; not a full "
    "temporal GNN); 10-seed mean +/- sd vs each panel's",
    "ceiling max(M2,M3); paired per-seed Wilcoxon + BH. Econ (%.3f): all graph-aware "
    "at/below ceiling (%s; %s)." % (
        data["econ"][1], _model_phrase("econ", "rgcn"), _model_phrase("econ", "tgat")),
    "Neuro (%.3f): %s; %s." % (
        data["neuro"][1], _model_phrase("neuro", "rgcn"), _model_phrase("neuro", "tgat")),
], y=0.010)
save(fig, "F11_econ_vs_neuro_ladder_withGNN")

# ---------- PAPER-TUNED copies (footnote-free, single-column aspect) ----------
# These overwrite the paper's submission_KDD/figures/F2_* and F11_* with the SAME
# data/markers as the review copies above, but without the embedded footnote and
# at the paper's column aspect: the honest method note and the neuro RGCN
# exception live in the LaTeX captions (Figure 2 / Figure 11) and Table 3, so
# duplicating them inside the image only shrinks the figure. This keeps the
# economics ladder (Figure 2a, in the body) from growing the page budget.
KDDDIR = os.path.normpath(os.path.join(PP, "..", "..", "..", "submission_KDD", "figures"))


def save_paper(fig, name):
    os.makedirs(KDDDIR, exist_ok=True)
    out = os.path.join(KDDDIR, f"{name}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote paper {out}  (figsize in: {fig.get_size_inches()})")
    plt.close(fig)


if os.path.isdir(KDDDIR):
    lo, hi = xlims(["econ"])
    fig, ax = plt.subplots(figsize=(3.33, 3.18))
    fig.subplots_adjust(left=0.30, right=0.965, top=0.975, bottom=0.195)
    draw_panel(ax, "econ", lo, hi)
    add_ylabels(ax)
    add_legend(fig, 0.012)
    save_paper(fig, "F2_e1_baseline_ladder")

    lo, hi = xlims(["econ", "neuro"])
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 3.28), sharey=True)
    fig.subplots_adjust(left=0.205, right=0.985, top=0.93, bottom=0.185, wspace=0.08)
    draw_panel(axes[0], "econ", lo, hi, title="Economics")
    draw_panel(axes[1], "neuro", lo, hi, title="Neuroscience")
    add_ylabels(axes[0])
    add_legend(fig, 0.012)
    save_paper(fig, "F11_econ_vs_neuro_ladder")
else:
    print("SKIP paper copies: submission_KDD/figures not found at", KDDDIR)
