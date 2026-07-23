#!/usr/bin/env python3
"""Certificate geometry dot plot across the five disciplines, in the paper house
style (serif type, darkblue/darkred with a gray neutral, no top or right spines,
faint horizontal grid only). One horizontal band per discipline, ordered by early
co-authorship, five points per band on a Test AUC-PR axis: the M0 prior (gray
square), the cohort-matched advisor placebo from e10 (gray triangle), the best
student-only floor from e14 (darkblue circle), the single-scalar advisor overlap
model M1 (open darkred triangle), and the best tabular rung (filled darkred
triangle). A light gray band behind each row spans the prior to the cohort
placebo, the compositional stretch, so the placebo visibly hugs the bottom of
every band. The same geometry repeats in all five disciplines: the placebo sits
near the prior and every advisor-informed model sits far to the right, while the
position of the student-only floor between them is the branch structure of the
self-persistence test. Reads results/results_<field>/e1_baselines.json,
e10_advisor_placebo.json, and e14_self_persistence.json. Writes a vector PDF for
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
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


# (field key, display name), ordered by early co-authorship as in the ladder
PANELS = [("econ", "economics"), ("math", "mathematics"),
          ("neuro", "neuroscience"), ("physics", "physics"),
          ("chemistry", "chemistry")]
TABULAR = ["M2_logit_tabular", "M3_gbdt_tabular", "M5_gbdt_nfa"]
STUDENT_ONLY = ["M1s_typicality", "M1s_scalars", "M4s_tfidf", "M3s_gbdt"]
DARKBLUE = "#00008B"
DARKRED = "#8B0000"
DIMGRAY = "#555555"


def load(field):
    base = RES / f"results_{field}"
    e1 = json.load(open(base / "e1_baselines.json"))["summary"]
    e10 = json.load(open(base / "e10_advisor_placebo.json"))["summary"]
    e14 = json.load(open(base / "e14_self_persistence.json"))["a_student_only_ladder"]["summary"]
    return {
        "prior": e1["M0_prior"]["auc_pr"]["mean"],
        "placebo": e10["placebo_cohort"]["L1_logit_overlap"]["auc_pr"]["mean"],
        "floor": max(e14[m]["auc_pr"]["mean"] for m in STUDENT_ONLY),
        "m1": e1["M1_logit_overlap"]["auc_pr"]["mean"],
        "best": max(e1[m]["auc_pr"]["mean"] for m in TABULAR),
    }


def build():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ys = list(range(len(PANELS)))[::-1]  # economics on top
    for y, (field, name) in zip(ys, PANELS):
        d = load(field)
        pts = [d["prior"], d["placebo"], d["floor"], d["m1"], d["best"]]
        ax.plot([min(pts), max(pts)], [y, y], lw=0.7, color="#cccccc", zorder=2)
        # placebo band: the compositional stretch from the prior to the cohort
        # placebo, shaded so the placebo visibly hugs the bottom of each band
        ax.plot([d["prior"], d["placebo"]], [y, y], lw=7, color="#e4e4e4",
                solid_capstyle="round", zorder=1)
        ax.plot(d["prior"], y, marker="s", ms=4.5, color=DIMGRAY, ls="none", zorder=4)
        ax.plot(d["placebo"], y, marker="v", ms=5, color=DIMGRAY, ls="none", zorder=4)
        ax.plot(d["floor"], y, marker="o", ms=5, color=DARKBLUE, ls="none", zorder=4)
        ax.plot(d["m1"], y, marker="^", ms=5.5, mfc="white", mec=DARKRED,
                mew=1.0, ls="none", zorder=4)
        ax.plot(d["best"], y, marker="^", ms=5.5, color=DARKRED, ls="none", zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([name for _, name in PANELS])
    ax.set_ylim(-0.6, len(PANELS) - 0.4)
    ax.set_xlabel("Test AUC-PR")
    ax.set_xlim(0.15, 0.68)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=2.5)
    ax.tick_params(axis="y", length=0)
    handles = [
        mpl.lines.Line2D([], [], marker="s", ms=4.5, color=DIMGRAY, ls="none",
                         label="M0 prior"),
        mpl.lines.Line2D([], [], marker="v", ms=5, color=DIMGRAY, ls="none",
                         label="placebo advisor (e10)"),
        mpl.lines.Line2D([], [], marker="o", ms=5, color=DARKBLUE, ls="none",
                         label="student-only floor (e14)"),
        mpl.lines.Line2D([], [], marker="^", ms=5.5, mfc="white", mec=DARKRED,
                         mew=1.0, ls="none", label="M1 advisor overlap"),
        mpl.lines.Line2D([], [], marker="^", ms=5.5, color=DARKRED, ls="none",
                         label="best tabular"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.55, -0.30), handletextpad=0.3, columnspacing=1.0,
               labelspacing=0.5)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "F13_certificate_geometry"
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
