"""F0 - the benchmark framework in five panels, redrawn for legibility.

2026-07-14: the findings strip is removed from the figure - its text now
lives in the caption (KDD reviewer feedback: five panels + strip packed too
much into one graphic); the freed height goes to the panels, whose fonts grow
accordingly. Panel 5 wording scoped to the STRUCTURAL findings (the
divergence-citation estimate does not transfer to neuroscience, E6-neuro).
Drawn at 2x print size (14 x 3.6 in for a 7-in \\textwidth).

Numbers shown are the paper's headline values; sources:
results_econ/e1_baselines.json, results_econ/e10_advisor_placebo.json, e3_ablation.json,
e12_hgt_vs_baselines(_tuned).json, e6_innovation_premium.json,
results_neuro/e1_baselines.json, results_neuro/e12_hgt_vs_baselines.json.
Output: paper_pipeline/figures_econ/F0_framework.pdf/.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "figures_econ")

ECON_BLUE = "#1f77b4"
NEURO_ORANGE = "#ff7f0e"
NAVY, TEAL, RED, GREY = "#3C5488", "#00A087", "#E64B35", "#5A6472"

fig, ax = plt.subplots(figsize=(14, 3.6))
ax.set_xlim(0, 14); ax.set_ylim(0, 3.6); ax.axis("off")
plt.rcParams["font.family"] = "Arial"

def panel(x, y, w, h, edge, title, lines, title_fs=15.0, body_fs=12.4, lh=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.055,rounding_size=0.09",
                         linewidth=2.0, edgecolor=edge, facecolor="#FFFFFF")
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.30, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color=edge)
    n = len(lines)
    lh = lh or (h - 1.05) / max(n - 1, 1)
    y0 = y + h - 0.70
    for i, (txt, kw) in enumerate(lines):
        ax.text(x + w/2, y0 - i*lh, txt, ha="center", va="center",
                fontsize=kw.get("fs", body_fs), color=kw.get("c", "#222222"),
                fontweight=kw.get("w", "normal"), style=kw.get("st", "normal"))

def arrow(x0, y, x1, color="#666666"):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=17, linewidth=1.8, color=color))

TOP_Y, TOP_H, W = 0.20, 3.20, 2.44
xs = [0.18, 2.94, 5.70, 8.46, 11.22]

panel(xs[0], TOP_Y, W, TOP_H, NAVY, "1  Data",
      [("Academic Family Tree", {}),
       ("$\\oplus$ OpenAlex records", {}),
       ("frozen benchmark", {}),
       ("6,043 pairs, hash-pinned", {"w": "bold", "c": ECON_BLUE}),
       ("(economics)", {"fs": 10.2, "c": "#555555"})])

panel(xs[1], TOP_Y, W, TOP_H, TEAL, "2  Temporal contract",
      [("inputs: records $\\leq t_0{+}5$", {}),
       ("label: $(t_0{+}5,\\, t_0{+}15]$", {}),
       ("selection on validation only,", {}),
       ("test evaluated once", {}),
       ("6 leakage channels removed", {"w": "bold"})])

panel(xs[2], TOP_Y, W, TOP_H, RED, "3  Leakage audit",
      [("y-scrambling placebos", {}),
       ("(30 seeds, at chance)", {"fs": 11.2, "c": "#555555"}),
       ("advisor-disjoint split", {}),
       ("advisor + student-only", {}),
       ("controls (E10, E14):", {}),
       ("0.347 $\\to$ 0.158 collapse", {"w": "bold", "c": RED})])

panel(xs[3], TOP_Y, W, TOP_H, ECON_BLUE, "4  Models",
      [("tabular ladder M0–M5", {}),
       ("ceiling AUC-PR 0.390", {"w": "bold", "c": ECON_BLUE}),
       ("HGT standard 0.351", {}),
       ("HGT tuned 0.352", {}),
       ("both significantly below", {"st": "italic"})])

panel(xs[4], TOP_Y, W, TOP_H, NEURO_ORANGE, "5  Second discipline",
      [("identical pipeline $\\to$", {}),
       ("neuroscience, 21,844", {"w": "bold", "c": NEURO_ORANGE}),
       ("ladder + placebo + HGT", {}),
       ("HGT ties ceiling (0.429 $\\approx$ 0.426)", {"fs": 11.4}),
       ("structural findings replicate", {"st": "italic", "w": "bold"})])

for i in range(4):
    arrow(xs[i] + W + 0.07, TOP_Y + TOP_H/2, xs[i+1] - 0.09)

# findings strip removed: its text lives in the Figure 1 caption now.

fig.tight_layout(pad=0.25)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGDIR, f"F0_framework.{ext}"), dpi=220,
                bbox_inches="tight", facecolor="white")
print("wrote F0_framework.pdf/.png")
