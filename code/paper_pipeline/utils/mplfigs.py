"""Publication figures - SEC-notebook aesthetic (seaborn darkgrid + matplotlib
path-effect drop shadows, the "3D" look). This module is now authoritative for
the paper's PNG/PDF figures; the plotly HTMLs remain for interactive viewing.

Style fingerprint (from SEC_Filing_Compliance_Analysis.ipynb):
  sns.set_theme() darkgrid - panel #EAEAF2, white gridlines, no spines/ticks
  palette: navy #1f4e79 (dark #0c2b4a) / darkred #8b0000 / grays #555555...
  bars/boxes: white edges (lw 1.5), SimplePatchShadow offset(3,-3) alpha .25
  lines: lw 3, white-edged markers, SimpleLineShadow offset(2,-2)
  typography: bold titles (13, pad 14-18), bold 11 axis labels #333333,
  bold tick labels; compact figsize so text reads large in print.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
# ACM camera-ready: embed TrueType (Type 42), never Type 3 bitmap-glyph fonts.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

sns.set_theme()

NAVY, DARKNAVY = "#1f4e79", "#0c2b4a"
RED = "#8b0000"
GRAY = "#555555"
GRAYS = ["#4a4a4a", "#8c8c8c", "#c6c6c6"]
INK = "#333333"

BAR_FX = [pe.SimplePatchShadow(offset=(3, -3), shadow_rgbFace=(0, 0, 0), alpha=0.25),
          pe.Normal()]
LINE_FX = [pe.SimpleLineShadow(offset=(2, -2), shadow_color="black", alpha=0.2,
                               linewidth=4), pe.Normal()]
ERR_KW = dict(ecolor="#2b2b2b", capsize=4, capthick=1.6, elinewidth=1.6, zorder=10)


def _fmt(ax, title=None, xlabel=None, ylabel=None, title_size=13):
    if title:
        ax.set_title(title, fontsize=title_size, fontweight="bold", pad=16,
                     color="#222222")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold", labelpad=10, color=INK)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold", labelpad=10, color=INK)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold"); lab.set_fontsize(10.5); lab.set_color(INK)
    sns.despine(ax=ax, left=True, bottom=True)


def _save(fig, name):
    fig.savefig(C.FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(C.FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name} (png+pdf)")


def _shadow(ax):
    for p in ax.patches:
        p.set_path_effects(BAR_FX)


# ---------------- F2: baseline ladder ----------------
def fig_baselines(summary, fname="F2_e1_baseline_ladder"):
    order = ["M0_prior", "M1_logit_overlap", "M2_logit_tabular",
             "M3_gbdt_tabular", "M4_logit_tfidf", "M5_gbdt_nfa"]
    labels = ["Prior", "Logit\n(overlap)", "Logit\n(tabular)",
              "GBDT\n(tabular)", "Logit\n(+TF-IDF)", "GBDT\n+ NFA"]
    mean = [summary[m]["auc_pr"]["mean"] for m in order]
    std = [summary[m]["auc_pr"]["std"] for m in order]
    colors = [RED if m in ("M3_gbdt_tabular", "M5_gbdt_nfa") else GRAY for m in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, mean, width=0.62, color=colors, alpha=0.95,
           edgecolor="white", linewidth=1.5,
           yerr=std, error_kw=ERR_KW)
    _shadow(ax)
    for x, (m, s) in enumerate(zip(mean, std)):
        ax.annotate(f"{m:.3f}", (x, m + s + 0.012), ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color=INK, zorder=11)
    ln = ax.axhline(0.135, color=RED, linewidth=2, linestyle="--")
    ln.set_path_effects(LINE_FX)
    ax.annotate("base rate = 0.135", (0.02, 0.196), xycoords=("axes fraction", "data"),
                fontsize=10.5, fontweight="bold", color=RED, zorder=11)
    ax.set_ylim(0, 0.47)
    _fmt(ax, "Baseline ladder: test AUC-PR (temporal split, 10 seeds)",
         None, "AUC-PR")
    fig.tight_layout()
    _save(fig, fname)


# ---------------- F10: advisor placebo ----------------
def fig_advisor_placebo(summary, fname="F10_e10_advisor_placebo"):
    models = [("L1_logit_overlap", "Logit, overlap only"),
              ("L2_logit_tabular", "Logit, all tabular"),
              ("G3_gbdt_tabular", "GBDT, all tabular")]
    variants = [("true", "true advisor", RED),
                ("placebo_cohort", "cohort-matched placebo", GRAYS[0]),
                ("placebo_random", "random placebo", GRAYS[1]),
                ("field_mean", "discipline-mean profile", GRAYS[2])]
    x = np.arange(len(models)); w = 0.19
    fig, ax = plt.subplots(figsize=(8.6, 5))
    for i, (v, lab, color) in enumerate(variants):
        means = [summary[v][m]["auc_pr"]["mean"] for m, _ in models]
        stds = [summary[v][m]["auc_pr"]["std"] for m, _ in models]
        ax.bar(x + (i - 1.5) * w, means, width=w * 0.92, color=color, alpha=0.95,
               edgecolor="white", linewidth=1.5, label=lab,
               yerr=stds, error_kw=dict(ERR_KW, capsize=3))
    _shadow(ax)
    ln = ax.axhline(0.135, color=RED, linewidth=2, linestyle="--")
    ln.set_path_effects(LINE_FX)
    ax.annotate("base rate = 0.135", (0.99, 0.122), xycoords=("axes fraction", "data"),
                ha="right", va="top", fontsize=10.5, fontweight="bold", color=RED)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in models])
    ax.set_ylim(0, 0.45)
    ax.legend(loc="upper left", fontsize=9.5, frameon=False, ncol=2,
              columnspacing=1.2, handlelength=1.4)
    _fmt(ax, "Advisor-placebo control (E10)", None, "AUC-PR")
    fig.tight_layout()
    _save(fig, fname)


# ---------------- F3: placebo boxes ----------------
def fig_placebo(global_rows, cohort_rows, fname="F3_e9a_placebo"):
    import pandas as pd
    df = pd.DataFrame({
        "AUC-ROC": [r["auc_roc"] for r in global_rows] +
                   [r["auc_roc"] for r in cohort_rows],
        "design": (["global shuffle"] * len(global_rows) +
                   ["within-cohort shuffle"] * len(cohort_rows))})
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.boxplot(data=df, x="design", y="AUC-ROC", hue="design",
                palette=[NAVY, RED], width=0.42, ax=ax, legend=False,
                linewidth=1.5, fliersize=4,
                boxprops=dict(alpha=0.95, edgecolor="white"),
                medianprops=dict(color="white", linewidth=2.5),
                whiskerprops=dict(color="#444444", linewidth=1.6),
                capprops=dict(color="#444444", linewidth=1.6),
                flierprops=dict(markerfacecolor="#888888",
                                markeredgecolor="#888888", alpha=0.8))
    _shadow(ax)
    ln = ax.axhline(0.5, color=RED, linewidth=2.2, linestyle="--")
    ln.set_path_effects(LINE_FX)
    ax.annotate("chance (0.5)", (0.02, 0.508), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", fontsize=10.5, fontweight="bold", color=RED,
                zorder=11)
    ax.set_ylim(0.36, 0.67)
    _fmt(ax, "Leakage certificate: placebo models vs true test labels\n(30 seeds each)",
         "", "Test AUC-ROC")
    fig.tight_layout()
    _save(fig, fname)


# ---------------- F9: robustness ----------------
def fig_robustness(variants, fname="F9_e9_robustness"):
    pretty = {"ref_temporal": "Reference (temporal split)",
              "b_advisor_disjoint": "Advisor-disjoint split",
              "c_drop_capped": "Drop 400-capped students",
              "d_t0_ge_1960": "Drop t0 < 1960",
              "e_no_overlap_feature": "Remove early-overlap feature"}
    names = list(variants)
    mean = [variants[n]["summary"]["auc_pr"]["mean"] for n in names]
    std = [variants[n]["summary"]["auc_pr"]["std"] for n in names]
    colors = [RED if n.startswith("ref") else GRAY for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh([pretty.get(n, n) for n in names], mean, height=0.55, color=colors,
            alpha=0.95, edgecolor="white", linewidth=1.5,
            xerr=std, error_kw=ERR_KW)
    _shadow(ax)
    for y, (m, s) in enumerate(zip(mean, std)):
        ax.annotate(f"{m:.3f}", (m + s + 0.012, y), va="center", ha="left",
                    fontsize=10.5, fontweight="bold", color=INK, zorder=11)
    ln = ax.axvline(0.135, color=RED, linewidth=2, linestyle="--")
    ln.set_path_effects(LINE_FX)
    ax.annotate("base rate\n= 0.135", (0.142, -0.35), fontsize=9.5,
                fontweight="bold", color=RED, va="top")
    ax.set_xlim(0, 0.48)
    _fmt(ax, "Stress tests: AUC-PR under design perturbations (10 seeds each)",
         "AUC-PR", None)
    fig.tight_layout()
    _save(fig, fname)


# ---------------- F4 / e4: theta stability ----------------
def _line(ax, x, y, color, label, ms=8):
    l, = ax.plot(x, y, marker="o", markersize=ms, markeredgecolor="white",
                 markeredgewidth=2, linewidth=3, color=color, label=label, zorder=3)
    l.set_path_effects(LINE_FX)
    return l


def fig_theta_stability(sweep, fname="F4_e4_theta_stability"):
    thetas = sorted(sweep, key=float)
    xv = [float(t) for t in thetas]
    fig, ax = plt.subplots(figsize=(8, 5))
    _line(ax, xv, [sweep[t]["auc_roc"]["mean"] for t in thetas], NAVY, "AUC-ROC")
    _line(ax, xv, [sweep[t]["auc_pr"]["mean"] for t in thetas], RED, "AUC-PR")
    l, = ax.plot(xv, [sweep[t]["base_rate"] for t in thetas], marker="o",
                 markersize=6, markeredgecolor="white", markeredgewidth=1.5,
                 linewidth=2, linestyle="--", color="#777777", label="base rate")
    ax.set_ylim(0, 0.9)
    ax.legend(loc="center right", fontsize=10, frameon=False)
    _fmt(ax, "Label-threshold robustness (GBDT, temporal split, 10 seeds)",
         r"Jaccard threshold $\theta$", "Score")
    fig.tight_layout()
    _save(fig, fname)


def fig_theta_single(thetas, vals, fname="e4_theta_stability"):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    _line(ax, thetas, vals, NAVY, "M3 GBDT")
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.set_ylim(0, max(vals) * 1.12)
    _fmt(ax, "Label-threshold robustness (AUC-PR)",
         r"Jaccard threshold $\theta$", "AUC-PR")
    fig.tight_layout()
    _save(fig, fname)


# ---------------- F6: innovation premium spec curve ----------------
def fig_coefficient_ladder(specs, fname="F6_e6_innovation_premium"):
    names = list(specs)
    pretty = {"1_raw": "raw", "2_controls": "+ controls",
              "3_controls_FE": "+ cohort &\ninstitution FE",
              "4_structural_FE": "structural\ndivergence + FE"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, s in enumerate(names):
        coef, se, pval = specs[s]["coef"], specs[s]["se"], specs[s]["p"]
        color = RED if pval < 0.05 else GRAY
        ax.errorbar([i], [coef], yerr=[1.96 * se], fmt="o", markersize=11,
                    markeredgecolor="white", markeredgewidth=2, color=color,
                    ecolor=color, elinewidth=2.4, capsize=6, capthick=2.2, zorder=3)
        ax.annotate(f"{coef:+.4f}\np={pval:.3f}", (i + 0.14, coef), fontsize=9.5,
                    fontweight="bold", color=color, va="center")
    ln = ax.axhline(0, color="#666666", linewidth=1.6, linestyle="--")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([pretty.get(s, s) for s in names])
    ax.set_xlim(-0.5, len(names) - 0.3)
    _fmt(ax, "Innovation premium: divergence coefficient across specifications",
         None, r"$\beta$ (per SD divergence)")
    fig.tight_layout()
    _save(fig, fname)


# ---------------- F8: cohort trends ----------------
def fig_cohort_trends(trends, fname="F8_e8_cohort_trends"):
    dec = [t["decade"] for t in trends]
    x = np.arange(len(dec))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()
    ax2.bar(x, [t["n"] for t in trends], width=0.55, color="#c3cbd9", alpha=0.55,
            edgecolor="white", linewidth=1.2, zorder=1)
    ax2.set_ylabel("cohort size", fontsize=11, fontweight="bold", labelpad=10,
                   color=INK)
    ax2.grid(False)
    for lab in ax2.get_yticklabels():
        lab.set_fontweight("bold"); lab.set_fontsize(10.5); lab.set_color(INK)
    if all("retention_ci95" in t for t in trends):
        lo = [t["retention_rate"] - t["retention_ci95"][0] for t in trends]
        hi = [t["retention_ci95"][1] - t["retention_rate"] for t in trends]
        ax.errorbar(x, [t["retention_rate"] for t in trends], yerr=[lo, hi],
                    fmt="none", **ERR_KW)
    _line(ax, x, [t["retention_rate"] for t in trends], NAVY, "field retention")
    _line(ax, x, [t["coauth_rate"] for t in trends], RED, "early co-authorship")
    ax.set_xticks(x); ax.set_xticklabels(dec)
    ax.set_ylim(0, 0.5)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    _fmt(ax, "Six decades: retention flat since the 1990s;\nearly co-authorship rose to 0.43",
         None, "Rate")
    sns.despine(ax=ax2, left=True, right=True, bottom=True)
    fig.tight_layout()
    _save(fig, fname)


# ---------------- F7e: ablation forest ----------------
def fig_forest(rows, fname="F7e_ablation_forest"):
    pretty = {"student_concept": "student–concept", "advisor_concept": "advisor–concept",
              "coauth": "co-authorship", "institution": "institution",
              "advising": "advising", "social": "social (all)"}
    rows = sorted(rows, key=lambda r: r["mean_diff"])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for y, r in enumerate(rows):
        p = r["p_adj"]
        c = RED if p <= 0.05 else (NAVY if p <= 0.10 else "#8c8c8c")
        ax.errorbar([r["mean_diff"]], [y],
                    xerr=[[r["mean_diff"] - r["ci95"][0]], [r["ci95"][1] - r["mean_diff"]]],
                    fmt="o", markersize=10, markeredgecolor="white",
                    markeredgewidth=2, color=c, ecolor=c, elinewidth=2.2,
                    capsize=5, capthick=2, zorder=3)
        ax.annotate(f"p={p:.2f}", (1.01, y), xycoords=("axes fraction", "data"),
                    fontsize=9.5, fontweight="bold", color="#666666", va="center")
    ax.axvline(0, color="#333333", linewidth=1.6, linestyle="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([pretty.get(r["ablation"], r["ablation"]) for r in rows])
    ax.invert_yaxis()
    _fmt(ax, r"Edge-type ablations: $\Delta$ AUC-PR vs full model (95% CI)",
         r"$\Delta$ AUC-PR (ablated $-$ full)", None)
    fig.tight_layout()
    _save(fig, fname)


# ---------------- quintiles ----------------
def fig_quintiles(labels, means, counts, fname="e1_overlap_quintiles"):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(labels, means, width=0.6, color=NAVY, alpha=0.95,
           edgecolor="white", linewidth=1.5)
    _shadow(ax)
    for x, (m, n) in enumerate(zip(means, counts)):
        ax.annotate(f"{m:.1%}\n(n={n})", (x, m + 0.008), ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=INK, zorder=11)
    ax.set_ylim(0, max(means) * 1.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _fmt(ax, "Field retention by early-overlap quintile",
         "Early advisor-overlap quintile", "P(y = 1)")
    fig.tight_layout()
    _save(fig, fname)


# ---------------- F1b: top-50 PageRank ----------------
def fig_top50(top50, fname="F1b_top50_pagerank"):
    import pandas as pd
    t = pd.DataFrame(top50).sort_values("pagerank")
    vals = t["pagerank"].values
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "navy", ["#9fb6cd", NAVY, DARKNAVY])
    norm = (vals - vals.min()) / (vals.max() - vals.min())
    fig, ax = plt.subplots(figsize=(7, 11))
    ax.barh(t["name"], vals, height=0.72, color=cmap(norm),
            edgecolor="white", linewidth=0.8)
    _fmt(ax, "Top 50 economists by PageRank (descriptive)",
         "PageRank score", None)
    for lab in ax.get_yticklabels():
        lab.set_fontsize(8.2)
    fig.tight_layout()
    _save(fig, fname)


# ---------------- F7a-d: HGT panels ----------------
def fig_hgt_training(h, fname="F7a_hgt_training_dynamics"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()
    l1, = ax.plot(h["epoch"], h["train_loss"], color=RED, linewidth=3,
                  label="Training loss")
    l1.set_path_effects(LINE_FX)
    l2, = ax2.plot(h["epoch"], h["val_f1"], color=NAVY, linewidth=3,
                   label="Validation F1")
    l2.set_path_effects(LINE_FX)
    ax.set_ylabel("Loss", fontsize=11, fontweight="bold", color=RED, labelpad=10)
    ax2.set_ylabel("F1 score", fontsize=11, fontweight="bold", color=NAVY, labelpad=10)
    ax2.grid(False)
    for lab in ax.get_yticklabels():
        lab.set_color(RED)
    for lab in ax2.get_yticklabels():
        lab.set_fontweight("bold"); lab.set_fontsize(10.5); lab.set_color(NAVY)
    ax.legend(handles=[l1, l2], loc="center right", fontsize=10, frameon=False)
    _fmt(ax, "HGT training dynamics (seed 0)", "Epoch", None)
    sns.despine(ax=ax2, left=True, right=True, bottom=True)
    fig.tight_layout()
    _save(fig, fname)


def fig_hgt_roc(y, p, fname="F7b_hgt_roc"):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y, p)
    A = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.8)
    l, = ax.plot(fpr, tpr, color=NAVY, linewidth=3, label=f"HGT (AUC = {A:.3f})")
    l.set_path_effects(LINE_FX)
    ax.fill_between(fpr, tpr, alpha=0.15, color=NAVY)
    ax.legend(loc="lower right", fontsize=10.5, frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    _fmt(ax, "HGT test ROC (seed 0)", "False positive rate", "True positive rate")
    fig.tight_layout()
    _save(fig, fname)


def fig_hgt_pr(y, p, fname="F7c_hgt_pr"):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    base = float(np.mean(y))
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ln = ax.axhline(base, color=RED, linestyle="--", linewidth=2)
    ln.set_path_effects(LINE_FX)
    ax.annotate(f"base rate = {base:.3f}", (0.98, base + 0.015),
                xycoords=("axes fraction", "data"), ha="right", fontsize=10,
                fontweight="bold", color=RED)
    l, = ax.plot(rec, prec, color=NAVY, linewidth=3, label=f"HGT (AUC-PR = {ap:.3f})")
    l.set_path_effects(LINE_FX)
    ax.fill_between(rec, prec, alpha=0.15, color=NAVY)
    ax.legend(loc="upper right", fontsize=10.5, frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    _fmt(ax, "HGT test precision-recall (seed 0)", "Recall", "Precision")
    fig.tight_layout()
    _save(fig, fname)


def fig_hgt_confusion(y, p, thr, fname="F7d_hgt_confusion"):
    y = np.asarray(y); yhat = (np.asarray(p) >= thr).astype(int)
    cm = np.array([[np.sum((y == a) & (yhat == b)) for b in (0, 1)] for a in (0, 1)])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "navy", ["#EAEAF2", NAVY])
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.imshow(cm, cmap=cmap)
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / cm.max()
            ax.annotate(f"{cm[i, j]:,}", (j, i), ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if frac > 0.5 else INK)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticklabels(["true 0", "true 1"])
    ax.grid(False)
    _fmt(ax, f"HGT confusion matrix (val-optimal threshold = {thr:.2f})", None, None)
    fig.tight_layout()
    _save(fig, fname)
