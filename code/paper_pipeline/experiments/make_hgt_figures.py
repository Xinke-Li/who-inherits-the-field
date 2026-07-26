"""HGT figure panel - exact reproduction of the itr4 visual style.

Reads results_hgt/hgt_none_seed<S>.json (produced by the extended e2_hgt.py,
which now saves training history + test scores). Generates:
  F7a_hgt_training_dynamics  dual-axis: dark-red loss / gold validation F1
  F7b_hgt_roc                blue ROC with light fill + dashed random line
  F7c_hgt_pr                 precision-recall with base-rate line (AUC-PR primary)
  F7d_hgt_confusion          confusion matrix at the val-optimal threshold
Run after the Colab grid: python make_hgt_figures.py [--seed 0]
Then run e3_aggregate.py for the ablation forest (F7e).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as CFG

DARK_RED, GOLD, BLUE = "#8B0000", "#1F4E79", "#1F4E79"  # seaborn-darkgrid house palette


def _theme(fig):
    """Seaborn-darkgrid house style, applied to every panel at save time."""
    fig.update_layout(plot_bgcolor="#EAEAF2", paper_bgcolor="white",
                      font=dict(family="Arial", size=13, color="#333333"))
    t = fig.layout.title.text or ""
    fig.update_layout(title=dict(text=t, x=0.5, xanchor="center",
                                 font=dict(size=16, color="#222222")))
    fig.update_xaxes(showline=False, ticks="", showgrid=True,
                     gridcolor="white", gridwidth=1.4, zeroline=False,
                     tickfont=dict(size=12, color="#333333"))
    fig.update_yaxes(showline=False, ticks="", showgrid=True,
                     gridcolor="white", gridwidth=1.4, zeroline=False,
                     tickfont=dict(size=12, color="#333333"))
    return fig


def save(fig, name):
    _theme(fig)
    p = CFG.FIGURES_DIR / f"{name}.html"
    fig.write_html(p, include_plotlyjs="cdn")
    try:
        fig.write_image(CFG.FIGURES_DIR / f"{name}.png", width=800, height=520, scale=2)
    except Exception:
        pass
    print(f"[fig] {p}")


def training_dynamics(h):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["epoch"], y=h["train_loss"], name="Training Loss",
                             line=dict(color=DARK_RED, width=3)))
    fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_f1"], name="Validation F1",
                             line=dict(color=GOLD, width=3), yaxis="y2"))
    fig.update_layout(
        title=dict(text="<b>Training Dynamics</b>", font=dict(size=20), x=0.02),
        xaxis=dict(title="Epoch", showgrid=True, gridcolor="#EEEEEE"),
        yaxis=dict(title=dict(text="Loss", font=dict(color=DARK_RED)),
                   tickfont=dict(color=DARK_RED), showgrid=True, gridcolor="#EEEEEE"),
        yaxis2=dict(title=dict(text="F1 Score", font=dict(color="#1F4E79")),
                    tickfont=dict(color="#1F4E79"), overlaying="y", side="right"),
        legend=dict(orientation="h", x=0.3, y=-0.25),
        plot_bgcolor="white", paper_bgcolor="white")
    save(fig, "F7a_hgt_training_dynamics")


def roc_curve_fig(y, p):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y, p)
    A = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", mode="lines",
                             line=dict(color="#999999", dash="dash")))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {A:.3f}", mode="lines",
                             line=dict(color=BLUE, width=3),
                             fill="tozeroy", fillcolor="rgba(31,119,180,0.10)"))
    fig.update_layout(
        title=dict(text="<b>ROC Curve</b>", font=dict(size=20), x=0.02),
        xaxis=dict(title="False Positive Rate", showgrid=True, gridcolor="#EEEEEE"),
        yaxis=dict(title="True Positive Rate", showgrid=True, gridcolor="#EEEEEE"),
        legend=dict(x=0.62, y=0.08, bordercolor="#CCCCCC", borderwidth=1),
        plot_bgcolor="white", paper_bgcolor="white")
    save(fig, "F7b_hgt_roc")


def pr_curve_fig(y, p):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    base = float(np.mean(y))
    fig = go.Figure()
    fig.add_hline(y=base, line_dash="dash", line_color="#999999",
                  annotation_text=f"base rate {base:.3f}")
    fig.add_trace(go.Scatter(x=rec, y=prec, name=f"AUC-PR = {ap:.3f}", mode="lines",
                             line=dict(color="#E64B35", width=3),
                             fill="tozeroy", fillcolor="rgba(230,75,53,0.08)"))
    fig.update_layout(
        title=dict(text="<b>Precision-Recall Curve</b> (primary metric)",
                   font=dict(size=20), x=0.02),
        xaxis=dict(title="Recall", showgrid=True, gridcolor="#EEEEEE"),
        yaxis=dict(title="Precision", showgrid=True, gridcolor="#EEEEEE"),
        legend=dict(x=0.6, y=0.95, bordercolor="#CCCCCC", borderwidth=1),
        plot_bgcolor="white", paper_bgcolor="white")
    save(fig, "F7c_hgt_pr")


def confusion_fig(y, p, y_val, p_val):
    from sklearn.metrics import confusion_matrix, precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    t = float(thr[max(np.nanargmax(f1[:-1]), 0)])   # threshold tuned on VAL only
    cm = confusion_matrix(y, np.array(p) >= t)
    labels = [["TN", "FP"], ["FN", "TP"]]
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Pred: leaves field", "Pred: stays"],
        y=["True: leaves field", "True: stays"],
        text=[[f"{labels[i][j]}<br>{cm[i][j]}" for j in range(2)] for i in range(2)],
        texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(
        title=dict(text=f"<b>Confusion Matrix</b> (val-optimal threshold = {t:.3f})",
                   font=dict(size=20), x=0.02),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white")
    save(fig, "F7d_hgt_confusion")


def main(seed=0):
    p = CFG.RESULTS_DIR / "results_hgt" / f"hgt_none_seed{seed}.json"
    if not p.exists():
        sys.exit(f"{p} not found - run the Colab grid first (colab_hgt_runner.md)")
    r = json.loads(p.read_text())
    if "history" not in r:
        sys.exit("run was made with the old e2_hgt.py - rerun with the extended version")
    training_dynamics(r["history"])
    roc_curve_fig(r["test_labels"], r["test_scores"])
    pr_curve_fig(r["test_labels"], r["test_scores"])
    confusion_fig(r["test_labels"], r["test_scores"], r["val_labels"], r["val_scores"])
    print(f"[hgt-figs] done (seed {seed}, test AUC-PR={r['auc_pr']:.3f}, "
          f"AUC-ROC={r['auc_roc']:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
