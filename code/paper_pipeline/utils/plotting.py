"""House plotting style — faithful to the IC2S2 / MACS-40123 notebooks.

Style DNA (extracted from 1_community_detection_Colab.ipynb / Node2Vec_Colab.ipynb):
- NPG palette (#3C5488 navy, #E64B35 red, #00A087 teal, #4DBBD5 sky, ...)
- networks: spring_layout with fixed seed, white background, thin light-gray edges
  (#BBBBBB, width 0.5), node size ~ degree*1.5+8, white node outline (width 1),
  hub labels in Arial 10 #333333 offset above the node, hidden axes, title size 18.

Every figure is written as both interactive HTML and static PNG (if kaleido present),
into config.FIGURES_DIR.
"""
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

# defensive defaults (house style) in case config is partially loaded
EDGE_COLOR = getattr(C, "EDGE_COLOR", "#BBBBBB")
EDGE_WIDTH = getattr(C, "EDGE_WIDTH", 0.5)
LABEL_FONT = getattr(C, "LABEL_FONT", dict(size=10, color="#333333", family="Arial"))
TITLE_FONT_SIZE = getattr(C, "TITLE_FONT_SIZE", 18)


def _save(fig, name: str):
    html = C.FIGURES_DIR / f"{name}.html"
    fig.write_html(html, include_plotlyjs="cdn")
    try:
        w = int(fig.layout.width or 1100)
        h = int(fig.layout.height or 800)
        fig.write_image(C.FIGURES_DIR / f"{name}.png", width=w, height=h, scale=2)
    except Exception:
        pass  # kaleido not installed - HTML is authoritative
    print(f"[fig] {html}")
    return fig


def base_layout(title: str, **kw):
    return go.Layout(
        title=dict(text=title, font=dict(size=TITLE_FONT_SIZE)),
        showlegend=kw.pop("showlegend", False), hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white", paper_bgcolor="white", **kw)


def plot_network(G, communities: dict, name_map: dict, title: str, fname: str,
                 palette=None, layout_seed=42, layout_k=None, layout_iter=60,
                 n_labels=60, label_min_degree=5):
    """Interactive community-colored network in the house style.

    G: networkx graph (undirected or directed); communities: node -> community id;
    name_map: node -> display name. Labels go to the n_labels highest-degree nodes
    with degree >= label_min_degree.
    """
    import networkx as nx
    palette = palette or C.PALETTE
    Gu = G.to_undirected() if nx.is_directed(G) else G
    pos = nx.spring_layout(Gu, k=layout_k, seed=layout_seed, iterations=layout_iter)

    edge_x, edge_y = [], []
    for a, b in Gu.edges():
        if a in pos and b in pos:
            edge_x += [pos[a][0], pos[b][0], None]
            edge_y += [pos[a][1], pos[b][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none",
                            line=dict(width=EDGE_WIDTH, color=EDGE_COLOR))

    nx_, ny_, txt, col, size = [], [], [], [], []
    for n in Gu.nodes():
        x, y = pos[n]
        d = Gu.degree[n]
        cid = communities.get(n, 0)
        nx_.append(x); ny_.append(y)
        txt.append(f"<b>{name_map.get(n, n)}</b><br>Community: {cid}<br>Degree: {d}")
        col.append(palette[cid % len(palette)])
        size.append(max(d * 1.5 + 8, 10))
    node_trace = go.Scatter(x=nx_, y=ny_, mode="markers", hoverinfo="text", text=txt,
                            marker=dict(size=size, color=col, line_width=1,
                                        line_color="white"))

    hubs = sorted(Gu.nodes(), key=lambda n: Gu.degree[n], reverse=True)
    hubs = [h for h in hubs if Gu.degree[h] >= label_min_degree][:n_labels]
    label_trace = go.Scatter(
        x=[pos[h][0] for h in hubs], y=[pos[h][1] + 0.03 for h in hubs],
        mode="text", text=[name_map.get(h, h) for h in hubs],
        textposition="top center", textfont=dict(**LABEL_FONT), hoverinfo="skip")

    fig = go.Figure(data=[edge_trace, node_trace, label_trace],
                    layout=base_layout(title))
    return _save(fig, fname)


def plot_theta_stability(theta_grid, results_by_model: dict, fname="e4_theta_stability"):
    """E4: primary metric across the label-threshold sweep (publication style:
    darkblue thick line, Arial, clean axes)."""
    LINE_COLORS = ["#1F4E79", "#8B0000", "#555555"]
    fig = go.Figure()
    for i, (model, vals) in enumerate(results_by_model.items()):
        c = LINE_COLORS[i % len(LINE_COLORS)]
        fig.add_trace(go.Scatter(x=list(theta_grid), y=vals, mode="lines+markers",
                                 name=model, line=dict(color=c, width=4),
                                 marker=dict(size=9, color=c,
                                             line=dict(color="white", width=1.5))))
    fig.update_layout(
        font=dict(family="Arial", size=14, color="#1A1A1A"),
        title=dict(text="<b>Label-threshold robustness (AUC-PR)</b>",
                   font=dict(size=16, family="Arial", color="#222222"),
                   x=0.5, xanchor="center"),
        plot_bgcolor="#EAEAF2", paper_bgcolor="white", width=880, height=540,
        margin=dict(t=72, r=30, b=62, l=72),
        showlegend=True,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.09,
                    yanchor="bottom", font=dict(size=13),
                    bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(title=dict(text="Jaccard threshold θ", font=dict(size=15)),
                     showline=False, ticks="",
                     showgrid=True, gridcolor="white", gridwidth=1.3,
                     zeroline=False)
    fig.update_yaxes(title=dict(text="AUC-PR", font=dict(size=15)),
                     showline=False, ticks="",
                     showgrid=True, gridcolor="white", gridwidth=1.3,
                     zeroline=False, rangemode="tozero")
    return _save(fig, fname)

def plot_ablation_forest(labels, mean_diffs, ci_los, ci_his, pvals_adj,
                         fname="e3_ablation_forest",
                         title="Edge-type ablations: Δ AUC-PR vs full model (95% CI)"):
    """E3: forest plot (publication style); significant effects in darkred,
    p_adj printed in the right margin, pretty edge-type names."""
    pretty = {"student_concept": "student–concept", "advisor_concept": "advisor–concept",
              "coauth": "co-authorship", "institution": "institution",
              "advising": "advising", "social": "social (all)"}
    order = sorted(range(len(labels)), key=lambda i: mean_diffs[i])
    fig = go.Figure()
    for row, i in enumerate(order):
        # color by significance tier so rows are visually distinguishable:
        # p_adj<=.05 darkred | <=.10 suggestive darkyellow | else slate gray
        c = ("#8B0000" if pvals_adj[i] <= 0.05
             else "#1F4E79" if pvals_adj[i] <= 0.10 else "#8C8C8C")
        fig.add_trace(go.Scatter(
            x=[mean_diffs[i]], y=[row], mode="markers",
            error_x=dict(type="data", symmetric=False,
                         array=[ci_his[i] - mean_diffs[i]],
                         arrayminus=[mean_diffs[i] - ci_los[i]],
                         color=c, thickness=2.2, width=5),
            marker=dict(size=11, color=c, line=dict(color="white", width=1.2)),
            showlegend=False,
            hovertext=f"{labels[i]}: {mean_diffs[i]:+.4f} (p_adj={pvals_adj[i]:.3f})"))
        fig.add_annotation(x=1.0, xref="paper", y=row, xanchor="left",
                           text=f"p={pvals_adj[i]:.2f}", showarrow=False,
                           font=dict(size=12, family="Arial", color="#6E6E6E"))
    fig.add_vline(x=0, line_dash="dash", line_color="#1A1A1A", line_width=1.3)
    fig.update_layout(
        font=dict(family="Arial", size=14, color="#1A1A1A"),
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, family="Arial",
                   color="#222222"), x=0.5, xanchor="center"),
        plot_bgcolor="#EAEAF2", paper_bgcolor="white", width=880, height=440,
        margin=dict(t=72, r=80, b=62, l=170), showlegend=False)
    fig.update_yaxes(tickvals=list(range(len(order))),
                     ticktext=[pretty.get(labels[i], labels[i]) for i in order],
                     showline=False, showgrid=False, zeroline=False,
                     tickfont=dict(size=14))
    fig.update_xaxes(title=dict(text="Δ AUC-PR (ablated − full)", font=dict(size=15)),
                     showline=False, ticks="",
                     showgrid=True, gridcolor="white", gridwidth=1.3,
                     zeroline=False)
    return _save(fig, fname)

def plot_calibration_by_quintile(df, fname="e1_overlap_quintiles",
                                 title="Field retention by early-overlap quintile"):
    """The signal-at-a-glance bar chart for the results section."""
    import pandas as pd
    q = pd.qcut(df.early_overlap, 5, duplicates="drop")
    g = df.groupby(q, observed=True).y.agg(["mean", "count"])
    labels = [f"Q{i+1}" for i in range(len(g))]
    fig = go.Figure(go.Bar(x=labels, y=g["mean"],
                           marker=dict(color="#1F4E79", line=dict(color="white", width=1.5)),
                           text=[f"<b>{v:.1%}</b><br>(n={n})" for v, n in zip(g["mean"], g["count"])],
                           textposition="outside",
                           textfont=dict(size=12, color="#333333"), cliponaxis=False))
    fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center",
                                 font=dict(size=16, color="#222222")),
                      yaxis=dict(title="<b>P(y=1)</b>", tickformat=".0%", showgrid=True,
                                 gridcolor="white", gridwidth=1.4, ticks="",
                                 showline=False, zeroline=False),
                      xaxis=dict(title="<b>Early advisor-overlap quintile</b>",
                                 showgrid=False, ticks="", showline=False),
                      font=dict(family="Arial", size=13, color="#333333"),
                      plot_bgcolor="#EAEAF2", paper_bgcolor="white",
                      width=880, height=540, margin=dict(t=66))
    return _save(fig, fname)
