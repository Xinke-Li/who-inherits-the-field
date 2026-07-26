"""Publication-grade figures - faithful reproduction of the MACS-40123 visual system.

Reverse-engineered from 1_community_detection_Colab.ipynb:
- dual-graph strategy: viz subgraph via degree-threshold scan targeting 800-1000 nodes;
- layout: spring_layout ON THE MAXIMUM SPANNING TREE (k and iterations adaptive to
  size: <400 -> k=2.0/it=200, <800 -> 1.5/150, <1200 -> 1.2/120, else 0.8/100),
  then ALL edges drawn over the tree layout - this produces the organic look;
- palettes per algorithm: RED (Louvain), GREEN (Girvan-Newman), BLUE (Spectral);
- nodes: size = degree*1.5+8 (min 10), white outline w=1; edges #BBBBBB w=0.5;
- labels: top hubs, up to min(200, 25% of nodes), Arial 10 #333 offset +0.03;
- title: f"{Algo} (k={k}, Q={q:.3f}) - {n} nodes", size 18.
"""
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

RED_PALETTE = ["#E64B35", "#DC0000", "#E18727", "#BC3C29", "#FF7F0E",
               "#F39B7F", "#8C564B", "#D62728", "#FFDB6D", "#B40F20"]
GREEN_PALETTE = ["#004949", "#20854E", "#009292", "#59A14F", "#8CD17D",
                 "#4E79A7", "#B6992D", "#499894", "#86BCB6", "#117733"]
BLUE_PALETTE = ["#3C5488", "#4DBBD5", "#00A087", "#3B9AB2", "#8491B4",
                "#00468B", "#42B540", "#0099B4", "#925E9F", "#5F559B"]
NPG = ["#3C5488", "#E64B35", "#00A087", "#4DBBD5", "#F39B7F",
       "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"]
GRID = "#EEEEEE"

# publication style (2026-07-05 pass): Arial, white bg, outside legends,
# darkred/darkgray semantic palette requested for the camera-ready figures
# seaborn-darkgrid house style (matched to the SEC notebook: sns.set_theme(),
# navy #1F4E79 / darkred #8b0000 / gray #555555, bold #333333 typography)
DARKRED = "#8B0000"
DARKGRAY = "#555555"
LIGHTGRAY = "#8C8C8C"
PALEGRAY = "#C6C6C6"
DARKBLUE = "#1F4E79"
NAVY = "#1F4E79"
INK = "#333333"
PUBGRID = "white"


PLOTBG = "#EAEAF2"   # seaborn darkgrid panel


def _pub(fig, title, w=880, h=540, xtitle=None, ytitle=None, xgrid=True):
    fig.update_layout(
        font=dict(family="Arial", size=13.5, color=INK),
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, family="Arial",
                   color="#222222"), x=0.5, xanchor="center",
                   y=0.985, yanchor="top"),
        plot_bgcolor=PLOTBG, paper_bgcolor="white", width=w, height=h,
        margin=dict(t=66, r=30, b=62, l=76))
    fig.update_xaxes(showline=False, ticks="", showgrid=xgrid,
                     gridcolor="white", gridwidth=1.4, zeroline=False,
                     tickfont=dict(size=12.5, color=INK),
                     title=dict(text=f"<b>{xtitle}</b>",
                                font=dict(size=13.5)) if xtitle else None)
    fig.update_yaxes(showline=False, ticks="", showgrid=True,
                     gridcolor="white", gridwidth=1.4, zeroline=False,
                     tickfont=dict(size=12.5, color=INK),
                     title=dict(text=f"<b>{ytitle}</b>",
                                font=dict(size=13.5)) if ytitle else None)
    return fig


def save(fig, name):
    html = C.FIGURES_DIR / f"{name}.html"
    fig.write_html(html, include_plotlyjs="cdn")
    try:
        w = int(fig.layout.width or 1100)
        h = int(fig.layout.height or 850)
        fig.write_image(C.FIGURES_DIR / f"{name}.png", width=w, height=h, scale=2)
    except Exception:
        pass
    print(f"[fig] {html}")
    return fig


def _clean_layout(title, **kw):
    return dict(title=dict(text=title, font=dict(size=18, color="#2F3B52")),
                showlegend=kw.pop("showlegend", False), hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=70),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="white", paper_bgcolor="white", **kw)


# ---------------- network machinery (old recipe, verbatim behavior) ----------------
def viz_subgraph(G, target=(560, 1000)):
    """Degree-threshold scan for a readable 800-1000 node viz subgraph."""
    import networkx as nx
    Gu = G.to_undirected() if nx.is_directed(G) else G
    deg = dict(Gu.degree())
    lo, hi = target
    best_t, best_nodes = 1, list(Gu.nodes())
    for t in range(1, max(deg.values()) + 1):
        nodes = [n for n, d in deg.items() if d >= t]
        if len(nodes) <= hi:
            best_t, best_nodes = t, nodes
            if len(nodes) >= lo:
                break
    if len(best_nodes) > hi:
        best_nodes = sorted(best_nodes, key=lambda n: deg[n], reverse=True)[:hi]
    Gv = Gu.subgraph(best_nodes).copy()
    print(f"[viz] threshold degree>={best_t}: {Gv.number_of_nodes()} nodes, "
          f"{Gv.number_of_edges()} edges")
    return Gv


def tree_layout(Gv, seed=42):
    """Spring layout computed on the maximum spanning tree (the old secret sauce)."""
    import networkx as nx
    n = Gv.number_of_nodes()
    k, it = (2.0, 200) if n < 400 else (1.5, 150) if n < 800 else \
            (1.2, 120) if n < 1200 else (0.8, 100)
    T = nx.maximum_spanning_tree(Gv)
    return nx.spring_layout(T, k=k, seed=seed, iterations=it)


def community_gradient(communities, colorscale="Reds", lo=0.30, hi=0.95):
    """Map community ids onto a CONTINUOUS colorscale by size rank (largest =
    darkest). This - not a 10-color cycle - is what keeps 30-70 communities
    visually coherent in the old Louvain/GN figures."""
    from collections import Counter
    from plotly.colors import sample_colorscale
    sizes = Counter(communities.values())
    order = [c for c, _ in sizes.most_common()]
    n = max(len(order) - 1, 1)
    vals = [hi - (hi - lo) * i / n for i in range(len(order))]
    cols = sample_colorscale(colorscale, vals)
    return dict(zip(order, cols))


def plot_network_pub(Gv, pos, communities, name_map, title, fname,
                     palette=RED_PALETTE, colorscale=None,
                     top_hub_labels=120, label_min_degree=6):
    """colorscale (e.g. 'Reds', 'YlGnBu'): size-ranked gradient (recommended -
    matches the old figures); palette: legacy discrete cycle fallback."""
    import networkx as nx
    cmap = community_gradient(communities, colorscale) if colorscale else None
    edge_x, edge_y = [], []
    for a, b in Gv.edges():
        if a in pos and b in pos:
            edge_x += [pos[a][0], pos[b][0], None]
            edge_y += [pos[a][1], pos[b][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none",
                            line=dict(width=0.5, color="#BBBBBB"))

    xs, ys, txt, col, size = [], [], [], [], []
    for n in Gv.nodes():
        if n not in pos:
            continue
        x, y = pos[n]
        d = Gv.degree[n]
        cid = communities.get(n, 0)
        xs.append(x); ys.append(y)
        txt.append(f"<b>{name_map.get(n, n)}</b><br>Community: {cid}<br>Degree: {d}")
        col.append(cmap[cid] if cmap else palette[cid % len(palette)])
        size.append(max(d * 1.5 + 8, 10))
    node_trace = go.Scatter(x=xs, y=ys, mode="markers", hoverinfo="text", text=txt,
                            marker=dict(size=size, color=col, line_width=1,
                                        line_color="white"))

    deg_sorted = sorted(Gv.degree, key=lambda x: x[1], reverse=True)
    hubs = [n for n, d in deg_sorted[:top_hub_labels]]
    hubs += [n for n, d in deg_sorted if d >= label_min_degree]
    hubs = list(dict.fromkeys(hubs))
    max_labels = min(200, int(Gv.number_of_nodes() * 0.25))
    hubs = hubs[:max_labels]
    label_trace = go.Scatter(
        x=[pos[h][0] for h in hubs if h in pos],
        y=[pos[h][1] + 0.03 for h in hubs if h in pos],
        mode="text", text=[name_map.get(h, h) for h in hubs if h in pos],
        textposition="top center",
        textfont=dict(size=10, color="#333333", family="Arial"), hoverinfo="skip")

    fig = go.Figure(data=[edge_trace, node_trace, label_trace])
    fig.update_layout(**_clean_layout(title))
    return save(fig, fname)


# ---------------- result-panel figures ----------------
def fig_baselines(summary, fname="F2_e1_baseline_ladder"):
    """E1: AUC-PR bars (darkred = GBDT family, darkgray = linear/prior)."""
    order = ["M0_prior", "M1_logit_overlap", "M2_logit_tabular",
             "M3_gbdt_tabular", "M4_logit_tfidf", "M5_gbdt_nfa"]
    labels = ["Prior", "Logit<br>(overlap only)", "Logit<br>(tabular)",
              "GBDT<br>(tabular)", "Logit<br>(+concept TF-IDF)", "GBDT + NFA"]
    order = [m for m in order if m in summary]
    labels = labels[:len(order)]
    mean = [summary[m]["auc_pr"]["mean"] for m in order]
    std = [summary[m]["auc_pr"]["std"] for m in order]
    bar_colors = [DARKRED if m in ("M3_gbdt_tabular", "M5_gbdt_nfa") else DARKGRAY
                  for m in order]
    fig = go.Figure(go.Bar(
        x=labels, y=mean,
        error_y=dict(type="data", array=std, color=INK, thickness=1.3, width=5),
        marker=dict(color=bar_colors, line=dict(color="white", width=1.5)),
        width=0.62, cliponaxis=False))
    for lab, m, s in zip(labels, mean, std):
        fig.add_annotation(x=lab, y=m + s + 0.012, text=f"<b>{m:.3f}</b>",
                           showarrow=False, yanchor="bottom",
                           font=dict(size=13, family="Arial", color=INK))
    fig.add_hline(y=0.135, line_dash="dash", line_color="#666666", line_width=1.4)
    fig.add_annotation(x=labels[0], y=0.178, text="base rate 0.135",
                       showarrow=False, yanchor="bottom",
                       font=dict(size=12, family="Arial", color="#6E6E6E"))
    _pub(fig, "Baseline ladder — test AUC-PR (temporal split, 10 seeds)",
         w=880, h=540, ytitle="AUC-PR")
    fig.update_yaxes(range=[0, 0.47])
    fig.update_layout(showlegend=False, bargap=0.34)
    return save(fig, fname)

def fig_theta_stability(sweep, fname="F4_e4_theta_stability"):
    """E4: AUC-ROC stability vs base-rate collapse across theta (pub style)."""
    thetas = sorted(sweep, key=float)
    roc = [sweep[t]["auc_roc"]["mean"] for t in thetas]
    pr = [sweep[t]["auc_pr"]["mean"] for t in thetas]
    base = [sweep[t]["base_rate"] for t in thetas]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thetas, y=roc, name="AUC-ROC", mode="lines+markers",
                             line=dict(color=DARKBLUE, width=3.5),
                             marker=dict(size=9, color=DARKBLUE,
                                         line=dict(color="white", width=1.5))))
    fig.add_trace(go.Scatter(x=thetas, y=pr, name="AUC-PR", mode="lines+markers",
                             line=dict(color=DARKRED, width=3.5),
                             marker=dict(size=9, color=DARKRED,
                                         line=dict(color="white", width=1.5))))
    fig.add_trace(go.Scatter(x=thetas, y=base, name="base rate", mode="lines+markers",
                             line=dict(color="#8C8C8C", width=2, dash="dash"),
                             marker=dict(size=6, color="#8C8C8C")))
    _pub(fig, "Label-threshold robustness (GBDT, temporal split, 10 seeds)",
         w=880, h=540, xtitle="Jaccard threshold θ", ytitle="Score", xgrid=True)
    fig.update_yaxes(range=[0, 0.9])
    fig.update_layout(showlegend=True, margin=dict(t=104),
                      legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.012, yanchor="bottom",
                                  font=dict(size=12.5), bgcolor="rgba(0,0,0,0)"))
    return save(fig, fname)

def fig_coefficient_ladder(specs, fname="F6_e6_innovation_premium",
                           title="Innovation premium — divergence coefficient across specifications"):
    """E6: spec curve, publication style. Significant specs darkred, n.s. darkgray;
    95% CI whiskers with caps; beta labels beside markers."""
    names = list(specs)
    coefs = [specs[s]["coef"] for s in names]
    ses = [specs[s]["se"] for s in names]
    pretty = {"1_raw": "raw", "2_controls": "+ controls",
              "3_controls_FE": "+ cohort &<br>institution FE",
              "4_structural_FE": "structural<br>divergence + FE"}
    fig = go.Figure()
    for i, s in enumerate(names):
        sig = specs[s]["p"] < 0.05
        c = DARKRED if sig else DARKGRAY
        fig.add_trace(go.Scatter(
            x=[i], y=[coefs[i]], mode="markers",
            error_y=dict(type="data", array=[1.96 * ses[i]], color=c,
                         thickness=2.4, width=7),
            marker=dict(size=13, color=c, line=dict(color="white", width=1.5)),
            showlegend=False,
            hovertext=f"{s}: {coefs[i]:+.4f} (p={specs[s]['p']:.4f})"))
        fig.add_annotation(x=i + 0.13, y=coefs[i],
                           text=f"<b>{coefs[i]:+.4f}</b><br>p={specs[s]['p']:.3f}",
                           showarrow=False, xanchor="left", align="left",
                           font=dict(size=12, family="Arial", color=c))
    fig.add_hline(y=0, line_dash="dash", line_color="#666666", line_width=1.4)
    _pub(fig, title, w=880, h=500, ytitle="β (per SD divergence)")
    fig.update_xaxes(tickvals=list(range(len(names))),
                     ticktext=[pretty.get(s, s) for s in names],
                     range=[-0.45, len(names) - 0.35], tickfont=dict(size=13))
    fig.update_layout(showlegend=False)
    return save(fig, fname)

def fig_cohort_trends(trends, fname="F8_e8_cohort_trends"):
    """E8 cohort trends (seaborn-darkgrid style)."""
    dec = [t["decade"] for t in trends]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dec, y=[t["n"] for t in trends], name="cohort size",
                         yaxis="y2", marker=dict(color="#C9CFDB",
                         line=dict(color="white", width=1.2)), opacity=0.7))
    err = None
    if all("retention_ci95" in t for t in trends):
        err = dict(type="data", symmetric=False, color=NAVY, thickness=1.4, width=5,
                   array=[t["retention_ci95"][1] - t["retention_rate"] for t in trends],
                   arrayminus=[t["retention_rate"] - t["retention_ci95"][0] for t in trends])
    fig.add_trace(go.Scatter(x=dec, y=[t["retention_rate"] for t in trends],
                             name="field retention", mode="lines+markers", error_y=err,
                             line=dict(color=NAVY, width=3.5),
                             marker=dict(size=10, color=NAVY,
                                         line=dict(color="white", width=2))))
    fig.add_trace(go.Scatter(x=dec, y=[t["coauth_rate"] for t in trends],
                             name="early co-authorship", mode="lines+markers",
                             line=dict(color=DARKRED, width=3.5),
                             marker=dict(size=10, color=DARKRED,
                                         line=dict(color="white", width=2))))
    _pub(fig, "Six decades: retention flat since the 1990s; early co-authorship rose to 0.43",
         w=880, h=540, ytitle="Rate")
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(
        yaxis2=dict(overlaying="y", side="right",
                    title=dict(text="<b>cohort size</b>", font=dict(size=13.5)),
                    showgrid=False, ticks="", tickfont=dict(size=12.5, color=INK)),
        showlegend=True, margin=dict(t=104),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.012,
                    yanchor="bottom", font=dict(size=12.5), bgcolor="rgba(0,0,0,0)"))
    return save(fig, fname)

def fig_robustness(variants, fname="F9_e9_robustness"):
    """E9: horizontal bars, reference in darkred, publication proportions."""
    pretty = {"ref_temporal": "Reference (temporal split)",
              "b_advisor_disjoint": "Advisor-disjoint split",
              "c_drop_capped": "Drop 400-capped students",
              "d_t0_ge_1960": "Drop t0 < 1960",
              "e_no_overlap_feature": "Remove early-overlap feature"}
    names = list(variants)
    mean = [variants[n]["summary"]["auc_pr"]["mean"] for n in names]
    std = [variants[n]["summary"]["auc_pr"]["std"] for n in names]
    colors = [DARKRED if n.startswith("ref") else DARKGRAY for n in names]
    fig = go.Figure(go.Bar(
        y=[pretty.get(n, n) for n in names], x=mean, orientation="h",
        error_x=dict(type="data", array=std, color=INK, thickness=1.2, width=4),
        marker=dict(color=colors, line=dict(color="white", width=1.5)),
        width=0.55, cliponaxis=False))
    for n, m, s in zip(names, mean, std):
        fig.add_annotation(x=m + s + 0.013, y=pretty.get(n, n), text=f"<b>{m:.3f}</b>",
                           showarrow=False, xanchor="left",
                           font=dict(size=13, family="Arial", color=INK))
    fig.add_vline(x=0.135, line_dash="dash", line_color="#666666", line_width=1.4,
                  annotation_text="base rate 0.135",
                  annotation_position="bottom right",
                  annotation_font=dict(size=12, color="#6E6E6E", family="Arial"))
    _pub(fig, "Stress tests — AUC-PR under design perturbations (10 seeds each)",
         w=880, h=460, xtitle="AUC-PR")
    fig.update_xaxes(range=[0, 0.48], showgrid=True, gridcolor="white",
                     gridwidth=1.3)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(showlegend=False, margin=dict(l=225), bargap=0.42)
    return save(fig, fname)

def fig_placebo(global_rows, cohort_rows, fname="F3_e9a_placebo"):
    """E9a placebo distributions (seaborn-darkgrid style: solid boxes with
    white border/median, gray outlier dots)."""
    fig = go.Figure()
    fig.add_trace(go.Box(y=[r["auc_roc"] for r in global_rows], name="global shuffle",
                         fillcolor=NAVY, line=dict(color="white", width=2),
                         marker=dict(color="#777777", size=4, opacity=0.8),
                         boxpoints="outliers", whiskerwidth=0.5))
    fig.add_trace(go.Box(y=[r["auc_roc"] for r in cohort_rows],
                         name="within-cohort shuffle",
                         fillcolor=DARKRED, line=dict(color="white", width=2),
                         marker=dict(color="#777777", size=4, opacity=0.8),
                         boxpoints="outliers", whiskerwidth=0.5))
    fig.add_hline(y=0.5, line_dash="dash", line_color=DARKRED, line_width=2)
    fig.add_annotation(xref="paper", x=0.99, y=0.503, text="<b>chance (0.5)</b>",
                       showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(size=12, family="Arial", color=DARKRED))
    _pub(fig, "Leakage certificate: placebo models vs true test labels (30 seeds each)",
         w=880, h=540, ytitle="Test AUC-ROC")
    fig.update_yaxes(range=[0.36, 0.66])
    fig.update_xaxes(showgrid=False)
    fig.update_layout(showlegend=False)
    return save(fig, fname)

def fig_institution_map(df, fname="F1_institution_map"):
    """Descriptive: PhD-institution countries, old choropleth style (Blues)."""
    counts = df.institution_country.dropna().value_counts()
    fig = go.Figure(go.Choropleth(
        locations=counts.index, locationmode="country names", z=counts.values,
        colorscale="Blues", marker_line_color="#DDDDDD", marker_line_width=0.4,
        colorbar=dict(title="Students")))
    fig.update_layout(
        title=dict(text="<b>Global Distribution of PhD Institutions</b><br>"
                        f"<sup>clean dataset, n={int(counts.sum())} students with "
                        "known country (52.8% coverage)</sup>",
                   font=dict(size=18), x=0.5),
        geo=dict(showframe=False, showcoastlines=False, projection_type="natural earth",
                 bgcolor="#F8F9FA"),
        paper_bgcolor="white", margin=dict(t=90, b=10, l=10, r=10))
    return save(fig, fname)


def fig_advisor_placebo(summary, fname="F10_e10_advisor_placebo"):
    """E10: true advisor (navy) vs placebo variants (gray family)."""
    models = [("L1_logit_overlap", "Logit, overlap only"),
              ("L2_logit_tabular", "Logit, all tabular"),
              ("G3_gbdt_tabular", "GBDT, all tabular")]
    # true advisor = the single darkred focal bar; placebos = gray gradient
    variants = [("true", "true advisor", "#8B0000"),                      # darkred
                ("placebo_cohort", "cohort-matched placebo", "#4A4A4A"),  # dark gray
                ("placebo_random", "random placebo", "#8C8C8C"),          # mid gray
                ("field_mean", "discipline-mean profile", "#C6C6C6")]     # light gray
    fig = go.Figure()
    for v, vlab, color in variants:
        fig.add_trace(go.Bar(
            x=[mlab for _, mlab in models],
            y=[summary[v][m]["auc_pr"]["mean"] for m, _ in models],
            error_y=dict(type="data",
                         array=[summary[v][m]["auc_pr"]["std"] for m, _ in models],
                         color=INK, thickness=1.1, width=3),
            name=vlab, marker=dict(color=color, line=dict(color="white", width=1.5))))
    fig.add_hline(y=0.135, line_dash="dash", line_color="#666666", line_width=1.4)
    fig.add_annotation(xref="paper", x=0.36, y=0.122, text="base rate 0.135",
                       showarrow=False, yanchor="top",
                       font=dict(size=12, family="Arial", color="#6E6E6E"))
    _pub(fig, "Advisor-placebo control (E10)", w=920, h=540, ytitle="AUC-PR")
    fig.update_yaxes(range=[0, 0.45])
    fig.update_layout(barmode="group", bargap=0.30, bargroupgap=0.06,
                      showlegend=True, margin=dict(t=104),
                      legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.012, yanchor="bottom",
                                  font=dict(size=12.5), bgcolor="rgba(0,0,0,0)"))
    return save(fig, fname)


def fig_influence_choropleth(country_rows, fname="F1_pagerank_choropleth"):
    """F1: PageRank-by-country choropleth with continent/ocean anchors and
    top-country callouts (publication style, 2026-07-05)."""
    import pandas as pd
    ci = pd.DataFrame(country_rows)
    # low-value countries were near-white (#F2F4F7) and vanished into the light
    # land fill; deepen the two lightest stops to a visible light gray-blue so
    # every country with data reads clearly against the #F5F5F5 land.
    # pure blue ramp: low end a clearly-visible light blue (distinct from the
    # #F5F5F5 no-data land), deepening to navy
    scale = [[0.0, "#C6DBEF"], [0.25, "#9ECAE1"], [0.50, "#6BAED6"],
             [0.75, "#2171B5"], [1.0, "#0B2265"]]
    fig = go.Figure(go.Choropleth(
        locations=ci["country"], z=ci["total_pagerank"],
        locationmode="country names", colorscale=scale,
        colorbar=dict(title=dict(text="Total<br>influence", font=dict(size=13)),
                      thickness=14, len=0.62, outlinewidth=0,
                      tickfont=dict(size=12)),
        marker_line_color="white", marker_line_width=0.5,
        customdata=ci[["scholar_count", "avg_pagerank"]].values,
        hovertemplate="<b>%{location}</b><br>Total PageRank: %{z:.3f}<br>"
                      "Scholars: %{customdata[0]}<br>"
                      "Avg PageRank: %{customdata[1]:.5f}<extra></extra>"))

    continents = [("NORTH AMERICA", -103, 52), ("SOUTH AMERICA", -59, -14),
                  ("EUROPE", 28, 58), ("AFRICA", 17, 5),
                  ("ASIA", 95, 46), ("OCEANIA", 146, -30)]
    oceans = [("PACIFIC  OCEAN", -152, -8), ("ATLANTIC  OCEAN", -33, -3),
              ("INDIAN  OCEAN", 80, -26)]
    fig.add_trace(go.Scattergeo(
        lon=[c[1] for c in continents], lat=[c[2] for c in continents],
        text=[c[0] for c in continents], mode="text", hoverinfo="skip",
        textfont=dict(family="Arial", size=14, color="rgba(120,120,120,0.65)"),
        showlegend=False))
    fig.add_trace(go.Scattergeo(
        lon=[o[1] for o in oceans], lat=[o[2] for o in oceans],
        text=[f"<i>{o[0]}</i>" for o in oceans], mode="text", hoverinfo="skip",
        textfont=dict(family="Arial", size=13, color="rgba(140,160,180,0.85)"),
        showlegend=False))

    # Alaska is filled like the mainland (same country) - label it to avoid
    # readers mistaking it for a separate dark country; Hawaii is the Pacific dot
    fig.add_trace(go.Scattergeo(
        lon=[-151], lat=[65], text=["Alaska (U.S.)"], mode="text",
        hoverinfo="skip", showlegend=False,
        textfont=dict(family="Arial", size=10, color="white")))

    top = ci.sort_values("total_pagerank", ascending=False).head(5)
    coords = {"United States": (-98, 38.5), "Canada": (-112, 60),
              "Netherlands": (-1, 59), "United Kingdom": (-16, 49),
              "Hong Kong": (119, 17), "Germany": (18, 44),
              "Australia": (134, -24)}
    for _, r in top.iterrows():
        if r["country"] not in coords:
            continue
        x, y = coords[r["country"]]
        if r["country"] == "United States":
            t, c, s = (f"<b>United States<br>{r['total_pagerank']:.3f}</b>",
                       "white", 14)
        else:
            t, c, s = (f"{r['country']}  {r['total_pagerank']:.3f}", "#333333", 11)
        fig.add_trace(go.Scattergeo(lon=[x], lat=[y], text=[t], mode="text",
                                    hoverinfo="skip", showlegend=False,
                                    textfont=dict(family="Arial", size=s, color=c)))

    fig.update_layout(
        title=dict(text="Global distribution of economic influence<br>"
                        "<sup>Aggregated PageRank by country "
                        "(descriptive; full-career weighted network)</sup>",
                   font=dict(size=19, family="Arial", color=INK),
                   x=0.02, xanchor="left"),
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor="#D8D8D8",
                 coastlinewidth=0.6, projection_type="natural earth",
                 landcolor="#F5F5F5", showcountries=True, countrycolor="#D0D0D0",
                 countrywidth=0.4, showocean=True, oceancolor="#F7FAFC",
                 showlakes=False),
        width=1000, height=560, margin=dict(r=10, t=85, l=10, b=10),
        paper_bgcolor="white", font=dict(family="Arial", size=12, color=INK))
    return save(fig, fname)
