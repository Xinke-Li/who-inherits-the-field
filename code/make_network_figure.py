#!/usr/bin/env python3
"""Five advising-genealogy networks, one per discipline, in the ITR-Lab visual
system of the original Network_Figures_Colab notebook: a maximum-spanning-tree
plus raw-k spring layout (calculate_exact_layout), Plotly rendering with light
edges and white-outlined nodes (create_base_plot), Louvain communities colored by
the GREEN_PALETTE. Runs locally (no GPU): only Louvain, no Node2Vec. Produces one
full-size labeled figure per discipline (supplement) and one horizontal five-panel
composite for the paper appendix, plus a provenance JSON of Q, community count, and
subsample size. Descriptive only: no predictive model uses this network.
"""
import json, os
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import community.community_louvain as community_louvain

# ----- verbatim from the original notebook -----
GREEN_PALETTE = ["#004949", "#20854E", "#009292", "#59A14F", "#8CD17D",
                 "#4E79A7", "#B6992D", "#499894", "#86BCB6", "#117733"]
VIZ_MIN_DEGREE, VIZ_TARGET_NODES, VIZ_MAX_NODES = 4, 800, 1000
RANDOM_SEED = 42

DATA = "KDD New dataset 2027/data"
FIGDIR = "KDD New dataset 2027/paper/figures"
FULLDIR = os.path.join(FIGDIR, "networks_full")
PANELS = [("econ", "economics", 0.58), ("math", "mathematics", 0.68),
          ("neuro", "neuroscience", 0.76), ("physics", "physics", 0.79),
          ("chemistry", "chemistry", 0.79)]


def load_graph(field):
    df = pd.read_parquet(f"{DATA}/pairs_resolved_{field}.parquet")

    def full(fn, ln):
        parts = [str(x).strip() for x in (fn, ln)
                 if isinstance(x, str) and str(x).strip() and str(x).lower() != "nan"]
        return " ".join(parts)

    name_map = {}
    for r in df.itertuples():
        for pid, fn, ln in ((r.advisor_pid, r.adv_firstname, r.adv_lastname),
                            (r.student_pid, r.stu_firstname, r.stu_lastname)):
            if pd.notna(pid):
                key = str(pid)
                nm = full(fn, ln)
                if nm and key not in name_map:
                    name_map[key] = nm
    # advising graph; no co-authorship or citation columns exist here, so weight=1
    G = nx.DiGraph()
    for a, s in df[["advisor_pid", "student_pid"]].dropna().itertuples(index=False):
        u, v = str(a), str(s)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v, weight=1.0)
    G_und = G.to_undirected()
    G_main = G_und.subgraph(max(nx.connected_components(G_und), key=len)).copy()
    return G_main, name_map


def select_visualization_subgraph(G_main):   # verbatim
    degrees = dict(G_main.degree())
    degree_values = sorted(set(degrees.values()), reverse=True)
    best_threshold, best_count = VIZ_MIN_DEGREE, 0
    for threshold in range(1, max(degree_values) + 1):
        count = sum(1 for d in degrees.values() if d >= threshold)
        if count <= VIZ_MAX_NODES:
            if count >= VIZ_TARGET_NODES * 0.7:
                best_threshold = threshold; break
            elif count > best_count:
                best_threshold, best_count = threshold, count
    selected = [n for n, d in degrees.items() if d >= best_threshold]
    if len(selected) > VIZ_MAX_NODES:
        selected = sorted(selected, key=lambda n: degrees[n], reverse=True)[:VIZ_MAX_NODES]
    return G_main.subgraph(sorted(selected)).copy()


def bfs_sample(G, start, n):
    """Connected neighborhood of n nodes by breadth-first search from start."""
    seen = {start}; queue = [start]
    while queue and len(seen) < n:
        u = queue.pop(0)
        for v in G.neighbors(u):
            if v not in seen:
                seen.add(v); queue.append(v)
                if len(seen) >= n:
                    break
    return G.subgraph(seen).copy()


def connected_viz(G_main):
    """The degree-threshold selection fragments on pure advising forests (the
    sciences), so keep its connected core when that core is large, and otherwise
    grow a connected BFS neighborhood from the top hub. The spanning-tree layout
    then produces the branchy shape rather than a disc of isolated hubs."""
    G_viz = select_visualization_subgraph(G_main)
    if not nx.is_connected(G_viz):
        G_viz = G_viz.subgraph(max(nx.connected_components(G_viz), key=len)).copy()
    if G_viz.number_of_nodes() < 0.6 * VIZ_TARGET_NODES:
        hub = max(G_main.degree, key=lambda kv: kv[1])[0]
        G_viz = bfs_sample(G_main, hub, VIZ_TARGET_NODES)
    return G_viz


def calculate_exact_layout(G_viz, seed=RANDOM_SEED):   # verbatim recipe
    n = G_viz.number_of_nodes()
    T = nx.maximum_spanning_tree(G_viz)
    if n < 400: k, iters = 2.0, 200
    elif n < 800: k, iters = 1.5, 150
    elif n < 1200: k, iters = 1.2, 120
    else: k, iters = 0.8, 100
    pos = nx.spring_layout(T, k=k, seed=seed, iterations=iters)
    P = np.array(list(pos.values())); r = np.linalg.norm(P - P.mean(0), axis=1)
    disp = float(r.std() / (r.mean() + 1e-12))
    if disp < 0.35:
        pos = nx.spring_layout(T, k=k / np.sqrt(n), seed=seed, iterations=iters)
    return pos


def edge_node_traces(G_viz, pos, part, name_map, node_scale, edge_width, with_hover,
                     node_base, node_min, node_cap):
    ex, ey = [], []
    for a, b in G_viz.edges():
        if a in pos and b in pos:
            ex += [pos[a][0], pos[b][0], None]; ey += [pos[a][1], pos[b][1], None]
    edge = go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="none",
                      line=dict(width=edge_width, color="rgba(150, 150, 150, 0.55)"))
    nodes = list(G_viz.nodes())
    colors = [GREEN_PALETTE[part.get(n, 0) % len(GREEN_PALETTE)] for n in nodes]
    # degree-scaled with a cap: the science super-hubs reach degree 200+, which
    # would swamp a panel; the cap keeps the size hierarchy readable.
    sizes = [min(max(G_viz.degree[n] * node_scale + node_base, node_min), node_cap)
             for n in nodes]
    texts = ([f"<b>{name_map.get(n, n)}</b><br>Community: {part.get(n, 0)}"
              f"<br>Degree: {G_viz.degree[n]}" for n in nodes] if with_hover else None)
    node = go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
                      mode="markers", hoverinfo="text" if with_hover else "none",
                      text=texts, marker=dict(size=sizes, color=colors,
                                              line=dict(width=0.5, color="white")))
    return edge, node


def full_size_figure(field, name, G_viz, pos, part, name_map, Q, k_comm):
    edge, node = edge_node_traces(G_viz, pos, part, name_map, 1.5, 1.0, True, 8, 10, 55)
    hubs = sorted(G_viz.nodes(), key=lambda n: G_viz.degree[n], reverse=True)[:20]
    label = go.Scatter(x=[pos[h][0] for h in hubs], y=[pos[h][1] + 0.04 for h in hubs],
                       mode="text", text=[name_map.get(h, h) for h in hubs],
                       textposition="top center",
                       textfont=dict(size=11, color="#333", family="Arial"),
                       hoverinfo="skip")
    fig = go.Figure(data=[edge, node, label], layout=go.Layout(
        title=dict(text=f"<b>{name}</b>   Louvain Q = {Q:.2f}, k = {k_comm} "
                        f"({G_viz.number_of_nodes()} nodes)", font=dict(size=18, family="Arial")),
        showlegend=False, hovermode="closest", margin=dict(b=20, l=5, r=5, t=70),
        plot_bgcolor="white", width=1050, height=900,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    os.makedirs(FULLDIR, exist_ok=True)
    base = os.path.join(FULLDIR, f"network_{field}")
    try:
        fig.write_image(base + ".pdf", format="pdf")
    except Exception as e:
        print(f"  [{field}] PDF export failed ({e}); PNG only")
    fig.write_image(base + ".png", scale=2)


def main():
    prov = {}
    panel_data = []
    for field, name, coauth in PANELS:
        G_main, name_map = load_graph(field)
        part = community_louvain.best_partition(G_main, random_state=0)
        Q = community_louvain.modularity(part, G_main)
        k_comm = len(set(part.values()))
        G_viz = connected_viz(G_main)
        pos = calculate_exact_layout(G_viz)
        prov[field] = {"largest_cc_nodes": G_main.number_of_nodes(),
                       "largest_cc_edges": G_main.number_of_edges(),
                       "louvain_Q": round(float(Q), 3), "n_communities": int(k_comm),
                       "viz_nodes": G_viz.number_of_nodes()}
        print(f"[{field}] CC {G_main.number_of_nodes()} | Q {Q:.3f} k {k_comm} | viz {G_viz.number_of_nodes()}")
        full_size_figure(field, name, G_viz, pos, part, name_map, Q, k_comm)
        panel_data.append((name, G_viz, pos, part, name_map, Q, k_comm))

    # composite: two rows (three panels above, two below, bottom row centered),
    # each panel roughly three times the area of the old one-row layout, so the
    # community colors and hub structure stay readable at print size. Same data,
    # same layouts, same palette; only the paste-up changes.
    titles = [f"{nm}<br>Q = {Q:.2f}, k = {k}" for nm, _, _, _, _, Q, k in panel_data]
    comp = make_subplots(rows=2, cols=3, subplot_titles=titles + [""],
                         horizontal_spacing=0.02, vertical_spacing=0.09)
    cells = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    for (r, c), (nm, G_viz, pos, part, name_map, Q, k) in zip(cells, panel_data):
        edge, node = edge_node_traces(G_viz, pos, part, name_map, 0.9, 0.8, False, 5, 6, 26)
        comp.add_trace(edge, row=r, col=c)
        comp.add_trace(node, row=r, col=c)
    # center the bottom row: move panels 4 and 5 (axes 4 and 5) inward
    comp.update_layout(xaxis4=dict(domain=[0.17, 0.49]),
                       xaxis5=dict(domain=[0.51, 0.83]))
    for ann, cx in zip(comp.layout.annotations[3:5], (0.33, 0.67)):
        ann.update(x=cx)
    comp.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    comp.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    comp.update_annotations(font=dict(size=16, family="Arial"))
    comp.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                       width=1750, height=1150, margin=dict(b=10, l=5, r=5, t=55))
    base = os.path.join(FIGDIR, "F12_five_discipline_networks")
    try:
        comp.write_image(base + ".pdf", format="pdf")
        print(f"wrote {base}.pdf (vector)")
    except Exception as e:
        print(f"composite PDF failed ({e}); high-DPI PNG only")
    comp.write_image(base + ".png", scale=2)
    json.dump(prov, open(os.path.join(DATA, "network_modularity.json"), "w"), indent=2)
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
