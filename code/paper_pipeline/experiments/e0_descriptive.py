"""E0 - Descriptive influence topology (faithful port of ITR Lab 2 / PageRank Colab).

DESCRIPTIVE ONLY. This stage characterizes the discipline's influence topology on
the FULL genealogy with full-career edge weights (co-authorship / citation flags).
That is legitimate for describing the network as of today, and matches the original
IC2S2 Figure 3. Nothing computed here may enter the prediction models (E1/E2): the
edge weights use whole-career information, which is exactly what the leakage
protocol bans from features. The paper says this explicitly in the Fig-1 caption.

Faithful to the old notebook:
  - weighted DiGraph: advisor->student 1/n_students(advisor); co-authorship 0.5 in
    both directions; student-cites-advisor 0.3; advisor-cites-student 0.3
  - PageRank alpha=0.85, weight='weight'
  - geography from the pairs file's institution_city/country/latitude/longitude
    (~11.7k scholars), with the same country-name standardization map
  - choropleth: the custom blue-grey scale, natural-earth projection, Arial Black
    title, #FAFAFA paper - the exact old look

Outputs:
  figures_econ/F1_pagerank_choropleth.html   (replaces the useless student-count map)
  figures_econ/F1b_top50_pagerank.html       (horizontal Reds bar, old VIZ-2 style)
  results_econ/e0_descriptive.json           (top-50 table, country table, graph stats)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as CFG

COUNTRY_MAP = {
    'U.S.A.': 'United States', 'USA': 'United States', 'US': 'United States',
    'United States of America': 'United States', 'U.S.': 'United States',
    'UK': 'United Kingdom', 'U.K.': 'United Kingdom', 'Great Britain': 'United Kingdom',
    'England': 'United Kingdom', 'Scotland': 'United Kingdom', 'Wales': 'United Kingdom',
    "People's Republic of China": 'China', 'PRC': 'China',
    'Korea': 'South Korea', 'Republic of Korea': 'South Korea',
    'The Netherlands': 'Netherlands', 'Brasil': 'Brazil',
}
BLUE_GREY_SCALE = [
    [0.0, 'rgb(207, 216, 220)'], [0.2, 'rgb(179, 229, 252)'],
    [0.4, 'rgb(100, 181, 246)'], [0.6, 'rgb(33, 150, 243)'],
    [0.8, 'rgb(21, 101, 192)'], [1.0, 'rgb(13, 71, 161)'],
]


def standardize_country(c):
    if pd.isna(c):
        return None
    c = str(c).strip()
    if c == '' or c.upper() in ('NULL', 'NONE'):
        return None
    return COUNTRY_MAP.get(c, c)


def build_weighted_graph(df):
    G = nx.DiGraph()
    def add(src, dst, w):
        if pd.isna(src) or pd.isna(dst) or src == dst:
            return
        if G.has_edge(src, dst):
            G[src][dst]['weight'] += w
        else:
            G.add_edge(src, dst, weight=w)

    gen = df[['advisor_pid', 'student_pid']].drop_duplicates()
    n_students = gen.groupby('advisor_pid').size().to_dict()
    for r in gen.itertuples():
        add(r.advisor_pid, r.student_pid, 1.0 / n_students.get(r.advisor_pid, 1))
    if 'co_authored' in df.columns:
        for r in df[df.co_authored == True][['advisor_pid', 'student_pid']].drop_duplicates().itertuples():
            add(r.advisor_pid, r.student_pid, 0.5)
            add(r.student_pid, r.advisor_pid, 0.5)
    if 'st_cites_adv_broad' in df.columns:
        for r in df[df.st_cites_adv_broad == True][['student_pid', 'advisor_pid']].drop_duplicates().itertuples():
            add(r.student_pid, r.advisor_pid, 0.3)
    if 'adv_cites_st_broad' in df.columns:
        for r in df[df.adv_cites_st_broad == True][['advisor_pid', 'student_pid']].drop_duplicates().itertuples():
            add(r.advisor_pid, r.student_pid, 0.3)
    return G


def main():
    df = pd.read_parquet(CFG.PAIRS_FILE)

    names = {}
    for r in df.itertuples():
        if pd.notna(getattr(r, 'adv_name', None)):
            names[r.advisor_pid] = r.adv_name
        if pd.notna(getattr(r, 'st_name', None)):
            names[r.student_pid] = r.st_name

    G = build_weighted_graph(df)
    print(f"[e0] weighted graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    pr = nx.pagerank(G, alpha=0.85, max_iter=100, weight='weight')

    top50 = sorted(pr.items(), key=lambda kv: kv[1], reverse=True)[:50]
    top50 = [{"pid": str(p), "name": names.get(p, str(p)), "pagerank": round(s, 6)}
             for p, s in top50]

    # ---- geography (pairs-file columns, as in the old notebook) ----
    geo_cols = ['institution_city', 'institution_country',
                'institution_latitude', 'institution_longitude']
    have_geo = all(c in df.columns for c in geo_cols)
    country_rows = []
    if have_geo:
        gs = df[['student_pid'] + geo_cols].rename(columns={'student_pid': 'id'})
        ga = df[['advisor_pid'] + geo_cols].rename(columns={'advisor_pid': 'id'})
        geo = pd.concat([gs, ga], ignore_index=True)
        geo['country'] = geo['institution_country'].apply(standardize_country)
        geo = geo[geo.country.notna()].drop_duplicates(subset=['id'])
        geo['pagerank'] = geo['id'].map(pr)
        geo = geo[geo.pagerank.notna()]
        print(f"[e0] scholars with valid geography: {len(geo):,}")

        ci = geo.groupby('country').agg(
            total_pagerank=('pagerank', 'sum'),
            avg_pagerank=('pagerank', 'mean'),
            scholar_count=('pagerank', 'size')).reset_index()
        ci = ci.sort_values('total_pagerank', ascending=False)
        country_rows = ci.round(5).to_dict('records')

        from utils import pubfigs as PF
        PF.fig_influence_choropleth(ci.to_dict("records"))
        print(f"[fig] {CFG.FIGURES_DIR / 'F1_pagerank_choropleth.html'}")
    else:
        print("[e0] WARNING: pairs file lacks geography columns - choropleth skipped")

    # ---- top-50 bar, old VIZ-2 style ----
    t = pd.DataFrame(top50)[::-1]
    fig2 = go.Figure(go.Bar(
        y=t['name'], x=t['pagerank'], orientation='h',
        marker=dict(color=t['pagerank'],
                    colorscale=[[0, "#9FB6CD"], [0.5, "#1F4E79"], [1, "#0C2B4A"]],
                    showscale=True,
                    colorbar=dict(title="PageRank", thickness=12, len=0.7),
                    line=dict(color="white", width=1)),
        text=t['pagerank'].round(4), textposition='outside',
        textfont=dict(size=9, color="#333333"),
        hovertemplate='<b>%{y}</b><br>PageRank: %{x:.4f}<extra></extra>'))
    fig2.update_layout(
        title=dict(text="<b>Top 50 Economists by PageRank</b><br><sub>Weighted "
                        "network analysis (descriptive)</sub>",
                   x=0.5, xanchor="center", font=dict(size=16, color="#222222")),
        xaxis=dict(title="<b>PageRank Score</b>", showgrid=True, gridcolor="white",
                   gridwidth=1.4, ticks="", showline=False, zeroline=False),
        yaxis=dict(showgrid=False, ticks="", showline=False),
        height=750, width=750,
        paper_bgcolor="white", plot_bgcolor="#EAEAF2",
        margin=dict(l=180, r=80, t=90, b=60),
        font=dict(family="Arial", size=10, color="#333333"))
    fig2.write_html(CFG.FIGURES_DIR / "F1b_top50_pagerank.html",
                    include_plotlyjs="cdn")
    try:
        fig2.write_image(CFG.FIGURES_DIR / "F1b_top50_pagerank.png",
                         width=1100, height=850, scale=2)
    except Exception:
        pass  # kaleido not installed - HTML is authoritative
    print(f"[fig] {CFG.FIGURES_DIR / 'F1b_top50_pagerank.html'}")

    out = {"experiment": "E0_descriptive",
           "note": "descriptive topology only - full-career weighted edges; "
                   "never used as model features (leakage protocol)",
           "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()},
           "pagerank_alpha": 0.85, "top50": top50,
           "countries": country_rows[:25]}
    (CFG.RESULTS_DIR / "e0_descriptive.json").write_text(json.dumps(out, indent=2))
    print("[e0] done")


if __name__ == "__main__":
    main()
