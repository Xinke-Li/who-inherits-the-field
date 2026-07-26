"""E5 - Community structure done right (replaces the old Q=0.86 claim).

DATA NOTE: topology claims concern the WHOLE discipline network, so this runs on
the FULL genealogy (Econ_Pairs_With_Citations.parquet, ~17.8k pairs), not the
filtered modeling sample (whose graph is deliberately fragmented by the
window-quality filters - its LCC has only ~23 nodes).

Measured facts this script establishes (verified on the real data):
  - the genealogy is a FOREST: ~21.2k scholars, 17,826 edges; the largest
    lineage component covers 5,953 scholars (~28%); Q(LCC)=0.939 with ~71 communities;
  - BUT the random-genealogy null
    (same students-per-advisor branching, random assignment) already gives
    Q ~ 0.88: high modularity is a property of trees, not evidence of silos.

Honest reporting rule (pre-registered): 'knowledge silo' language is kept only if
observed Q exceeds BOTH nulls by z > 2; otherwise the paper reports (a) extreme
fragmentation as the real structural finding and (b) institutional anchoring via
NMI(community, institution).

Full run (~30-60 min for 100 null draws): python e5_communities.py
Smoke run: python e5_communities.py --draws 5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as CFG

import networkx as nx
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score

N_LOUVAIN_SEEDS = 10
SEED = 0  # frozen for camera-ready reproducibility; DO NOT change (see note below)


def canonical_lcc(G):
    """Largest connected component with a DETERMINISTIC node+adjacency order.

    Reproducibility note: networkx's `connected_components`/`subgraph` derive
    node iteration order from Python set iteration, which depends on
    PYTHONHASHSEED. python-louvain's `best_partition` shuffles `graph.nodes()`
    from that base order, so even with a fixed `random_state` the partition (and
    thus k, NMI, and the configuration-null degree sequence) varies run-to-run.
    Rebuilding the LCC with sorted nodes and sorted adjacency removes that
    dependence, so E5's numbers are byte-reproducible across processes.
    """
    lcc = max(nx.connected_components(G), key=len)
    H = nx.Graph()
    H.add_nodes_from(sorted(lcc))
    H.add_edges_from(sorted(tuple(sorted(e)) for e in G.subgraph(lcc).edges()))
    return H


def louvain_Q(G, seed):
    part = community_louvain.best_partition(G, random_state=seed)
    return community_louvain.modularity(part, G), part


def configuration_null(G, n_draws, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    deg_seq = [d for _, d in G.degree()]
    qs = []
    for i in range(n_draws):
        H = nx.configuration_model(deg_seq, seed=int(rng.integers(1e9)))
        H = nx.Graph(H)
        H.remove_edges_from(nx.selfloop_edges(H))
        qs.append(louvain_Q(H, seed=i)[0])
    return float(np.mean(qs)), float(np.std(qs, ddof=1))


def random_genealogy_null(pairs, n_draws, seed=SEED + 2):
    """Preserve each advisor's number of students; assign students randomly.
    The tree-specific null: same branching process, no sociology."""
    rng = np.random.default_rng(seed)
    sizes = pairs.groupby("advisor_pid").size().values
    students = pairs.student_pid.values
    qs = []
    for i in range(n_draws):
        perm = rng.permutation(students)
        G = nx.Graph()
        k = 0
        advisors = pairs.groupby("advisor_pid").size().index
        for a, size in zip(advisors, sizes):
            for _ in range(size):
                G.add_edge(a, perm[k]); k += 1
        Gl = canonical_lcc(G)
        qs.append(louvain_Q(Gl, seed=i)[0])
    return float(np.mean(qs)), float(np.std(qs, ddof=1))


def main(n_draws=100):
    pairs = pd.read_parquet(CFG.PAIRS_FILE)
    G = nx.Graph()
    G.add_edges_from(zip(pairs.student_pid, pairs.advisor_pid))  # NO prefixes: one person = one node
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    lcc = comps[0]
    Gl = canonical_lcc(G)  # deterministic node/adjacency order (see canonical_lcc)
    frag = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "n_components": len(comps), "lcc_size": len(lcc),
            "lcc_coverage": round(len(lcc) / G.number_of_nodes(), 4),
            "is_forest": G.number_of_edges() < G.number_of_nodes()}
    print(f"[e5] {frag}")

    obs = [louvain_Q(Gl, seed=s) for s in range(N_LOUVAIN_SEEDS)]
    Qs = [q for q, _ in obs]
    Q_mean, Q_std = float(np.mean(Qs)), float(np.std(Qs, ddof=1))
    best_part = obs[int(np.argmax(Qs))][1]
    print(f"[e5] LCC Louvain Q = {Q_mean:.3f}±{Q_std:.3f} "
          f"| {len(set(best_part.values()))} communities")

    cfg_mu, cfg_sd = configuration_null(Gl, n_draws)
    gen_mu, gen_sd = random_genealogy_null(pairs, n_draws)
    z_cfg = (Q_mean - cfg_mu) / cfg_sd if cfg_sd > 0 else float("inf")
    z_gen = (Q_mean - gen_mu) / gen_sd if gen_sd > 0 else float("inf")

    # institutional anchoring within the LCC
    st = pairs[pairs.institution_name.notna()].copy()
    st = st[st.student_pid.isin(best_part)]
    st["comm"] = st.student_pid.map(best_part)
    nmi = float(normalized_mutual_info_score(st.institution_name, st.comm))

    verdict = ("'knowledge silos' SUPPORTED (z>2 vs both nulls)"
               if (z_cfg > 2 and z_gen > 2) else
               "high Q is a tree artifact - report fragmentation + NMI, drop silo language")
    out = {"experiment": "E5_communities", "graph": frag,
           "louvain_Q": {"mean": Q_mean, "std": Q_std,
                         "n_communities_best": len(set(best_part.values()))},
           "null_configuration": {"mean": cfg_mu, "std": cfg_sd, "z": round(z_cfg, 2)},
           "null_random_genealogy": {"mean": gen_mu, "std": gen_sd, "z": round(z_gen, 2)},
           "institution_NMI": round(nmi, 4), "n_null_draws": n_draws,
           "verdict": verdict}
    (CFG.RESULTS_DIR / "e5_communities.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # Figures live in experiments/make_network_figures.py (single source of
    # truth, line-faithful old ITR-Lab style: weighted spanning-tree layout,
    # top-20 hub labels, rgba(200,200,200,0.3) edges). E5 owns statistics only.
    print("[e5] stats saved; render panels via: "
          "python experiments/make_network_figures.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=100)
    main(ap.parse_args().draws)
