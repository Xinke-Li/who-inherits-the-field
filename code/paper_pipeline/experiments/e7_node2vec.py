"""E7 - Node2Vec structural divergence on the PRE-WINDOW graph, multi-seed.

Old flaws: embeddings trained on the full-career graph (label-window edges included),
single run (no embedding uncertainty), raw correlation only.

New design:
  graph    : student-advisor advising edges + coauth_early edges + student-institution
             (all pre-window; NO concept edges, so structural divergence is not
             mechanically coupled to the concept-based label)
  embed    : node2vec via gensim word2vec on random walks, 10 seeds
  measure  : n2v_divergence = 1 - cosine(student, advisor), averaged over seeds;
             per-seed spread reported (a result is only kept if |rho| > 2*seed-std)
  export   : results_econ/e7_divergence.parquet -> consumed by e6 regression ladder

Requires gensim (pip install gensim). Runs on CPU in ~10 min.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D

WALK_LEN, N_WALKS, DIM, WINDOW, EPOCHS_W2V = 40, 10, 64, 5, 5


def build_edges(df):
    edges = []
    for r in df.itertuples():
        s, a = f"s{r.student_pid}", f"a{r.advisor_pid}"
        edges.append((s, a))                      # advising (predetermined)
        if r.coauth_early:
            edges.append((s, a))                  # early co-authorship (duplicates = weight)
        if pd.notna(r.institution_name):
            edges.append((s, f"i{r.institution_name}"))
    return edges


def random_walks(adj, rng):
    nodes = list(adj)
    walks = []
    for _ in range(N_WALKS):
        rng.shuffle(nodes)
        for start in nodes:
            walk, cur = [start], start
            for _ in range(WALK_LEN - 1):
                nbrs = adj[cur]
                if not nbrs:
                    break
                cur = nbrs[rng.integers(len(nbrs))]
                walk.append(cur)
            walks.append(walk)
    return walks


def embed_seed(adj, seed):
    from gensim.models import Word2Vec
    rng = np.random.default_rng(seed)
    walks = random_walks(adj, rng)
    model = Word2Vec(walks, vector_size=DIM, window=WINDOW, min_count=1,
                     sg=1, workers=4, seed=seed, epochs=EPOCHS_W2V)
    return model.wv


def main():
    df = D.load_dataset()
    adj = {}
    for u, v in build_edges(df):
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    print(f"[e7] graph: {len(adj)} nodes")

    div_by_seed = []
    for seed in C.SEEDS:
        wv = embed_seed(adj, seed)
        div = []
        for r in df.itertuples():
            s, a = f"s{r.student_pid}", f"a{r.advisor_pid}"
            if s in wv and a in wv:
                cos = np.dot(wv[s], wv[a]) / (np.linalg.norm(wv[s]) * np.linalg.norm(wv[a]))
                div.append(1.0 - float(cos))
            else:
                div.append(np.nan)
        div_by_seed.append(div)
        print(f"[e7] seed {seed} embedded")

    M = np.array(div_by_seed)                     # seeds x students
    out_df = pd.DataFrame({"student_pid": df.student_pid,
                           "n2v_divergence": np.nanmean(M, axis=0),
                           "n2v_divergence_seedstd": np.nanstd(M, axis=0, ddof=1)})
    out_df.to_parquet(C.RESULTS_DIR / "e7_divergence.parquet")

    # per-seed Spearman with the two outcomes (uncertainty the old abstract lacked)
    from scipy.stats import spearmanr
    oc = pd.read_parquet(C.OUTCOMES)
    merged = df.merge(oc, on=["student_pid", "st_openalex_id"])
    rhos_h, rhos_pct = [], []
    for i in range(len(C.SEEDS)):
        d = pd.Series(M[i], index=df.student_pid).reindex(merged.student_pid).values
        ok = ~np.isnan(d)
        rhos_h.append(float(spearmanr(d[ok], merged.h_index_full_career.values[ok]).statistic))
        m2 = ok & merged.late_cite_pct_mean.notna().values
        rhos_pct.append(float(spearmanr(d[m2], merged.late_cite_pct_mean.values[m2]).statistic))

    def summ(r):
        return {"mean": round(float(np.mean(r)), 4), "std": round(float(np.std(r, ddof=1)), 4),
                "stable": bool(abs(np.mean(r)) > 2 * np.std(r, ddof=1))}

    out = {"experiment": "E7_node2vec",
           "spearman_vs_h_full": summ(rhos_h),
           "spearman_vs_late_cite_pct": summ(rhos_pct),
           "note": "divergence exported to e7_divergence.parquet; causal-adjacent claims "
                   "come from the e6 FE regression, not from these raw correlations"}
    (C.RESULTS_DIR / "e7_node2vec.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
