"""E14d - M4b: modern text embeddings (SPECTER2) vs the tabular ceiling (P1-1/W2).

Consumes m4b_embeddings_<disc>.npz produced by M4b_SPECTER2_Colab.ipynb
(SPECTER2 base + proximity adapter, mean-pooled over each student's pre-window
works; the notebook re-asserts year <= t0+5 per work before encoding). This
script never re-derives text; it only aligns the embedding matrix to the frozen
table by st_openalex_id and evaluates under the standard protocol
(temporal split, thresholds on VAL only, AUC-PR primary).

Models:
  M4b_logit      logistic on the 768-d embedding alone (deterministic;
                 mirrors M4's protocol with a modern representation)
  M4b_gbdt       HistGradientBoosting on the embedding alone (10 seeds)
  M4b_gbdt_tab   HistGradientBoosting on embedding + tabular features
                 (10 seeds; the strongest text-augmented rung)

Tests (P0-3 conventions): stochastic-vs-deterministic comparisons are
one-sample Wilcoxon signed-rank against the deterministic value (M2); the
M4b_gbdt* vs M3 comparisons are paired per-seed; BH across the family.

PRE-REGISTERED DECISION RULE (written before the embeddings existed):
  Let best = max seed-mean AUC-PR over the three M4b rungs and
      ceiling = max(M2, M3) seed-mean AUC-PR from the frozen E1 json.
  - best <= ceiling (no significant positive difference) -> the tabular
    ceiling holds for modern text representations too; the second finding's
    wording upgrades from "TF-IDF" to "text representations including
    SPECTER2".
  - best > ceiling with p_adj < 0.05 and a positive bootstrap CI -> a NEW
    positive finding; report it as such and reword the second finding.

Output: results_econ/e14d_m4b.json (or results_neuro/ under NEURO_DATASET).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S

from scipy import stats as sps
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import os
SUFFIX = "neuro" if os.environ.get("NEURO_DATASET") else "econ"
NPZ = C.RESULTS_DIR / f"m4b_embeddings_{SUFFIX}.npz"


def main():
    assert NPZ.exists(), (f"{NPZ} missing - run M4b_SPECTER2_Colab.ipynb and "
                          "place its output there first")
    df = D.load_dataset()
    df_split = D.temporal_split(df)

    z = np.load(NPZ, allow_pickle=False)
    aids = z["st_openalex_id"].astype(str)
    emb = z["emb"].astype(np.float32)
    idx = {a: i for i, a in enumerate(aids)}
    missing = [a for a in df_split.st_openalex_id if a not in idx]
    assert not missing, f"{len(missing)} students missing from the embedding file"
    E = emb[[idx[a] for a in df_split.st_openalex_id]]
    print(f"[e14d] embeddings aligned: {E.shape}")

    Xt, _ = D.build_features(df_split, concepts="none")
    masks = {k: (df_split.split == k).values for k in ("train", "val", "test")}
    y = {k: df_split.loc[m, "y"].values for k, m in masks.items()}

    def eval_model(m, X):
        """-> (metrics, test scores). Scores are kept for the seed-0 bootstrap
        gate of the pre-registered rule, which needs score-level draws and not
        just the seed-mean metrics."""
        m.fit(X[masks["train"]], y["train"])
        thr = S.best_f1_threshold(y["val"], m.predict_proba(X[masks["val"]])[:, 1])
        s_test = m.predict_proba(X[masks["test"]])[:, 1]
        return S.evaluate(y["test"], s_test, thr), s_test

    per_model, seed0_scores = {}, {}
    # deterministic logistic on the embedding (M4 protocol: scaled, C=0.5)
    r, s = eval_model(make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000, C=0.5,
                                                       random_state=0)), E)
    per_model["M4b_logit"] = [r]
    seed0_scores["M4b_logit"] = s
    print(f"[e14d] M4b_logit: PR={r['auc_pr']:.4f}")

    X_embtab = np.hstack([Xt, E]).astype(np.float32)
    for name, X in (("M4b_gbdt", E), ("M4b_gbdt_tab", X_embtab)):
        runs = []
        for seed in C.SEEDS:
            m = HistGradientBoostingClassifier(random_state=seed, max_iter=500,
                                               early_stopping=True,
                                               validation_fraction=0.15)
            rr, ss = eval_model(m, X)
            runs.append(rr)
            if seed == C.SEEDS[0]:
                seed0_scores[name] = ss
        per_model[name] = runs
        print(f"[e14d] {name}: PR={np.mean([r['auc_pr'] for r in runs]):.4f}")

    # M3 recomputed in-script for the paired comparison (same protocol as E1)
    m3_runs = []
    for seed in C.SEEDS:
        m = HistGradientBoostingClassifier(random_state=seed, max_iter=500,
                                           early_stopping=True,
                                           validation_fraction=0.15)
        rr, ss = eval_model(m, Xt)
        m3_runs.append(rr)
        if seed == C.SEEDS[0]:
            seed0_scores["M3_ref"] = ss
    per_model["M3_ref"] = m3_runs

    summary = {k: S.summarize_seeds(v) for k, v in per_model.items()}

    e1 = json.loads((C.RESULTS_DIR / "e1_baselines.json").read_text())
    m2_pr = e1["summary"]["M2_logit_tabular"]["auc_pr"]["mean"]
    m3_frozen = e1["summary"]["M3_gbdt_tabular"]["auc_pr"]["mean"]
    dev = abs(summary["M3_ref"]["auc_pr"]["mean"] - m3_frozen)
    assert dev < 5e-3, f"M3 reference drifted from frozen E1 ({dev})"
    ceiling = max(m2_pr, m3_frozen)

    comparisons, pvals = [], []
    for name in ("M4b_logit", "M4b_gbdt", "M4b_gbdt_tab"):
        prs = [r["auc_pr"] for r in per_model[name]]
        # vs deterministic M2: one-sample signed-rank (skip if model deterministic)
        if np.std(prs) > 0:
            w = sps.wilcoxon(np.array(prs) - m2_pr)
            comparisons.append({"model": name, "against": "M2(det)",
                                "delta": float(np.mean(prs) - m2_pr),
                                "test": "one-sample signed-rank", "p_raw": float(w.pvalue)})
            pvals.append(float(w.pvalue))
        # vs M3: paired per-seed where stochastic
        if np.std(prs) > 0:
            t = S.paired_wilcoxon(prs, [r["auc_pr"] for r in m3_runs])
            comparisons.append({"model": name, "against": "M3(paired)",
                                "delta": t["mean_diff"], "test": "paired signed-rank",
                                "p_raw": t["p"]})
            pvals.append(t["p"])
    if pvals:
        p_adj, rej = S.bh_correction(pvals)
        for c, pa, rj in zip(comparisons, p_adj, rej):
            c["p_adj"] = round(float(pa), 6)
            c["significant"] = bool(rj)

    best_name = max(("M4b_logit", "M4b_gbdt", "M4b_gbdt_tab"),
                    key=lambda n: summary[n]["auc_pr"]["mean"])
    best = summary[best_name]["auc_pr"]["mean"]

    # Second gate of the pre-registered rule: a positive seed-0 score-level
    # bootstrap CI of (best - M3), 2000 draws, seed 0. Mirrors E14a's
    # paired_bootstrap_delta and E12's HGT-M5 bootstrap.
    boot = None
    if best_name in seed0_scores and "M3_ref" in seed0_scores:
        yt = np.asarray(y["test"])
        s_alt, s_ref = seed0_scores[best_name], seed0_scores["M3_ref"]
        rng = np.random.default_rng(0)
        n, deltas = len(yt), []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            if len(np.unique(yt[idx])) < 2:
                continue
            deltas.append(average_precision_score(yt[idx], s_alt[idx])
                          - average_precision_score(yt[idx], s_ref[idx]))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        boot = {"model": best_name, "against": "M3_ref (seed 0, score-level)",
                "delta_mean": float(np.mean(deltas)),
                "ci95": [float(lo), float(hi)],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "ci_positive": bool(lo > 0), "n_boot": len(deltas)}

    sig_positive = any(c["against"].startswith("M3") and c["model"] == best_name
                       and c["delta"] > 0 and c.get("significant")
                       for c in comparisons)
    # BOTH gates, as pre-registered: p_adj < 0.05 AND a positive bootstrap CI.
    beats = bool(best > ceiling and sig_positive and boot is not None
                 and boot["ci_positive"])
    verdict = ("NEW FINDING: modern text embedding beats the tabular ceiling"
               if beats else
               "tabular ceiling HOLDS for modern text representations (SPECTER2)")
    gates = {"best_gt_ceiling": bool(best > ceiling),
             "paired_p_adj_significant": bool(sig_positive),
             "bootstrap_ci_positive": bool(boot["ci_positive"]) if boot else None}

    out = {"experiment": "E14d_m4b_specter2", "embedding_file": str(NPZ),
           "n": len(df_split), "summary": summary, "per_seed": per_model,
           "comparisons": comparisons, "bootstrap_vs_M3": boot,
           "ceiling": {"M2": m2_pr, "M3": m3_frozen},
           "verdict": {"text": verdict, "best_model": best_name,
                       "best_auc_pr": round(best, 4), "gates": gates}}
    (C.RESULTS_DIR / "e14d_m4b.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["verdict"], indent=2))


if __name__ == "__main__":
    main()
