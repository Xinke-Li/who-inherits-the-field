"""E1 - Baseline ladder (the honest floor).

Models, all trained on the temporal split, early stopping / threshold tuning on VAL only:
  M0  majority / prior            (sanity floor)
  M1  logistic: early_overlap only  (the single-scalar baseline reviewers will ask about)
  M2  logistic: all tabular pre-window features
  M3  HistGradientBoosting: same features
  M4  logistic: tabular + TF-IDF(early concepts, advisor profile)
  M5  GBDT + Neighbor-Feature Aggregation (temporally guarded prior/closed
      same-advisor siblings; the strong-baseline standard for graphs with
      tabular features - TabGraphs 2024, BGNN 2021)

Decision rule (pre-registered, updated 2026-07-05): any GNN contribution in
the paper is claimed RELATIVE TO M5 (the strongest graph-aware non-GNN
baseline), with M3/M4 reported for the ladder.

Outputs: results_econ/e1_baselines.json, results_econ/e1_baselines.md (paper-ready table).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def run(seed: int, df_split, cache={}):
    """Train all baselines for one seed; return {model: {metric: value}}."""
    res = {}

    # design matrices (built once, cached across seeds - they are deterministic)
    if "tab" not in cache:
        cache["tab"] = D.build_features(df_split, concepts="none")
        cache["tfidf"] = D.build_features(df_split, concepts="tfidf")
    Xt, _ = cache["tab"]
    Xf, _ = cache["tfidf"]
    tab = D.split_xy(df_split, Xt)
    tfi = D.split_xy(df_split, Xf)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = tab["train"], tab["val"], tab["test"]

    # M0 majority/prior: constant score = train base rate
    prior = np.full(len(yte), ytr.mean())
    res["M0_prior"] = S.evaluate(yte, prior)

    # M1 logistic on early_overlap only
    i = C.TABULAR_FEATURES.index("early_overlap")
    m1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    m1.fit(Xtr[:, [i]], ytr)
    thr = S.best_f1_threshold(yva, m1.predict_proba(Xva[:, [i]])[:, 1])
    res["M1_logit_overlap"] = S.evaluate(yte, m1.predict_proba(Xte[:, [i]])[:, 1], thr)

    # M2 logistic, all tabular
    m2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    m2.fit(Xtr, ytr)
    thr = S.best_f1_threshold(yva, m2.predict_proba(Xva)[:, 1])
    res["M2_logit_tabular"] = S.evaluate(yte, m2.predict_proba(Xte)[:, 1], thr)

    # M3 GBDT, all tabular (early stopping on internal val split of TRAIN)
    m3 = HistGradientBoostingClassifier(random_state=seed, max_iter=500,
                                        early_stopping=True, validation_fraction=0.15)
    m3.fit(Xtr, ytr)
    thr = S.best_f1_threshold(yva, m3.predict_proba(Xva)[:, 1])
    res["M3_gbdt_tabular"] = S.evaluate(yte, m3.predict_proba(Xte)[:, 1], thr)

    # M4 logistic + TF-IDF concepts
    (Xtr4, _), (Xva4, _), (Xte4, _) = tfi["train"], tfi["val"], tfi["test"]
    m4 = make_pipeline(StandardScaler(with_mean=False),
                       LogisticRegression(max_iter=1000, C=0.5, random_state=seed,
                                          solver="liblinear"))
    m4.fit(Xtr4, ytr)
    thr = S.best_f1_threshold(yva, m4.predict_proba(Xva4)[:, 1])
    res["M4_logit_tfidf"] = S.evaluate(yte, m4.predict_proba(Xte4)[:, 1], thr)

    # M5 GBDT + temporally-guarded neighbor-feature aggregation
    if "nfa" not in cache:
        nfa = D.build_nfa_features(df_split)
        cache["nfa"] = np.hstack([Xt, nfa.values.astype(float)])
    X5 = cache["nfa"]
    p5 = D.split_xy(df_split, X5)
    (Xtr5, _), (Xva5, _), (Xte5, _) = p5["train"], p5["val"], p5["test"]
    m5 = HistGradientBoostingClassifier(random_state=seed, max_iter=500,
                                        early_stopping=True, validation_fraction=0.15)
    m5.fit(Xtr5, ytr)
    thr = S.best_f1_threshold(yva, m5.predict_proba(Xva5)[:, 1])
    res["M5_gbdt_nfa"] = S.evaluate(yte, m5.predict_proba(Xte5)[:, 1], thr)

    # bootstrap CI for ALL ladder models (seed 0 only; 1000 draws each).
    # M1/M2/M4 are deterministic given the split (std across seeds = 0 by
    # construction); their uncertainty is test-set sampling error, which the
    # bootstrap CI quantifies - the paper table footnotes this.
    if seed == C.SEEDS[0]:
        for key, scores in [("M1_logit_overlap", m1.predict_proba(Xte[:, [i]])[:, 1]),
                            ("M2_logit_tabular", m2.predict_proba(Xte)[:, 1]),
                            ("M3_gbdt_tabular", m3.predict_proba(Xte)[:, 1]),
                            ("M4_logit_tfidf", m4.predict_proba(Xte4)[:, 1]),
                            ("M5_gbdt_nfa", m5.predict_proba(Xte5)[:, 1])]:
            lo, hi = S.bootstrap_ci(yte, scores, metric=C.PRIMARY_METRIC,
                                    n_boot=1000, seed=0)
            res[key]["auc_pr_ci95"] = [lo, hi]
    return res


def main():
    df = D.load_dataset()
    df_split = D.temporal_split(df)

    cache = {}
    per_model = {}
    for seed in C.SEEDS:
        for model, metrics in run(seed, df_split, cache).items():
            per_model.setdefault(model, []).append(metrics)
        print(f"[e1] seed {seed} done")

    summary = {m: S.summarize_seeds(v) for m, v in per_model.items()}
    # keep seed-0 CIs
    for m, v in per_model.items():
        if "auc_pr_ci95" in v[0]:
            summary[m]["auc_pr_ci95_seed0"] = v[0]["auc_pr_ci95"]

    out = {"experiment": "E1_baselines", "split": "temporal",
           "seeds": C.SEEDS, "summary": summary, "per_seed": per_model}
    (C.RESULTS_DIR / "e1_baselines.json").write_text(json.dumps(out, indent=2))

    # paper-ready markdown table
    lines = ["| Model | AUC-PR | AUC-ROC | F1 |", "|---|---|---|---|"]
    for m in ["M0_prior", "M1_logit_overlap", "M2_logit_tabular",
              "M3_gbdt_tabular", "M4_logit_tfidf", "M5_gbdt_nfa"]:
        s = summary[m]
        f1 = f"{s['f1']['mean']:.3f}\u00b1{s['f1']['std']:.3f}" if "f1" in s else "\u2014"
        lines.append(f"| {m} | {s['auc_pr']['mean']:.3f}\u00b1{s['auc_pr']['std']:.3f} "
                     f"| {s['auc_roc']['mean']:.3f}\u00b1{s['auc_roc']['std']:.3f} | {f1} |")
    (C.RESULTS_DIR / "e1_baselines.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
