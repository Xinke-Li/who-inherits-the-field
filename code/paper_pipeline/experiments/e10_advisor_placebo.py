"""E10 - Advisor-placebo controls (anti-tautology test, added 2026-07-05).

Reviewer attack this experiment answers: "label and headline feature share the
same advisor profile A, so the task may reduce to generic topic persistence -
students keep working on what they started on - and the *advisor* is
epiphenomenal."

Design: keep the TRUE label y = 1[J(C_late, A_own) > theta] fixed, and replace
the early_overlap FEATURE with overlap against a placebo advisor profile:

  true            J(C_early, A_own)                       (reference)
  placebo_cohort  J(C_early, A_j) for a row j with |t0_j - t0_i| <= 3 and a
                  different advisor (cohort-matched: controls field drift)
  placebo_random  J(C_early, A_j) for a uniformly sampled row j with a
                  different advisor
  field_mean      J(C_early, top-10 most frequent advisor-profile concepts)
                  (deterministic generic "field typicality" feature)

If the placebo features predict retention as well as the true feature, the
predictive signal is generic field typicality and the advisor is irrelevant;
if the true feature dominates, the task measures advisor-specific inheritance.

Models per variant (mirroring E1): L1 = logistic on the overlap feature alone
(M1 analogue); L2 = logistic on all tabular features with early_overlap
swapped for the placebo feature (M2 analogue); G3 = GBDT likewise (M3
analogue). Temporal split, thresholds on VAL only, 10 seeds (placebo redraw +
GBDT seed), paired Wilcoxon true-vs-placebo with BH correction across the
comparison family.

Output: results_econ/e10_advisor_placebo.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COHORT_WINDOW = 3  # +/- years for the cohort-matched placebo


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def placebo_overlap(df, kind: str, seed: int) -> np.ndarray:
    """Overlap of each student's early concepts with a placebo advisor profile."""
    rng = np.random.default_rng(seed)
    early = [set(l) for l in df.early_concepts]
    profiles = [set(l) for l in df.adv_profile]
    adv = df.advisor_pid.values
    t0 = df.t0.values
    n = len(df)

    if kind == "field_mean":
        top10 = {c for c, _ in Counter(
            c for l in df.adv_profile for c in l).most_common(10)}
        return np.array([jaccard(e, top10) for e in early])

    order = np.argsort(t0, kind="stable")
    out = np.empty(n)
    for i in range(n):
        if kind == "cohort":
            lo = np.searchsorted(t0[order], t0[i] - COHORT_WINDOW, "left")
            hi = np.searchsorted(t0[order], t0[i] + COHORT_WINDOW, "right")
            pool = order[lo:hi]
        else:  # random
            pool = np.arange(n)
        pool = pool[adv[pool] != adv[i]]
        j = int(rng.choice(pool))
        out[i] = jaccard(early[i], profiles[j])
    return out


def run_variant(df_split, overlap_feature, seed, thr_models=("L1", "L2", "G3")):
    """Train the three model analogues on one placebo (or true) feature draw."""
    Xt, cols = D.build_features(df_split, concepts="none")
    i_ov = cols.index("early_overlap")
    Xt = Xt.copy()
    Xt[:, i_ov] = overlap_feature
    parts = D.split_xy(df_split, Xt)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = parts["train"], parts["val"], parts["test"]
    res = {}

    if "L1" in thr_models:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,
                                                               random_state=seed))
        m.fit(Xtr[:, [i_ov]], ytr)
        thr = S.best_f1_threshold(yva, m.predict_proba(Xva[:, [i_ov]])[:, 1])
        res["L1_logit_overlap"] = S.evaluate(yte, m.predict_proba(Xte[:, [i_ov]])[:, 1], thr)
    if "L2" in thr_models:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,
                                                               random_state=seed))
        m.fit(Xtr, ytr)
        thr = S.best_f1_threshold(yva, m.predict_proba(Xva)[:, 1])
        res["L2_logit_tabular"] = S.evaluate(yte, m.predict_proba(Xte)[:, 1], thr)
    if "G3" in thr_models:
        m = HistGradientBoostingClassifier(random_state=seed, max_iter=300,
                                           early_stopping=True, validation_fraction=0.15)
        m.fit(Xtr, ytr)
        thr = S.best_f1_threshold(yva, m.predict_proba(Xva)[:, 1])
        res["G3_gbdt_tabular"] = S.evaluate(yte, m.predict_proba(Xte)[:, 1], thr)
    return res


def main():
    df = D.load_dataset()
    df_split = D.temporal_split(df)

    # sanity: recomputed Jaccard reproduces the frozen early_overlap column
    recomputed = np.array([jaccard(set(e), set(a))
                           for e, a in zip(df_split.early_concepts, df_split.adv_profile)])
    max_dev = float(np.abs(recomputed - df_split.early_overlap.values).max())
    # frozen column is stored rounded; 1e-3 tolerance (observed max dev ~5e-5)
    assert max_dev < 1e-3, f"early_overlap not reproducible (max dev {max_dev})"

    variants = {"true": None, "placebo_cohort": "cohort",
                "placebo_random": "random", "field_mean": "field_mean"}
    per_variant, feat_stats = {}, {}
    for name, kind in variants.items():
        runs = []
        feats = []
        for seed in C.SEEDS:
            if kind is None:
                ov = df_split.early_overlap.values
            else:
                ov = placebo_overlap(df_split, kind, seed)
            feats.append(ov)
            runs.append(run_variant(df_split, ov, seed))
            if kind in (None, "field_mean"):  # deterministic feature: 10 GBDT seeds only
                pass
        per_variant[name] = runs
        fv = np.concatenate(feats)
        feat_stats[name] = {"mean": round(float(fv.mean()), 4),
                            "std": round(float(fv.std()), 4),
                            "corr_with_true": round(float(np.corrcoef(
                                np.tile(df_split.early_overlap.values, len(feats)), fv)[0, 1]), 4)}
        print(f"[e10] {name} done")

    summary = {v: {m: S.summarize_seeds([r[m] for r in runs], keys=("auc_pr", "auc_roc"))
                   for m in runs[0]} for v, runs in per_variant.items()}

    # paired Wilcoxon true vs each placebo, BH across the family
    comparisons, pvals = [], []
    for v in ("placebo_cohort", "placebo_random", "field_mean"):
        for m in ("L1_logit_overlap", "L2_logit_tabular", "G3_gbdt_tabular"):
            a = [r[m]["auc_pr"] for r in per_variant["true"]]
            b = [r[m]["auc_pr"] for r in per_variant[v]]
            t = S.paired_wilcoxon(a, b)
            comparisons.append({"variant": v, "model": m,
                                "true_mean": round(float(np.mean(a)), 4),
                                "placebo_mean": round(float(np.mean(b)), 4),
                                "mean_diff": round(t["mean_diff"], 4), "p_raw": t["p"]})
            pvals.append(t["p"])
    p_adj, reject = S.bh_correction(pvals)
    for c, pa, rj in zip(comparisons, p_adj, reject):
        c["p_adj"] = round(float(pa), 6)
        c["significant"] = bool(rj)

    out = {"experiment": "E10_advisor_placebo", "seeds": C.SEEDS,
           "cohort_window_years": COHORT_WINDOW,
           "feature_stats": feat_stats, "summary": summary,
           "comparisons": comparisons,
           "reading": ("If true-advisor overlap significantly outperforms cohort-matched "
                       "placebo overlap, the predictive signal is advisor-specific, not "
                       "generic field typicality; the task is not a persistence tautology.")}
    (C.RESULTS_DIR / "e10_advisor_placebo.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("feature_stats", "comparisons")}, indent=2))


if __name__ == "__main__":
    main()
