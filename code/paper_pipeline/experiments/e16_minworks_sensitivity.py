"""E16 - works-per-window sensitivity: >=1 / >=3 / >=5 (reviewer R6).

WHY. The frozen sample conditions on >=3 works in BOTH the early and late window.
Conditioning on a post-treatment quantity in both windows is a potential collider:
early leavers (few late works) are dropped, and leaving may correlate with not
inheriting, so the 13.9% retention rate and the E10/E14 geometry could be
selection artifacts. We vary the min-works filter and re-measure.

WHAT RUNS HERE (CPU, real):
  >=3  the frozen table itself (base rate 0.139).
  >=5  a strict SUBSET of the frozen table (early_prod>=5 AND late_prod>=5); this
       is a pure filter, no re-fetch, so it is exact.
For each, on its own temporal split we recompute: base rate (retention); M1, the
true-advisor single-feature AUC-PR; the cohort-placebo single-feature AUC-PR
(E10 analogue, 10 seeds) and the M1-minus-placebo gap; and a student-only floor
(logistic on advisor-free pre-window scalars + early typicality, the E14 proxy).

WHAT IS DEFERRED (>=1): the informative, collider-RELAXING direction needs the
students the >=3 filter dropped (<3 works in a window), whose windowed works are
not in the frozen works store. It requires a builder re-run with MIN_WORKS=1
(re-fetch + re-window). Scaffolded here as a no-op branch that records the TODO;
see PENDING_TODO.md. We do NOT fabricate its numbers.

Frozen table untouched (>=5 is an in-memory filter). Output:
results_<disc>/e16_minworks_sensitivity.json
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

COHORT_WINDOW = 3


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def logit_auc(Xtr, ytr, Xte, yte, seed=0):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    m.fit(Xtr, ytr)
    return S.evaluate(yte, m.predict_proba(Xte)[:, 1])["auc_pr"]


def measure(df):
    """All the geometry for one min-works subsample."""
    ds = D.temporal_split(df.reset_index(drop=True))
    tr, te = (ds.split == "train").values, (ds.split == "test").values
    y = ds.y.values
    eo = ds.early_overlap.values.reshape(-1, 1)

    # student-only floor features: advisor-free pre-window scalars + typicality
    early_sets = [set(l) for l in ds.early_concepts]
    disc_top = set(c for c, _ in Counter(c for s in early_sets for c in s).most_common(10))
    typ = np.array([jaccard(s, disc_top) for s in early_sets]).reshape(-1, 1)
    Xso = np.hstack([ds[["early_prod", "early_breadth", "coauth_early"]].astype(float).values, typ])

    m1 = logit_auc(eo[tr], y[tr], eo[te], y[te])
    so = logit_auc(Xso[tr], y[tr], Xso[te], y[te])

    t0 = ds.t0.values; adv = ds.advisor_pid.values
    adv_profiles = [set(l) for l in ds.adv_profile]
    placebo = []
    for seed in C.SEEDS:
        rng = np.random.default_rng(seed)
        eop = np.empty(len(ds))
        for i in range(len(ds)):
            cand = np.where((np.abs(t0 - t0[i]) <= COHORT_WINDOW) & (adv != adv[i]))[0]
            if len(cand) == 0:
                cand = np.where(adv != adv[i])[0]
            j = cand[rng.integers(len(cand))]
            eop[i] = jaccard(early_sets[i], adv_profiles[j])
        placebo.append(logit_auc(eop[tr].reshape(-1, 1), y[tr], eop[te].reshape(-1, 1), y[te], seed))
    pl = float(np.mean(placebo))
    return {"n": int(len(df)), "base_rate": round(float(df.y.mean()), 4),
            "M1_true_auc_pr": round(m1, 4),
            "cohort_placebo_auc_pr": round(pl, 4),
            "M1_minus_placebo_gap": round(m1 - pl, 4),
            "student_only_floor_auc_pr": round(so, 4)}


def main():
    disc = "neuro" if os.environ.get("NEURO_DATASET") else "econ"
    df = D.load_dataset()
    out = {"experiment": "E16_minworks_sensitivity", "discipline": disc, "thresholds": {}}

    out["thresholds"][">=3_frozen"] = measure(df)
    print(f"[e16:{disc}] >=3 (frozen): {out['thresholds']['>=3_frozen']}")

    sub5 = df[(df.early_prod >= 5) & (df.late_prod >= 5)].copy()
    out["thresholds"][">=5_subset"] = measure(sub5)
    print(f"[e16:{disc}] >=5 (subset n={len(sub5)}): {out['thresholds']['>=5_subset']}")

    out["thresholds"][">=1"] = {
        "status": "DEFERRED - needs builder re-run with MIN_WORKS=1 (dropped "
                  "students' windowed works not in the frozen works store)",
        "todo": "python experiments/e16_minworks_sensitivity.py --minworks 1 "
                "after rebuilding data_econ/clean_dataset_min1.parquet via the "
                "OpenAlex_Dataset_Builder with MIN_WORKS=1 (see PENDING_TODO.md)",
    }
    # honest reading of the runnable directions
    g3 = out["thresholds"][">=3_frozen"]["M1_minus_placebo_gap"]
    g5 = out["thresholds"][">=5_subset"]["M1_minus_placebo_gap"]
    out["reading"] = (f"the advisor-vs-placebo gap is stable under tightening "
                      f"(>=3: {g3:+.3f}; >=5: {g5:+.3f}); the collider-relaxing "
                      f">=1 direction is deferred (scaffold + TODO)")
    (C.RESULTS_DIR / "e16_minworks_sensitivity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
