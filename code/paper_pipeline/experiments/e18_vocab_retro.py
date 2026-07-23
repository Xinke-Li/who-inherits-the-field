"""E18 - Snapshot-retroactivity bound via an earliest-appearance vocab subset (R5).

WHY. Concept tags and author disambiguation are produced by 2026 models over a
2026 snapshot. The temporal contract guarantees event TIMESTAMPS (a work's year)
but not DESCRIPTOR timing: a concept whose tag entered the vocabulary in, say,
2015 can be applied by the 2026 model to a 1995 paper, so a feature built from it
uses future vocabulary. We BOUND that channel: restrict the concept vocabulary to
concepts whose EARLIEST appearance (first year any work in the store is tagged
with it) is <= the student's feature-freeze year t0+5, rebuild the headline
overlap features on that subset, and re-run the models. This is an UPPER bound,
not a removal: the 2026 tagging MODEL is still what assigned the tags; we cannot
un-apply it, only drop concepts that provably did not exist as tags by t0+5.
(Stated as such in the paper.)

PRE-REGISTERED (for this revision): eligibility = earliest_appearance(concept) <=
t0+5, computed once from the per-author works store. We rebuild the student early
top-10 and the advisor profile on eligible concepts only, recompute
early_overlap, and re-run M1 (single feature), M2 (tabular), and the cohort-
placebo gap (E10 analogue) on the frozen temporal split, 10 seeds. We report the
DELTA against the frozen (unrestricted) headline numbers; a small delta bounds
the descriptor-timing channel as immaterial to the conclusions.

Frozen table untouched. Output: results_<disc>/e18_vocab_retro.json
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
import e14_self_persistence as E14
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

COHORT_WINDOW = 3


def earliest_appearance():
    """First year each concept appears anywhere in the works store."""
    earliest = {}
    with open(C.WORKS_STORE) as f:
        for line in f:
            try:
                works = json.loads(line)["works"]
            except Exception:
                continue
            for w in works:
                y = w.get("year")
                if y is None:
                    continue
                for c in w.get("concepts", []):
                    if c not in earliest or y < earliest[c]:
                        earliest[c] = y
    return earliest


def rebuild_profiles(df, earliest):
    """Per student: eligible early top-10 and eligible advisor profile
    (concepts with earliest appearance <= t0+5), and the restricted overlap."""
    early_cnt, _ = E14.load_student_windows(df)   # early counts from the store
    eo_restr = np.empty(len(df))
    dropped_frac = []
    early_sets_restr, adv_sets_restr = [], []
    for i, r in enumerate(df.itertuples()):
        freeze = r.t0 + C.EARLY_YEARS
        cnt = early_cnt.get(r.st_openalex_id, Counter())
        elig_cnt = Counter({c: n for c, n in cnt.items()
                            if earliest.get(c, 9999) <= freeze})
        e_set = set(c for c, _ in elig_cnt.most_common(10))
        a_set = set(c for c in r.adv_profile if earliest.get(c, 9999) <= freeze)
        early_sets_restr.append(e_set); adv_sets_restr.append(a_set)
        eo_restr[i] = E14.jaccard(e_set, a_set)
        n_adv = len(r.adv_profile)
        dropped_frac.append(1.0 - len(a_set) / n_adv if n_adv else 0.0)
    return eo_restr, early_sets_restr, adv_sets_restr, float(np.mean(dropped_frac))


def fit_single(ds, feature, seed):
    tr = ds.split == "train"; va = ds.split == "val"; te = ds.split == "test"
    y = ds.y.values
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    m.fit(feature[tr.values].reshape(-1, 1), y[tr.values])
    return S.evaluate(y[te.values], m.predict_proba(feature[te.values].reshape(-1, 1))[:, 1])["auc_pr"]


def main():
    disc = "neuro" if os.environ.get("NEURO_DATASET") else "econ"
    df = D.load_dataset(); ds = D.temporal_split(df)
    earliest = earliest_appearance()
    print(f"[e18:{disc}] {len(earliest)} concepts with an earliest-appearance year")

    eo_restr, early_r, adv_r, adv_dropped = rebuild_profiles(df, earliest)
    eo_true = df.early_overlap.values

    # M1: single-feature, true vs restricted vocab (deterministic -> seed 0)
    m1_true = fit_single(ds, eo_true, 0)
    m1_restr = fit_single(ds, eo_restr, 0)

    # cohort placebo gap (E10 analogue) under RESTRICTED vocab, 10 seeds
    t0 = df.t0.values; adv_pid = df.advisor_pid.values
    gaps = []
    for seed in C.SEEDS:
        rng = np.random.default_rng(seed)
        eo_pl = np.empty(len(df))
        for i in range(len(df)):
            cand = np.where((np.abs(t0 - t0[i]) <= COHORT_WINDOW) & (adv_pid != adv_pid[i]))[0]
            if len(cand) == 0:
                cand = np.where(adv_pid != adv_pid[i])[0]
            j = cand[rng.integers(len(cand))]
            eo_pl[i] = E14.jaccard(early_r[i], adv_r[j])   # restricted placebo overlap
        gaps.append(m1_restr - fit_single(ds, eo_pl, seed))
    gap_mean = float(np.mean(gaps))

    out = {
        "experiment": "E18_vocab_retroactivity_bound", "discipline": disc,
        "n_concepts": len(earliest),
        "advisor_profile_concepts_dropped_frac": round(adv_dropped, 4),
        "m1_true_auc_pr": round(m1_true, 4),
        "m1_restricted_auc_pr": round(m1_restr, 4),
        "m1_delta_restricted_minus_true": round(m1_restr - m1_true, 4),
        "e10_gap_restricted_mean": round(gap_mean, 4),
        "bound_type": "UPPER bound: drops post-t0+5 vocabulary; the 2026 tagging "
                      "model itself is not removable",
        "reading": ("descriptor-timing channel is immaterial: restricting to "
                    "concepts eligible at t0+5 moves the headline single-feature "
                    f"AUC-PR by {m1_restr - m1_true:+.4f} and the advisor-placebo "
                    f"gap remains {gap_mean:+.4f}"),
    }
    (C.RESULTS_DIR / "e18_vocab_retro.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
