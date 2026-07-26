"""E9 (b-e) - Stress tests. One 5-row stress-test table.

  b: advisor-disjoint split          (advisor memorization ruled out)
  c: drop 400-cap students           (fetch-truncation sensitivity)
  d: drop t0 < 1960                  (residual disambiguation sensitivity)
  e: drop early_overlap feature      (tests the "the model is just one scalar" concern)

Each variant: M3 GBDT, 10 seeds, same protocol as E1. Compare to the E1 M3 row.
(E9a placebo lives in e9a_placebo.py; run it first.)
Output: results_econ/e9_robustness.json + results_econ/e9_robustness.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S

from sklearn.ensemble import HistGradientBoostingClassifier


def run_variant(df_split, drop_features=None, seeds=None):
    X, _ = D.build_features(df_split, concepts="none", drop=drop_features)
    parts = D.split_xy(df_split, X)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = parts["train"], parts["val"], parts["test"]
    rows = []
    for seed in (seeds or C.SEEDS):
        m = HistGradientBoostingClassifier(random_state=seed, max_iter=300,
                                           early_stopping=True, validation_fraction=0.15)
        m.fit(Xtr, ytr)
        thr = S.best_f1_threshold(yva, m.predict_proba(Xva)[:, 1])
        rows.append(S.evaluate(yte, m.predict_proba(Xte)[:, 1], thr))
    return S.summarize_seeds(rows), len(df_split[df_split.split == "test"])


def capped_students(df):
    """Students whose works fetch hit the MAX_WORKS cap (~400 works in store)."""
    import json as _json
    capped = set()
    if C.WORKS_STORE.exists():
        with open(C.WORKS_STORE) as f:
            for line in f:
                try:
                    j = _json.loads(line)
                    if len(j["works"]) >= 400:
                        capped.add(j["aid"])
                except Exception:
                    pass
    return df.st_openalex_id.isin(capped)


def main():
    df = D.load_dataset()
    out, table = {"experiment": "E9_robustness"}, []

    # reference: temporal split, all features (matches E1 M3)
    ref, n_te = run_variant(D.temporal_split(df))
    out["ref_temporal"] = ref
    table.append(("reference (temporal, all features)", ref, n_te))

    # b: advisor-disjoint split
    s, n = run_variant(D.advisor_disjoint_split(df, seed=0))
    out["b_advisor_disjoint"] = s; table.append(("b: advisor-disjoint split", s, n))

    # c: drop capped students
    mask = capped_students(df)
    print(f"[e9c] capped students dropped: {int(mask.sum())}")
    s, n = run_variant(D.temporal_split(df[~mask].reset_index(drop=True)))
    out["c_drop_capped"] = s; table.append(("c: drop 400-cap students", s, n))

    # d: drop t0 < 1960
    s, n = run_variant(D.temporal_split(df[df.t0 >= 1960].reset_index(drop=True)))
    out["d_t0_ge_1960"] = s; table.append(("d: drop t0<1960", s, n))

    # e: remove early_overlap
    s, n = run_variant(D.temporal_split(df), drop_features=["early_overlap"])
    out["e_no_overlap_feature"] = s; table.append(("e: no early_overlap feature", s, n))

    (C.RESULTS_DIR / "e9_robustness.json").write_text(json.dumps(out, indent=2))
    lines = ["| Variant | AUC-PR | AUC-ROC | n_test |", "|---|---|---|---|"]
    for name, s, n in table:
        lines.append(f"| {name} | {s['auc_pr']['mean']:.3f}±{s['auc_pr']['std']:.3f} "
                     f"| {s['auc_roc']['mean']:.3f}±{s['auc_roc']['std']:.3f} | {n} |")
    (C.RESULTS_DIR / "e9_robustness.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
