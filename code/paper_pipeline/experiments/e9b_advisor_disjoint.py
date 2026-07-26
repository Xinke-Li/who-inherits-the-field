"""E9b - Advisor-disjoint split certificate (advisor memorization ruled out).

All students of one advisor land in the same fold (utils/data.advisor_disjoint_split,
seed 0), so no advisor-specific information crosses folds. The strongest pure-tabular
ceiling model (M3 GBDT, the E1/E9 configuration: max_iter=300, internal early
stopping) is retrained at 10 seeds on that split and compared to the same model on
the frozen temporal split.

DECISION RULE (fixed before the five-discipline run, mirroring the v1 economics
reading where the shift was +0.006): the certificate PASSES when the
advisor-disjoint mean test AUC-PR does not fall more than 0.05 below the temporal
reference mean. A larger drop would indicate that the temporal ceiling relies on
advisor identity shared across folds (memorization) rather than on the pre-window
features. The two splits carry different test cohorts and base rates, so both are
reported alongside the shift.

Usage (per discipline): DATASET=<field> DATASET_PATH=... python e9b_advisor_disjoint.py
Output: results_<field>/e9b_advisor_disjoint.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S

from sklearn.ensemble import HistGradientBoostingClassifier


def run_split(df_split, seeds):
    X, _ = D.build_features(df_split, concepts="none")
    parts = D.split_xy(df_split, X)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = parts["train"], parts["val"], parts["test"]
    rows = []
    for seed in seeds:
        m = HistGradientBoostingClassifier(random_state=seed, max_iter=300,
                                           early_stopping=True, validation_fraction=0.15)
        m.fit(Xtr, ytr)
        thr = S.best_f1_threshold(yva, m.predict_proba(Xva)[:, 1])
        rows.append(S.evaluate(yte, m.predict_proba(Xte)[:, 1], thr))
    return S.summarize_seeds(rows), int(len(yte)), float(yte.mean())


def main():
    df = D.load_dataset()
    ref, n_ref, br_ref = run_split(D.temporal_split(df), C.SEEDS)
    dis, n_dis, br_dis = run_split(D.advisor_disjoint_split(df, seed=0), C.SEEDS)
    shift = dis["auc_pr"]["mean"] - ref["auc_pr"]["mean"]
    out = {
        "experiment": "E9b_advisor_disjoint",
        "model": "M3 GBDT (E1 configuration), 10 seeds per split",
        "reference_temporal": {**ref, "n_test": n_ref, "test_base_rate": round(br_ref, 4)},
        "advisor_disjoint": {**dis, "n_test": n_dis, "test_base_rate": round(br_dis, 4)},
        "shift_auc_pr": round(float(shift), 4),
        "rule": "PASS if disjoint mean AUC-PR >= temporal reference mean - 0.05",
        "verdict": "PASS" if shift >= -0.05 else "FAIL",
    }
    (C.RESULTS_DIR / "e9b_advisor_disjoint.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
