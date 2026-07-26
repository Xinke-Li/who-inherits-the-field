"""R15c - full eq. (2) protocol for the one screen-passing extra rung (B8).

Neuro CatBoost beat frozen M3 and M5 per-seed (r15 screen); the post hoc
rule requires the full corrected protocol before any paper row: per-seed
scores vs the validation-symmetric ceiling M5' (refit via
e12_corrected_aggregation.fit_val_symmetric, identical to the stored e12
run), paired Wilcoxon on the ten seed pairs, and the 2000-draw paired
student-level bootstrap pooled over seeds. Single comparison, no family:
the raw p is reported as such.

Usage: DATASET=neuro DATASET_PATH=data/clean_dataset_neuro.parquet \
       python r15c_catboost_eq2.py
Output: results/robustness/catboost_eq2_neuro.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from e12_corrected_aggregation import N_BOOT, fast_auc_pr, fit_val_symmetric


def main():
    from catboost import CatBoostClassifier
    field = C.CLEAN_DATASET.stem.replace("clean_dataset_", "")
    df_split = D.temporal_split(D.load_dataset())
    Xt, _ = D.build_features(df_split, concepts="none")
    nfa = D.build_nfa_features(df_split)
    X5 = np.hstack([Xt, nfa.values.astype(float)])
    tab = D.split_xy(df_split, Xt)
    p5 = D.split_xy(df_split, X5)
    (Xtr, ytr), (Xte, yte) = tab["train"], tab["test"]
    (Xtr5, _), (Xva5, yva), (Xte5, _) = p5["train"], p5["val"], p5["test"]

    cb_scores, cb_ap, m5p_scores, m5p_ap = {}, [], {}, []
    for seed in C.SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(ytr))
        cut = int(0.85 * len(ytr))
        m = CatBoostClassifier(iterations=500, random_seed=seed, verbose=0,
                               early_stopping_rounds=20)
        m.fit(Xtr[idx[:cut]], ytr[idx[:cut]],
              eval_set=(Xtr[idx[cut:]], ytr[idx[cut:]]), use_best_model=True)
        cb_scores[seed] = m.predict_proba(Xte)[:, 1]
        cb_ap.append(fast_auc_pr(yte, cb_scores[seed]))
        p_te, _, it, _ = fit_val_symmetric(Xtr5, ytr, Xva5, yva, Xte5, seed)
        m5p_scores[seed] = p_te
        m5p_ap.append(fast_auc_pr(yte, p_te))
        print(f"[r15c] seed {seed}: catboost {cb_ap[-1]:.4f} M5' {m5p_ap[-1]:.4f}",
              flush=True)

    w = S.paired_wilcoxon(cb_ap, m5p_ap)
    rng = np.random.default_rng(0)
    n = len(yte)
    pooled = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = yte[idx]
        if yb.min() == yb.max():
            pooled[b] = np.nan
            continue
        pooled[b] = float(np.mean([fast_auc_pr(yb, cb_scores[s][idx])
                                   - fast_auc_pr(yb, m5p_scores[s][idx])
                                   for s in C.SEEDS]))
    pooled = pooled[~np.isnan(pooled)]
    ci = [round(float(np.percentile(pooled, 2.5)), 4),
          round(float(np.percentile(pooled, 97.5)), 4)]
    gates = {"seed_mean_gt_ceiling": bool(np.mean(cb_ap) > np.mean(m5p_ap)),
             "p_lt_0.05": bool(w["p"] < 0.05),
             "student_ci_lower_gt_0": bool(ci[0] > 0)}
    out = {"experiment": "R15c_catboost_eq2", "field": field,
           "catboost_mean": round(float(np.mean(cb_ap)), 4),
           "M5prime_mean": round(float(np.mean(m5p_ap)), 4),
           "delta": round(float(np.mean(cb_ap) - np.mean(m5p_ap)), 4),
           "wilcoxon_p_raw": w["p"], "student_ci95": ci,
           "gates": gates, "exceeds": all(gates.values())}
    (REPO / "results" / "robustness" / f"catboost_eq2_{field}.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
