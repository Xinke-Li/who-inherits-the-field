"""E4 - Label robustness: theta sweep + continuous-label regression.

(i)  Re-run the M3 baseline over theta in THETA_GRID (label = late_overlap > theta).
     Pre-registered criterion: the feature-importance ORDERING and the sign of the
     headline effects must agree on >= 4/5 thresholds.
(ii) Fractional-logit regression of the continuous late_overlap on the same
     pre-window features (threshold-free evidence).

Outputs: results_econ/e4_label_robustness.json + figures_econ/e4_theta_stability.html
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
from sklearn.inspection import permutation_importance


def main():
    df = D.load_dataset()
    df_split = D.temporal_split(df)
    X, names = D.build_features(df_split, concepts="none")
    parts = D.split_xy(df_split, X)
    (Xtr, _), (Xva, _), (Xte, _) = parts["train"], parts["val"], parts["test"]
    m_tr = (df_split.split == "train").values
    m_te = (df_split.split == "test").values

    sweep, importances = {}, {}
    for theta in C.THETA_GRID:
        y_all = (df_split.late_overlap > theta).astype(int).values
        ytr, yte = y_all[m_tr], y_all[m_te]
        per_seed = []
        for seed in C.SEEDS:  # full 10-seed protocol (was 5; unified 2026-07-05)
            m = HistGradientBoostingClassifier(random_state=seed, max_iter=300,
                                               early_stopping=True, validation_fraction=0.15)
            m.fit(Xtr, ytr)
            per_seed.append(S.evaluate(yte, m.predict_proba(Xte)[:, 1]))
            if seed == 0:
                imp = permutation_importance(m, Xte, yte, n_repeats=10, random_state=0,
                                             scoring="average_precision")
                importances[theta] = dict(zip(names, imp.importances_mean.round(4)))
        sweep[theta] = S.summarize_seeds(per_seed, keys=("auc_pr", "auc_roc"))
        sweep[theta]["base_rate"] = float(y_all.mean())
        print(f"[e4] theta={theta}: base={sweep[theta]['base_rate']:.3f} "
              f"AUC-PR={sweep[theta]['auc_pr']['mean']:.3f}")

    # ranking agreement across thresholds
    rank_ref = sorted(importances[C.JACCARD_THETA], key=importances[C.JACCARD_THETA].get,
                      reverse=True)
    top3_agree = sum(
        set(sorted(importances[t], key=importances[t].get, reverse=True)[:3])
        == set(rank_ref[:3]) for t in C.THETA_GRID)

    # (ii) fractional logit on continuous late_overlap (GLM binomial + logit link)
    import statsmodels.api as sm
    Z = sm.add_constant((X - X.mean(0)) / np.clip(X.std(0), 1e-9, None))
    frac = sm.GLM(df_split.late_overlap.values, Z,
                  family=sm.families.Binomial()).fit(cov_type="HC1")
    frac_coefs = dict(zip(["const"] + names, np.round(frac.params, 4)))
    frac_p = dict(zip(["const"] + names, np.round(frac.pvalues, 5)))

    out = {"experiment": "E4_label_robustness",
           "theta_sweep": {str(k): v for k, v in sweep.items()},
           "perm_importance_by_theta": {str(k): v for k, v in importances.items()},
           "top3_importance_agreement": f"{top3_agree}/{len(C.THETA_GRID)}",
           "fractional_logit": {"coef": frac_coefs, "p_HC1": frac_p}}
    (C.RESULTS_DIR / "e4_label_robustness.json").write_text(json.dumps(out, indent=2))

    try:
        from utils import plotting as P
        P.plot_theta_stability(C.THETA_GRID,
                               {"M3 GBDT": [sweep[t]["auc_pr"]["mean"] for t in C.THETA_GRID]})
    except Exception as e:
        print(f"[e4] plotting skipped: {e}")
    print(f"[e4] top-3 importance agreement: {top3_agree}/{len(C.THETA_GRID)}")


if __name__ == "__main__":
    main()
