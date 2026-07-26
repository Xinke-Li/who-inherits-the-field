#!/usr/bin/env python3
"""Shared evaluation helpers for the revision scripts.

Two things the revision kept paying for twice, fixed once here.

1. fit_once_eval_masks: analyses that differ only in which test rows they score
   must not re-fit the model. Fit each (arm, seed) once, then evaluate the same
   score vector against every mask.

2. fast_auc_pr: the vectorised average precision already in the repo
   (e12_corrected_aggregation.py:82), tie-aware and identical to sklearn's
   average_precision_score. A 2000-draw bootstrap over 10 seeds is 20,000 AP
   computations per cell, and sklearn's Python-level implementation dominates
   the runtime there.

paired_bootstrap draws the resample indices once and reuses them across arms, so
the comparison is paired by construction rather than by coincidence.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e12_corrected_aggregation import fast_auc_pr  # noqa: E402,F401


def fit_once_eval_masks(fit_fn, seeds, masks):
    """fit_fn(seed) -> (y_test, scores). Returns {mask_name: (y, [scores...])}.

    The model is fitted once per seed; each mask only slices the resulting score
    vector. masks maps a name to a boolean array over the test rows, or None for
    the full test set.
    """
    y_full, per_seed = None, []
    for seed in seeds:
        y, s = fit_fn(seed)
        y_full = y
        per_seed.append(s)
    out = {}
    for name, m in masks.items():
        if m is None:
            out[name] = (y_full, per_seed)
        else:
            out[name] = (y_full[m], [s[m] for s in per_seed])
    return out


def paired_bootstrap(y, arm_scores, n_boot=2000, seed=0):
    """Paired student-level bootstrap. arm_scores: {arm: [scores per seed]}.

    Returns {arm: array (n_seeds, n_boot)} of AUC-PR, with one shared set of
    resample indices so differences are paired.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=(n_boot, n))
    names = list(arm_scores)
    n_seeds = len(arm_scores[names[0]])
    boot = {a: np.full((n_seeds, n_boot), np.nan) for a in names}
    for b in range(n_boot):
        i = idx[b]
        yb = y[i]
        if yb.sum() in (0, len(yb)):
            continue
        for a in names:
            ss = arm_scores[a]
            for s in range(n_seeds):
                boot[a][s, b] = fast_auc_pr(yb, ss[s][i])
    return boot


def tost_verdict(mean, lo95, hi95, exceeds, margins=(0.01, 0.02, 0.03)):
    """r11_power_tost.py's machinery, verbatim."""
    sigma = (hi95 - lo95) / 3.92
    lo90, hi90 = mean - 1.645 * sigma, mean + 1.645 * sigma
    eq_at = next((m for m in margins if -m < lo90 and hi90 < m), None)
    return {"delta": round(mean, 4), "sigma": round(sigma, 4),
            "MDE_80pct": round(2.80 * sigma, 4),
            "ci90": [round(lo90, 4), round(hi90, 4)],
            "equivalent_at_margin": eq_at,
            "largest_gain_excluded": round(hi90, 4),
            "verdict": ("exceeds" if exceeds else
                        f"equivalent at {eq_at}" if eq_at is not None else
                        "not identifiable at this design")}
