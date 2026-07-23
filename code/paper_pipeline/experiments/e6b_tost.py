"""E6b - TOST equivalence test for the divergence-citation estimate (reviewer R7).

WHY. The paper phrased the divergence-citation result as "no citation penalty".
That is an EQUIVALENCE claim, but it was supported only by a small, non-
significant coefficient (accepting the null from failure to reject it - a
fallacy). We replace it with a two-one-sided-tests (TOST) equivalence test: does
the divergence coefficient lie inside a pre-specified band of practical
negligibility?

PRE-REGISTERED (for this revision; declared post-hoc relative to the original
submission, NOT pre-registered at first submission - stated honestly in the
paper):
  equivalence bound  Delta = 1.0 percentile point of late-window citation
                     percentile per 1 SD of divergence. On the [0,1] percentile
                     scale this is 0.01 in the coefficient's native units.
  alpha              0.05 (each one-sided test); equivalence declared iff the
                     90% CI (beta +- 1.645*se) lies entirely within (-Delta,+Delta).
  We run it on the SAME FE ladder used for the point estimate (advisor-clustered
  SE), for BOTH disciplines and BOTH divergence constructs (concept-space =
  spec 3_controls_FE; structural Node2Vec = spec 4_structural_FE), so the
  equivalence verdict is reported like-for-like with the estimate.

Reads the frozen table + outcomes through e6.prep/e6.fit_ladder (no new data).
Output: results_<disc>/e6b_tost.json
"""
import json
import os
import sys
from pathlib import Path

from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as CFG
import e6_innovation_premium as E6

BOUND = 0.01          # 1.0 percentile point per SD, on the [0,1] outcome scale
ALPHA = 0.05


def tost(coef, se, delta=BOUND, alpha=ALPHA):
    """Two one-sided tests. H1 (equivalence): -delta < beta < delta."""
    z_lower = (coef + delta) / se          # test beta > -delta
    z_upper = (coef - delta) / se          # test beta <  delta
    p_lower = 1.0 - norm.cdf(z_lower)
    p_upper = norm.cdf(z_upper)
    p_tost = max(p_lower, p_upper)
    ci90 = (coef - 1.645 * se, coef + 1.645 * se)
    equivalent = (ci90[0] > -delta) and (ci90[1] < delta)
    return {"coef": round(coef, 4), "se": round(se, 4),
            "delta_bound": delta, "p_lower": round(p_lower, 5),
            "p_upper": round(p_upper, 5), "p_tost": round(p_tost, 5),
            "ci90": [round(ci90[0], 4), round(ci90[1], 4)],
            "equivalent_to_zero_within_bound": bool(equivalent)}


def main():
    disc = "neuro" if os.environ.get("NEURO_DATASET") else "econ"
    df = E6.prep()
    out_range = [float(df.late_cite_pct_mean.min()), float(df.late_cite_pct_mean.max())]
    assert out_range[1] <= 1.0001, f"outcome not on [0,1] scale: {out_range}; re-scale BOUND"
    ladder = E6.fit_ladder(df, "late_cite_pct_mean")

    res = {"experiment": "E6b_TOST_equivalence", "discipline": disc,
           "outcome_range": [round(x, 4) for x in out_range],
           "bound_pctile_points_per_SD": 1.0, "alpha": ALPHA,
           "prereg": "bound fixed before running; post-hoc vs original submission",
           "tests": {}}
    for label, spec in [("concept_space_FE", "3_controls_FE"),
                        ("structural_n2v_FE", "4_structural_FE")]:
        if spec in ladder:
            r = ladder[spec]
            res["tests"][label] = tost(r["coef"], r["se"])
            res["tests"][label]["p_point_estimate"] = r["p"]
    # summary line for the paper
    ct = res["tests"].get("concept_space_FE", {})
    res["summary"] = (
        f"concept-space divergence: coef {ct.get('coef')} (90% CI {ct.get('ci90')}), "
        f"{'equivalent to zero within +-1 pctile pt (TOST p=%s)' % ct.get('p_tost') if ct.get('equivalent_to_zero_within_bound') else 'NOT equivalent within bound'}")
    (CFG.RESULTS_DIR / "e6b_tost.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
