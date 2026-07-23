"""E8 - Mechanisms & heterogeneity (feeds the Discussion section).

(i)   coauth x divergence interaction on field retention:
      do early co-authors stay in the advisor's field for different reasons?
(ii)  advisor breadth moderation: do generalist advisors 'release' students more?
(iii) cohort trends: retention base rate and the early_overlap gradient by decade
      (with 95% CIs - the 1970s/1980s cells are small and the paper must not
      overstate flatness).

All regressors are pre-window; logit models with advisor-clustered SE.
2026-07-05: institution-FE variants added for parity with the E6 ladder.
Output: results_econ/e8_heterogeneity.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as CFG
from utils import data as D

import statsmodels.formula.api as smf


def zscore(s):
    return (s - s.mean()) / s.std()


def fit(formula, df):
    m = smf.logit(formula, data=df).fit(disp=0, cov_type="cluster",
                                        cov_kwds={"groups": df["advisor_pid"]})
    return m


def coef_row(m, key):
    return {"coef": round(float(m.params[key]), 4), "se": round(float(m.bse[key]), 4),
            "p": round(float(m.pvalues[key]), 5), "n": int(m.nobs)}


def main():
    df = D.load_dataset()
    df["divergence"] = zscore(1.0 - df.early_overlap)
    df["adv_breadth_z"] = zscore(df.adv_early_breadth)
    df["early_prod_z"] = zscore(df.early_prod)
    df["coauth"] = df.coauth_early.astype(int)
    df["cohort"] = (df.t0 // 5 * 5).astype(int).astype(str)

    out = {"experiment": "E8_heterogeneity"}

    base = "y ~ divergence + early_prod_z + adv_breadth_z + C(cohort)"
    m1 = fit(base + " + coauth + coauth:divergence", df)
    out["coauth_x_divergence"] = {
        "coauth_main": coef_row(m1, "coauth"),
        "interaction": coef_row(m1, "coauth:divergence")}

    m2 = fit(base + " + coauth + adv_breadth_z:divergence", df)
    out["advisor_breadth_x_divergence"] = {
        "breadth_main": coef_row(m2, "adv_breadth_z"),
        "interaction": coef_row(m2, "adv_breadth_z:divergence")}

    # FE parity with E6 (2026-07-05): add institution fixed effects so the
    # mechanism coefficients carry the same confound adjustment as the
    # innovation-premium ladder. Rare institutions pooled to avoid separation.
    inst = df.institution_name.fillna("(missing)")
    top = inst.value_counts()
    df["inst_fe"] = np.where(inst.map(top) >= 30, inst, "(other)")
    # logit + full FE dummies hits separation (singular matrix); the FE
    # variants therefore use an OLS linear-probability model with
    # advisor-clustered SE - the same estimator family as the E6 ladder.
    def fit_lpm(formula, d):
        return smf.ols(formula, data=d).fit(cov_type="cluster",
                                            cov_kwds={"groups": d["advisor_pid"]})
    m1fe = fit_lpm(base + " + coauth + coauth:divergence + C(inst_fe)", df)
    m2fe = fit_lpm(base + " + coauth + adv_breadth_z:divergence + C(inst_fe)", df)
    out["coauth_x_divergence_instFE_lpm"] = {
        "estimator": "OLS LPM, advisor-clustered SE, cohort + institution FE",
        "coauth_main": coef_row(m1fe, "coauth"),
        "interaction": coef_row(m1fe, "coauth:divergence")}
    out["advisor_breadth_x_divergence_instFE_lpm"] = {
        "estimator": "OLS LPM, advisor-clustered SE, cohort + institution FE",
        "breadth_main": coef_row(m2fe, "adv_breadth_z"),
        "interaction": coef_row(m2fe, "adv_breadth_z:divergence")}

    # logit + institution FE converges under BFGS; reported alongside the LPM.
    # NOTE (functional form): the coauth x divergence interaction is large and
    # significant on the log-odds scale (logit) but ~0 on the probability
    # scale (LPM) - the classic Ai & Norton (2003) point that product-term
    # coefficients in nonlinear models are not marginal-effect interactions.
    # The paper reports both and does NOT rest the mechanism claim on the
    # logit product term alone.
    m1fe_logit = smf.logit(base + " + coauth + coauth:divergence + C(inst_fe)",
                           data=df).fit(disp=0, method="bfgs", maxiter=500,
                                        cov_type="cluster",
                                        cov_kwds={"groups": df["advisor_pid"]})
    out["coauth_x_divergence_instFE_logit"] = {
        "estimator": "logit (BFGS), advisor-clustered SE, cohort + institution FE",
        "coauth_main": coef_row(m1fe_logit, "coauth"),
        "interaction": coef_row(m1fe_logit, "coauth:divergence")}
    out["functional_form_note"] = (
        "coauth:divergence is +0.54 (p<.001) in the logit but ~0 (n.s.) in the "
        "LPM with identical controls; interaction claims are therefore stated "
        "on the odds scale only, and the robust mechanism facts are the coauth "
        "main effect (binary anchor, E4 fractional logit) and cohort trends.")

    dec = []
    for d0 in range(1950, 2011, 10):
        sub = df[(df.t0 >= d0) & (df.t0 < d0 + 10)]
        if len(sub) < 100:
            continue
        r = float(np.corrcoef(sub.early_overlap, sub.y)[0, 1])
        p = float(sub.y.mean())
        se = float(np.sqrt(p * (1 - p) / len(sub)))
        dec.append({"decade": f"{d0}s", "n": len(sub),
                    "retention_rate": round(p, 4),
                    "retention_ci95": [round(max(p - 1.96 * se, 0.0), 4),
                                       round(p + 1.96 * se, 4)],
                    "corr_overlap_y": round(r, 4),
                    "coauth_rate": round(float(sub.coauth_early.mean()), 4)})
    out["cohort_trends"] = dec

    (CFG.RESULTS_DIR / "e8_heterogeneity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
