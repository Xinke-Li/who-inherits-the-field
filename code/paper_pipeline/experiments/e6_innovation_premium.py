"""E6 - Innovation Premium, redone with fixed effects (replaces rho = -0.109).

Old flaws: full-career h-index outcome (confounded by cohort/career length),
raw Spearman with no controls, single Node2Vec run.

New design:
  outcome  : late_cite_pct_mean  (within-publication-year citation percentile of the
             student's late-window works, from outcomes.parquet)  [primary]
             h_index_full_career [reported as legacy comparison only]
  treatment: divergence = 1 - early_overlap  (concept-space divergence; the
             embedding-space version comes from e7 and is merged if present)
  controls : early_prod, early_breadth, adv_early_prod, adv_early_breadth,
             adv_career_age_at_t0, coauth_early
  FE       : t0 cohort (5-year bins) + institution (top-K, rest pooled)
  SE       : clustered by advisor
  extras   : divergence quintile bins -> non-linearity check (inverted-U per
             Lienard et al. 2018)

Pre-registered rule: 'Innovation Premium' survives only if the divergence
coefficient is significant WITH the full FE set; otherwise report that the raw
correlation is explained by cohort/institution composition.
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

TOP_K_INST = 30
CONTROLS = ["early_prod", "early_breadth", "adv_early_prod",
            "adv_early_breadth", "adv_career_age_at_t0"]


def prep():
    df = D.load_dataset()
    oc = pd.read_parquet(CFG.OUTCOMES)
    df = df.merge(oc, on=["student_pid", "st_openalex_id"], how="left")
    df = df[df.late_cite_pct_mean.notna()].copy()

    df["divergence"] = 1.0 - df.early_overlap
    df["cohort"] = (df.t0 // 5 * 5).astype(int).astype(str)
    top_inst = df.institution_name.value_counts().head(TOP_K_INST).index
    df["inst_fe"] = np.where(df.institution_name.isin(top_inst),
                             df.institution_name, "OTHER")
    df["coauth"] = df.coauth_early.astype(int)
    for c in CONTROLS + ["divergence"]:
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    # merge Node2Vec divergence if e7 has run
    n2v = CFG.RESULTS_DIR / "e7_divergence.parquet"
    if n2v.exists():
        df = df.merge(pd.read_parquet(n2v), on="student_pid", how="left")
        # standardize like every other regressor, so spec-4's beta is per-SD and
        # directly comparable to spec 1-3 (raw-scale betas are NOT comparable)
        m, s = df.n2v_divergence.mean(), df.n2v_divergence.std()
        df["n2v_divergence"] = (df.n2v_divergence - m) / s
        print("[e6] merged Node2Vec structural divergence from e7 (z-scored)")
    return df


def fit_ladder(df, outcome):
    """Specification ladder: raw -> +controls -> +FE. Cluster SE by advisor."""
    ctrl = " + ".join(CONTROLS + ["coauth"])
    specs = {
        "1_raw": f"{outcome} ~ divergence",
        "2_controls": f"{outcome} ~ divergence + {ctrl}",
        "3_controls_FE": f"{outcome} ~ divergence + {ctrl} + C(cohort) + C(inst_fe)",
    }
    if "n2v_divergence" in df.columns and df.n2v_divergence.notna().any():
        specs["4_structural_FE"] = (f"{outcome} ~ n2v_divergence + {ctrl}"
                                    " + C(cohort) + C(inst_fe)")
    rows = {}
    base_vars = [outcome, "divergence", "coauth", "cohort", "inst_fe",
                 "advisor_pid"] + CONTROLS
    for name, f in specs.items():
        need = base_vars + (["n2v_divergence"] if "n2v_divergence" in f else [])
        d = df.dropna(subset=[v for v in need if v in df.columns])
        m = smf.ols(f, data=d).fit(cov_type="cluster",
                                   cov_kwds={"groups": d["advisor_pid"]})
        key = "n2v_divergence" if "n2v_divergence" in f else "divergence"
        rows[name] = {"coef": round(float(m.params[key]), 4),
                      "se": round(float(m.bse[key]), 4),
                      "p": round(float(m.pvalues[key]), 5),
                      "n": int(m.nobs), "r2": round(float(m.rsquared), 4)}
        print(f"[e6] {outcome} | {name:<16} beta={rows[name]['coef']:+.4f} "
              f"(p={rows[name]['p']:.4f}, n={rows[name]['n']})")
    return rows


def nonlinearity(df, outcome):
    """Divergence quintiles -> outcome means (inverted-U check)."""
    q = pd.qcut(df.divergence, 5, duplicates="drop")
    g = df.groupby(q, observed=True)[outcome].agg(["mean", "sem", "count"])
    return {str(k): {"mean": round(v["mean"], 4), "sem": round(v["sem"], 4),
                     "n": int(v["count"])} for k, v in g.iterrows()}


def main():
    df = prep()
    out = {"experiment": "E6_innovation_premium", "n": len(df),
           "primary_outcome": {}, "legacy_outcome": {}, "nonlinearity": {}}
    out["primary_outcome"]["late_cite_pct_mean"] = fit_ladder(df, "late_cite_pct_mean")
    out["legacy_outcome"]["h_index_full_career"] = fit_ladder(df, "h_index_full_career")
    out["nonlinearity"]["late_cite_pct_mean"] = nonlinearity(df, "late_cite_pct_mean")

    fe = out["primary_outcome"]["late_cite_pct_mean"]["3_controls_FE"]
    p_fe, b_fe = fe["p"], fe["coef"]
    if p_fe >= 0.05:
        out["verdict"] = "raw correlation not robust to FE - report compositional explanation"
    elif b_fe > 0:
        out["verdict"] = "Innovation Premium SUPPORTED with FE (positive divergence coefficient)"
    else:
        out["verdict"] = ("DIVERGENCE PENALTY: negative and significant with FE - "
                          "premium NOT supported in this discipline; report as discipline difference")

    (CFG.RESULTS_DIR / "e6_innovation_premium.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
    