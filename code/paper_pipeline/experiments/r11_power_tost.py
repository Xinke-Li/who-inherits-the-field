"""R11 - power analysis for the negative-control claim (task B7).

POST-HOC ANALYSIS of the existing bootstrap artifacts; no model is retrained.
Question: where the paper reports "no graph gain", how large a gain could the
design actually have detected, and can any cell claim EQUIVALENCE rather than
merely a failure to reject?

Per (discipline, architecture), from the stored student-level pooled bootstrap
of e12_corrected_vs_m5.json (gap vs M5'):
  sigma  the bootstrap SD of the pooled gap, recovered from the stored 95
         percent interval as (hi - lo) / 3.92 (normal approximation);
  MDE    the minimum detectable effect at two-sided alpha 0.05 and 80 percent
         power, (1.96 + 0.84) * sigma = 2.80 * sigma;
  TOST   equivalence at margin m iff the 90 percent interval
         (mean +/- 1.645 * sigma) lies inside (-m, +m), evaluated at
         m in {0.01, 0.02, 0.03};
  cell verdict: "exceeds" (the eq. 2 verdict), "equivalent at m", or
         "not identifiable at this design" (neither exceeds nor equivalent at
         the largest margin).

Margin justification recorded in the output: the M1-to-best-tabular gap per
discipline, the size of a whole ladder step, which the margins bracket from
below.

Output: results/robustness/power_tost.json
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
MODELS = ["hgt", "hgt_tuned", "rgcn", "gat_cohort_time"]
MARGINS = [0.01, 0.02, 0.03]


def main():
    out = {"experiment": "R11_power_tost",
           "design": ("sigma from the stored student-level pooled bootstrap "
                      "CI of e12_corrected_vs_m5.json; MDE = 2.80 sigma "
                      "(alpha 0.05 two-sided, power 0.80); TOST equivalent "
                      "at m iff mean +/- 1.645 sigma inside (-m, +m)"),
           "margins": MARGINS, "fields": {}}
    for f in FIELDS:
        d = json.loads((REPO / "results" / f"results_{f}" /
                        "e12_corrected_vs_m5.json").read_text())
        e1 = json.loads((REPO / "results" / f"results_{f}" /
                         "e1_baselines.json").read_text())["summary"]
        m1 = e1["M1_logit_overlap"]["auc_pr"]["mean"]
        best_tab = max(e1[m]["auc_pr"]["mean"]
                       for m in ("M2_logit_tabular", "M3_gbdt_tabular"))
        rows = {}
        for mname in MODELS:
            m = d["models"][mname]
            lo, hi = m["bootstrap"]["vs_M5prime"]["pooled_ci95"]
            mean = m["delta_vs_M5prime"]
            sigma = (hi - lo) / 3.92
            mde = 2.80 * sigma
            lo90, hi90 = mean - 1.645 * sigma, mean + 1.645 * sigma
            eq_at = next((mm for mm in MARGINS
                          if -mm < lo90 and hi90 < mm), None)
            exceeds = m.get("exceeds_fair", False)
            verdict = ("exceeds" if exceeds else
                       f"equivalent at {eq_at}" if eq_at is not None else
                       "not identifiable at this design")
            rows[mname] = {"delta_vs_M5prime": mean, "sigma": round(sigma, 4),
                           "MDE_80pct": round(mde, 4),
                           "ci90": [round(lo90, 4), round(hi90, 4)],
                           "equivalent_at_margin": eq_at, "verdict": verdict}
        out["fields"][f] = {
            "margin_justification_M1_to_best_tabular": round(best_tab - m1, 4),
            "models": rows}
    p = REPO / "results" / "robustness" / "power_tost.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"{'field':10} {'model':16} {'delta':>8} {'MDE':>7} "
          f"{'ci90':>20} verdict")
    for f in FIELDS:
        for mname in MODELS:
            r = out["fields"][f]["models"][mname]
            print(f"{f:10} {mname:16} {r['delta_vs_M5prime']:>+8.4f} "
                  f"{r['MDE_80pct']:>7.4f} {str(r['ci90']):>20} {r['verdict']}")
        print(f"{'':10} margin ref (best_tab - M1): "
              f"{out['fields'][f]['margin_justification_M1_to_best_tabular']}")


if __name__ == "__main__":
    main()
