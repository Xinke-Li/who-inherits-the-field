#!/usr/bin/env python3
"""T2.8: does the cross-discipline co-authorship ordering survive stratification?

Reviewer C1 objects that the ordering, with economics lowest, assumes selection
bias runs the same way in all five disciplines and offers no evidence. Resolution
selects on publishing, so the cheapest defensible test stratifies each discipline
by the student's early-window productivity and recomputes the early-window
co-authorship rate inside each stratum. If economics stays lowest within strata,
the ordering is not an artifact of differing productivity distributions.

Terciles are computed within discipline, so each stratum holds the same share of
that discipline rather than a common absolute cut.

Output: results/revision/T2_8_coverage_sensitivity/strata.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "T2_8_coverage_sensitivity"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, pooled = {}, {}
    for f in FIELDS:
        df = pd.read_parquet(ROOT / "data" / f"clean_dataset_{f}.parquet")
        rate = df.coauth_early.astype(float)
        pooled[f] = round(float(rate.mean()), 4)
        q = df.early_prod.quantile([1 / 3, 2 / 3]).values
        lab = np.where(df.early_prod <= q[0], "low",
                       np.where(df.early_prod <= q[1], "mid", "high"))
        rows[f] = {}
        for s in ("low", "mid", "high"):
            m = lab == s
            rows[f][s] = {
                "n": int(m.sum()),
                "coauth_early_rate": round(float(rate[m].mean()), 4),
                "early_prod_range": [float(df.early_prod[m].min()),
                                     float(df.early_prod[m].max())],
            }
        print(f"[T2.8] {f:10} pooled {pooled[f]:.4f}  "
              + "  ".join(f"{s} {rows[f][s]['coauth_early_rate']:.4f} "
                          f"(n={rows[f][s]['n']})" for s in ("low", "mid", "high")),
              flush=True)

    # does economics stay lowest inside every stratum?
    holds = {}
    for s in ("low", "mid", "high"):
        vals = {f: rows[f][s]["coauth_early_rate"] for f in FIELDS}
        lo = min(vals, key=vals.get)
        holds[s] = {"lowest": lo, "econ_is_lowest": lo == "econ",
                    "rates": vals,
                    "econ_margin_to_next": round(
                        sorted(vals.values())[1] - vals["econ"], 4)}
        print(f"[T2.8] stratum {s:5} lowest={lo:10} econ_lowest={lo == 'econ'} "
              f"margin={holds[s]['econ_margin_to_next']:+.4f}", flush=True)

    all_hold = all(v["econ_is_lowest"] for v in holds.values())
    out = {
        "task": "T2.8", "reviewer_point": "C1",
        "question": ("does the early co-authorship ordering, economics lowest, "
                     "hold within early-productivity strata rather than only in "
                     "the pooled sample?"),
        "stratifier": "early_prod terciles, computed within discipline",
        "pooled_rates": pooled,
        "strata": rows,
        "ordering_within_strata": holds,
        "econ_lowest_in_all_strata": all_hold,
        "reading": ("the ordering survives stratification and is not an artifact "
                    "of differing productivity distributions"
                    if all_hold else
                    "the ordering does not survive stratification; it is reported "
                    "as an exploratory observation rather than a finding"),
    }
    (OUT / "strata.json").write_text(json.dumps(out, indent=2),
                                     encoding="utf-8", newline="\n")
    print(f"\n[T2.8] econ lowest in all three strata: {all_hold}")
    print(f"[T2.8] -> {OUT / 'strata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
