#!/usr/bin/env python3
"""T2.8, second variable: does the co-authorship ordering survive stratification
on index coverage rather than on productivity?

WHAT r30 DID AND WHY IT IS NOT THIS. r30 stratified each discipline by the
student's early-window productivity. That tests whether the ordering is an
artifact of differing productivity distributions. The reviewer asked a different
question: whether it is an artifact of differing INDEX COVERAGE, since resolution
selects on being findable in OpenAlex and the five genealogies are not indexed
equally well. Productivity terciles cannot answer that.

THE VARIABLE, AND WHY IT IS NOT A COLUMN. st_id_coverage is a single scalar per
discipline in data/funnel_<field>.json, the share of that discipline's raw pairs
whose student resolved to an OpenAlex author. One number per discipline cannot
stratify within a discipline. The per-row form of the same quantity is the
resolution rate of the student's own institution, computed from the frozen
resolved-pairs table: for each institution, the share of its raw pairs whose
student resolved. Every modeled student inherits their institution's rate. This
is the reviewer's second named variable, institutional index density, and it is
also the decomposition of the first: averaging it over raw pairs returns
st_id_coverage exactly.

Terciles are computed within discipline, as in r30, so each stratum holds the
same share of that discipline rather than a common absolute cut. Rows whose
institution appears in no resolved pair carry no density and are excluded and
counted; institutions with few raw pairs give a noisy rate, so the whole
comparison is repeated restricted to institutions with at least ten raw pairs.

Reads only frozen artifacts. Nothing is written outside results/revision/.

Output: results/revision/T2_8_coverage_sensitivity/coverage_strata.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "T2_8_coverage_sensitivity"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
MIN_RAW_PAIRS = 10          # the sensitivity restriction, not the primary cut


def density_by_institution(field):
    """Institution -> share of its raw pairs whose student resolved."""
    pr = pd.read_parquet(ROOT / "data" / f"pairs_resolved_{field}.parquet",
                         columns=["institution_name", "st_openalex_id"])
    pr["resolved"] = (pr.st_openalex_id.notna()
                      & (pr.st_openalex_id.astype(str) != "")
                      & (pr.st_openalex_id.astype(str) != "None"))
    g = pr.groupby("institution_name").agg(raw=("resolved", "size"),
                                           res=("resolved", "sum"))
    g["density"] = g.res / g.raw
    return g, float(pr.resolved.mean())


def strata_for(field, min_raw):
    g, overall = density_by_institution(field)
    keep = g[g.raw >= min_raw] if min_raw else g
    df = pd.read_parquet(ROOT / "data" / f"clean_dataset_{field}.parquet",
                         columns=["institution_name", "coauth_early"])
    j = df.merge(keep["density"], left_on="institution_name",
                 right_index=True, how="left")
    n_all = len(j)
    j = j[j.density.notna()].reset_index(drop=True)
    q = j.density.quantile([1 / 3, 2 / 3]).values
    lab = np.where(j.density <= q[0], "low",
                   np.where(j.density <= q[1], "mid", "high"))
    rate = j.coauth_early.astype(float)
    out = {"n_modeled": n_all, "n_with_density": int(len(j)),
           "n_without_density": int(n_all - len(j)),
           "overall_student_resolution_rate": round(overall, 4),
           "tercile_cuts": [round(float(q[0]), 4), round(float(q[1]), 4)],
           "strata": {}}
    for s in ("low", "mid", "high"):
        m = lab == s
        out["strata"][s] = {
            "n": int(m.sum()),
            "coauth_early_rate": round(float(rate[m].mean()), 4),
            "density_range": [round(float(j.density[m].min()), 4),
                              round(float(j.density[m].max()), 4)],
        }
    return out


def compare(per_field):
    holds = {}
    for s in ("low", "mid", "high"):
        vals = {f: per_field[f]["strata"][s]["coauth_early_rate"] for f in FIELDS}
        lo = min(vals, key=vals.get)
        ordered = sorted(vals.values())
        holds[s] = {
            "lowest": lo, "econ_is_lowest": lo == "econ", "rates": vals,
            "econ_margin_to_next": round(ordered[1] - vals["econ"], 4),
            "n_by_field": {f: per_field[f]["strata"][s]["n"] for f in FIELDS}}
    return holds


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"task": "T2.8",
           "question": ("does the cross-discipline early co-authorship ordering, "
                        "economics lowest, survive stratification on index "
                        "coverage rather than on productivity?"),
           "variable": ("institutional index density: the share of an "
                        "institution's raw pairs whose student resolved to an "
                        "OpenAlex author, from the frozen resolved-pairs table"),
           "why_not_st_id_coverage": ("st_id_coverage is one scalar per "
                                      "discipline in data/funnel_<field>.json "
                                      "and cannot stratify within a discipline; "
                                      "this is its per-row decomposition"),
           "arms": {}}
    for tag, min_raw in (("all_institutions", 0),
                         (f"institutions_with_at_least_{MIN_RAW_PAIRS}_raw_pairs",
                          MIN_RAW_PAIRS)):
        per_field = {f: strata_for(f, min_raw) for f in FIELDS}
        holds = compare(per_field)
        res["arms"][tag] = {"per_field": per_field, "by_stratum": holds,
                            "econ_lowest_in_all_strata": all(
                                v["econ_is_lowest"] for v in holds.values())}
        print(f"\n=== {tag} ===")
        for f in FIELDS:
            d = per_field[f]
            print(f"[T2.8c] {f:10} cuts {d['tercile_cuts']}  "
                  + "  ".join(f"{s} {d['strata'][s]['coauth_early_rate']:.4f} "
                              f"(n={d['strata'][s]['n']})"
                              for s in ("low", "mid", "high"))
                  + f"  no-density {d['n_without_density']}")
        for s in ("low", "mid", "high"):
            h = holds[s]
            print(f"[T2.8c] stratum {s:5} lowest={h['lowest']:10} "
                  f"econ_lowest={h['econ_is_lowest']} "
                  f"margin={h['econ_margin_to_next']:+.4f}")
        print(f"[T2.8c] econ lowest in all three strata: "
              f"{res['arms'][tag]['econ_lowest_in_all_strata']}")
    p = OUT / "coverage_strata.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
