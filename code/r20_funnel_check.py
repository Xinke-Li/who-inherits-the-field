#!/usr/bin/env python3
"""T2.6: back the funnel caption's reproduction claim with a result file.

The neuro window-filter counter cannot be rebuilt from the release, because it
needs per-author publication years and work counts that live only in that build's
works store. The two counters that ARE recorded can be rebuilt from the shipped
resolved-pairs table, and the caption says so; this script is what makes that
claim checkable.

  python code/r20_funnel_check.py
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "T2_6_funnel_check"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "task": "T2.6",
        "claim": ("the two recorded neuroscience funnel counters reproduce from "
                  "the shipped resolved-pairs table"),
        "not_reproducible": {
            "counters": ["both_in_works_store", "survive_window_filters",
                         "drop_breakdown.no_years", "drop_breakdown.adv_no_years",
                         "drop_breakdown.too_recent", "drop_breakdown.implausible_t0",
                         "drop_breakdown.span_gt_max", "drop_breakdown.sparse_windows"],
            "reason": ("each needs per-author publication years and work counts; "
                       "pairs_resolved_*.parquet carries resolver output only "
                       "(OpenAlex ids, match method, anchor hits) and no "
                       "publication records"),
        },
        "fields": {},
    }

    for f in FIELDS:
        pairs = ROOT / "data" / f"pairs_resolved_{f}.parquet"
        funnel = ROOT / "data" / f"funnel_{f}.json"
        if not (pairs.exists() and funnel.exists()):
            continue
        df = pd.read_parquet(pairs)
        rec = json.load(open(funnel))
        raw = len(df)
        both = int((df.adv_openalex_id.notna() & df.st_openalex_id.notna()).sum())
        h = hashlib.sha256(pairs.read_bytes()).hexdigest()
        report["fields"][f] = {
            "source_file": f"data/pairs_resolved_{f}.parquet",
            "source_sha256": h,
            "columns_present": sorted(df.columns.tolist()),
            "raw_pairs": {"recorded": rec["raw_pairs"], "recomputed": raw,
                          "match": raw == rec["raw_pairs"]},
            "both_resolvable": {"recorded": rec["both_resolvable"],
                                "recomputed": both,
                                "match": both == rec["both_resolvable"]},
        }
        print(f"[T2.6] {f:10} raw {raw:>7} vs {rec['raw_pairs']:>7} "
              f"{'OK' if raw == rec['raw_pairs'] else 'MISMATCH'}   "
              f"both_resolvable {both:>7} vs {rec['both_resolvable']:>7} "
              f"{'OK' if both == rec['both_resolvable'] else 'MISMATCH'}")

    report["all_match"] = all(
        v["raw_pairs"]["match"] and v["both_resolvable"]["match"]
        for v in report["fields"].values())
    json.dump(report, open(OUT / "funnel_reproduction.json", "w"), indent=2)
    print(f"\n[T2.6] all fields match: {report['all_match']}")
    print(f"[T2.6] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
