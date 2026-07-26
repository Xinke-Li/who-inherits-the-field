#!/usr/bin/env python3
"""T2.4 step 1 - the fifth student-side feature, rebuilt from the r5 cache.

WHY THIS EXISTS. The frozen e14 certificate scores a student-only floor on five
pre-window student features. The fifth, early_concentration, is a Herfindahl
index over the student's early-window concept occurrence counts, read from the
builder's per-author works store. That store is in neither the repository nor
the archive, so r1_theta_sweep and r6_topk_sweep run the floor on four features
and say so (r1_theta_sweep.py:26-32). This script supplies the missing column
from the ONE per-author cache the repository does carry, the r5 OpenAlex fetch
at results/robustness/openalex_cache/, which covers every student and every
advisor of all five frozen tables with zero misses.

PROVENANCE, stated because it bounds every reading downstream. The r5 cache is
a 2026-07 live-API fetch, not the frozen 2026 snapshot the builder used. The
number here is therefore a RECONSTRUCTION of the frozen feature, not the frozen
feature. r6_topk_sweep.py:9-25 states the same rule for its own rebuilds. The
calibration that bounds the drift is r48's, which rescores the frozen e14 cell
(theta 0.2, k 10) with this column and compares against the frozen floor.

EXACTNESS. The Herfindahl is e14_self_persistence.herfindahl verbatim, over
e14_self_persistence.profile_counts verbatim, on the builder's concept view
recovered by r6_topk_sweep.builder_view (score >= min_score, first three, in
API descending-score order). No re-implementation: the three functions are
imported from the modules that own them, so this file cannot drift from them.

An author cached with zero works yields an empty counter, and e14's herfindahl
returns 1.0 there. That is the frozen code path, kept rather than special-cased;
the count of such rows is recorded per field.

Outputs
  data/supplement/early_concentration_<field>.parquet
      student_pid (the key) + early_concentration, one row per frozen row,
      at the builder's concept_min_score of 0.3.
  results/revision/T2_4_e14_full5/early_concentration_variants.parquet
      the same index plus the 0.2 and 0.4 min-score variants that r6's
      concept-min-score arm needs. Kept out of the supplement because T2.4
      specifies the supplement table as one column.
  results/revision/T2_4_e14_full5/concentration_manifest.json

Usage:  python code/r47_early_concentration.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline" / "experiments"))

from e14_self_persistence import herfindahl, profile_counts   # noqa: E402
from r6_topk_sweep import builder_view                        # noqa: E402

CACHE_DIR = ROOT / "results" / "robustness" / "openalex_cache"
SUPP = ROOT / "data" / "supplement"
OUT = ROOT / "results" / "revision" / "T2_4_e14_full5"
FIELDS = ["chemistry", "econ", "math", "neuro", "physics"]
EARLY_YEARS = 5                      # config.EARLY_YEARS
BUILDER_MIN_SCORE = 0.3              # r6_topk_sweep.BUILDER_MIN_SCORE
MIN_SCORES = [0.2, 0.3, 0.4]         # builder default plus r6's two variants


def load_cache(aids):
    """aid -> raw works list, only the requested authors held in memory."""
    out = {}
    for p in sorted(CACHE_DIR.glob("works_*.parquet")):
        df = pd.read_parquet(p)
        hit = df[df.aid.isin(aids)]
        for aid, wj in zip(hit.aid, hit.works_json):
            out[aid] = json.loads(wj)
    return out


def main():
    SUPP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"task": "T2.4",
                "feature": "early_concentration",
                "definition": ("Herfindahl index of the student's early-window "
                               "concept occurrence counts, window [t0, t0+5] "
                               "inclusive, on the builder's concept view"),
                "source_cache": "results/robustness/openalex_cache/",
                "source_is_frozen_snapshot": False,
                "source_note": ("r5 live-API fetch of 2026-07; the builder's "
                                "works store is absent from repository and "
                                "archive. Reconstruction, not the frozen "
                                "feature; drift bounded by r48's calibration"),
                "builder_min_score": BUILDER_MIN_SCORE,
                "min_score_variants": MIN_SCORES,
                "imported_verbatim": {
                    "herfindahl": "e14_self_persistence.herfindahl",
                    "profile_counts": "e14_self_persistence.profile_counts",
                    "builder_view": "r6_topk_sweep.builder_view"},
                "fields": {}}

    for f in FIELDS:
        df = pd.read_parquet(ROOT / "data" / f"clean_dataset_{f}.parquet",
                             columns=["student_pid", "st_openalex_id", "t0"])
        cache = load_cache(set(df.st_openalex_id))
        missing = [a for a in df.st_openalex_id if a not in cache]
        assert not missing, f"{f}: {len(missing)} students absent from the r5 cache"

        cols = {}
        for ms in MIN_SCORES:
            vals, empty = [], 0
            for aid, t0 in zip(df.st_openalex_id, df.t0):
                view = builder_view(cache[aid], ms)
                cnt, _ = profile_counts(view, int(t0), int(t0) + EARLY_YEARS)
                if not cnt:
                    empty += 1
                vals.append(herfindahl(cnt))
            cols[ms] = vals
            if ms == BUILDER_MIN_SCORE:
                n_empty = empty

        one = pd.DataFrame({"student_pid": df.student_pid.values,
                            "early_concentration": cols[BUILDER_MIN_SCORE]})
        p_one = SUPP / f"early_concentration_{f}.parquet"
        one.to_parquet(p_one, index=False)

        var = pd.DataFrame({"student_pid": df.student_pid.values,
                            **{f"early_concentration_ms{ms}": cols[ms]
                               for ms in MIN_SCORES}})
        var.to_parquet(OUT / f"early_concentration_variants_{f}.parquet", index=False)

        s = one.early_concentration
        manifest["fields"][f] = {
            "n_rows": int(len(one)),
            "n_students_unique": int(one.student_pid.nunique()),
            "n_empty_early_window": int(n_empty),
            "mean": round(float(s.mean()), 6),
            "median": round(float(s.median()), 6),
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
            "supplement_file": f"data/supplement/early_concentration_{f}.parquet"}
        print(f"[r47] {f}: n={len(one)} mean={s.mean():.4f} "
              f"median={s.median():.4f} empty_window={n_empty}", flush=True)

    (OUT / "concentration_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[r47] manifest ->", OUT / "concentration_manifest.json")


if __name__ == "__main__":
    main()
