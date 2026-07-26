"""R12 - measured snapshot-retroactivity bound from the r5 cache (task B4).

Design restated from e18_vocab_retro.py (source of the eligibility rule and
the upper-bound framing), executed on the r5 author cache instead of the
absent works store. A concept is ELIGIBLE at a student's feature freeze iff
its earliest corpus appearance (first year any work in the full 89,374-author
cache carries it at the builder view) is <= t0+5. The student early profile
and the advisor profile are rebuilt on eligible concepts only, the overlap
scalar is recomputed, and the single-feature AUC-PR (equivalent to M1: the
logistic is a monotone transform of the scalar) is compared restricted vs
unrestricted WITHIN the rebuilt family, label untouched. This BOUNDS the
descriptor-timing channel; it cannot un-apply the 2026 tagging model, only
drop concepts that provably did not exist as tags by t0+5.

Output: results/robustness/vocab_bound.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

from r6_topk_sweep import (BUILDER_MIN_SCORE, CACHE_DIR, builder_view,
                           load_cache_for, profile_counts, topk_set)
from e14_self_persistence import jaccard

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from e12_corrected_aggregation import fast_auc_pr

REPO = Path(__file__).resolve().parents[3]
FIELDS = ["chemistry", "econ", "math", "neuro", "physics"]
K = 10


def earliest_appearance() -> dict:
    """One streaming pass over every cache shard: concept -> first year."""
    first = {}
    shards = sorted(CACHE_DIR.glob("works_*.parquet"))
    for i, p in enumerate(shards):
        df = pd.read_parquet(p)
        for wj in df.works_json:
            for w in json.loads(wj):
                y = w.get("year")
                if y is None:
                    continue
                for c, s in w["concepts"]:
                    if s >= BUILDER_MIN_SCORE:
                        if c not in first or y < first[c]:
                            first[c] = y
        if (i + 1) % 20 == 0:
            print(f"[r12] scanned {i + 1}/{len(shards)} shards, "
                  f"{len(first)} concepts", flush=True)
    return first


def run_field(field, first):
    df = pd.read_parquet(REPO / "data" / f"clean_dataset_{field}.parquet")
    df = df[df.early_concepts.apply(len) > 0].reset_index(drop=True)
    q1, q2 = np.quantile(df.t0, C.SPLIT_QUANTILES)
    is_test = (df.t0 > int(q2)).values

    aids = set(df.st_openalex_id) | set(df.adv_openalex_id)
    cache = load_cache_for(aids)
    view = {a: builder_view(cache[a], BUILDER_MIN_SCORE) for a in aids if a in cache}
    del cache

    eo_full = np.empty(len(df)); eo_res = np.empty(len(df))
    y = np.empty(len(df), int)
    n_dropped_student, n_profiles = 0, 0
    for i, r in enumerate(df.itertuples()):
        sw, aw = view[r.st_openalex_id], view[r.adv_openalex_id]
        cut = r.t0 + C.EARLY_YEARS
        e_cnt, _ = profile_counts(sw, r.t0, cut)
        l_cnt, _ = profile_counts(sw, cut + 1, r.t0 + C.LATE_YEARS)
        a_cnt, _ = profile_counts(aw, None, cut)
        early, late, advp = topk_set(e_cnt, K), topk_set(l_cnt, K), topk_set(a_cnt, K)
        y[i] = int(jaccard(late, advp) > C.JACCARD_THETA)
        eo_full[i] = jaccard(early, advp)
        # restricted: rebuild top-10 over eligible concepts only
        e_ok = {c: n for c, n in e_cnt.items() if first.get(c, 9999) <= cut}
        a_ok = {c: n for c, n in a_cnt.items() if first.get(c, 9999) <= cut}
        from collections import Counter
        early_r = topk_set(Counter(e_ok), K)
        advp_r = topk_set(Counter(a_ok), K)
        eo_res[i] = jaccard(early_r, advp_r)
        n_profiles += 2
        n_dropped_student += len(e_cnt) - len(e_ok)
    auc_full = fast_auc_pr(y[is_test], eo_full[is_test])
    auc_res = fast_auc_pr(y[is_test], eo_res[is_test])
    changed = float(np.mean(np.abs(eo_full - eo_res) > 1e-9))
    return {"n": len(df), "n_test": int(is_test.sum()),
            "auc_pr_unrestricted": round(auc_full, 4),
            "auc_pr_restricted": round(auc_res, 4),
            "shift": round(auc_res - auc_full, 4),
            "rows_with_changed_feature": round(changed, 4),
            "corr_restricted_vs_full": round(float(np.corrcoef(eo_full, eo_res)[0, 1]), 4)}


def main():
    print("[r12] computing earliest corpus appearance over the full cache",
          flush=True)
    first = earliest_appearance()
    out = {"experiment": "R12_vocab_bound",
           "design": ("e18 eligibility rule on the r5 cache: concept eligible "
                      "iff earliest corpus appearance <= t0+5; single-feature "
                      "AUC-PR (M1-equivalent: raw scalar as score), rebuilt "
                      "family, label untouched; an upper bound, the 2026 "
                      "tagging model itself cannot be un-applied"),
           "n_concepts": len(first),
           "fields": {}}
    for f in FIELDS:
        print(f"[r12] {f}", flush=True)
        out["fields"][f] = run_field(f, first)
        print(f"[r12] {f}: full {out['fields'][f]['auc_pr_unrestricted']} "
              f"restricted {out['fields'][f]['auc_pr_restricted']} "
              f"shift {out['fields'][f]['shift']:+.4f}", flush=True)
    (REPO / "results" / "robustness" / "vocab_bound.json").write_text(
        json.dumps(out, indent=2))
    print("[r12] written", flush=True)


if __name__ == "__main__":
    main()
