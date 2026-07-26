#!/usr/bin/env python3
"""T2.2b, step 1: the lineage table the strict_lineage graph contract consumes.

The graph arm never sees the multi-generation genealogy of Figure 2. Reviewer B1
is right that this makes "negative control for graph learning" rest on a graph
that cannot consume the structure the paper measures. T2.2a answered the
question on the tabular side; this prepares the graph side.

Two things are precomputed here, on CPU, so the GPU leg needs neither the
1.2 GB OpenAlex cache nor the resolver tables:

  grand_adv_pid       the AFT parent of the focal student's advisor, from
                      pairs_resolved_<field>.parquet, the same map
                      r23_genealogy_tabular.build_genealogy reads
  grand_adv_concepts  that person's top-10 concepts computed from works dated
                      at or before the FOCAL STUDENT's t0+5, not the
                      grand-advisor's own career horizon

The second point is the whole temporal argument. A grand-advisor supervises
many people across decades; a profile taken at their career end would carry
information from after the focal student's freeze date. Every profile here is
keyed to one focal cohort, so the same person yields different profiles for
different cohorts, exactly as build_graph_v2(contract="strict") already does
for direct advisors.

The ancestry relation itself carries no year stamp. Its temporal compliance
rests on the data-model assumption that a person's own doctoral advisor
predates them, not on a checkable constraint, which is the same standing that
student --at--> institution has in the frozen graph. Recorded in the manifest
and printed.

  python code/r31_lineage_table.py
  python code/r31_lineage_table.py --field chemistry

Output: data/supplement/lineage_<field>.parquet
        data/supplement/lineage_manifest.json
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "robustness" / "openalex_cache"
OUT = ROOT / "data" / "supplement"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
EARLY, TOPK = 5, 10          # r23_genealogy_tabular.EARLY, .TOPK
# The cache stores every concept scoring at least 0.2 (r5_fetch_author_works
# STORE_MIN_SCORE). The builder's own view of a work's concepts, the one that
# produced early_concepts and adv_profile, is the first three scoring at least
# 0.3 (build_neuro_dataset.CONCEPT_MIN_SCORE and the [:3] collapse). The
# grand-advisor profile uses the builder's rule, so the graph's two concept
# relations are built on one vocabulary rather than two.
#
# r23_genealogy_tabular does NOT apply this filter; it takes every cached
# concept. The two arms therefore read the same genealogy, the same parent map
# and the same cache, but not the same concept view, and their Jaccard columns
# are not comparable to these edges.
CONCEPT_MIN_SCORE, WORK_TOP_CONCEPTS = 0.3, 3


def bare(x):
    return x.rsplit("/", 1)[-1] if isinstance(x, str) else x


def parent_map(field):
    """student_pid -> (advisor_pid, advisor_openalex_id), first occurrence wins.

    Byte-for-byte the loop in r23_genealogy_tabular.build_genealogy, so the
    graph arm and the tabular arm read the same genealogy.
    """
    pr = pd.read_parquet(ROOT / "data" / f"pairs_resolved_{field}.parquet")
    par, par_aid = {}, {}
    for sp, ap, aoa in zip(pr.student_pid, pr.advisor_pid, pr.adv_openalex_id):
        if pd.notna(sp) and pd.notna(ap) and sp not in par:
            par[sp], par_aid[sp] = ap, bare(aoa)
    return par, par_aid


def load_cache(aids):
    """One pass over the 91 cache shards for the union of wanted author ids."""
    want, got = {a for a in aids if isinstance(a, str)}, {}
    files = sorted(glob.glob(str(CACHE / "*.parquet")))
    if not files:
        raise SystemExit(
            f"no OpenAlex cache shards under {CACHE}. This script cannot "
            f"invent grand-advisor works; run it in the full repository.")
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=["aid", "works_json"])
        hit = df[df.aid.isin(want)]
        for aid, wj in zip(hit.aid, hit.works_json):
            try:
                works = json.loads(wj)
            except Exception:
                continue
            got[aid] = [
                (w.get("year"),
                 [c[0] for c in (w.get("concepts") or [])
                  if c[1] >= CONCEPT_MIN_SCORE][:WORK_TOP_CONCEPTS])
                for w in works if w.get("year")]
        if len(got) == len(want):
            print(f"  cache: all {len(want)} authors found after {i + 1} shards")
            break
    else:
        print(f"  cache: {len(got)} of {len(want)} authors found "
              f"after all {len(files)} shards")
    return got


def profile_asof(works, cutoff):
    """r23_genealogy_tabular.profile_asof verbatim, including its guard."""
    sel = [(y, cs) for (y, cs) in works if y is not None and y <= cutoff]
    assert all(y <= cutoff for y, _ in sel), "leakage guard: work after cutoff"
    if not sel:
        return 0, 0, [], None
    counts = {}
    for _, cs in sel:
        for c in cs:
            counts[c] = counts.get(c, 0) + 1
    top = [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOPK]]
    return len(sel), len(counts), top, min(y for y, _ in sel)


def build(field, cache=None):
    ds = pd.read_parquet(ROOT / "data" / f"clean_dataset_{field}.parquet")
    ds = ds[ds.early_concepts.apply(len) > 0].reset_index(drop=True)
    par, par_aid = parent_map(field)

    gpid = [par.get(a) for a in ds.advisor_pid]
    gaid = [par_aid.get(a) for a in ds.advisor_pid]
    if cache is None:
        cache = load_cache({a for a in gaid if isinstance(a, str)})

    prod, breadth, age, concepts, has = [], [], [], [], []
    late = 0
    for aid, t0 in zip(gaid, ds.t0):
        works = cache.get(aid) if isinstance(aid, str) else None
        if not works:
            prod.append(0); breadth.append(0); age.append(0)
            concepts.append([]); has.append(0)
            continue
        cutoff = int(t0) + EARLY
        late += sum(1 for y, _ in works if y is not None and y > cutoff)
        p, b, top, first = profile_asof(works, cutoff)
        prod.append(p); breadth.append(b)
        age.append(max(0, int(t0) - first) if first else 0)
        concepts.append(top)
        has.append(1 if p > 0 else 0)

    g = pd.DataFrame({
        "student_pid": ds.student_pid.values,
        "t0": ds.t0.values.astype(int),
        "advisor_pid": ds.advisor_pid.values,
        "grand_adv_pid": gpid,
        "grand_adv_aid": gaid,
        "grand_adv_early_prod": prod,
        "grand_adv_early_breadth": breadth,
        "grand_adv_career_age_at_t0": age,
        "grand_adv_concepts": concepts,
        "has_grandadv_works": has,
    })

    # The guard that matters. profile_asof already asserts per row; this is the
    # table-level restatement, and it is what the manifest reports.
    for cs, t0, aid in zip(g.grand_adv_concepts, g.t0, g.grand_adv_aid):
        if not isinstance(cs, list):
            raise SystemExit(f"{field}: grand_adv_concepts is not a list for {aid}")
    self_loops = int(sum(1 for a, gp in zip(g.advisor_pid, g.grand_adv_pid)
                         if gp is not None and gp == a))

    meta = {
        "field": field,
        "n_rows": int(len(g)),
        # Two coverages, because they answer different questions and differ.
        # The ancestry edge needs only a parent pid; the grand-advisor concept
        # and feature channels additionally need a resolved OpenAlex id.
        # r23_genealogy_tabular reports the second one as its coverage.
        "grand_advisor_coverage": round(float(pd.notna(g.grand_adv_pid).mean()), 4),
        "grand_advisor_aid_coverage": round(
            float(pd.Series(gaid).notna().mean()), 4),
        "grand_advisor_with_cached_works": round(float(np.mean(has)), 4),
        "n_distinct_grand_advisors": int(pd.Series(gpid).dropna().nunique()),
        "n_lineage_cohort_keys": int(
            len({(gp, int(t)) for gp, t in zip(g.grand_adv_pid, g.t0)
                 if gp is not None})),
        "advisor_is_own_grand_advisor_rows": self_loops,
        "works_excluded_as_after_focal_t0_plus5": int(late),
        "concept_window": "works dated <= focal student t0+5",
        "topk": TOPK,
        "concept_min_score": CONCEPT_MIN_SCORE,
        "work_top_concepts": WORK_TOP_CONCEPTS,
        "concept_view": ("the builder's rule, first 3 concepts scoring >= 0.3 "
                         "per work, so these edges share a vocabulary with "
                         "adv_profile; r23_genealogy_tabular uses the cache's "
                         "unfiltered list and is not comparable"),
        "ancestry_edge_has_no_year_stamp": True,
        "ancestry_note": (
            "the advisor--advisor relation is predetermined in the same sense "
            "as advisor--advises--student; its compliance with the time "
            "contract rests on a data-model assumption, not a checkable "
            "constraint, exactly as student--at--institution does"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    g.to_parquet(OUT / f"lineage_{field}.parquet", index=False)
    print(f"[{field:10}] rows {meta['n_rows']:6d}  coverage "
          f"{meta['grand_advisor_coverage']:.4f}  with works "
          f"{meta['grand_advisor_with_cached_works']:.4f}  "
          f"cohort keys {meta['n_lineage_cohort_keys']:6d}  "
          f"works dropped as post-t0+5 {meta['works_excluded_as_after_focal_t0_plus5']}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default=None, choices=FIELDS)
    args = ap.parse_args()
    fields = [args.field] if args.field else FIELDS

    # One cache pass for the union across every field asked for; the shards are
    # 1.2 GB and rereading them per field is the slowest part of this script.
    wanted = set()
    for f in fields:
        _, par_aid = parent_map(f)
        ds = pd.read_parquet(ROOT / "data" / f"clean_dataset_{f}.parquet",
                             columns=["advisor_pid", "early_concepts"])
        ds = ds[ds.early_concepts.apply(len) > 0]
        wanted |= {par_aid.get(a) for a in ds.advisor_pid
                   if isinstance(par_aid.get(a), str)}
    print(f"cache pass for {len(wanted)} distinct grand-advisor author ids")
    cache = load_cache(wanted)

    metas = {f: build(f, cache) for f in fields}
    mpath = OUT / "lineage_manifest.json"
    prev = json.loads(mpath.read_text()) if mpath.exists() else {}
    prev.update(metas)
    mpath.write_text(json.dumps(prev, indent=2))
    print(f"\n-> {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
