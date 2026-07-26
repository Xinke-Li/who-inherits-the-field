#!/usr/bin/env python3
"""T2.6 - the eight neuroscience funnel counters the frozen build never wrote.

WHAT IS MISSING. data/funnel_neuro.json records raw_pairs 120,347, both
resolvable 61,724 and modeled 21,846, and writes ZERO for both_in_works_store,
survive_window_filters and all six drop_breakdown entries. econ carries all
eight. Table 25 of the paper therefore prints "not recorded" in one cell.
r20_funnel_check.py already proves the two recorded counters rebuild from the
shipped resolved-pairs table; it does not and cannot supply the missing eight,
because they need per-author publication years and work counts.

WHAT THIS DOES. Reruns the builder's counting loop and NOTHING ELSE. No table is
built, no parquet is written, no hash moves. stage_table's filter cascade is
reproduced by importing build_neuro_dataset's own profile() and the module's own
window constants, so the thresholds cannot drift from the builder; only the row
accumulation is dropped, since the counters are all this task needs.

PROVENANCE, and it bounds every number here. The builder's works store is in
neither the repository nor the archive. The works are refetched from the live
OpenAlex API of 2026-07, which is not the 2026 frozen snapshot: tags,
disambiguation and records drift. These counters are therefore a RECONSTRUCTION
of the frozen build's counters, not the frozen build's counters. The one honest
way to read them is against the three counters the file does record, which this
script recomputes on the same live works and reports side by side; the agreement
on modeled is the measurement that says how far the reconstruction sits from the
build. Recorded values are never overwritten and never dropped.

Cache: results/revision/T2_6_funnel_neuro/works_cache/works_*.parquet, in r5's
layout and resumable the same way. The r5 cache at
results/robustness/openalex_cache/ is read first and never written to; it holds
29,443 of the 64,476 authors, all of them survivors, so the fetch here is
exactly the authors the filters dropped.

Usage
  python code/r49_funnel_neuro_counters.py --stage fetch
  python code/r49_funnel_neuro_counters.py --stage count
"""
import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline" / "experiments"))

import build_neuro_dataset as bnd                     # noqa: E402
from r6_topk_sweep import builder_view                # noqa: E402

OUT = ROOT / "results" / "revision" / "T2_6_funnel_neuro"
WORK_CACHE = OUT / "works_cache"
R5_CACHE = ROOT / "results" / "robustness" / "openalex_cache"
SUPP = ROOT / "data" / "supplement"
PAIRS = ROOT / "data" / "pairs_resolved_neuro.parquet"
FUNNEL = ROOT / "data" / "funnel_neuro.json"
SHARD = 1000


def needed_aids():
    """Authors of every pair with both endpoints resolved, in the builder's
    bare-id form (stage_table's _aid)."""
    df = pd.read_parquet(PAIRS, columns=["st_openalex_id", "adv_openalex_id"])
    sub = df.dropna(subset=["st_openalex_id", "adv_openalex_id"])
    return sorted(set(sub.st_openalex_id.map(bnd._aid)) |
                  set(sub.adv_openalex_id.map(bnd._aid)))


def cached_aids():
    done = set()
    for d in (R5_CACHE, WORK_CACHE):
        if d.exists():
            for p in sorted(d.glob("works_*.parquet")):
                done |= set(pd.read_parquet(p, columns=["aid"]).aid)
    return done


def stage_fetch():
    import r5_fetch_author_works as r5
    WORK_CACHE.mkdir(parents=True, exist_ok=True)
    done = cached_aids()          # hoisted: one scan of the shards, not one per author
    todo = [a for a in needed_aids() if a not in done]
    n0 = len(list(WORK_CACHE.glob("works_*.parquet")))
    print(f"[r49] {len(todo)} authors to fetch | {n0} shards present", flush=True)
    buf, shard, t0 = [], n0, time.time()

    def flush():
        nonlocal buf, shard
        if not buf:
            return
        pd.DataFrame(buf, columns=["aid", "works_json"]).to_parquet(
            WORK_CACHE / f"works_{shard:05d}.parquet", index=False)
        shard += 1
        buf = []

    with ThreadPoolExecutor(max_workers=r5.N_WORKERS) as ex:
        for i, (aid, works) in enumerate(zip(todo, ex.map(r5.fetch_author_works, todo))):
            buf.append((aid, json.dumps(works)))
            if len(buf) >= SHARD:
                flush()
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                print(f"[r49] {i+1}/{len(todo)} | {rate:.1f} authors/s | "
                      f"ETA {(len(todo)-i-1)/rate/3600:.2f} h", flush=True)
    flush()
    print(f"[r49] fetch complete in {(time.time()-t0)/3600:.2f} h", flush=True)


def load_all(aids):
    """aid -> works at the builder's concept view (score >= 0.3, first three),
    which is the form the builder's own store held."""
    out = {}
    for d in (R5_CACHE, WORK_CACHE):
        if not d.exists():
            continue
        for p in sorted(d.glob("works_*.parquet")):
            df = pd.read_parquet(p)
            hit = df[df.aid.isin(aids)]
            for aid, wj in zip(hit.aid, hit.works_json):
                w = json.loads(wj)
                out[aid] = builder_view(w, bnd.CONCEPT_MIN_SCORE) if w else []
    return out


def stage_count():
    OUT.mkdir(parents=True, exist_ok=True)
    recorded = json.loads(FUNNEL.read_text())
    pairs = pd.read_parquet(PAIRS)
    raw = len(pairs)
    sub = pairs.dropna(subset=["st_openalex_id", "adv_openalex_id"]).copy()
    sub["st_aid"] = sub.st_openalex_id.map(bnd._aid)
    sub["adv_aid"] = sub.adv_openalex_id.map(bnd._aid)
    both_resolvable = len(sub)

    works = load_all(set(sub.st_aid) | set(sub.adv_aid))
    have = {a for a, w in works.items() if w}
    sub_store = sub[sub.st_aid.isin(have) & sub.adv_aid.isin(have)]
    both_in_store = len(sub_store)
    print(f"[r49] authors with works: {len(have)} / {len(works)}", flush=True)

    EY, LY = bnd.EARLY_YEARS, bnd.LATE_YEARS
    drop = Counter()
    survivors = []
    for r in sub_store.itertuples():
        sw, aw = works[r.st_aid], works[r.adv_aid]
        years = [w["year"] for w in sw if w.get("year")]
        if not years:
            drop["no_years"] += 1
            continue
        years_a = [w["year"] for w in aw if w.get("year")]
        if not years_a:
            drop["adv_no_years"] += 1
            continue
        t0 = min(years)
        if t0 + LY > bnd.OBS_YEAR:
            drop["too_recent"] += 1
            continue
        if t0 < bnd.T0_MIN:
            drop["implausible_t0"] += 1
            continue
        if (max(years) - min(years) > bnd.MAX_CAREER_SPAN or
                max(years_a) - min(years_a) > bnd.MAX_CAREER_SPAN):
            drop["span_gt_max"] += 1
            continue
        _, n_e, _ = bnd.profile(sw, t0, t0 + EY)
        _, n_l, _ = bnd.profile(sw, t0 + EY + 1, t0 + LY)
        _, n_a, _ = bnd.profile(aw, None, t0 + EY)
        if n_e < bnd.MIN_WORKS_PER_WINDOW or n_l < bnd.MIN_WORKS_PER_WINDOW or n_a < 3:
            drop["sparse_windows"] += 1
            continue
        survivors.append((r.student_pid, n_e))
    survive = len(survivors)
    modeled = len({p for p, _ in survivors})     # stage_table's drop_duplicates

    frozen = pd.read_parquet(ROOT / "data" / "clean_dataset_neuro.parquet",
                             columns=["student_pid"])
    frozen_pids = set(frozen.student_pid)
    rec_pids = {p for p, _ in survivors}
    out = {
        "task": "T2.6",
        "field": "neuro",
        "what": ("the eight counters data/funnel_neuro.json records as zero, "
                 "reconstructed by rerunning the builder's counting loop only"),
        "frozen_tables_unchanged": True,
        "provenance": {
            "works_source": "live OpenAlex API, 2026-07",
            "is_frozen_snapshot": False,
            "why": ("the builder's works store is in neither the repository nor "
                    "the archive"),
            "caches_read": ["results/robustness/openalex_cache/",
                            "results/revision/T2_6_funnel_neuro/works_cache/"],
            "reading_rule": ("a reconstruction, not the build's own counters; "
                             "judge it by the agreement block below")},
        "recorded": {
            "raw_pairs": recorded["raw_pairs"],
            "both_resolvable": recorded["both_resolvable"],
            "both_in_works_store": recorded["both_in_works_store"],
            "survive_window_filters": recorded["survive_window_filters"],
            "modeled": recorded["modeled"],
            "drop_breakdown": recorded["drop_breakdown"],
            "zero_means": ("uninstrumented in this build, not measured as zero: "
                           "econ's funnel carries all eight")},
        "reconstructed": {
            "raw_pairs": raw,
            "both_resolvable": both_resolvable,
            "both_in_works_store": both_in_store,
            "survive_window_filters": survive,
            "modeled": modeled,
            "drop_breakdown": {k: int(drop[k]) for k in
                               ("no_years", "adv_no_years", "too_recent",
                                "implausible_t0", "span_gt_max", "sparse_windows")}},
        "agreement": {
            "raw_pairs": {"recorded": recorded["raw_pairs"], "reconstructed": raw,
                          "match": raw == recorded["raw_pairs"]},
            "both_resolvable": {"recorded": recorded["both_resolvable"],
                                "reconstructed": both_resolvable,
                                "match": both_resolvable == recorded["both_resolvable"]},
            "modeled": {"recorded": recorded["modeled"], "reconstructed": modeled,
                        "delta": modeled - recorded["modeled"],
                        "ratio": round(modeled / recorded["modeled"], 4)},
            "frozen_students_recovered": {
                "n_frozen": len(frozen_pids),
                "n_recovered": len(frozen_pids & rec_pids),
                "share": round(len(frozen_pids & rec_pids) / len(frozen_pids), 4)},
            "students_new_to_reconstruction": len(rec_pids - frozen_pids)},
        "builder_params": {"early_years": EY, "late_years": LY,
                           "min_works_per_window": bnd.MIN_WORKS_PER_WINDOW,
                           "t0_min": bnd.T0_MIN, "obs_year": bnd.OBS_YEAR,
                           "max_career_span": bnd.MAX_CAREER_SPAN,
                           "concept_min_score": bnd.CONCEPT_MIN_SCORE,
                           "max_works_per_author": bnd.MAX_WORKS_PER_AUTHOR},
    }
    SUPP.mkdir(parents=True, exist_ok=True)
    (SUPP / "funnel_neuro_complete.json").write_text(json.dumps(out, indent=2))
    (OUT / "funnel_neuro_complete.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"reconstructed": out["reconstructed"],
                      "agreement": out["agreement"]}, indent=2))
    print("[r49] ->", SUPP / "funnel_neuro_complete.json")


def stage_years():
    """The 288 MB fetch cache, reduced to the only thing the counters read.

    Every one of the eight counters is a function of publication years alone.
    stage_table's cascade takes t0 from min(years), tests t0 against the census
    year and T0_MIN, tests max(years) minus min(years) against the span cap, and
    then counts works falling inside three year windows; build_neuro_dataset's
    profile() returns that count as its second value and never consults a
    concept to produce it. So a per-author year list reproduces the cascade
    exactly, and a reader can check 24,128 without the cache it came from.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    aids = needed_aids()
    works = load_all(set(aids))
    rows = []
    for a in aids:
        w = works.get(a) or []
        ys = sorted(int(x["year"]) for x in w if x.get("year") is not None)
        rows.append((a, len(w), ys))
    df = pd.DataFrame(rows, columns=["aid", "n_works", "years"])
    p = SUPP / "funnel_neuro_years.parquet"
    df.to_parquet(p, index=False, compression="zstd")
    print(f"[r49] {len(df)} authors, "
          f"{int(df.years.apply(len).sum())} dated works -> {p} "
          f"({p.stat().st_size / 1e6:.1f} MB)", flush=True)


def stage_verify():
    """Recompute the eight counters from the year file alone and compare."""
    rec = json.loads((SUPP / "funnel_neuro_complete.json").read_text())
    years = {a: list(y) for a, y in zip(
        *[pd.read_parquet(SUPP / "funnel_neuro_years.parquet")[c]
          for c in ("aid", "years")])}
    pairs = pd.read_parquet(PAIRS)
    sub = pairs.dropna(subset=["st_openalex_id", "adv_openalex_id"]).copy()
    sub["st_aid"] = sub.st_openalex_id.map(bnd._aid)
    sub["adv_aid"] = sub.adv_openalex_id.map(bnd._aid)
    have = {a for a, y in years.items() if y}
    sub_store = sub[sub.st_aid.isin(have) & sub.adv_aid.isin(have)]
    EY, LY = bnd.EARLY_YEARS, bnd.LATE_YEARS
    drop, survive = Counter(), 0
    win = lambda ys, lo, hi: sum(1 for y in ys if lo <= y <= hi)   # noqa: E731
    for r in sub_store.itertuples():
        ys, ya = years[r.st_aid], years[r.adv_aid]
        if not ys:
            drop["no_years"] += 1
            continue
        if not ya:
            drop["adv_no_years"] += 1
            continue
        t0 = min(ys)
        if t0 + LY > bnd.OBS_YEAR:
            drop["too_recent"] += 1
            continue
        if t0 < bnd.T0_MIN:
            drop["implausible_t0"] += 1
            continue
        if (max(ys) - min(ys) > bnd.MAX_CAREER_SPAN
                or max(ya) - min(ya) > bnd.MAX_CAREER_SPAN):
            drop["span_gt_max"] += 1
            continue
        if (win(ys, t0, t0 + EY) < bnd.MIN_WORKS_PER_WINDOW
                or win(ys, t0 + EY + 1, t0 + LY) < bnd.MIN_WORKS_PER_WINDOW
                or win(ya, -10**9, t0 + EY) < 3):
            drop["sparse_windows"] += 1
            continue
        survive += 1
    got = {"both_in_works_store": len(sub_store),
           "survive_window_filters": survive,
           **{k: int(drop[k]) for k in rec["reconstructed"]["drop_breakdown"]}}
    want = {"both_in_works_store": rec["reconstructed"]["both_in_works_store"],
            "survive_window_filters": rec["reconstructed"]["survive_window_filters"],
            **rec["reconstructed"]["drop_breakdown"]}
    ok = True
    for k in want:
        m = got[k] == want[k]
        ok &= m
        print(f"  {'OK  ' if m else 'FAIL'} {k:24s} year file {got[k]:7d}  "
              f"recorded {want[k]:7d}")
    verdict = ("all eight counters reproduce from the year file alone"
               if ok else "MISMATCH against funnel_neuro_complete.json")
    print(f"[r49:verify] {verdict}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fetch", "count", "years", "verify"],
                    required=True)
    a = ap.parse_args()
    {"fetch": stage_fetch, "count": stage_count,
     "years": stage_years, "verify": stage_verify}[a.stage]()
