#!/usr/bin/env python3
"""P1(b) label-feasibility probe for a new discipline (default: math).

Question this answers BEFORE committing to a full build: are OpenAlex concept
annotations dense enough in this tree for the top-k concept-overlap label to be
meaningful, or does it degrade (mostly-empty windows, base rate collapsing to
~0 or ~1)?  Reuses the EXACT window/label functions and parameters of the frozen
build (imported from build_neuro_dataset), so the estimated base rate is directly
comparable to econ (0.139) and neuro (0.251).

PRE-REGISTERED FEASIBILITY RULE (fixed here before the numbers are seen):
  PASS  if estimated base rate in [0.05, 0.40] AND empty-window fraction < ~15%.
  FLAG  otherwise -> report; do not force a degraded label. Options on FLAG are
        (i) OpenAlex Topics layer instead of Concepts (denser; footnote the
        different taxonomy layer), or (ii) admit math with a "labels limited by
        concept sparsity" limitation. This script only reports; it decides nothing.

Runs on a RESOLVED sample (both ends have an OpenAlex id). Fetches per-author
works into a resumable store, then computes the table exactly like stage_table.

Usage:
  python probe_labels.py --ultimate suite_data/math/Math_sample_Ultimate.parquet \
      --wstore suite_data/math/works_math_sample.jsonl --field math \
      --out suite_data/math/math_label_probe.json
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "Neuro Data")
import build_neuro_dataset as bnd   # identical windows/labels/params/fetch logic

EY, LY = bnd.EARLY_YEARS, bnd.LATE_YEARS
MWW, OBS = bnd.MIN_WORKS_PER_WINDOW, bnd.OBS_YEAR
T0MIN, SPAN = bnd.T0_MIN, bnd.MAX_CAREER_SPAN
THETA, TOPK = bnd.JACCARD_THETA, bnd.PROFILE_TOPK


def _aid(u):
    return str(u).rstrip("/").split("/")[-1] if u is not None and pd.notna(u) else None


def load_store(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                j = json.loads(line); d[j["aid"]] = j["works"]
            except Exception:
                pass
    return d


def fetch_works(ids, path):
    import requests
    store = load_store(path)
    todo = [i for i in ids if i not in store]
    print(f"[probe-fetch] {len(ids)} authors | {len(store)} cached | {len(todo)} to fetch", flush=True)
    s = requests.Session()
    with open(path, "a") as out:
        for k, aid in enumerate(todo):
            works = bnd.fetch_author_works(aid, s)   # EXACT build fetch: concepts>=0.3, cap 400
            out.write(json.dumps({"aid": aid, "works": works}) + "\n")
            if (k + 1) % 50 == 0:
                out.flush(); os.fsync(out.fileno()); print(f"[probe-fetch] {k+1}/{len(todo)}", flush=True)
    return load_store(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ultimate", required=True)
    ap.add_argument("--wstore", required=True)
    ap.add_argument("--field", default="math")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    det = pd.read_parquet(args.ultimate)
    sub = det.dropna(subset=["st_openalex_id", "adv_openalex_id"]).copy()
    sub["st_aid"] = sub.st_openalex_id.map(_aid)
    sub["adv_aid"] = sub.adv_openalex_id.map(_aid)
    sub = sub.dropna(subset=["st_aid", "adv_aid"])
    n_sampled = len(det)
    n_both_resolved = len(sub)

    ids = pd.unique(pd.concat([sub.st_aid, sub.adv_aid]).dropna())
    works = fetch_works(list(ids), args.wstore)

    # per-author concept-density stats (students, whose windows drive the label)
    stu_distinct, stu_early_empty, stu_late_empty, n_stu_scored = [], 0, 0, 0
    # base-rate table, identical logic to bnd.stage_table
    rows = []
    drop = {"no_years": 0, "adv_no_years": 0, "too_recent": 0,
            "implausible_t0": 0, "span_gt_max": 0, "sparse_windows": 0,
            "author_missing_works": 0}
    for r in sub.itertuples():
        sw = works.get(r.st_aid); aw = works.get(r.adv_aid)
        if sw is None or aw is None:
            drop["author_missing_works"] += 1; continue
        years = [w["year"] for w in sw if w.get("year")]
        if not years:
            drop["no_years"] += 1; continue
        years_a = [w["year"] for w in aw if w.get("year")]
        if not years_a:
            drop["adv_no_years"] += 1; continue
        t0 = min(years)
        # concept-density diagnostics for the student (independent of downstream drops)
        _, _, distinct = bnd.profile(sw, None, None, topk=10**6)
        early_set, n_e, _ = bnd.profile(sw, t0, t0 + EY)
        late_set, n_l, _ = bnd.profile(sw, t0 + EY + 1, t0 + LY)
        stu_distinct.append(distinct)
        stu_early_empty += int(len(early_set) == 0)
        stu_late_empty += int(len(late_set) == 0)
        n_stu_scored += 1
        # downstream window/label filters (identical to build)
        if t0 + LY > OBS: drop["too_recent"] += 1; continue
        if t0 < T0MIN: drop["implausible_t0"] += 1; continue
        if (max(years) - min(years) > SPAN or max(years_a) - min(years_a) > SPAN):
            drop["span_gt_max"] += 1; continue
        advp, n_a, _ = bnd.profile(aw, None, t0 + EY)
        if n_e < MWW or n_l < MWW or n_a < 3:
            drop["sparse_windows"] += 1; continue
        rows.append({"student_pid": r.student_pid,
                     "late_overlap": round(bnd.jaccard(late_set, advp), 4),
                     "y": int(bnd.jaccard(late_set, advp) > THETA)})
    lab = pd.DataFrame(rows).drop_duplicates("student_pid") if rows else pd.DataFrame(columns=["y"])
    base_rate = float(lab.y.mean()) if len(lab) else None

    out = {
        "field": args.field,
        "n_sampled_pairs": int(n_sampled),
        "n_both_resolved": int(n_both_resolved),
        "n_modeled_pairs": int(len(lab)),
        "estimated_base_rate": round(base_rate, 4) if base_rate is not None else None,
        "compare_base_rate": {"econ": 0.139, "neuro": 0.251},
        "student_concepts_distinct_median": float(np.median(stu_distinct)) if stu_distinct else None,
        "student_concepts_distinct_mean": round(float(np.mean(stu_distinct)), 2) if stu_distinct else None,
        "n_students_scored": int(n_stu_scored),
        "early_window_empty_frac": round(stu_early_empty / n_stu_scored, 4) if n_stu_scored else None,
        "late_window_empty_frac": round(stu_late_empty / n_stu_scored, 4) if n_stu_scored else None,
        "drop_breakdown": drop,
        "params": {"early_years": EY, "late_years": LY, "min_works_per_window": MWW,
                   "theta": THETA, "topk": TOPK, "concept_min_score": bnd.CONCEPT_MIN_SCORE,
                   "obs_year": OBS, "t0_min": T0MIN, "max_career_span": SPAN},
        "preregistered_rule": "PASS if base_rate in [0.05,0.40] and empty-window frac < ~0.15",
    }
    # verdict is descriptive only (operator decides)
    br_ok = base_rate is not None and 0.05 <= base_rate <= 0.40
    ew = out["early_window_empty_frac"]; lw = out["late_window_empty_frac"]
    empty_ok = (ew is not None and lw is not None and max(ew, lw) < 0.15)
    out["auto_verdict"] = "PASS" if (br_ok and empty_ok) else "FLAG"
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print(f"[probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
