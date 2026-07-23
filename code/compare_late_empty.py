#!/usr/bin/env python3
"""Read-only: late/early-window empty fractions + base rate for econ & neuro,
to contextualize the math probe's 16.2% late-empty. Uses the frozen works stores
WITHOUT fetching or writing anything. Same window/label logic as the build
(imported from build_neuro_dataset), so the base rate here should reproduce the
frozen summary (econ 0.139 / neuro 0.251) as a logic check.
"""
import argparse, json, sys, random
import numpy as np
import pandas as pd

sys.path.insert(0, "Neuro Data")
import build_neuro_dataset as bnd
EY, LY = bnd.EARLY_YEARS, bnd.LATE_YEARS
MWW, OBS, T0MIN, SPAN, THETA = (bnd.MIN_WORKS_PER_WINDOW, bnd.OBS_YEAR,
                                bnd.T0_MIN, bnd.MAX_CAREER_SPAN, bnd.JACCARD_THETA)


def _aid(u):
    return str(u).rstrip("/").split("/")[-1] if u is not None and pd.notna(u) else None


def load_store_subset(path, needed):
    d = {}
    with open(path) as f:
        for line in f:
            try:
                j = json.loads(line)
                if j["aid"] in needed:
                    d[j["aid"]] = j["works"]
            except Exception:
                pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ultimate", required=True)
    ap.add_argument("--wstore", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    det = pd.read_parquet(args.ultimate)
    sub = det.dropna(subset=["st_openalex_id", "adv_openalex_id"]).copy()
    sub["st_aid"] = sub.st_openalex_id.map(_aid)
    sub["adv_aid"] = sub.adv_openalex_id.map(_aid)
    sub = sub.dropna(subset=["st_aid", "adv_aid"])
    if len(sub) > args.sample:
        sub = sub.sample(n=args.sample, random_state=args.seed)
    needed = set(sub.st_aid) | set(sub.adv_aid)
    works = load_store_subset(args.wstore, needed)

    n_scored = early_empty = late_empty = 0
    rows = []
    miss = 0
    for r in sub.itertuples():
        sw = works.get(r.st_aid); aw = works.get(r.adv_aid)
        if sw is None or aw is None:
            miss += 1; continue
        years = [w["year"] for w in sw if w.get("year")]
        if not years:
            continue
        years_a = [w["year"] for w in aw if w.get("year")]
        if not years_a:
            continue
        t0 = min(years)
        early_set, n_e, _ = bnd.profile(sw, t0, t0 + EY)
        late_set, n_l, _ = bnd.profile(sw, t0 + EY + 1, t0 + LY)
        n_scored += 1
        early_empty += int(len(early_set) == 0)
        late_empty += int(len(late_set) == 0)
        if t0 + LY > OBS or t0 < T0MIN:
            continue
        if max(years) - min(years) > SPAN or max(years_a) - min(years_a) > SPAN:
            continue
        advp, n_a, _ = bnd.profile(aw, None, t0 + EY)
        if n_e < MWW or n_l < MWW or n_a < 3:
            continue
        rows.append(int(bnd.jaccard(late_set, advp) > THETA))
    out = {
        "field": args.field, "sample_pairs": int(len(sub)),
        "both_resolved_with_works": int(n_scored), "missing_works": int(miss),
        "early_window_empty_frac": round(early_empty / n_scored, 4) if n_scored else None,
        "late_window_empty_frac": round(late_empty / n_scored, 4) if n_scored else None,
        "n_modeled": len(rows),
        "base_rate": round(float(np.mean(rows)), 4) if rows else None,
    }
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
