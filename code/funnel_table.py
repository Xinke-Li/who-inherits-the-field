#!/usr/bin/env python3
"""Post-build attrition funnel for a suite discipline, mirroring Appendix A.

Reads the discipline's Detailed (raw pairs), resolved Ultimate (id coverage), the
per-author works store, and the frozen clean_dataset, and reports the exact stage
counts using the SAME window/label logic as the build (imported from
build_neuro_dataset). Purely read-only; writes suite_data/<field>/funnel_<field>.json.

  raw_pairs               both-ends AFT pairs (Detailed rows)
  both_resolvable         both advisor & student got an OpenAlex id
  both_in_works_store     both ids have fetched works
  survive_window_filters  pass t0-range / career-span / >=3 works both windows
  modeled                 unique students in the frozen clean_dataset

Usage:
  python funnel_table.py --field math
"""
import argparse, json, os, sys
import pandas as pd

sys.path.insert(0, "Neuro Data")
import build_neuro_dataset as bnd
EY, LY = bnd.EARLY_YEARS, bnd.LATE_YEARS
MWW, OBS, T0MIN, SPAN = bnd.MIN_WORKS_PER_WINDOW, bnd.OBS_YEAR, bnd.T0_MIN, bnd.MAX_CAREER_SPAN


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--detailed", default=None)
    ap.add_argument("--ultimate", default=None)
    ap.add_argument("--min-works", type=int, default=3)
    args = ap.parse_args()
    field = args.field
    mww = args.min_works
    base_root = args.base or os.path.join("suite_data", field)
    cap = {"econ": "Econ"}.get(field, field.capitalize())
    detailed = args.detailed or os.path.join(base_root, f"{cap}_Pairs_Detailed.parquet")
    ultimate = args.ultimate or os.path.join(base_root, f"{cap}_Pairs_Openalex_Ultimate.parquet")
    store = os.path.join(base_root, "build", "cache", "works_store.jsonl")
    clean = os.path.join("artifact", "code", f"data_{field}", "clean_dataset.parquet")

    raw = len(pd.read_parquet(detailed, columns=["connection_id"]))
    ult = pd.read_parquet(ultimate)
    ult["st_aid"] = ult.st_openalex_id.map(_aid)
    ult["adv_aid"] = ult.adv_openalex_id.map(_aid)
    both_resolvable = int((ult.st_aid.notna() & ult.adv_aid.notna()).sum())

    works = load_store(store)
    sub = ult.dropna(subset=["st_aid", "adv_aid"])
    both_in_store = 0
    survive = 0
    drop = {"no_years": 0, "adv_no_years": 0, "too_recent": 0, "implausible_t0": 0,
            "span_gt_max": 0, "sparse_windows": 0}
    for r in sub.itertuples():
        sw = works.get(r.st_aid); aw = works.get(r.adv_aid)
        if sw is None or aw is None:
            continue
        both_in_store += 1
        years = [w["year"] for w in sw if w.get("year")]
        if not years: drop["no_years"] += 1; continue
        years_a = [w["year"] for w in aw if w.get("year")]
        if not years_a: drop["adv_no_years"] += 1; continue
        t0 = min(years)
        if t0 + LY > OBS: drop["too_recent"] += 1; continue
        if t0 < T0MIN: drop["implausible_t0"] += 1; continue
        if max(years) - min(years) > SPAN or max(years_a) - min(years_a) > SPAN:
            drop["span_gt_max"] += 1; continue
        _, n_e, _ = bnd.profile(sw, t0, t0 + EY)
        _, n_l, _ = bnd.profile(sw, t0 + EY + 1, t0 + LY)
        _, n_a, _ = bnd.profile(aw, None, t0 + EY)
        if n_e < mww or n_l < mww or n_a < 3:
            drop["sparse_windows"] += 1; continue
        survive += 1

    modeled = len(pd.read_parquet(clean, columns=["student_pid"])) if os.path.exists(clean) else None
    out = {"field": field, "min_works": mww,
           "raw_pairs": raw, "both_resolvable": both_resolvable,
           "both_in_works_store": both_in_store,
           "survive_window_filters": survive, "modeled": modeled,
           "drop_breakdown": drop,
           "adv_id_coverage": round(float(ult.adv_aid.notna().mean()), 4),
           "st_id_coverage": round(float(ult.st_aid.notna().mean()), 4)}
    outp = os.path.join(base_root, f"funnel_{field}.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"[funnel] wrote {outp}")


if __name__ == "__main__":
    main()
