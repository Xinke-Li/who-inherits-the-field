"""R6 - Top-k profile sweep from the r5 OpenAlex cache (P0-4).

POST-HOC ROBUSTNESS CHECK, layered on top of the frozen artifact; NOT part of
the pre-registered protocol. The frozen tables store only the k=10
late_overlap, so the profile-size sweep rebuilds concept profiles from the
per-author works cache fetched ONCE by r5_fetch_author_works.py. The API is
never touched here; every k and every concept-min-score variant derives
offline from the same cache.

PROVENANCE RULE (stated in the paper's Appendix G). The live API is not the
2026 frozen snapshot: tags, disambiguation, and records drift. The sweep
therefore has two stages with different comparators:

  calibrate  rebuild at k=10 with the builder's exact view (concept score
             >= 0.3, first three, profile top-10, identical windows) and
             report drift AGAINST THE FROZEN TABLES: fraction of rows whose
             recomputed late_overlap matches within 1e-6, the |delta|
             distribution, and per-discipline label agreement (Cohen's
             kappa). This is a transparency exhibit, not a robustness claim.
  sweep      k in {5, 15, 20} at theta = 0.2, read AGAINST THE REBUILT k=10
             REFERENCE, never against frozen numbers, so snapshot drift
             cancels within the rebuilt family. Frozen-table numbers and
             rebuilt-family numbers are never mixed in one comparison.

Per (discipline, k): base rate, the tabular ladder (e1_baselines.run imported
verbatim on a dataframe whose profile-derived columns are rebuilt at that k),
the advisor-placebo gap (e10 L1 rung, as in r1), the student-only floor (r1's
reduced e14 ladder), and the Table 6 mechanism branch under the unchanged
pre-registered 0.05 thresholds.

Rebuild details (all mirroring build_neuro_dataset.stage_table, the source):
  * profiles via Counter.most_common(k) over works in store order (the cache
    preserves publication-date-ascending order, so ties break as the builder
    breaks them);
  * early profile [t0, t0+5], late (t0+5, t0+15], advisor profile <= t0+5;
  * rebuilt columns: early_concepts, adv_profile, early_overlap, early_prod,
    early_breadth, adv_early_prod, adv_early_breadth, adv_career_age_at_t0,
    late_overlap, y. The coauth columns CANNOT be rebuilt from this cache
    (the builder's coauth stage used a separate endpoint); they are carried
    over frozen and are identical across k, so they cancel within the
    rebuilt family. Documented in the output.
  * the frozen row set is kept at every k (same students); rows whose rebuilt
    windows would now fail the builder's min-works filters are counted and
    reported, not dropped, so the k cells stay row-comparable.

Also produced (cheap once the cache exists): the concept-min-score variant
{0.2, 0.4} at k=10, same outputs, one summary row each.

Usage (one discipline per process, then merge):
  DATASET=<field> DATASET_PATH=data/clean_dataset_<field>.parquet \
      python code/paper_pipeline/experiments/r6_topk_sweep.py
  python code/paper_pipeline/experiments/r6_topk_sweep.py --merge

Outputs: results/robustness/topk_partial/<field>_k<k>.json etc.,
         results/robustness/topk_<k>.json, topk_sweep_summary.json (--merge).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S

import e1_baselines as E1
from r1_theta_sweep import advisor_placebo_L1, student_only_floor
from e14_self_persistence import LEGACY_STUDENT_TFIDF, student_tfidf, jaccard
from sklearn.metrics import cohen_kappa_score

REPO = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO / "results" / "robustness" / "openalex_cache"
OUT_DIR = REPO / "results" / "robustness"
PARTIAL_DIR = OUT_DIR / "topk_partial"
FIELDS = ["chemistry", "econ", "math", "neuro", "physics"]

K_GRID = [5, 10, 15, 20]
K_REF = 10
MIN_SCORE_GRID = [0.2, 0.4]        # builder default 0.3; variants at k=10
BUILDER_MIN_SCORE = 0.3
EARLY_YEARS, LATE_YEARS = C.EARLY_YEARS, C.LATE_YEARS


def load_cache_for(aids: set) -> dict:
    """aid -> list of (year, [concept names at builder view], full concept
    list with scores). Only the requested authors are kept in memory."""
    out = {}
    for p in sorted(CACHE_DIR.glob("works_*.parquet")):
        df = pd.read_parquet(p)
        hit = df[df.aid.isin(aids)]
        for aid, wj in zip(hit.aid, hit.works_json):
            out[aid] = json.loads(wj)
    return out


def builder_view(works, min_score):
    """The builder's fetch-time collapse, applied offline: concepts with
    score >= min_score, first three, in API (descending-score) order."""
    return [{"year": w["year"],
             "concepts": [c for c, s in w["concepts"] if s >= min_score][:3]}
            for w in works]


def profile_counts(works, y0=None, y1=None):
    """build_neuro_dataset.profile's counting loop (source), minus the topk."""
    cnt, n = Counter(), 0
    for w in works:
        y = w.get("year")
        if y is None:
            continue
        if y0 is not None and y < y0:
            continue
        if y1 is not None and y > y1:
            continue
        n += 1
        for c in w["concepts"]:
            cnt[c] += 1
    return cnt, n


def topk_set(cnt, k):
    return set(c for c, _ in cnt.most_common(k))


def rebuild(df, cache, k, min_score):
    """Rebuilt per-row quantities at profile size k. Returns a dataframe of
    rebuilt columns plus per-row diagnostics."""
    view = {a: builder_view(cache[a], min_score) for a in
            set(df.st_openalex_id) | set(df.adv_openalex_id) if a in cache}
    rows = []
    for r in df.itertuples():
        sw = view.get(r.st_openalex_id)
        aw = view.get(r.adv_openalex_id)
        if sw is None or aw is None or not sw or not aw:
            rows.append(None)
            continue
        t0 = r.t0
        e_cnt, n_e = profile_counts(sw, t0, t0 + EARLY_YEARS)
        l_cnt, n_l = profile_counts(sw, t0 + EARLY_YEARS + 1, t0 + LATE_YEARS)
        a_cnt, n_a = profile_counts(aw, None, t0 + EARLY_YEARS)
        early, late, advp = topk_set(e_cnt, k), topk_set(l_cnt, k), topk_set(a_cnt, k)
        years_a = [w["year"] for w in aw if w.get("year") is not None]
        rows.append({
            "early_concepts": sorted(early), "adv_profile": sorted(advp),
            "early_overlap": round(jaccard(early, advp), 4),
            "early_prod": n_e, "early_breadth": len(e_cnt),
            "adv_early_prod": n_a, "adv_early_breadth": len(a_cnt),
            "adv_career_age_at_t0": int(t0 - min(years_a)) if years_a else 0,
            "late_overlap": round(jaccard(late, advp), 4),
            "late_prod": n_l,
            "min_works_ok": bool(n_e >= 3 and n_l >= 3 and n_a >= 3),
        })
    return rows


def apply_rebuild(df, rows):
    """Frozen df copy with the profile-derived columns replaced by the rebuilt
    values; rows without cache coverage are dropped (counted by caller)."""
    keep = [i for i, r in enumerate(rows) if r is not None]
    out = df.iloc[keep].reset_index(drop=True).copy()
    rr = [rows[i] for i in keep]
    for col in ("early_concepts", "adv_profile"):
        out[col] = [r[col] for r in rr]
    for col in ("early_overlap", "early_prod", "early_breadth", "adv_early_prod",
                "adv_early_breadth", "adv_career_age_at_t0", "late_overlap",
                "late_prod"):
        out[col] = [r[col] for r in rr]
    out["y"] = (out.late_overlap > C.JACCARD_THETA).astype(int)
    # empty rebuilt early profiles would break the TF-IDF arm; drop and count
    n0 = len(out)
    out = out[out.early_concepts.apply(len) > 0].reset_index(drop=True)
    return out, {"n_no_cache": int(len(rows) - len(keep)),
                 "n_empty_early": int(n0 - len(out)),
                 "n_fail_min_works": int(sum(not r["min_works_ok"] for r in rr))}


def evaluate_variant(dfk):
    """Base rate + ladder + placebo gap + floor + branch on a rebuilt table."""
    df_split = D.temporal_split(dfk)
    out = {"n": len(dfk), "base_rate": round(float(dfk.y.mean()), 4)}
    cache, per_model = {}, {}
    for seed in C.SEEDS:
        for model, metrics in E1.run(seed, df_split, cache).items():
            per_model.setdefault(model, []).append(metrics)
    out["ladder"] = {m: S.summarize_seeds(v) for m, v in per_model.items()}
    best_tab = max(("M2_logit_tabular", "M3_gbdt_tabular"),
                   key=lambda m: out["ladder"][m]["auc_pr"]["mean"])
    out["best_pure_tabular"] = {"model": best_tab,
                                "auc_pr": out["ladder"][best_tab]["auc_pr"]["mean"]}
    out["advisor_placebo"] = advisor_placebo_L1(df_split)
    X_st = student_tfidf(df_split, legacy=LEGACY_STUDENT_TFIDF)
    out["self_persistence"] = student_only_floor(
        df_split, X_st, out["advisor_placebo"]["placebo_band_max_L1"])
    out["mechanism_branch"] = out["self_persistence"]["verdict"]["branch"]
    return out


def main_field():
    field = C.CLEAN_DATASET.stem.replace("clean_dataset_", "")
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    df = D.load_dataset()
    aids = set(df.st_openalex_id) | set(df.adv_openalex_id)
    print(f"[r6] {field}: loading cache for {len(aids)} authors", flush=True)
    cache = load_cache_for(aids)
    print(f"[r6] {field}: {len(cache)} authors in cache "
          f"({len(aids) - len(cache)} missing)", flush=True)

    # ---- calibration at k=10, builder view, vs the FROZEN tables ----
    rows = rebuild(df, cache, K_REF, BUILDER_MIN_SCORE)
    covered = [(i, r) for i, r in enumerate(rows) if r is not None]
    frozen_lo = df.late_overlap.values
    deltas = np.array([abs(r["late_overlap"] - frozen_lo[i]) for i, r in covered])
    y_frozen = df.y.values[[i for i, _ in covered]]
    y_rebuilt = np.array([int(r["late_overlap"] > C.JACCARD_THETA) for _, r in covered])
    calib = {
        "n_rows": len(df), "n_covered": len(covered),
        "match_within_1e6": round(float((deltas < 1e-6).mean()), 4),
        "abs_delta": {"mean": round(float(deltas.mean()), 4),
                      "median": round(float(np.median(deltas)), 4),
                      "p90": round(float(np.percentile(deltas, 90)), 4),
                      "max": round(float(deltas.max()), 4)},
        "label_agreement": round(float((y_frozen == y_rebuilt).mean()), 4),
        "label_kappa": round(float(cohen_kappa_score(y_frozen, y_rebuilt)), 4),
        "base_rate_frozen": round(float(y_frozen.mean()), 4),
        "base_rate_rebuilt": round(float(y_rebuilt.mean()), 4),
    }
    (PARTIAL_DIR / f"{field}_calibration_k10.json").write_text(json.dumps(calib, indent=2))
    print(f"[r6] {field} calibration: match {calib['match_within_1e6']}, "
          f"kappa {calib['label_kappa']}", flush=True)

    # ---- rebuilt family: k grid at builder min-score ----
    for k in K_GRID:
        if (PARTIAL_DIR / f"{field}_k{k}.json").exists():
            print(f"[r6] {field} k={k} already done, skipping", flush=True)
            continue
        rk = rows if k == K_REF else rebuild(df, cache, k, BUILDER_MIN_SCORE)
        dfk, drops = apply_rebuild(df, rk)
        res = evaluate_variant(dfk)
        res.update({"field": field, "k": k, "min_score": BUILDER_MIN_SCORE,
                    "provenance": "rebuilt from live-API cache (r5); compare "
                                  "within the rebuilt family only",
                    "row_diagnostics": drops})
        p = PARTIAL_DIR / f"{field}_k{k}.json"
        p.write_text(json.dumps(res, indent=2))
        print(f"[r6] {field} k={k}: base {res['base_rate']} best "
              f"{res['best_pure_tabular']['auc_pr']:.4f} branch "
              f"{res['mechanism_branch']}", flush=True)

    # ---- min-score variants at k=10 ----
    for ms in MIN_SCORE_GRID:
        if (PARTIAL_DIR / f"{field}_k{K_REF}_ms{ms}.json").exists():
            print(f"[r6] {field} ms={ms} already done, skipping", flush=True)
            continue
        rk = rebuild(df, cache, K_REF, ms)
        dfk, drops = apply_rebuild(df, rk)
        res = evaluate_variant(dfk)
        res.update({"field": field, "k": K_REF, "min_score": ms,
                    "provenance": "rebuilt family", "row_diagnostics": drops})
        p = PARTIAL_DIR / f"{field}_k{K_REF}_ms{ms}.json"
        p.write_text(json.dumps(res, indent=2))
        print(f"[r6] {field} min_score={ms}: base {res['base_rate']} best "
              f"{res['best_pure_tabular']['auc_pr']:.4f} branch "
              f"{res['mechanism_branch']}", flush=True)


def main_merge():
    summary = {"experiment": "R6_topk_sweep", "k_grid": K_GRID, "k_ref": K_REF,
               "min_score_grid": MIN_SCORE_GRID, "theta": C.JACCARD_THETA,
               "note": ("rebuilt-family comparisons only; calibration vs frozen "
                        "reported separately; coauth columns frozen (identical "
                        "across k); student floor without early_concentration"),
               "calibration_k10": {}, "fields": {}}
    for k in K_GRID:
        merged = {f: json.loads((PARTIAL_DIR / f"{f}_k{k}.json").read_text())
                  for f in FIELDS}
        (OUT_DIR / f"topk_{k}.json").write_text(json.dumps(merged, indent=2))
    for f in FIELDS:
        summary["calibration_k10"][f] = json.loads(
            (PARTIAL_DIR / f"{f}_calibration_k10.json").read_text())
        rows = {}
        for k in K_GRID:
            r = json.loads((PARTIAL_DIR / f"{f}_k{k}.json").read_text())
            rows[f"k{k}"] = {
                "base_rate": r["base_rate"],
                "best_tabular_auc_pr": round(r["best_pure_tabular"]["auc_pr"], 4),
                "M5_auc_pr": round(r["ladder"]["M5_gbdt_nfa"]["auc_pr"]["mean"], 4),
                "placebo_gap": r["advisor_placebo"]["gap_true_minus_cohort"],
                "floor": r["self_persistence"]["verdict"]["best_student_auc_pr"],
                "branch": r["mechanism_branch"]}
        for ms in MIN_SCORE_GRID:
            r = json.loads((PARTIAL_DIR / f"{f}_k{K_REF}_ms{ms}.json").read_text())
            rows[f"k{K_REF}_ms{ms}"] = {
                "base_rate": r["base_rate"],
                "best_tabular_auc_pr": round(r["best_pure_tabular"]["auc_pr"], 4),
                "M5_auc_pr": round(r["ladder"]["M5_gbdt_nfa"]["auc_pr"]["mean"], 4),
                "placebo_gap": r["advisor_placebo"]["gap_true_minus_cohort"],
                "floor": r["self_persistence"]["verdict"]["best_student_auc_pr"],
                "branch": r["mechanism_branch"]}
        ref_branch = rows[f"k{K_REF}"]["branch"]
        summary["fields"][f] = {
            "rows": rows,
            "branch_stable_all_k": all(rows[f"k{k}"]["branch"] == ref_branch
                                       for k in K_GRID)}
    kappas = {f: summary["calibration_k10"][f]["label_kappa"] for f in FIELDS}
    summary["min_kappa"] = min(kappas.values())
    summary["kappa_gate_0.8"] = ("PASS" if summary["min_kappa"] >= 0.8 else
                                 "FAIL - k-sweep confounded by snapshot drift; "
                                 "present the drift measurement instead")
    (OUT_DIR / "topk_sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("calibration_k10", "min_kappa",
                                              "kappa_gate_0.8")}, indent=2))
    for f in FIELDS:
        print(f, json.dumps(summary["fields"][f]["rows"], indent=1)[:400])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        main_merge()
    else:
        main_field()
