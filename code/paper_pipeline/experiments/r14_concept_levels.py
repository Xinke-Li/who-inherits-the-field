"""R14 - concept-level composition of the top-10 profiles (task B6).

Phase --fetch: one pass over the OpenAlex concepts endpoint (about 65k
concepts, 200 per page) into results/robustness/concept_levels.json
(display-name lowercased -> level; name collisions keep the LOWEST level,
the conservative choice for the generic-concept diagnosis).

Phase --analyze: histogram of levels over the frozen student early and
advisor top-10 profiles per discipline. If levels 0 and 1 dominate (over
half the profile mass pooled), rerun base rate, M1 (raw-scalar AUC-PR,
M1-equivalent), and M3 (GBDT on the tabular block with the restricted
overlap substituted, 10 seeds) under a level >= 2 restriction, computed
within the rebuilt-k10 family (r6 convention, never against frozen numbers).

Output: results/robustness/concept_levels.json (map),
        results/robustness/concept_level_analysis.json (histograms + rerun)
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code" / "paper_pipeline"))
sys.path.insert(0, str(REPO / "code" / "paper_pipeline" / "experiments"))

FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
MAP_PATH = REPO / "results" / "robustness" / "concept_levels.json"
OUT_PATH = REPO / "results" / "robustness" / "concept_level_analysis.json"


def fetch_map():
    import requests
    from r5_fetch_author_works import API_KEY, MAILTO
    s = requests.Session()
    levels, cursor, n = {}, "*", 0
    while cursor:
        u = (f"https://api.openalex.org/concepts?per-page=200&cursor={cursor}"
             f"&select=display_name,level&mailto={MAILTO}&api_key={API_KEY}")
        r = s.get(u, timeout=30)
        if r.status_code == 429:
            time.sleep(5)
            continue
        r.raise_for_status()
        js = r.json()
        for c in js.get("results", []):
            nm = c["display_name"].lower()
            lv = int(c["level"])
            if nm not in levels or lv < levels[nm]:
                levels[nm] = lv
        n += len(js.get("results", []))
        cursor = js.get("meta", {}).get("next_cursor")
        if n % 5000 < 200:
            print(f"[r14] {n} concepts", flush=True)
        time.sleep(0.15)
    MAP_PATH.write_text(json.dumps(levels))
    print(f"[r14] wrote {len(levels)} concept levels", flush=True)


def analyze():
    import config as C
    from r6_topk_sweep import (BUILDER_MIN_SCORE, builder_view, load_cache_for,
                               profile_counts, topk_set)
    from e14_self_persistence import jaccard
    from e12_corrected_aggregation import fast_auc_pr
    from sklearn.ensemble import HistGradientBoostingClassifier

    levels = {k: int(v) for k, v in json.loads(MAP_PATH.read_text()).items()}
    out = {"experiment": "R14_concept_levels", "histograms": {}, "rerun": {}}
    pooled = Counter()
    for f in FIELDS:
        df = pd.read_parquet(REPO / "data" / f"clean_dataset_{f}.parquet")
        h = Counter()
        for col in ("early_concepts", "adv_profile"):
            for lst in df[col]:
                for c in lst:
                    h[levels.get(c, -1)] += 1
        tot = sum(h.values())
        out["histograms"][f] = {str(k): round(v / tot, 4)
                                for k, v in sorted(h.items())}
        pooled.update(h)
        print(f"[r14] {f}: {out['histograms'][f]}", flush=True)
    ptot = sum(pooled.values())
    share01 = (pooled.get(0, 0) + pooled.get(1, 0)) / ptot
    out["pooled_level01_share"] = round(share01, 4)
    dominate = share01 > 0.5
    out["level01_dominates"] = bool(dominate)

    if dominate:
        for f in FIELDS:
            df = pd.read_parquet(REPO / "data" / f"clean_dataset_{f}.parquet")
            df = df[df.early_concepts.apply(len) > 0].reset_index(drop=True)
            q1, q2 = np.quantile(df.t0, C.SPLIT_QUANTILES)
            split = np.where(df.t0 <= int(q1), "train",
                             np.where(df.t0 <= int(q2), "val", "test"))
            aids = set(df.st_openalex_id) | set(df.adv_openalex_id)
            cache = load_cache_for(aids)
            view = {a: builder_view(cache[a], BUILDER_MIN_SCORE)
                    for a in aids if a in cache}
            del cache

            def profiles(r, min_level):
                sw, aw = view[r.st_openalex_id], view[r.adv_openalex_id]
                cut = r.t0 + C.EARLY_YEARS

                def flt(cnt):
                    return Counter({c: n for c, n in cnt.items()
                                    if levels.get(c, -1) >= min_level})
                e, _ = profile_counts(sw, r.t0, cut)
                l, _ = profile_counts(sw, cut + 1, r.t0 + C.LATE_YEARS)
                a, _ = profile_counts(aw, None, cut)
                if min_level > 0:
                    e, l, a = flt(e), flt(l), flt(a)
                return topk_set(e, 10), topk_set(l, 10), topk_set(a, 10)

            res = {}
            for tag, ml in (("k10_reference", 0), ("level_ge2", 2)):
                eo = np.empty(len(df)); y = np.empty(len(df), int)
                for i, r in enumerate(df.itertuples()):
                    E, L, A = profiles(r, ml)
                    eo[i] = jaccard(E, A)
                    y[i] = int(jaccard(L, A) > C.JACCARD_THETA)
                te = split == "test"
                # M3: tabular block with the (restricted) overlap substituted
                X = df[["early_prod", "early_breadth", "adv_early_prod",
                        "adv_early_breadth", "adv_career_age_at_t0",
                        "coauth_early_n"]].astype(float).values
                X = np.hstack([eo[:, None], X, df.coauth_early.astype(float).values[:, None]])
                m3 = []
                for seed in C.SEEDS:
                    m = HistGradientBoostingClassifier(random_state=seed, max_iter=500,
                                                       early_stopping=True,
                                                       validation_fraction=0.15)
                    m.fit(X[split == "train"], y[split == "train"])
                    m3.append(fast_auc_pr(y[te], m.predict_proba(X[te])[:, 1]))
                res[tag] = {"base_rate": round(float(y.mean()), 4),
                            "M1_auc_pr": round(fast_auc_pr(y[te], eo[te]), 4),
                            "M3_auc_pr_mean": round(float(np.mean(m3)), 4),
                            "M3_auc_pr_std": round(float(np.std(m3, ddof=1)), 4)}
            out["rerun"][f] = res
            del view
            print(f"[r14] {f}: ref {res['k10_reference']} | ge2 {res['level_ge2']}",
                  flush=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print("[r14] written", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch_map()
    if args.analyze:
        analyze()
