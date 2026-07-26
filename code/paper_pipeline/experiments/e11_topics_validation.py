"""E11 - Topics re-validation of the concept-based label (Q12, added 2026-07-05).

OpenAlex has deprecated Concepts (unmaintained since 2024); Topics is the
successor taxonomy. The frozen benchmark label is built on legacy concepts, so
the paper must show the task survives the taxonomy change. This script, run on
a random subsample of students (default 500), does:

  1. refetch each student's + advisor's works from the OpenAlex API with
     `topics` (subfield level), date-ascending, capped at 400 works;
  2. rebuild the SAME temporal windows (W_E = [t0, t0+5], W_L = (t0+5, t0+15],
     advisor profile <= t0+5) using top-10 subfield sets;
  3. recompute early_overlap_topics and y_topics = 1[J > theta];
  4. report: label agreement rate y vs y_topics, Cohen's kappa, base rates,
     and the M3 GBDT AUC-PR when trained/evaluated on y_topics.

Run OUTSIDE the sandbox (needs api.openalex.org):
    python experiments/e11_topics_validation.py --n 500 --email you@uni.edu
Resumable: per-author fetches are cached in cache/topics_cache.jsonl.

Output: results_econ/e11_topics_validation.json
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S

API = "https://api.openalex.org/works"
LEVEL = "subfield"          # default granularity; overridden by --level
TOP_K = 10


def cache_path(level):
    # per-level caches: the fetch stores only the chosen level's names
    suffix = "" if level == "subfield" else f"_{level}"
    return C.DATA_DIR / "cache" / f"topics_cache{suffix}.jsonl"


def fetch_author_topic_years(openalex_id: str, email: str, max_works=400,
                             level=None):
    """[(year, [subfield names]), ...] for one author, date-ascending."""
    rows, cursor = [], "*"
    while cursor and len(rows) < max_works:
        q = {"filter": f"authorships.author.id:{openalex_id}",
             "select": "publication_year,topics", "sort": "publication_date:asc",
             "per-page": "200", "cursor": cursor, "mailto": email}
        url = API + "?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        for w in d["results"]:
            lv = level or LEVEL
            if lv == "topic":
                names = [t["display_name"] for t in (w.get("topics") or [])]
            else:
                names = [t[lv]["display_name"] for t in (w.get("topics") or [])
                         if t.get(lv)]
            rows.append((w.get("publication_year"), names))
        cursor = d.get("meta", {}).get("next_cursor")
        time.sleep(0.15)
    return rows[:max_works]


def topk_set(rows, lo, hi, k=TOP_K):
    cnt = Counter(n for y, names in rows if y is not None and lo <= y <= hi
                  for n in names)
    return {n for n, _ in cnt.most_common(k)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--email", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", default="subfield",
                    choices=["topic", "subfield", "field"],
                    help="Topics granularity; 'topic' (~4.5k labels) is closest "
                         "to legacy concept granularity and is the camera-ready "
                         "re-validation level; 'subfield' (~250) is coarse")
    args = ap.parse_args()
    global LEVEL
    LEVEL = args.level

    df = D.load_dataset()
    rng = np.random.default_rng(args.seed)
    sub = df.iloc[rng.choice(len(df), size=min(args.n, len(df)), replace=False)]

    CACHE = cache_path(args.level)
    cache = {}
    if CACHE.exists():
        for l in CACHE.read_text().splitlines():
            d = json.loads(l)
            cache[d["id"]] = d["rows"]

    def get(aid):
        if aid not in cache:
            rows = fetch_author_topic_years(aid, args.email, level=args.level)
            cache[aid] = rows
            with open(CACHE, "a") as f:
                f.write(json.dumps({"id": aid, "rows": rows}) + "\n")
        return cache[aid]

    recs = []
    for i, r in enumerate(sub.itertuples()):
        try:
            st = get(r.st_openalex_id)
            ad = get(r.adv_openalex_id)
        except Exception as e:
            print(f"[e11] skip {r.student_pid}: {e}")
            continue
        A = topk_set(ad, -10**9, r.t0 + C.EARLY_YEARS)
        early = topk_set(st, r.t0, r.t0 + C.EARLY_YEARS)
        late = topk_set(st, r.t0 + C.EARLY_YEARS + 1, r.t0 + C.LATE_YEARS)
        if not early or not late or not A:
            continue
        recs.append({"student_pid": r.student_pid, "t0": int(r.t0),
                     "y_concepts": int(r.y),
                     "early_overlap_topics": jaccard(early, A),
                     "late_overlap_topics": jaccard(late, A),
                     "y_topics": int(jaccard(late, A) > C.JACCARD_THETA)})
        if (i + 1) % 25 == 0:
            print(f"[e11] {i+1}/{len(sub)} pairs done")

    yc = np.array([x["y_concepts"] for x in recs])
    yt = np.array([x["y_topics"] for x in recs])
    agree = float((yc == yt).mean())
    po, pe = agree, float(yc.mean() * yt.mean() + (1 - yc.mean()) * (1 - yt.mean()))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0

    # theta-calibrated comparison (added 2026-07-05, reviewer point R1):
    # the subfield taxonomy is far coarser than legacy concepts, so at the SAME
    # theta the topics label has a 5x higher base rate - raw agreement/kappa
    # then mostly measure the granularity/theta interaction, not label drift.
    # Calibrate theta_topics so the topics label matches the concept-label base
    # rate on the same subsample, then recompute agreement/kappa; also report
    # the threshold-free concordance AUC (does the continuous topics overlap
    # rank concept-label positives highly?).
    lot = np.array([x["late_overlap_topics"] for x in recs])
    theta_cal = float(np.quantile(lot, 1.0 - yc.mean()))
    yt_cal = (lot > theta_cal).astype(int)
    agree_cal = float((yc == yt_cal).mean())
    pe_cal = float(yc.mean() * yt_cal.mean() + (1 - yc.mean()) * (1 - yt_cal.mean()))
    kappa_cal = (agree_cal - pe_cal) / (1 - pe_cal) if pe_cal < 1 else 0.0
    from sklearn.metrics import roc_auc_score
    auc_concord = float(roc_auc_score(yc, lot)) if 0 < yc.mean() < 1 else None

    # M3 on the topic label within the subsample (temporal split by t0)
    perf = None
    ids = {x["student_pid"] for x in recs}
    dsub = df[df.student_pid.isin(ids)].copy()
    ymap = {x["student_pid"]: x["y_topics"] for x in recs}
    dsub["y_topics"] = dsub.student_pid.map(ymap)
    dsp = D.temporal_split(dsub)
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, _ = D.build_features(dsp, concepts="none")
    m_tr, m_te = (dsp.split == "train").values, (dsp.split == "test").values
    if m_te.sum() > 30 and dsp.loc[m_te, "y_topics"].nunique() == 2:
        per_seed = []
        for seed in C.SEEDS:
            m = HistGradientBoostingClassifier(random_state=seed, max_iter=300,
                                               early_stopping=True,
                                               validation_fraction=0.15)
            m.fit(X[m_tr], dsp.loc[m_tr, "y_topics"])
            per_seed.append(S.evaluate(dsp.loc[m_te, "y_topics"].values,
                                       m.predict_proba(X[m_te])[:, 1]))
        perf = S.summarize_seeds(per_seed, keys=("auc_pr", "auc_roc"))
        perf["base_rate_test"] = float(dsp.loc[m_te, "y_topics"].mean())

    out = {"experiment": "E11_topics_validation", "n_pairs": len(recs),
           "granularity": args.level, "theta": C.JACCARD_THETA,
           "label_agreement": round(agree, 4), "cohen_kappa": round(kappa, 4),
           "base_rate_concepts": round(float(yc.mean()), 4),
           "base_rate_topics": round(float(yt.mean()), 4),
           "theta_calibrated": {
               "theta_topics": round(theta_cal, 4),
               "base_rate_topics_cal": round(float(yt_cal.mean()), 4),
               "label_agreement": round(agree_cal, 4),
               "cohen_kappa": round(kappa_cal, 4)},
           "concordance_auc_topics_overlap_vs_concept_label":
               round(auc_concord, 4) if auc_concord is not None else None,
           "m3_on_topics_label": perf}
    suffix = "" if args.level == "subfield" else f"_{args.level}"
    (C.RESULTS_DIR / f"e11_topics_validation{suffix}.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
