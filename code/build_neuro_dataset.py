#!/usr/bin/env python3
"""Neuro modeling-dataset builder — faithful port of OpenAlex_Dataset_Builder_final.ipynb.

Identical windows/labels/features/filters to the econ build (that is the
identical-protocol claim). ONLY the following differ from the notebook, all
documented for the night-run decision list:
  * paths: BASE points to Neuro Data/build_neuro so the per-author works cache,
    co-authorship cache, and outputs never collide with the frozen econ ones.
  * pairs file: Neuro_Pairs_Openalex_Ultimate.parquet (pub-ID-anchored ids).
  * credentials read from OPENALEX_API_KEY / OPENALEX_MAILTO env (local fast pool).
  * cell-12 meta merge: the AFT neuro Detailed table has no st_name/adv_name/
    institution_country columns (econ got them from OpenAlex); they are filled
    with NA. None of them is a TABULAR_FEATURE, so E1/E10 are unaffected.

Stages (all resumable; safe to Ctrl-C and rerun):
  fetch   — per-author works from OpenAlex (long; the checkpoint bottleneck)
  table   — temporal feature/label table (offline, from the works cache)
  coauth  — early-window co-authorship counts (count API; long, resumable)
  build   — merge/clean/leakage-guard/export clean_dataset.parquet + summary
  outcomes— outcomes.parquet (offline)
Run e.g.:  python build_neuro_dataset.py fetch
           python build_neuro_dataset.py table coauth build outcomes
           python build_neuro_dataset.py all
"""
import os, sys, json, time, hashlib
from collections import Counter
import numpy as np
import pandas as pd

# ---------- paths (neuro-isolated) ----------
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "build_neuro")
os.makedirs(BASE, exist_ok=True)
PAIRS_FILE = os.path.join(HERE, "Neuro_Pairs_Openalex_Ultimate.parquet")

OPENALEX_MAILTO  = os.environ.get("OPENALEX_MAILTO", "")
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "")

# ---------- window & label parameters (IDENTICAL to econ) ----------
EARLY_YEARS          = 5
LATE_YEARS           = 15
MIN_WORKS_PER_WINDOW = 3
OBS_YEAR             = 2026
PROFILE_TOPK         = 10
CONCEPT_MIN_SCORE    = 0.3
JACCARD_THETA        = 0.2
MAX_WORKS_PER_AUTHOR = 400
FETCH_SLEEP          = 0.01 if OPENALEX_API_KEY else 0.06
MAX_AUTHORS          = None
T0_MIN               = 1950
MAX_CAREER_SPAN      = 60
FETCH_COAUTH_EARLY   = True

CACHE_DIR = os.path.join(BASE, "cache"); os.makedirs(CACHE_DIR, exist_ok=True)
STORE        = os.path.join(CACHE_DIR, "works_store.jsonl")
COAUTH_CACHE = os.path.join(CACHE_DIR, "coauth_early.jsonl")
LABELS_CACHE = os.path.join(CACHE_DIR, "temporal_labels_v2.parquet")

def _aid(u): return str(u).rstrip("/").split("/")[-1] if u else None

# ================= stage: fetch (per-author works) =================

def load_store():
    d = {}
    if os.path.exists(STORE):
        with open(STORE) as f:
            for line in f:
                try:
                    j = json.loads(line); d[j["aid"]] = j["works"]
                except Exception:
                    pass
    return d

def fetch_author_works(aid, session):
    out, cursor, fetched = [], "*", 0
    base_url = ("https://api.openalex.org/works?filter=authorships.author.id:{aid}"
                "&per-page=200&sort=publication_date:asc"
                "&select=publication_year,concepts,cited_by_count&cursor={cur}")
    while cursor and fetched < MAX_WORKS_PER_AUTHOR:
        try:
            u = base_url.format(aid=aid, cur=cursor)
            if OPENALEX_MAILTO:  u += f"&mailto={OPENALEX_MAILTO}"
            if OPENALEX_API_KEY: u += f"&api_key={OPENALEX_API_KEY}"
            r = session.get(u, timeout=30)
            if r.status_code == 403 and OPENALEX_API_KEY:
                print("[warn] api_key rejected - falling back to polite pool")
                globals()["OPENALEX_API_KEY"] = ""; continue
            if r.status_code == 429:
                time.sleep(5); continue
            r.raise_for_status()
            js = r.json()
        except Exception as e:
            print(f"[warn] {aid}: {e}"); break
        for w in js.get("results", []):
            cs = [c["display_name"].lower() for c in w.get("concepts", [])
                  if c.get("score", 0) >= CONCEPT_MIN_SCORE][:3]
            out.append({"year": w.get("publication_year"), "concepts": cs,
                        "cited_by_count": w.get("cited_by_count", 0)})
        fetched += len(js.get("results", []))
        cursor = js.get("meta", {}).get("next_cursor")
        if not js.get("results"): break
        time.sleep(FETCH_SLEEP)
    return out

def stage_fetch():
    import requests
    pairs = pd.read_parquet(PAIRS_FILE)
    if os.path.exists(STORE) and os.path.getsize(STORE) > 0:
        with open(STORE, "rb+") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n": f.write(b"\n")
    store_dict = load_store()
    ids = pd.unique(pd.concat([pairs.st_openalex_id, pairs.adv_openalex_id]).dropna().map(_aid))
    if MAX_AUTHORS: ids = ids[:MAX_AUTHORS]
    todo = [i for i in ids if i not in store_dict]
    print(f"[fetch] {len(ids)} authors | {len(store_dict)} in store | {len(todo)} new to fetch", flush=True)
    s = requests.Session()
    with open(STORE, "a") as store:
        for k, aid in enumerate(todo):
            works = fetch_author_works(aid, s)
            store.write(json.dumps({"aid": aid, "works": works}) + "\n")
            if (k + 1) % 50 == 0:
                store.flush(); os.fsync(store.fileno())
                print(f"[fetch] {k+1}/{len(todo)}", flush=True)
    print("[fetch] complete.", flush=True)

# ================= stage: temporal table =================

def profile(works, y0=None, y1=None, topk=PROFILE_TOPK):
    cnt, n = Counter(), 0
    for w in works:
        y = w.get("year")
        if y is None: continue
        if y0 is not None and y < y0: continue
        if y1 is not None and y > y1: continue
        n += 1
        for c in w["concepts"]: cnt[c] += 1
    return set(c for c, _ in cnt.most_common(topk)), n, len(cnt)

def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0

def stage_table():
    pairs = pd.read_parquet(PAIRS_FILE)
    works_by_aid = load_store()
    print(f"[table] authors in store: {len(works_by_aid)}", flush=True)
    sub = pairs.dropna(subset=["st_openalex_id", "adv_openalex_id"]).copy()
    sub["st_aid"], sub["adv_aid"] = sub.st_openalex_id.map(_aid), sub.adv_openalex_id.map(_aid)
    sub = sub[sub.st_aid.isin(works_by_aid) & sub.adv_aid.isin(works_by_aid)]
    print(f"[table] pairs with both authors in store: {len(sub)}", flush=True)
    rows = []
    drop = {"no_years": 0, "adv_no_years": 0, "too_recent": 0,
            "implausible_t0": 0, "span_gt_max": 0, "sparse_windows": 0}
    for _, r in sub.iterrows():
        sw, aw = works_by_aid[r.st_aid], works_by_aid[r.adv_aid]
        years = [w["year"] for w in sw if w.get("year")]
        if not years: drop["no_years"] += 1; continue
        years_a = [w["year"] for w in aw if w.get("year")]
        if not years_a: drop["adv_no_years"] += 1; continue
        t0 = min(years)
        if t0 + LATE_YEARS > OBS_YEAR: drop["too_recent"] += 1; continue
        if t0 < T0_MIN: drop["implausible_t0"] += 1; continue
        if (max(years) - min(years) > MAX_CAREER_SPAN or
            max(years_a) - min(years_a) > MAX_CAREER_SPAN): drop["span_gt_max"] += 1; continue
        early, n_e, br_e = profile(sw, t0, t0 + EARLY_YEARS)
        late,  n_l, _    = profile(sw, t0 + EARLY_YEARS + 1, t0 + LATE_YEARS)
        advp,  n_a, br_a = profile(aw, None, t0 + EARLY_YEARS)
        if n_e < MIN_WORKS_PER_WINDOW or n_l < MIN_WORKS_PER_WINDOW or n_a < 3:
            drop["sparse_windows"] += 1; continue
        rows.append({
            "student_pid": r.student_pid, "advisor_pid": r.advisor_pid,
            "st_openalex_id": r.st_aid, "adv_openalex_id": r.adv_aid,
            "institution_name": r.institution_name, "t0": t0,
            "early_prod": n_e, "early_breadth": br_e,
            "early_concepts": sorted(early), "adv_profile": sorted(advp),
            "adv_early_prod": n_a, "adv_early_breadth": br_a,
            "adv_career_age_at_t0": int(t0 - min(years_a)),
            "early_overlap": round(jaccard(early, advp), 4),
            "late_prod": n_l, "late_overlap": round(jaccard(late, advp), 4),
            "y": int(jaccard(late, advp) > JACCARD_THETA),
        })
    lab = pd.DataFrame(rows).sort_values("early_prod", ascending=False).drop_duplicates("student_pid")
    lab.to_parquet(LABELS_CACHE)
    print(f"[table] {len(lab)} students | base rate y = {lab.y.mean():.3f} | dropped {drop}", flush=True)
    return lab

# ================= stage: co-authorship =================

def stage_coauth():
    import requests
    lab = pd.read_parquet(LABELS_CACHE)
    coauth = {}
    if os.path.exists(COAUTH_CACHE):
        with open(COAUTH_CACHE) as f:
            for line in f:
                try:
                    j = json.loads(line)
                    if j["n"] >= 0: coauth[j["key"]] = j["n"]
                except Exception:
                    pass
    print(f"[coauth] already cached (valid): {len(coauth)}", flush=True)
    if FETCH_COAUTH_EARLY:
        s = requests.Session()
        if OPENALEX_MAILTO: s.headers.update({"User-Agent": f"mailto:{OPENALEX_MAILTO}"})
        todo = [(r.st_openalex_id, r.adv_openalex_id, r.t0) for r in lab.itertuples()
                if f"{r.st_openalex_id}|{r.adv_openalex_id}" not in coauth]
        print(f"[coauth] pairs still to query: {len(todo)}", flush=True)
        MAX_RETRIES, stop, done = 6, False, 0
        out = open(COAUTH_CACHE, "a")
        try:
            for st, adv, t0 in todo:
                key = f"{st}|{adv}"
                u = ("https://api.openalex.org/works?filter="
                     f"authorships.author.id:{st},authorships.author.id:{adv},"
                     f"publication_year:{t0}-{t0 + EARLY_YEARS}&per-page=1")
                if OPENALEX_API_KEY: u += f"&api_key={OPENALEX_API_KEY}"
                if OPENALEX_MAILTO:  u += f"&mailto={OPENALEX_MAILTO}"
                n = None
                for attempt in range(MAX_RETRIES):
                    try:
                        r = s.get(u, timeout=30)
                    except Exception:
                        time.sleep(2 * (attempt + 1)); continue
                    if r.status_code == 200:
                        n = int(r.json().get("meta", {}).get("count", 0)); break
                    if r.status_code == 429:
                        time.sleep(min(60, 2 ** attempt)); continue
                    if r.status_code in (401, 403, 409):
                        print(f"[coauth] HTTP {r.status_code} after {done} - stopping (resumable)", flush=True)
                        stop = True; break
                    time.sleep(2 * (attempt + 1))
                if stop: break
                if n is None: continue
                coauth[key] = n
                out.write(json.dumps({"key": key, "n": n}) + "\n"); out.flush()
                done += 1
                if done % 200 == 0: print(f"[coauth] +{done}", flush=True)
                time.sleep(FETCH_SLEEP)
        finally:
            out.close()
        print(f"[coauth] newly fetched: {done} | total valid cached: {len(coauth)}", flush=True)
    lab["coauth_early_n"] = [coauth.get(f"{r.st_openalex_id}|{r.adv_openalex_id}", -1)
                             for r in lab.itertuples()]
    lab["coauth_early"]   = (lab["coauth_early_n"] > 0).astype(bool)
    lab.to_parquet(LABELS_CACHE)
    print(f"[coauth] rate: {lab.loc[lab.coauth_early_n >= 0, 'coauth_early'].mean():.3f} "
          f"| still missing: {int((lab.coauth_early_n < 0).sum())}", flush=True)
    return lab

# ================= stage: build (merge/clean/export) =================

def stage_build():
    pairs = pd.read_parquet(PAIRS_FILE)
    lab = pd.read_parquet(LABELS_CACHE)
    if "coauth_early_n" not in lab.columns:   # coauth stage not yet run
        lab["coauth_early_n"] = -1; lab["coauth_early"] = False
    # NEURO adaptation: source lacks st_name/adv_name/institution_country -> NA
    for c in ["st_name", "adv_name", "institution_country"]:
        if c not in pairs.columns: pairs[c] = pd.NA
    meta_cols = ["student_pid", "st_name", "adv_name", "institution_country"]
    meta = pairs.drop_duplicates("student_pid")[meta_cols]
    clean = lab.merge(meta, on="student_pid", how="left")
    clean["institution_country"] = (clean["institution_country"].astype("object")
                                    .replace({"NULL": pd.NA, "None": pd.NA, "": pd.NA}))
    clean["institution_name"] = (clean["institution_name"].astype("object")
                                 .replace({"NULL": pd.NA, "": pd.NA}))
    ID_COLS      = ["student_pid", "advisor_pid", "st_openalex_id", "adv_openalex_id",
                    "st_name", "adv_name"]
    FEATURE_COLS = ["t0", "early_prod", "early_breadth", "early_concepts",
                    "adv_profile", "adv_early_prod", "adv_early_breadth",
                    "adv_career_age_at_t0", "early_overlap",
                    "coauth_early", "coauth_early_n",
                    "institution_name", "institution_country"]
    LABEL_COLS   = ["y", "late_overlap", "late_prod"]
    BANNED = {"st_h_index", "adv_h_index", "adv_total_citations", "st_h_recomp",
              "co_authored", "st_cites_adv_broad", "adv_cites_st_broad", "career_len"}
    assert not (BANNED & set(clean.columns)), f"leakage columns present"
    unclassified = set(clean.columns) - set(ID_COLS + FEATURE_COLS + LABEL_COLS)
    assert not unclassified, f"unclassified columns: {unclassified}"
    assert (clean.t0 + LATE_YEARS <= OBS_YEAR).all(), "right-censored rows present"
    assert (clean.t0 >= T0_MIN).all(), "implausible t0 present"
    assert ((clean.late_overlap > JACCARD_THETA).astype(int) == clean.y).all(), "label mismatch"
    assert clean.student_pid.is_unique, "duplicate students"
    print("[build] leakage guard passed.", flush=True)
    clean = clean[ID_COLS + FEATURE_COLS + LABEL_COLS]
    out_parquet = os.path.join(BASE, "clean_dataset.parquet")
    clean.to_parquet(out_parquet)
    sha = hashlib.sha256(open(out_parquet, "rb").read()).hexdigest()
    csv = clean.copy()
    csv["early_concepts"] = csv["early_concepts"].apply(lambda l: "; ".join(l))
    csv["adv_profile"]    = csv["adv_profile"].apply(lambda l: "; ".join(l))
    csv.to_csv(os.path.join(BASE, "clean_dataset.csv"), index=False)
    summary = {
        "schema_version": 2, "sha256_parquet": sha,
        "n_students": int(len(clean)), "n_advisors": int(clean.advisor_pid.nunique()),
        "n_institutions": int(clean.institution_name.nunique()),
        "label_base_rate": round(float(clean.y.mean()), 4),
        "t0_range": [int(clean.t0.min()), int(clean.t0.max())],
        "coauth_early_rate": round(float(clean.loc[clean.coauth_early_n >= 0, "coauth_early"].mean()), 4)
                             if (clean.coauth_early_n >= 0).any() else None,
        "country_missing_rate": round(float(clean.institution_country.isna().mean()), 4),
        "id_cols": ID_COLS, "feature_cols": FEATURE_COLS, "label_cols": LABEL_COLS,
        "params": {"early_years": EARLY_YEARS, "late_years": LATE_YEARS,
                   "min_works_per_window": MIN_WORKS_PER_WINDOW, "theta": JACCARD_THETA,
                   "topk": PROFILE_TOPK, "concept_min_score": CONCEPT_MIN_SCORE,
                   "t0_min": T0_MIN, "max_career_span": MAX_CAREER_SPAN,
                   "max_works_per_author": MAX_WORKS_PER_AUTHOR,
                   "sort": "publication_date:asc", "obs_year": OBS_YEAR},
        "discipline": "neuroscience",
    }
    json.dump(summary, open(os.path.join(BASE, "dataset_summary.json"), "w"), indent=2)
    print(f"[build] clean_dataset.parquet rows={len(clean)} base_rate={clean.y.mean():.3f} sha={sha[:12]}", flush=True)
    return clean

def stage_outcomes():
    works_by_aid = load_store()
    clean = pd.read_parquet(os.path.join(BASE, "clean_dataset.parquet"))
    yr_list, ct_list = [], []
    for w_list in works_by_aid.values():
        for w in w_list:
            y = w.get("year")
            if y and T0_MIN <= y <= OBS_YEAR:
                yr_list.append(y); ct_list.append(int(w.get("cited_by_count") or 0))
    uni = pd.DataFrame({"year": yr_list, "cites": ct_list})
    uni["pct"] = uni.groupby("year")["cites"].rank(pct=True)
    pct_map = uni.groupby(["year", "cites"])["pct"].max().to_dict()
    del uni, yr_list, ct_list
    out_rows = []
    for r in clean.itertuples():
        sw = works_by_aid.get(r.st_openalex_id, [])
        cites = sorted([int(w.get("cited_by_count") or 0) for w in sw], reverse=True)
        h_full = sum(1 for k, c in enumerate(cites) if c >= k + 1)
        late_pcts = [pct_map.get((w["year"], int(w.get("cited_by_count") or 0)))
                     for w in sw if w.get("year") and r.t0 + EARLY_YEARS < w["year"] <= r.t0 + LATE_YEARS]
        late_pcts = [p for p in late_pcts if p is not None]
        out_rows.append({"student_pid": r.student_pid, "st_openalex_id": r.st_openalex_id,
                         "h_index_full_career": h_full,
                         "late_cite_pct_mean": round(float(np.mean(late_pcts)), 4) if late_pcts else np.nan,
                         "late_cite_pct_n": len(late_pcts)})
    outcomes = pd.DataFrame(out_rows)
    outcomes.to_parquet(os.path.join(BASE, "outcomes.parquet"))
    print(f"[outcomes] {len(outcomes)} rows", flush=True)

STAGES = {"fetch": stage_fetch, "table": stage_table, "coauth": stage_coauth,
          "build": stage_build, "outcomes": stage_outcomes}

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    order = ["fetch", "table", "coauth", "build", "outcomes"]
    todo = order if args == ["all"] else args
    for st in todo:
        if st not in STAGES: print(f"unknown stage {st}"); continue
        print(f"==== stage {st} ====", flush=True); STAGES[st]()
