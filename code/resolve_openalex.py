#!/usr/bin/env python3
"""
AFT person -> OpenAlex author-ID disambiguation (fetch + apply).

Reproduces the upstream step that produced Econ_Pairs_Openalex_Ultimate.parquet,
so the same protocol can be applied to the neuroscience tree.

RUN ON YOUR OWN MACHINE (needs network to api.openalex.org).

Two-phase design so scoring can be tuned offline (see tune_resolver.py):
  Phase FETCH  : ORCID lookup + name search -> cache RAW candidates per person.
                 API-bound, resumable. Cache file: oa_candidates.jsonl
  Phase APPLY  : score cached candidates with a weight/threshold config and write
                 the resolved parquet. Pure local, instant, repeatable.

Typical flow
------------
  export OPENALEX_MAILTO=lixinke@uchicago.edu

  # 1. fetch candidates for econ (ground-truth set) and validate/tune
  python resolve_openalex.py fetch --detailed "../Econ Data/Econ_Pairs_Detailed.parquet" \
         --field economics --cache oa_candidates_econ.jsonl
  python tune_resolver.py --cache oa_candidates_econ.jsonl \
         --truth "../Econ Data/Econ_Pairs_Openalex_Ultimate.parquet" --field economics
  #    -> writes best_config.json

  # 2. fetch candidates for neuro and apply the tuned config
  python resolve_openalex.py fetch --detailed "Neuro_Pairs_Detailed.parquet" \
         --field neuroscience --cache oa_candidates_neuro.jsonl
  python resolve_openalex.py apply --detailed "Neuro_Pairs_Detailed.parquet" \
         --cache oa_candidates_neuro.jsonl --config best_config.json \
         --out "Neuro_Pairs_Openalex_Ultimate.parquet"
"""
import argparse, json, os, time, unicodedata, re
import pandas as pd
import requests

MAILTO = os.environ.get("OPENALEX_MAILTO", "")
API_KEY = os.environ.get("OPENALEX_API_KEY", "")
BASE = "https://api.openalex.org"
SLEEP = 0.01 if API_KEY else 0.11
NAME_SEARCH_N = 25
# minimal author fields for offline scoring (smaller payload = faster calls)
AUTHOR_SELECT = "id,display_name,works_count,x_concepts,affiliations,last_known_institutions"

# root-concept prior: OpenAlex concepts we expect this field's authors to touch
FIELD_CONCEPTS = {
    "economics":    {"Economics"},
    "neuroscience": {"Neuroscience", "Psychology", "Biology", "Medicine"},
}

# ---- default scoring (overwritten by best_config.json after tuning) ----
DEFAULT_WEIGHTS = {"lastname": 2.0, "firstinit": 0.5, "inst": 2.0, "field": 1.5, "works_ok": 0.5}
DEFAULT_THRESHOLD = 3.0

# ============================ shared helpers ============================

def _norm(s):
    if s is None or str(s) in ("NULL", "None", "nan"): return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip()

def _inst_tokens(s):
    s = _norm(s).lower()
    for junk in ["university of", "the ", "college", "school", "institute", ",", "."]:
        s = s.replace(junk, " ")
    return {t for t in s.split() if len(t) > 3}

def featurize(cand, last, first, inst_tokens, field):
    """Return raw 0/1 features for one candidate. Shared by resolve + tune so the
    grid tunes exactly what production scores."""
    dn = _norm(cand.get("display_name")).lower()
    f = {}
    f["lastname"]  = 1.0 if last and last.lower() in dn else 0.0
    f["firstinit"] = 1.0 if first and dn and first[:1].lower() == dn[:1].lower() else 0.0
    aff_txt = " ".join(_norm(x).lower() for x in (cand.get("inst_names") or []))
    aff_tokens = {t for t in re.split(r"\W+", aff_txt) if len(t) > 3}
    f["inst"] = 1.0 if inst_tokens and (inst_tokens & aff_tokens) else 0.0
    want = FIELD_CONCEPTS.get(field, set())
    f["field"] = 1.0 if want and (want & set(cand.get("concepts") or [])) else 0.0
    wc = cand.get("works_count") or 0
    f["works_ok"] = 1.0 if 3 <= wc <= 2000 else 0.0
    return f

def score(features, weights):
    return sum(weights.get(k, 0.0) * v for k, v in features.items())

def best_match(person, cands, weights, threshold, field):
    """Pick the best OpenAlex id for a person given cached candidates + config."""
    if person.get("orcid_hit"):
        return {"openalex_id": person["orcid_hit"], "method": "orcid", "confidence": 99.0}
    last = _norm(person.get("lastname")); first = _norm(person.get("firstname"))
    inst_tokens = _inst_tokens(person.get("institution"))
    best, best_s = None, -1.0
    for c in cands:
        s = score(featurize(c, last, first, inst_tokens, field), weights)
        if s > best_s: best, best_s = c, s
    if best is not None and best_s >= threshold:
        return {"openalex_id": best["id"], "method": "name", "confidence": round(best_s, 3)}
    return {"openalex_id": None, "method": "below_threshold", "confidence": round(best_s, 3)}

# ============================ person records ============================

def person_records(det):
    seen = {}
    for _, r in det.iterrows():
        for role, pidcol in (("adv", "advisor_pid"), ("stu", "student_pid")):
            pid = str(r[pidcol])
            if pid in seen: continue
            seen[pid] = {"pid": pid,
                         "firstname": r.get(f"{role}_firstname"),
                         "middlename": r.get(f"{role}_middlename"),
                         "lastname": r.get(f"{role}_lastname"),
                         "orcid": r.get(f"{role}_orcid"),
                         "institution": r.get("institution_name")}
    return seen

# ============================ FETCH phase ============================

def _sess():
    s = requests.Session()
    s.headers.update({"User-Agent": f"aft-resolver/1.0 (mailto:{MAILTO})"})
    return s

def _params(extra):
    p = dict(extra)
    if MAILTO: p["mailto"] = MAILTO
    if API_KEY: p["api_key"] = API_KEY
    return p

def _get(sess, url, params, tries=6):
    """Return (json_or_None, ok). ok=False means the call FAILED after retries
    (429/5xx/transport) and must NOT be cached as a genuine 'no result'.
    ok=True with None means a real 404 (no such record)."""
    for i in range(tries):
        try:
            r = sess.get(url, params=_params(params), timeout=45)
            if r.status_code == 200:
                return r.json(), True
            if r.status_code == 404:
                return None, True
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After", 0) or 0) or min(2 ** i, 30)
                time.sleep(wait); continue
            time.sleep(min(2 ** i, 30))
        except requests.RequestException:
            time.sleep(min(2 ** i, 30))
    return None, False

def _pool_check(sess):
    """Verify we're in the fast/polite pool before spending a long fetch.
    Prints x-api-pool and the credentials the script actually sees."""
    try:
        r = sess.get(f"{BASE}/authors/A5023888391", params=_params({"select": "id"}), timeout=30)
        pool = r.headers.get("x-api-pool")
        print(f"[pool check] HTTP {r.status_code} | x-api-pool={pool} | "
              f"api_key={'set' if API_KEY else 'MISSING'} | mailto={MAILTO or 'MISSING'}")
        if not API_KEY and pool in (None, "None", "common"):
            print("  WARNING: not in a fast pool. Set OPENALEX_API_KEY in THIS shell, "
                  "or expect heavy 429 throttling.")
        return pool
    except requests.RequestException as e:
        print(f"[pool check] failed: {e}")
        return None

def _slim(a):
    """Keep only fields needed for offline scoring, so the cache stays small."""
    inst = [x.get("institution", {}).get("display_name", "") for x in (a.get("affiliations") or [])]
    lki = (a.get("last_known_institutions") or [])
    inst += [x.get("display_name", "") for x in lki]
    return {"id": a.get("id"), "display_name": a.get("display_name"),
            "works_count": a.get("works_count"),
            "concepts": [c.get("display_name") for c in (a.get("x_concepts") or [])[:8]],
            "inst_names": [x for x in inst if x]}

def load_cache(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                j = json.loads(line); d[j["pid"]] = j
            except Exception:
                pass
    return d

def fetch(args):
    det = pd.read_parquet(args.detailed)
    persons = person_records(det)
    pids = list(persons)[: args.limit] if args.limit else list(persons)
    cache = load_cache(args.cache)
    print(f"{len(pids)} persons; {len(cache)} already cached; field={args.field}")
    sess = _sess()
    _pool_check(sess)
    n_done = n_fail = 0
    with open(args.cache, "a") as cf:
        for i, pid in enumerate(pids):
            if pid in cache: continue
            rec = persons[pid]
            entry = {"pid": pid, "firstname": rec["firstname"], "lastname": rec["lastname"],
                     "institution": rec["institution"], "orcid_hit": None, "candidates": []}
            orcid = _norm(rec.get("orcid"))
            if orcid:
                j, ok = _get(sess, f"{BASE}/authors/orcid:{orcid}", {"select": AUTHOR_SELECT})
                if ok and j and j.get("id"):
                    entry["orcid_hit"] = j["id"]
            if not entry["orcid_hit"]:
                last = _norm(rec.get("lastname"))
                name = " ".join(x for x in [_norm(rec.get("firstname")),
                                            _norm(rec.get("middlename")), last] if x)
                if last:
                    j, ok = _get(sess, f"{BASE}/authors",
                                 {"search": name, "per-page": NAME_SEARCH_N, "select": AUTHOR_SELECT})
                    if not ok:
                        # API FAILED (429/5xx/transport). Do NOT cache — leave the pid
                        # uncached so a rerun retries it. This is the bug that poisoned
                        # the first econ cache with throttle-failures stored as [].
                        n_fail += 1
                        continue
                    if j and j.get("results"):
                        entry["candidates"] = [_slim(a) for a in j["results"]]
            cache[pid] = entry
            cf.write(json.dumps(entry) + "\n"); cf.flush()
            n_done += 1
            time.sleep(SLEEP)
            if n_done % 200 == 0:
                print(f"  cached {n_done} (failed-so-far {n_fail})")
    print(f"FETCH done -> {args.cache} | newly cached {n_done} | API failures skipped {n_fail}")
    if n_fail:
        print(f"  {n_fail} persons hit API failures and were NOT cached; rerun the same "
              f"command to retry only those.")

# ============================ APPLY phase ============================

def apply(args):
    det = pd.read_parquet(args.detailed)
    cache = load_cache(args.cache)
    if args.config and os.path.exists(args.config):
        cfg = json.load(open(args.config)); weights = cfg["weights"]; thr = cfg["threshold"]
        print(f"loaded config {args.config}: threshold={thr} weights={weights}")
    else:
        weights, thr = DEFAULT_WEIGHTS, DEFAULT_THRESHOLD
        print("using DEFAULT config (no --config given)")
    resolved = {pid: best_match(e, e.get("candidates", []), weights, thr, args.field)
                for pid, e in cache.items()}
    rows = []
    for _, r in det.iterrows():
        a = resolved.get(str(r["advisor_pid"]), {}); s = resolved.get(str(r["student_pid"]), {})
        rows.append({**r.to_dict(),
                     "adv_openalex_id": a.get("openalex_id"), "adv_match_method": a.get("method"),
                     "adv_match_conf": a.get("confidence"),
                     "st_openalex_id": s.get("openalex_id"), "st_match_method": s.get("method"),
                     "st_match_conf": s.get("confidence")})
    out = pd.DataFrame(rows)
    cov_a = out.adv_openalex_id.notna().mean(); cov_s = out.st_openalex_id.notna().mean()
    print(f"advisor id coverage {cov_a:.1%} | student id coverage {cov_s:.1%}")
    out.to_parquet(args.out, index=False)
    print(f"WROTE {args.out}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("fetch"); pf.add_argument("--detailed", required=True)
    pf.add_argument("--cache", default="oa_candidates.jsonl")
    pf.add_argument("--field", default="neuroscience", choices=list(FIELD_CONCEPTS))
    pf.add_argument("--limit", type=int, default=None)
    pa = sub.add_parser("apply"); pa.add_argument("--detailed", required=True)
    pa.add_argument("--cache", default="oa_candidates.jsonl")
    pa.add_argument("--config", default="best_config.json")
    pa.add_argument("--field", default="neuroscience", choices=list(FIELD_CONCEPTS))
    pa.add_argument("--out", required=True)
    args = ap.parse_args()
    {"fetch": fetch, "apply": apply}[args.cmd](args)

if __name__ == "__main__":
    main()
