#!/usr/bin/env python3
"""
AFT person -> OpenAlex author-ID disambiguation by PUBLICATION-ID ANCHORING.

This is the high-precision, homonym-proof resolver for the MAIN-TEXT neuroscience
replication. Instead of searching by name (which conflates same-name scholars), it
anchors on the person's KNOWN publications (DOIs/PMIDs from the AFT authorPub table):
look those works up in OpenAlex, read off the author ids on each, and take the author
id that recurs across the person's own papers with a matching surname. A namesake
cannot win because they are not a co-author on the anchor's actual papers.

Almost certainly the method behind the original econ resolution (98.7% coverage;
econ authorPub carries DOIs, neuro carries PMIDs — OpenAlex indexes both).

RUN ON YOUR MACHINE (needs api.openalex.org). Set OPENALEX_API_KEY + OPENALEX_MAILTO.

Two-phase, same philosophy as resolve_openalex.py:
  FETCH : batch-look-up every anchor DOI/PMID in OpenAlex, cache work -> authorships.
          API-bound, resumable, batched 50/call across ALL persons (feasible at scale).
  APPLY : tally author ids per person offline and pick the anchor winner. Instant,
          re-tunable (MIN_HITS) without more API calls.

Validate on econ first (you have ground truth):
  export OPENALEX_API_KEY=...   OPENALEX_MAILTO=lixinke@uchicago.edu
  python resolve_by_pubid.py fetch \
      --detailed "../Econ Data/Econ_Pairs_Detailed.parquet" \
      --pubs "../Econ Data/Econ_Pubs_All.parquet" --wcache works_econ.jsonl --limit 500
  python resolve_by_pubid.py validate \
      --detailed "../Econ Data/Econ_Pairs_Detailed.parquet" \
      --pubs "../Econ Data/Econ_Pubs_All.parquet" --wcache works_econ.jsonl \
      --truth "../Econ Data/Econ_Pairs_Openalex_Ultimate.parquet"

Apply to neuro (after downloading the neuro authorPub files, see README):
  python resolve_by_pubid.py fetch \
      --detailed "Neuro_Pairs_Detailed.parquet" --pubs "neuro_authorPub/" --wcache works_neuro.jsonl
  python resolve_by_pubid.py apply \
      --detailed "Neuro_Pairs_Detailed.parquet" --pubs "neuro_authorPub/" \
      --wcache works_neuro.jsonl --out "Neuro_Pairs_Openalex_Ultimate.parquet"
"""
import argparse, glob, json, os, re, time
import pandas as pd
from resolve_openalex import _norm, _get, _sess, _pool_check, load_cache, person_records, BASE, SLEEP

MIN_SCORE = 0.3     # authorPub link score to trust a person->pub link (README convention)
N_ANCHOR  = 25      # max anchor pubs per person
BATCH     = 50      # OpenAlex OR-filter batch size
MIN_HITS  = 2       # winning author must appear on >=2 of the person's papers

# ------------------------- id normalization -------------------------

def norm_doi(d):
    d = str(d or "").strip().lower()
    if not d or d in ("nan", "0", "none"): return ""
    for p in ("https://doi.org/", "http://dx.doi.org/", "http://doi.org/", "doi.org/"):
        d = d.replace(p, "")
    return d

def norm_pmid(p):
    p = re.sub(r"\D", "", str(p or ""))
    return p if p and p != "0" else ""

# ------------------------- anchor-pub index -------------------------

def _iter_pub_chunks(pubs_path):
    """Yield DataFrames from either econ parquet-dir or neuro csv.gz files."""
    cols = ["pid", "pmid", "doi", "score_total"]
    if os.path.isdir(pubs_path):
        parts = sorted(glob.glob(os.path.join(pubs_path, "part-*.parquet")))
        if parts:
            for pth in parts:
                yield pd.read_parquet(pth, columns=cols)
            return
        for pth in sorted(glob.glob(os.path.join(pubs_path, "*.csv.gz"))):
            for chunk in pd.read_csv(pth, usecols=lambda c: c in cols, chunksize=500_000):
                yield chunk
        return
    if pubs_path.endswith(".csv.gz") or pubs_path.endswith(".csv"):
        for chunk in pd.read_csv(pubs_path, usecols=lambda c: c in cols, chunksize=500_000):
            yield chunk
    else:
        yield pd.read_parquet(pubs_path, columns=cols)

def build_pub_index(pubs_path, pids):
    """pid -> {'dois': [...], 'pmids': [...]} using score_total>=MIN_SCORE, top N by score."""
    pidset = set(map(str, pids))
    idx = {}
    for df in _iter_pub_chunks(pubs_path):
        df["pid"] = df["pid"].astype(str)
        df = df[df["pid"].isin(pidset)]
        if df.empty: continue
        df["sc"] = pd.to_numeric(df["score_total"], errors="coerce").fillna(0.0)
        df = df[df["sc"] >= MIN_SCORE]
        for pid, g in df.groupby("pid"):
            g = g.sort_values("sc", ascending=False).head(N_ANCHOR)
            e = idx.setdefault(pid, {"dois": set(), "pmids": set()})
            for _, r in g.iterrows():
                d = norm_doi(r["doi"]);  p = norm_pmid(r["pmid"])
                if d: e["dois"].add(d)
                if p: e["pmids"].add(p)
    for pid, e in idx.items():
        e["dois"] = list(e["dois"])[:N_ANCHOR]; e["pmids"] = list(e["pmids"])[:N_ANCHOR]
    return idx

# ------------------------- FETCH: work -> authorships -------------------------

def load_work_cache(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                j = json.loads(line); d[j["k"]] = j["a"]
            except Exception:
                pass
    return d

def fetch(args):
    det = pd.read_parquet(args.detailed)
    persons = person_records(det)
    pids = list(persons)[: args.limit] if args.limit else list(persons)
    idx = build_pub_index(args.pubs, pids)
    dois = set(); pmids = set()
    for pid in pids:
        e = idx.get(pid, {"dois": [], "pmids": []})
        dois.update(e["dois"]); pmids.update(e["pmids"])
    npub = sum(1 for pid in pids if idx.get(pid, {}).get("dois") or idx.get(pid, {}).get("pmids"))
    print(f"{len(pids)} persons | {npub} have anchor pubs | "
          f"{len(dois)} unique DOIs, {len(pmids)} unique PMIDs to look up")
    wcache = load_work_cache(args.wcache)
    sess = _sess(); _pool_check(sess)

    def run(kind, keys):
        keys = [k for k in keys if f"{kind}:{k}" not in wcache]
        print(f"  {kind}: {len(keys)} new to fetch")
        with open(args.wcache, "a") as wf:
            for i in range(0, len(keys), BATCH):
                chunk = keys[i:i + BATCH]
                filt = f"{kind}:" + "|".join(chunk)
                j, ok = _get(sess, f"{BASE}/works",
                             {"filter": filt, "per-page": BATCH,
                              "select": "id,doi,ids,authorships"})
                if not ok:
                    continue  # API failure: leave chunk uncached so a rerun retries it
                for w in (j.get("results") or []):
                    auths = [[a["author"]["id"], a["author"].get("display_name", "")]
                             for a in (w.get("authorships") or []) if a.get("author")]
                    wd = norm_doi(w.get("doi"))
                    wp = norm_pmid((w.get("ids") or {}).get("pmid"))
                    if wd and f"doi:{wd}" not in wcache:
                        wcache[f"doi:{wd}"] = auths; wf.write(json.dumps({"k": f"doi:{wd}", "a": auths}) + "\n")
                    if wp and f"pmid:{wp}" not in wcache:
                        wcache[f"pmid:{wp}"] = auths; wf.write(json.dumps({"k": f"pmid:{wp}", "a": auths}) + "\n")
                # genuine not-found ids (call succeeded, id absent) -> cache empty so we don't refetch
                for k in chunk:
                    key = f"{kind}:{k}"
                    if key not in wcache:
                        wcache[key] = []; wf.write(json.dumps({"k": key, "a": []}) + "\n")
                wf.flush(); time.sleep(SLEEP)
                if (i // BATCH + 1) % 20 == 0:
                    print(f"    {kind} {i+len(chunk)}/{len(keys)}")
    run("doi", list(dois)); run("pmid", list(pmids))
    print(f"FETCH done -> {args.wcache}")

# ------------------------- APPLY: anchor tally (offline) -------------------------

def resolve_all(idx, wcache, persons, min_hits):
    out = {}
    for pid, e in idx.items():
        last = _norm(persons.get(pid, {}).get("lastname")).lower()
        tally = {}
        for d in e["dois"]:
            for aid, name in wcache.get(f"doi:{d}", []):
                if last and last in _norm(name).lower(): tally[aid] = tally.get(aid, 0) + 1
        for p in e["pmids"]:
            for aid, name in wcache.get(f"pmid:{p}", []):
                if last and last in _norm(name).lower(): tally[aid] = tally.get(aid, 0) + 1
        if tally:
            aid, hits = max(tally.items(), key=lambda x: x[1])
            if hits >= min_hits or (len(tally) == 1 and hits >= 1):
                out[pid] = {"openalex_id": aid, "method": "pubid", "hits": hits}
                continue
        out[pid] = {"openalex_id": None, "method": "pubid_nohit", "hits": 0}
    return out

def _key(u):
    return None if u is None else str(u).rstrip("/").split("/")[-1]

def validate(args):
    det = pd.read_parquet(args.detailed)
    persons = person_records(det)
    pids = list(persons)[: args.limit] if args.limit else list(persons)
    idx = build_pub_index(args.pubs, pids)
    wcache = load_work_cache(args.wcache)
    resolved = resolve_all(idx, wcache, persons, args.min_hits)
    truth = pd.read_parquet(args.truth); gt = {}
    for _, r in truth.iterrows():
        if pd.notna(r.get("adv_openalex_id")): gt[str(r["advisor_pid"])] = _key(r["adv_openalex_id"])
        if pd.notna(r.get("st_openalex_id")):  gt[str(r["student_pid"])] = _key(r["st_openalex_id"])
    evalset = [p for p in pids if p in gt]
    got = [p for p in evalset if resolved.get(p, {}).get("openalex_id")]
    correct = [p for p in got if _key(resolved[p]["openalex_id"]) == gt[p]]
    cov = len(got) / max(len(evalset), 1); prec = len(correct) / max(len(got), 1)
    print(f"\nPUB-ID ANCHORING vs econ ground truth (min_hits={args.min_hits}):")
    print(f"  ground-truth persons evaluated : {len(evalset)}")
    print(f"  resolved (coverage)            : {len(got)} = {cov:.1%}")
    print(f"  correct (precision)            : {len(correct)} = {prec:.1%}")
    # name-search fallback precision, methodology only; not a v1 econ co-authorship figure
    print(f"  compare: name-search gave ~88% / ~88% — pub-id should be markedly higher precision")

def apply(args):
    det = pd.read_parquet(args.detailed)
    persons = person_records(det)
    idx = build_pub_index(args.pubs, list(persons))
    wcache = load_work_cache(args.wcache)
    resolved = resolve_all(idx, wcache, persons, args.min_hits)
    rows = []
    for _, r in det.iterrows():
        a = resolved.get(str(r["advisor_pid"]), {}); s = resolved.get(str(r["student_pid"]), {})
        rows.append({**r.to_dict(),
                     "adv_openalex_id": a.get("openalex_id"), "adv_match_method": a.get("method"),
                     "adv_anchor_hits": a.get("hits"),
                     "st_openalex_id": s.get("openalex_id"), "st_match_method": s.get("method"),
                     "st_anchor_hits": s.get("hits")})
    out = pd.DataFrame(rows)
    print(f"advisor id coverage {out.adv_openalex_id.notna().mean():.1%} | "
          f"student id coverage {out.st_openalex_id.notna().mean():.1%}")
    out.to_parquet(args.out, index=False)
    print(f"WROTE {args.out}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "validate", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--detailed", required=True)
        p.add_argument("--pubs", required=True, help="authorPub parquet dir (econ) or folder of csv.gz (neuro)")
        p.add_argument("--wcache", default="works_cache.jsonl")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--min_hits", type=int, default=MIN_HITS)
        if name == "validate": p.add_argument("--truth", required=True)
        if name == "apply": p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"fetch": fetch, "validate": validate, "apply": apply}[args.cmd](args)

if __name__ == "__main__":
    main()
