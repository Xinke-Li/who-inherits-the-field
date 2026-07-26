#!/usr/bin/env python3
"""R5 - One-time OpenAlex works fetch for the top-k robustness sweep (P0-4).

POST-HOC ROBUSTNESS INFRASTRUCTURE, not part of the pre-registered protocol.
The frozen tables store only the k=10 late_overlap; varying the profile size k
requires per-author work records. This script fetches every author appearing in
the five frozen modeling tables ONCE into a local parquet cache; all k variants
(and concept-min-score variants) are derived offline by r6_topk_sweep.py.
The API is never looped per k.

Differences from the builder's fetch (build_neuro_dataset.fetch_author_works,
copied here with modification - source named per revision policy):
  * concepts are stored as [display_name_lower, score] pairs for EVERY concept
    with score >= 0.2, in API order (descending score), WITHOUT the builder's
    ">= 0.3 then [:3]" collapse. The builder's exact view is recovered offline
    by  [c for c,s in concepts if s >= 0.3][:3];  min-score 0.2/0.4 variants
    likewise. Everything else (endpoint, select fields, publication_date:asc,
    per-page 200, 400-works cap) matches the builder verbatim.
  * a small thread pool with a global rate limiter replaces the serial loop
    (the builder fetched one discipline at a time; this run covers all five).

CAUTION (documented in the paper's Appendix G): the live API is NOT the 2026
frozen snapshot. r6 therefore first rebuilds k=10 as a calibration baseline and
reads the k sweep against that rebuilt reference, never against frozen numbers.

Cache layout: results/robustness/openalex_cache/works_NNNNN.parquet
  columns: aid (str), works_json (json string:
           [{"year": int|null, "concepts": [[name, score], ...], "cited_by_count": int}, ...])
Resumable: existing shards are scanned for done aids at startup; a shard is
written every SHARD_SIZE completed authors (crash loses at most one shard).

Usage:  python code/r5_fetch_author_works.py
Key:    .openalex_key at the project root (one line), else env OPENALEX_API_KEY.
        The key is never printed and never written into any output file.
"""
import json
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "results" / "robustness" / "openalex_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = ["chemistry", "econ", "math", "neuro", "physics"]
MAX_WORKS_PER_AUTHOR = 400      # builder: MAX_WORKS_PER_AUTHOR
STORE_MIN_SCORE = 0.2           # superset of every planned min-score variant
SHARD_SIZE = 1000
N_WORKERS = 6
RATE_PER_SEC = 8.0              # global cap across workers, below the API limit
MAILTO = "lixinke@uchicago.edu"  # builder's OPENALEX_MAILTO


def load_api_key() -> str:
    """Read .openalex_key (one line) or fall back to OPENALEX_API_KEY.
    Never print or persist the returned value."""
    p = REPO / ".openalex_key"
    if p.exists():
        key = p.read_text().strip()
    else:
        key = os.environ.get("OPENALEX_API_KEY", "").strip()
    assert key, "no OpenAlex key: .openalex_key missing/empty and OPENALEX_API_KEY unset"
    return key


API_KEY = load_api_key()


class RateLimiter:
    """Token bucket shared by all workers: at most RATE_PER_SEC requests/sec."""

    def __init__(self, rate):
        self.interval = 1.0 / rate
        self.lock = threading.Lock()
        self.next_t = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if self.next_t <= now:
                self.next_t = now + self.interval
                return
            delay = self.next_t - now
            self.next_t += self.interval
        time.sleep(delay)


LIMITER = RateLimiter(RATE_PER_SEC)
_local = threading.local()


def _session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def fetch_author_works(aid: str) -> list:
    """Copied from build_neuro_dataset.fetch_author_works, modified as documented
    in the module docstring (score-carrying concepts, shared rate limiter)."""
    out, cursor, fetched = [], "*", 0
    base_url = ("https://api.openalex.org/works?filter=authorships.author.id:{aid}"
                "&per-page=200&sort=publication_date:asc"
                "&select=publication_year,concepts,cited_by_count&cursor={cur}")
    use_key = True
    tries = 0
    while cursor and fetched < MAX_WORKS_PER_AUTHOR:
        try:
            u = base_url.format(aid=aid, cur=cursor) + f"&mailto={MAILTO}"
            if use_key:
                u += f"&api_key={API_KEY}"
            LIMITER.wait()
            r = _session().get(u, timeout=30)
            if r.status_code == 403 and use_key:
                print("[warn] api_key rejected - falling back to polite pool", flush=True)
                use_key = False
                continue
            if r.status_code == 429:
                time.sleep(5)
                continue
            r.raise_for_status()
            js = r.json()
            tries = 0
        except Exception as e:
            # transient network errors (keep-alive resets) get 3 retries with
            # backoff; only then keep whatever was fetched (builder behavior)
            tries += 1
            if tries <= 3:
                time.sleep(2 * tries)
                _local.s = requests.Session()  # drop the stale connection
                continue
            print(f"[warn] {aid}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            break
        for w in js.get("results", []):
            cs = [[c["display_name"].lower(), round(float(c.get("score", 0)), 4)]
                  for c in w.get("concepts", []) if c.get("score", 0) >= STORE_MIN_SCORE]
            out.append({"year": w.get("publication_year"), "concepts": cs,
                        "cited_by_count": w.get("cited_by_count", 0)})
        fetched += len(js.get("results", []))
        cursor = js.get("meta", {}).get("next_cursor")
        if not js.get("results"):
            break
    return out


def author_ids() -> list:
    ids = set()
    for f in FIELDS:
        df = pd.read_parquet(REPO / "data" / f"clean_dataset_{f}.parquet",
                             columns=["st_openalex_id", "adv_openalex_id"])
        ids |= set(df.st_openalex_id) | set(df.adv_openalex_id)
    return sorted(ids)


def done_aids() -> set:
    done = set()
    for p in sorted(CACHE_DIR.glob("works_*.parquet")):
        done |= set(pd.read_parquet(p, columns=["aid"]).aid)
    return done


def empty_aids() -> list:
    """Authors cached with zero works - candidates for the --repair pass
    (a truncated fetch and a legitimately workless author look the same in
    the cache, so both are refetched once; refetching an empty author is one
    cheap request)."""
    out = []
    for p in sorted(CACHE_DIR.glob("works_*.parquet")):
        df = pd.read_parquet(p)
        out += df.loc[df.works_json == "[]", "aid"].tolist()
    return sorted(set(out))


def repair():
    todo = empty_aids()
    print(f"[r5:repair] {len(todo)} zero-work authors to refetch", flush=True)
    buf = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for aid, works in zip(todo, ex.map(fetch_author_works, todo)):
            if works:
                buf.append((aid, json.dumps(works)))
    if buf:
        # name sorts after every works_NNNNN shard; r6's loader lets later
        # shards win, so these rows override the empty entries
        df = pd.DataFrame(buf, columns=["aid", "works_json"])
        path = CACHE_DIR / "works_zrepair_00000.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            df = pd.concat([old[~old.aid.isin(df.aid)], df], ignore_index=True)
        df.to_parquet(path, index=False)
    print(f"[r5:repair] {len(buf)} authors recovered with works", flush=True)


def main():
    ids = author_ids()
    done = done_aids()
    todo = [a for a in ids if a not in done]
    n_shard_start = len(list(CACHE_DIR.glob("works_*.parquet")))
    print(f"[r5] {len(ids)} authors total | {len(done)} cached | {len(todo)} to fetch "
          f"| {n_shard_start} shards present", flush=True)

    buf, shard_no = [], n_shard_start
    t0 = time.time()
    recent = deque(maxlen=1000)
    lock = threading.Lock()

    def flush_shard():
        nonlocal buf, shard_no
        if not buf:
            return
        df = pd.DataFrame(buf, columns=["aid", "works_json"])
        path = CACHE_DIR / f"works_{shard_no:05d}.parquet"
        df.to_parquet(path, index=False)
        shard_no += 1
        buf = []

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, (aid, works) in enumerate(
                zip(todo, ex.map(lambda a: fetch_author_works(a), todo))):
            with lock:
                buf.append((aid, json.dumps(works)))
                recent.append(time.time())
                if len(buf) >= SHARD_SIZE:
                    flush_shard()
            if (i + 1) % 1000 == 0:
                rate = len(recent) / max(recent[-1] - recent[0], 1e-9)
                eta_h = (len(todo) - i - 1) / max(rate, 1e-9) / 3600
                print(f"[r5] {i + 1}/{len(todo)} authors | {rate:.1f} authors/s "
                      f"| ETA {eta_h:.1f} h | elapsed {(time.time() - t0) / 3600:.1f} h",
                      flush=True)
    flush_shard()
    print(f"[r5] complete: {len(todo)} fetched, {shard_no} shards, "
          f"{(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair", action="store_true",
                    help="refetch authors cached with zero works")
    args = ap.parse_args()
    if args.repair:
        repair()
    else:
        main()
