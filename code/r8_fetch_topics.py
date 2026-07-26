#!/usr/bin/env python3
"""R8 - one-time OpenAlex TOPICS fetch for the parallel-label family (task B3).

POST-HOC ROBUSTNESS INFRASTRUCTURE. The r5 cache stores concepts only (its
select clause predates the Topics work), so the Topics parallel label needs
one more pass over the same 89,374 authors, fetching each work's topics with
scores at both granularities (topic ~4,500 labels; subfield ~250 labels).
Fetch-once rule as in r5: all label variants derive offline.

Reuses r5's rate limiter, session pool, author list, retry policy, and shard
pattern (source: r5_fetch_author_works.py); only the select clause and the
per-work parse differ. Same endpoint parameters otherwise (publication_date
ascending, per-page 200, 400-works cap).

Cache layout: results/robustness/openalex_topics_cache/topics_NNNNN.parquet
  columns: aid, works_json = [{"year": int|null,
            "topics": [[topic_id, subfield_id, score], ...]}, ...]
  ids are the short OpenAlex forms ("T10555", subfield integer id as string);
  display names are not needed for label construction and are not stored.

Usage:  python code/r8_fetch_topics.py            (resumable)
        python code/r8_fetch_topics.py --repair   (refetch zero-work authors)
"""
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r5_fetch_author_works as R5

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "results" / "robustness" / "openalex_topics_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SHARD_SIZE = 1000
MAX_WORKS_PER_AUTHOR = R5.MAX_WORKS_PER_AUTHOR


def _short(url_or_id):
    return str(url_or_id).rstrip("/").split("/")[-1] if url_or_id else None


def fetch_author_topics(aid: str) -> list:
    """r5.fetch_author_works with the topics select clause; same retry and
    politeness structure."""
    import requests
    out, cursor, fetched = [], "*", 0
    base_url = ("https://api.openalex.org/works?filter=authorships.author.id:{aid}"
                "&per-page=200&sort=publication_date:asc"
                "&select=publication_year,topics&cursor={cur}")
    use_key = True
    tries = 0
    while cursor and fetched < MAX_WORKS_PER_AUTHOR:
        try:
            u = base_url.format(aid=aid, cur=cursor) + f"&mailto={R5.MAILTO}"
            if use_key:
                u += f"&api_key={R5.API_KEY}"
            R5.LIMITER.wait()
            r = R5._session().get(u, timeout=30)
            if r.status_code == 403 and use_key:
                use_key = False
                continue
            if r.status_code == 429:
                time.sleep(5)
                continue
            r.raise_for_status()
            js = r.json()
            tries = 0
        except Exception as e:
            tries += 1
            if tries <= 3:
                time.sleep(2 * tries)
                R5._local.s = requests.Session()
                continue
            print(f"[warn] {aid}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            break
        for w in js.get("results", []):
            ts = [[_short(t.get("id")), _short((t.get("subfield") or {}).get("id")),
                   round(float(t.get("score", 0)), 4)]
                  for t in (w.get("topics") or [])]
            out.append({"year": w.get("publication_year"), "topics": ts})
        fetched += len(js.get("results", []))
        cursor = js.get("meta", {}).get("next_cursor")
        if not js.get("results"):
            break
    return out


def done_aids() -> set:
    done = set()
    for p in sorted(CACHE_DIR.glob("topics_*.parquet")):
        done |= set(pd.read_parquet(p, columns=["aid"]).aid)
    return done


def main():
    ids = R5.author_ids()
    done = done_aids()
    todo = [a for a in ids if a not in done]
    shard_no = len(list(CACHE_DIR.glob("topics_*.parquet")))
    print(f"[r8] {len(ids)} authors | {len(done)} cached | {len(todo)} to fetch",
          flush=True)
    buf = []
    t0 = time.time()
    recent = deque(maxlen=1000)

    def flush():
        nonlocal buf, shard_no
        if not buf:
            return
        pd.DataFrame(buf, columns=["aid", "works_json"]).to_parquet(
            CACHE_DIR / f"topics_{shard_no:05d}.parquet", index=False)
        shard_no += 1
        buf = []

    with ThreadPoolExecutor(max_workers=R5.N_WORKERS) as ex:
        for i, (aid, works) in enumerate(zip(todo, ex.map(fetch_author_topics, todo))):
            buf.append((aid, json.dumps(works)))
            recent.append(time.time())
            if len(buf) >= SHARD_SIZE:
                flush()
            if (i + 1) % 1000 == 0:
                rate = len(recent) / max(recent[-1] - recent[0], 1e-9)
                eta = (len(todo) - i - 1) / max(rate, 1e-9) / 3600
                print(f"[r8] {i + 1}/{len(todo)} | {rate:.1f}/s | ETA {eta:.1f} h",
                      flush=True)
    flush()
    print(f"[r8] complete in {(time.time() - t0) / 3600:.2f} h", flush=True)


def repair():
    empty = []
    for p in sorted(CACHE_DIR.glob("topics_*.parquet")):
        d = pd.read_parquet(p)
        empty += d.loc[d.works_json == "[]", "aid"].tolist()
    empty = sorted(set(empty))
    print(f"[r8:repair] {len(empty)} zero-work authors", flush=True)
    buf = []
    with ThreadPoolExecutor(max_workers=R5.N_WORKERS) as ex:
        for aid, works in zip(empty, ex.map(fetch_author_topics, empty)):
            if works:
                buf.append((aid, json.dumps(works)))
    if buf:
        path = CACHE_DIR / "topics_zrepair_00000.parquet"
        df = pd.DataFrame(buf, columns=["aid", "works_json"])
        if path.exists():
            old = pd.read_parquet(path)
            df = pd.concat([old[~old.aid.isin(df.aid)], df], ignore_index=True)
        df.to_parquet(path, index=False)
    print(f"[r8:repair] {len(buf)} recovered", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair", action="store_true")
    args = ap.parse_args()
    if args.repair:
        repair()
    else:
        main()
