#!/usr/bin/env python3
"""B1: one OpenAlex fetch serving T3.1 (LLM adjudication) and T3.2 (the M4b
abstract rung).

Read-only. Polite pool with the project key, which is read from .openalex_key
or OPENALEX_API_KEY and is never printed and never written into any output.
Nothing is pushed anywhere.

Structure and rate limiting are r5_fetch_author_works's, reused so the two
fetches behave identically against the API. What differs is the select list
(title and abstract_inverted_index rather than concepts) and, more
importantly, that records are filtered to the stated windows before anything
is written.

WINDOWS, applied at write time, per author and per role.

    student, titles      [t0, t0+15]     the early window T3.2 reads and the
                                         late window T3.1 judges, contiguous
    student, abstracts   [t0, t0+5]      the early window only: a text feature
                                         is bound by the same t0+5 freeze every
                                         other feature obeys
    advisor, titles      <= t0+5         the early profile T3.1 shows the judge

An author serving several cohorts gets the union of their windows, and the
year is stored alongside every record so the consumer re-filters to the focal
student's own window. T3.1's sampler must do that; it is not optional, because
the union is wider than any single cohort's window.

Abstracts are stored only for the student early window. T3.1's judge sees
titles and years and nothing else, by design, so it needs no abstracts, and
writing advisor abstracts would put a decade of post-freeze text in the
release for no consumer.

The 400-work cap is the builder's (audit finding F13, already disclosed in the
datasheet) and is applied before the window filter, so an author with more
than 400 works is truncated at the earliest 400 by publication date.

  python code/r35_fetch_titles_abstracts.py            # fetch, resumable
  python code/r35_fetch_titles_abstracts.py --merge    # write the per-field files

Cache:  data/supplement/_ta_cache/works_NNNNN.parquet
Output: data/supplement/titles_<field>.parquet
        data/supplement/abstracts_<field>.parquet
        data/supplement/titles_abstracts_manifest.json
"""
import argparse
import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
SUP = REPO / "data" / "supplement"
CACHE = SUP / "_ta_cache"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
EARLY_YEARS, LATE_YEARS = 5, 15
MAX_WORKS_PER_AUTHOR = 400        # builder's cap, audit finding F13
SHARD_SIZE = 500
N_WORKERS = 6
RATE_PER_SEC = 8.0                # r5_fetch_author_works.RATE_PER_SEC
MAILTO = "lixinke@uchicago.edu"


def load_api_key() -> str:
    """Read .openalex_key (one line) or fall back to OPENALEX_API_KEY.
    Never print or persist the returned value."""
    p = REPO / ".openalex_key"
    key = p.read_text().strip() if p.exists() else \
        os.environ.get("OPENALEX_API_KEY", "").strip()
    assert key, ("no OpenAlex key: .openalex_key missing or empty and "
                 "OPENALEX_API_KEY unset")
    return key


API_KEY = load_api_key()


class RateLimiter:
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


def bare(x):
    return x.rsplit("/", 1)[-1] if isinstance(x, str) else x


def fetch_author_works(aid: str) -> list:
    out, cursor, fetched = [], "*", 0
    base = ("https://api.openalex.org/works?filter=authorships.author.id:{aid}"
            "&per-page=200&sort=publication_date:asc"
            "&select=id,publication_year,title,abstract_inverted_index"
            "&cursor={cur}")
    use_key, tries = True, 0
    while cursor and fetched < MAX_WORKS_PER_AUTHOR:
        try:
            u = base.format(aid=aid, cur=cursor) + f"&mailto={MAILTO}"
            if use_key:
                u += f"&api_key={API_KEY}"
            LIMITER.wait()
            r = _session().get(u, timeout=60)
            if r.status_code == 403 and use_key:
                print("[warn] api_key rejected - falling back to polite pool",
                      flush=True)
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
                _local.s = requests.Session()
                continue
            print(f"[warn] {aid}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            break
        for w in js.get("results", []):
            out.append({"wid": bare(w.get("id")),
                        "year": w.get("publication_year"),
                        "title": w.get("title"),
                        "abs": w.get("abstract_inverted_index")})
        fetched += len(js.get("results", []))
        cursor = js.get("meta", {}).get("next_cursor")
        if not js.get("results"):
            break
    return out[:MAX_WORKS_PER_AUTHOR]


def windows():
    """author id -> {'t_lo': earliest t0 as a student, 't_hi': latest t0 as a
    student, 'adv_hi': latest t0+5 as an advisor}, plus the per-field role map
    the merge step needs."""
    per_field = {}
    win = {}
    for f in FIELDS:
        d = pd.read_parquet(REPO / "data" / f"clean_dataset_{f}.parquet",
                            columns=["student_pid", "st_openalex_id",
                                     "adv_openalex_id", "t0", "early_concepts"])
        d = d[d.early_concepts.apply(len) > 0].reset_index(drop=True)
        per_field[f] = d[["student_pid", "st_openalex_id", "adv_openalex_id",
                          "t0"]].copy()
        for sid, t0 in zip(d.st_openalex_id, d.t0):
            a = bare(sid)
            if not isinstance(a, str):
                continue
            w = win.setdefault(a, {})
            w["t_lo"] = min(w.get("t_lo", 10**6), int(t0))
            w["t_hi"] = max(w.get("t_hi", -10**6), int(t0))
        for aid, t0 in zip(d.adv_openalex_id, d.t0):
            a = bare(aid)
            if not isinstance(a, str):
                continue
            w = win.setdefault(a, {})
            w["adv_hi"] = max(w.get("adv_hi", -10**6), int(t0) + EARLY_YEARS)
    return win, per_field


def in_union_window(w, year):
    """The widest year an author may legitimately contribute to any consumer."""
    if year is None:
        return False
    if "t_lo" in w and w["t_lo"] <= year <= w["t_hi"] + LATE_YEARS:
        return True
    if "adv_hi" in w and year <= w["adv_hi"]:
        return True
    return False


def done_aids():
    done = set()
    for p in sorted(CACHE.glob("works_*.parquet")):
        done |= set(pd.read_parquet(p, columns=["aid"]).aid)
    return done


def fetch():
    CACHE.mkdir(parents=True, exist_ok=True)
    win, _ = windows()
    ids = sorted(win)
    done = done_aids()
    todo = [a for a in ids if a not in done]
    shard_no = len(list(CACHE.glob("works_*.parquet")))
    print(f"[b1] {len(ids)} authors | {len(done)} cached | {len(todo)} to fetch "
          f"| {shard_no} shards present", flush=True)

    buf = []
    t0 = time.time()
    recent = deque(maxlen=500)
    lock = threading.Lock()

    def flush_shard():
        nonlocal buf, shard_no
        if not buf:
            return
        pd.DataFrame(buf, columns=["aid", "works_json"]).to_parquet(
            CACHE / f"works_{shard_no:05d}.parquet", index=False,
            compression="zstd")
        shard_no += 1
        buf = []

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, (aid, works) in enumerate(zip(todo, ex.map(fetch_author_works,
                                                          todo))):
            # window filter at write time: nothing outside the union window is
            # ever persisted, so the cache itself respects the contract
            w = win[aid]
            keep = [x for x in works if in_union_window(w, x.get("year"))]
            with lock:
                buf.append((aid, json.dumps(keep)))
                recent.append(time.time())
                if len(buf) >= SHARD_SIZE:
                    flush_shard()
            if (i + 1) % 500 == 0:
                rate = len(recent) / max(recent[-1] - recent[0], 1e-9)
                eta = (len(todo) - i - 1) / max(rate, 1e-9) / 3600
                print(f"[b1] {i+1}/{len(todo)} | {rate:.1f} authors/s | "
                      f"ETA {eta:.1f} h | elapsed "
                      f"{(time.time()-t0)/3600:.2f} h", flush=True)
    flush_shard()
    print(f"[b1] fetch complete: {len(todo)} authors, "
          f"{(time.time()-t0)/3600:.2f} h", flush=True)
    return 0


def merge(only=None, titles_only=False):
    """Write the per-field files from the fetch cache.

    One discipline's slice of the cache fits in memory; five do not, which is
    what killed the first attempt with ArrowMemoryError at 179 shards. The loop
    below already drops each discipline's store before the next, and --field
    goes further by putting each discipline in its own process, so a failure on
    a large discipline cannot cost the ones that already succeeded."""
    win, per_field = windows()
    shards = sorted(CACHE.glob("works_*.parquet"))
    if not shards:
        raise SystemExit(f"no fetch shards under {CACHE}; run the fetch first")

    fields = FIELDS if only is None else [only]
    SUP.mkdir(parents=True, exist_ok=True)
    meta = {}
    for f in fields:
        d = per_field[f]
        # Load only the authors this discipline needs, then drop them before
        # the next one. Holding every author's parsed works at once died with
        # ArrowMemoryError at 179 shards; one discipline's slice fits.
        need = set()
        for sid, aid in zip(d.st_openalex_id, d.adv_openalex_id):
            for x in (bare(sid), bare(aid)):
                if isinstance(x, str):
                    need.add(x)
        store = {}
        for p in shards:
            dd = pd.read_parquet(p)
            for aid_, wj in zip(dd.aid, dd.works_json):
                if aid_ in need:
                    w = json.loads(wj)             # later shards win
                    if titles_only:
                        # The abstract inverted indices are the bulk of a
                        # parsed author and nothing downstream of a titles-only
                        # merge reads them. Dropping them here is what lets
                        # chemistry's 33,528 authors fit; keeping them is what
                        # killed the full merge twice on this machine.
                        w = [{"wid": x["wid"], "year": x.get("year"),
                              "title": x.get("title")} for x in w]
                    store[aid_] = w
            del dd
        print(f"[b1:merge] {f}: {len(store)} of {len(need)} needed authors "
              f"present in the cache", flush=True)
        trows, arows = [], []
        seen_t, seen_a = set(), set()
        miss_s = miss_a = 0
        for sid, aid, t0 in zip(d.st_openalex_id, d.adv_openalex_id, d.t0):
            t0 = int(t0)
            s, a = bare(sid), bare(aid)
            sw = store.get(s)
            if sw is None:
                miss_s += 1
            else:
                for x in sw:
                    y = x.get("year")
                    if y is None or not (t0 <= y <= t0 + LATE_YEARS):
                        continue
                    k = (s, x["wid"])
                    if k not in seen_t:
                        seen_t.add(k)
                        trows.append((s, "student", x["wid"], int(y),
                                      x.get("title")))
                    if y <= t0 + EARLY_YEARS and x.get("abs") and k not in seen_a:
                        seen_a.add(k)
                        arows.append((s, x["wid"], int(y),
                                      json.dumps(x["abs"], separators=(",", ":"))))
            aw = store.get(a)
            if aw is None:
                miss_a += 1
            else:
                for x in aw:
                    y = x.get("year")
                    if y is None or y > t0 + EARLY_YEARS:
                        continue
                    k = (a, x["wid"])
                    if k not in seen_t:
                        seen_t.add(k)
                        trows.append((a, "advisor", x["wid"], int(y),
                                      x.get("title")))
        tt = pd.DataFrame(trows, columns=["author_id", "role", "work_id",
                                          "publication_year", "title"])
        aa = pd.DataFrame(arows, columns=["author_id", "work_id",
                                          "publication_year",
                                          "abstract_inverted_index"])
        tp, apth = SUP / f"titles_{f}.parquet", SUP / f"abstracts_{f}.parquet"
        tt.to_parquet(tp, index=False, compression="zstd")
        if not titles_only:
            aa.to_parquet(apth, index=False, compression="zstd")
        meta[f] = {
            "field": f, "rows_in_frozen_table": int(len(d)),
            "titles": int(len(tt)), "titles_mb": round(tp.stat().st_size / 1e6, 1),
            "abstracts": None if titles_only else int(len(aa)),
            "abstracts_mb": (None if titles_only
                             else round(apth.stat().st_size / 1e6, 1)),
            "titles_only": bool(titles_only),
            "student_rows_with_no_cached_author": int(miss_s),
            "advisor_rows_with_no_cached_author": int(miss_a),
            "windows": {
                "student_titles": "[t0, t0+15]",
                "student_abstracts": "[t0, t0+5]",
                "advisor_titles": "<= t0+5"},
            "union_caveat": ("an author serving several cohorts carries the "
                             "union of their windows; a consumer must re-filter "
                             "to the focal student's own window using the "
                             "publication_year column"),
            "work_cap": MAX_WORKS_PER_AUTHOR,
        }
        _ab = ("abstracts   skipped        " if titles_only else
               f"abstracts {len(aa):8d} ({meta[f]['abstracts_mb']:6.1f} MB)")
        print(f"[{f:10}] titles {len(tt):8d} ({meta[f]['titles_mb']:6.1f} MB)  "
              f"{_ab}  "
              f"uncached student/advisor rows {miss_s}/{miss_a}", flush=True)

    mp = SUP / "titles_abstracts_manifest.json"
    prev = json.loads(mp.read_text()) if mp.exists() else {}
    prev.update(meta)
    mp.write_text(json.dumps(prev, indent=2))
    print(f"\n-> {mp}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true",
                    help="write the per-field files from the fetch cache")
    ap.add_argument("--titles-only", action="store_true",
                    help="skip abstracts; the titles file is byte-comparable "
                         "because the windowing and dedup are unchanged")
    ap.add_argument("--field", choices=FIELDS, default=None,
                    help="merge one discipline only, so each runs in its own "
                         "process and releases its store on exit")
    args = ap.parse_args()
    raise SystemExit(merge(args.field, args.titles_only)
                     if args.merge else fetch())
