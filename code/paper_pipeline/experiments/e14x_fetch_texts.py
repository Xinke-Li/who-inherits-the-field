"""E14x - fetch pre-window titles+abstracts for the M4b modern-text-embedding rung.

Pure I/O stage of the M4b pipeline (P1-1): for every student in the frozen
table, fetch their works with publication_year <= t0+5 from OpenAlex (title +
abstract_inverted_index, abstract rebuilt to plain text) and cache them as one
JSONL line per author. Embedding happens elsewhere (Colab, SPECTER2 - see
M4b_SPECTER2_Colab.ipynb); this script never touches the frozen table beyond
reading ids and t0.

TEMPORAL CONTRACT: the API filter requests publication_year < t0+6 only, so
nothing after the feature freeze is ever cached; the Colab notebook re-asserts
the same bound before encoding (belt and braces).

Resumable: authors already present in the cache are skipped; the cache is
flushed every 200 successful authors; safe to Ctrl-C and rerun. Work is keyed by
author, not by student row - a few students share an st_openalex_id, and a shared
id is fetched once.

RATE LIMITS (changed under us, 2026-07-15). OpenAlex is metered: the free tier is
~1000 requests/day, resetting midnight UTC, and mailto= no longer buys unlimited
access. One author costs >=1 request, so a full econ pass needs several days. On
budget exhaustion the API answers 429 with a Retry-After of many hours; this
script raises BudgetExhausted and stops with the reset time rather than sleeping
through it. Re-run after the reset to continue. Short 429s are retried up to
MAX_429 times; an unparseable Retry-After is treated as budget exhaustion (fail
closed) because assuming "5 seconds" would spin.

PARTIAL RECORDS ARE NEVER CACHED. A transient failure mid-pagination raises
FetchIncomplete and the author is skipped for this run, not written. Writing the
works fetched so far would be permanent - resume skips any cached aid, so that
author would keep a truncated corpus forever and the Colab encode would embed a
fraction of their text with no error. Skipped authors are counted and retried on
the next run; the script says so rather than printing "complete".

Output: <repo root>/cache/texts_econ.jsonl  (or texts_neuro.jsonl under the
NEURO_DATASET override). Line format:
  {"aid": "A...", "t0": 1998, "works": [{"year": 1996, "title": "...",
                                          "abstract": "..." | null}, ...]}
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D

import requests

MAILTO = "lixinke@uchicago.edu"
MAX_WORKS = 400          # mirrors the builder's per-author cap
SLEEP = 0.08
FLUSH_EVERY = 200
RETRY_MAX_WAIT = 600     # s. Above this, a Retry-After is the daily budget, not
                         # a burst limit: observed 2026-07-15 WITHOUT a key, the
                         # API answered `X-RateLimit-Limit: 1000`,
                         # `X-RateLimit-Limit-USD: 0.1`, `Retry-After: 35398`
                         # (~9.8 h, to midnight UTC) with "Insufficient budget".
                         # That was the no-key tier; with OPENALEX_API_KEY set the
                         # budget is whatever /rate-limit reports for the key, and
                         # this constant only distinguishes burst from budget.
MAX_429 = 5              # consecutive short 429s before we give up on an author

# Measured as we go, so the neuroscience fetch can be scheduled from observation
# rather than from a guess at a published tier.
STATS = {"calls": 0, "cost_usd": 0.0, "per_page": None, "remaining": None}

# The key is read at runtime, never written down. This repository ships publicly
# with the paper ("Experiment scripts, result files, and a reproduction notebook
# ship with the repository"), so a key committed here would be published to the
# world and would survive in git history after any later deletion.
#
# Two sources, in order: OPENALEX_API_KEY, then a one-line <repo root>/.openalex_key
# (gitignored). The file exists because an environment variable can only be
# inherited at process start, which is inconvenient for an already-running
# session; it is the same secret handled the same way, not a weaker standard.
KEY_FILE = C.REPO_ROOT.parents[2] / ".openalex_key"   # paper_pipeline -> code -> submission_V2 -> root


def _load_key():
    k = os.environ.get("OPENALEX_API_KEY")
    if k and k.strip():
        return k.strip(), "OPENALEX_API_KEY"
    if KEY_FILE.exists():
        k = KEY_FILE.read_text(encoding="utf-8").strip()
        if k:
            return k, str(KEY_FILE.name)
    return None, None


API_KEY, KEY_SOURCE = _load_key()


def require_key():
    if not API_KEY:
        empty = KEY_FILE.exists() and not KEY_FILE.read_text(encoding="utf-8").strip()
        sys.exit(
            f"No OpenAlex API key found"
            + (f" ({KEY_FILE.name} exists but is EMPTY).\n" if empty
               else f" (looked at $OPENALEX_API_KEY and {KEY_FILE}).\n")
            + "Provide it either way:\n"
              f"  file : write the key as one line into {KEY_FILE}\n"
              "  env  : $env:OPENALEX_API_KEY = '...'   (pwsh)\n"
              "Without a key the daily budget collapses to the no-key tier "
              "(~1000 requests/day), which is not enough to finish this fetch.")
    print(f"[e14x] key loaded from {KEY_SOURCE}", flush=True)


def auth(url):
    """Append credentials. mailto is the legacy polite-pool hint (harmless and
    kept); api_key is what actually carries the quota."""
    return f"{url}&mailto={MAILTO}&api_key={API_KEY}"


def redact(obj):
    """Strip the key before anything is written to disk or printed."""
    if isinstance(obj, dict):
        return {k: ("<redacted>" if "key" in k.lower() else redact(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, str) and API_KEY and API_KEY in obj:
        return obj.replace(API_KEY, "<redacted>")
    return obj


def report_budget(session):
    """Print the key's actual budget, and record it beside the cache.

    Print-only: never gates or pauses the run. Its purpose is to replace guessing
    with measurement - the numbers here are what we schedule the neuroscience
    fetch against, and they document the budget this data was pulled under.
    """
    try:
        r = session.get(auth("https://api.openalex.org/rate-limit?_=1"), timeout=30)
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        print(f"[e14x] budget probe failed ({e}); continuing anyway", flush=True)
        return
    js = redact(js)
    for k in ("daily_budget_usd", "daily_remaining_usd", "prepaid_remaining_usd",
              "endpoint_costs_usd"):
        if k in js:
            print(f"[budget] {k}: {js[k]}", flush=True)
    stamp = time.strftime("%Y%m%d", time.gmtime())
    dest = CACHE_DIR / f"openalex_budget_{stamp}.json"
    try:
        dest.write_text(json.dumps(js, indent=2), encoding="utf-8")
        print(f"[budget] recorded -> {dest}", flush=True)
    except Exception as e:
        print(f"[budget] could not record: {e}", flush=True)


class BudgetExhausted(Exception):
    """OpenAlex daily credit budget spent. Arg is the reset delay in seconds."""


class FetchIncomplete(Exception):
    """This author could not be fetched in full.

    Raised instead of returning a partial work list. Caching a truncated record
    would be permanent: the resume skips any aid already in the cache, so the
    author would keep its short text forever and the Colab encode would silently
    embed a fraction of their corpus. Skipping leaves them absent, so the next
    run retries them.
    """

CACHE_DIR = C.REPO_ROOT.parents[2] / "cache"
SUFFIX = "neuro" if os.environ.get("NEURO_DATASET") else "econ"
CACHE = CACHE_DIR / f"texts_{SUFFIX}.jsonl"


def rebuild_abstract(inv):
    """abstract_inverted_index -> plain text."""
    if not inv:
        return None
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    return " ".join(w for _, w in sorted(pos))


def fetch_author(aid, t0, session, probe=None):
    out, cursor = [], "*"
    # per-page=200 is the legacy parameter name and cap. Left as-is deliberately:
    # the loop runs until next_cursor is empty, so page length cannot cost us a
    # record even if the server ignores the parameter and serves its default.
    # It only affects the number of calls. probe[] records the page length the
    # server actually used, which is what the neuroscience estimate needs.
    base = ("https://api.openalex.org/works?filter=authorships.author.id:{aid},"
            "publication_year:<{ymax}&per-page=200&sort=publication_date:asc"
            "&select=publication_year,title,abstract_inverted_index"
            "&cursor={cur}")
    n_429 = 0
    while cursor and len(out) < MAX_WORKS:
        try:
            r = session.get(auth(base.format(aid=aid, ymax=t0 + C.EARLY_YEARS + 1,
                                             cur=cursor)), timeout=30)
        except Exception as e:
            raise FetchIncomplete(f"request failed: {e}")
        if probe is not None:
            probe.append(r.headers.get("X-RateLimit-Remaining"))
        if r.status_code == 429:
            # Honour Retry-After. A long wait is the daily budget: surface it so
            # main() stops cleanly rather than idling for hours in silence. A
            # short one may be a burst limit worth riding out - but only a few
            # times, and only when the header actually parses as seconds. RFC
            # 7231 also permits an HTTP-date; we do not parse that, and treating
            # an unreadable header as "5 seconds" would spin. Fail closed.
            # Two headers can say "wait": Retry-After, and X-RateLimit-Reset
            # (seconds to midnight UTC). Take the larger - a long wait means the
            # budget is gone, and sleeping through it would idle for hours in
            # silence. Neither parsing? Assume budget and fail closed; treating
            # an unreadable header as "5 seconds" would spin.
            def secs(name):
                v = r.headers.get(name)
                if v is None:
                    return None
                try:
                    return int(v)
                except ValueError:
                    return -1        # present but unparseable
            ra, rst = secs("Retry-After"), secs("X-RateLimit-Reset")
            if ra == -1 or rst == -1:
                raise BudgetExhausted(RETRY_MAX_WAIT + 1)
            waits = [w for w in (ra, rst) if w is not None]
            wait = max(waits) if waits else None
            if wait is not None and wait > RETRY_MAX_WAIT:
                raise BudgetExhausted(wait)
            n_429 += 1
            if n_429 > MAX_429:
                raise FetchIncomplete(f"{n_429} consecutive 429s (burst limit)")
            time.sleep(wait if wait else min(2 ** n_429, 60))   # exponential backoff
            continue
        n_429 = 0
        try:
            r.raise_for_status()
            js = r.json()
        except Exception as e:
            raise FetchIncomplete(f"bad response: {e}")
        meta = js.get("meta", {})
        if probe is not None:
            STATS["calls"] += 1
            STATS["cost_usd"] += float(meta.get("cost_usd") or 0)
            if STATS["per_page"] is None and meta.get("per_page"):
                STATS["per_page"] = meta["per_page"]
                print(f"[e14x] server per_page = {STATS['per_page']} "
                      f"(requested per-page=200; affects call count only)",
                      flush=True)
        for w in js.get("results", []):
            out.append({"year": w.get("publication_year"),
                        "title": w.get("title"),
                        "abstract": rebuild_abstract(w.get("abstract_inverted_index"))})
        if not js.get("results"):
            break
        cursor = meta.get("next_cursor")
        time.sleep(SLEEP)
    return out[:MAX_WORKS]


def main():
    require_key()
    df = D.load_dataset()

    # Work is per AUTHOR, not per student row: a few students share an
    # st_openalex_id (econ: 15 ids over 31 rows). Fetching a shared id twice buys
    # nothing and costs a request out of a daily budget that is now the binding
    # constraint, so de-duplicate. dict() keeps one (aid, t0) per aid; e14y
    # asserts shared ids agree on t0, so which one is kept does not matter.
    by_aid = dict(zip(df.st_openalex_id, df.t0))
    n_authors = len(by_aid)

    done = set()
    if CACHE.exists():
        with open(CACHE, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["aid"])
                except Exception:
                    pass
    todo = [(a, t) for a, t in by_aid.items() if a not in done]
    print(f"[e14x] {len(df)} student rows | {n_authors} distinct authors | "
          f"{len(done)} cached | {len(todo)} to fetch", flush=True)

    s = requests.Session()
    report_budget(s)
    probe = []
    n_ok = n_skipped = 0
    with open(CACHE, "a", encoding="utf-8") as out:
        for k, (aid, t0) in enumerate(todo):
            try:
                works = fetch_author(aid, int(t0), s, probe=probe)
            except BudgetExhausted as e:
                out.flush()
                os.fsync(out.fileno())
                hrs = int(e.args[0]) / 3600.0
                print(f"[e14x] STOPPED: OpenAlex daily budget exhausted after "
                      f"{n_ok} authors fetched this run "
                      f"({len(done) + n_ok}/{n_authors} authors cached overall, "
                      f"{len(todo) - n_ok} still to go). "
                      f"Budget resets in {hrs:.1f} h (midnight UTC). "
                      f"Re-run this script after the reset; it resumes from "
                      f"{CACHE.name} and re-fetches nothing already cached.",
                      flush=True)
                return
            except FetchIncomplete as e:
                # Leave the author OUT of the cache so a later run retries them.
                n_skipped += 1
                print(f"[warn] {aid}: {e} - skipped, will retry next run",
                      flush=True)
                continue
            out.write(json.dumps({"aid": aid, "t0": int(t0), "works": works},
                                 ensure_ascii=False) + "\n")
            n_ok += 1
            if n_ok % FLUSH_EVERY == 0:
                out.flush()
                os.fsync(out.fileno())
                rem = next((x for x in reversed(probe) if x is not None), "?")
                print(f"[e14x] {n_ok}/{len(todo)} | rate-limit remaining {rem} | "
                      f"calls {STATS['calls']} | cost ${STATS['cost_usd']:.4f}",
                      flush=True)
    cpa = STATS["calls"] / n_ok if n_ok else 0
    print(f"[e14x] pass complete: {n_ok} fetched, {n_skipped} skipped "
          f"(transient errors; re-run to retry) -> {CACHE}", flush=True)
    print(f"[e14x] measured: {STATS['calls']} calls for {n_ok} authors "
          f"= {cpa:.2f} calls/author | cost ${STATS['cost_usd']:.4f} | "
          f"server per_page {STATS['per_page']}", flush=True)
    if n_skipped:
        print(f"[e14x] NOT yet complete: {n_skipped} authors still missing.",
              flush=True)


if __name__ == "__main__":
    main()
