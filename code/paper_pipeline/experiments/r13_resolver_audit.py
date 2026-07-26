"""R13 - resolver audit against genealogy ORCIDs (task B5).

Audit population: every resolved student whose genealogy row carries a
syntactically valid ORCID (4,837 across the five disciplines). The resolved
OpenAlex author's ORCID is fetched in batches; agreement means the two
ORCIDs match. A seed-fixed sample of 50 disagreements is classified by
PRE-FIXED criteria, programmatic where possible:

  id_split        the genealogy ORCID resolves to an OpenAlex author whose
                  name matches the RESOLVED author's name (similarity >=
                  0.85): the same person exists under two OpenAlex records,
                  not a resolver error;
  original_id_error  the ORCID-holder's name matches neither the resolved
                  author nor the genealogy person: the genealogy's ORCID is
                  wrong at source;
  resolver_error  the ORCID-holder's name matches the genealogy person but
                  not the resolved author: the resolver picked the wrong
                  person;
  human_judgment  anything else (ORCID unindexed in OpenAlex, ambiguous
                  name similarity): flagged, not auto-classified.

Metrics: raw agreement over rows where both ORCIDs are present;
bias-corrected precision = agreements plus the extrapolated share of
disagreements whose classified cause is not the resolver's
(id_split + original_id_error), with human_judgment counted as error in the
conservative bound and excluded in the favorable bound; Wilson 95 percent
CI at the determinate sample size. The frozen 86.7 percent ground-truth
precision is reported alongside, untouched.

Output: results/robustness/resolver_audit.json
"""
import difflib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from r5_fetch_author_works import API_KEY, MAILTO

FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
ORCID_RE = re.compile(r"^(https?://orcid\.org/)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])$")
SAMPLE_SEED, SAMPLE_N = 0, 50
SIM_THRESHOLD = 0.85
SESSION = requests.Session()


def norm_orcid(x):
    m = ORCID_RE.match(str(x).strip())
    return m.group(2) if m else None


def norm_name(x):
    if not x:
        return ""
    x = unicodedata.normalize("NFKD", str(x))
    x = "".join(c for c in x if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", "", x.lower()).strip()


def sim(a, b):
    a, b = norm_name(a), norm_name(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def api(url):
    for tries in range(4):
        try:
            r = SESSION.get(url + f"&mailto={MAILTO}&api_key={API_KEY}", timeout=30)
            if r.status_code == 429:
                time.sleep(5)
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(2 * (tries + 1))
    return None


def main():
    rows = []
    for f in FIELDS:
        df = pd.read_parquet(REPO / "data" / f"pairs_resolved_{f}.parquet",
                             columns=["student_pid", "stu_firstname", "stu_lastname",
                                      "stu_orcid", "st_openalex_id"])
        for r in df.itertuples():
            o = norm_orcid(r.stu_orcid)
            aid = str(r.st_openalex_id).rstrip("/").split("/")[-1] if pd.notna(r.st_openalex_id) else ""
            if o and aid.startswith("A"):
                rows.append({"field": f, "pid": r.student_pid, "orcid": o,
                             "aid": aid,
                             "gen_name": f"{r.stu_firstname or ''} {r.stu_lastname or ''}".strip()})
    # one audit row per unique (aid, orcid)
    seen = {}
    for r in rows:
        seen.setdefault((r["aid"], r["orcid"]), r)
    rows = list(seen.values())
    print(f"[r13] audit population: {len(rows)} unique resolved-author/ORCID pairs",
          flush=True)

    # batched author fetch: ORCID + display name of each resolved author
    info = {}
    aids = sorted({r["aid"] for r in rows})
    for i in range(0, len(aids), 50):
        chunk = aids[i:i + 50]
        js = api("https://api.openalex.org/authors?filter=ids.openalex:"
                 + "|".join(chunk) + "&per-page=50&select=id,orcid,display_name")
        for a in (js or {}).get("results", []):
            short = a["id"].rstrip("/").split("/")[-1]
            info[short] = {"orcid": norm_orcid(a.get("orcid") or ""),
                           "name": a.get("display_name") or ""}
        time.sleep(0.25)
        if (i // 50) % 20 == 0:
            print(f"[r13] fetched {min(i + 50, len(aids))}/{len(aids)} authors",
                  flush=True)

    agree, disagree, no_link, missing = [], [], [], []
    for r in rows:
        a = info.get(r["aid"])
        if a is None:
            missing.append(r)
        elif a["orcid"] is None:
            no_link.append(r)
        elif a["orcid"] == r["orcid"]:
            agree.append(r)
        else:
            r["resolved_name"] = a["name"]
            disagree.append(r)
    n_det = len(agree) + len(disagree)
    raw_agreement = len(agree) / n_det if n_det else None
    print(f"[r13] agree {len(agree)} | disagree {len(disagree)} | "
          f"no OpenAlex ORCID link {len(no_link)} | author record missing "
          f"{len(missing)} | raw agreement {raw_agreement:.4f}", flush=True)

    rng = np.random.default_rng(SAMPLE_SEED)
    sample = list(rng.choice(len(disagree), min(SAMPLE_N, len(disagree)),
                             replace=False)) if disagree else []
    classified = []
    for idx in sample:
        r = disagree[int(idx)]
        js = api(f"https://api.openalex.org/authors?filter=orcid:{r['orcid']}"
                 "&per-page=5&select=id,display_name,orcid")
        cands = (js or {}).get("results", [])
        time.sleep(0.25)
        if not cands:
            cls = "human_judgment_orcid_unindexed"
        else:
            b = cands[0]
            s_res = sim(b["display_name"], r["resolved_name"])
            s_gen = sim(b["display_name"], r["gen_name"])
            if s_res >= SIM_THRESHOLD:
                cls = "id_split"
            elif s_gen >= SIM_THRESHOLD:
                cls = "resolver_error"
            elif s_gen < 0.5 and s_res < 0.5:
                cls = "original_id_error"
            else:
                cls = "human_judgment_ambiguous"
        classified.append({**{k: r[k] for k in ("field", "pid", "orcid", "aid",
                                                "gen_name")},
                           "resolved_name": r.get("resolved_name", ""),
                           "orcid_holder": cands[0]["display_name"] if cands else None,
                           "class": cls})
        print(f"[r13] {r['aid']} {cls}", flush=True)

    counts = {}
    for c in classified:
        counts[c["class"]] = counts.get(c["class"], 0) + 1
    n_s = len(classified)
    not_fault = counts.get("id_split", 0) + counts.get("original_id_error", 0)
    err = counts.get("resolver_error", 0)
    human = n_s - not_fault - err

    def wilson(p, n, z=1.96):
        if n == 0:
            return [None, None]
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return [round(float(c - h), 4), round(float(c + h), 4)]

    out = {"experiment": "R13_resolver_audit", "sample_seed": SAMPLE_SEED,
           "population": {"unique_pairs": len(rows), "agree": len(agree),
                          "disagree": len(disagree), "no_openalex_orcid_link": len(no_link),
                          "author_record_missing": len(missing)},
           "raw_agreement": round(raw_agreement, 4),
           "raw_agreement_wilson95": wilson(raw_agreement, n_det),
           "disagreement_sample": {"n": n_s, "counts": counts,
                                   "flagged_for_human_judgment": [
                                       c for c in classified
                                       if c["class"].startswith("human_judgment")]},
           "frozen_ground_truth_precision": 0.867}
    if n_s:
        for tag, extra_err in (("favorable_humans_excluded",
                                err / max(n_s - human, 1)),
                               ("conservative_humans_as_error",
                                (err + human) / n_s)):
            p = (len(agree) + len(disagree) * (1 - extra_err)) / n_det
            out[f"bias_corrected_precision_{tag}"] = round(p, 4)
            out[f"bias_corrected_precision_{tag}_wilson95"] = wilson(p, n_det)
    out["all_classified"] = classified
    (REPO / "results" / "robustness" / "resolver_audit.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "all_classified"
                      and not (isinstance(v, dict) and "flagged" in str(v))},
                     indent=2, default=str)[:1500], flush=True)


if __name__ == "__main__":
    main()
