#!/usr/bin/env python3
"""T3.1: LLM adjudication of advisor-to-student topic inheritance, with a
placebo control.

A third instrument, not human validation. The label is a Jaccard threshold on
concept sets; the swap-label control tests whether that threshold tracks advisor
identity; this asks a different question, whether a reader shown only titles
would judge the same pairs as inheriting. Agreement between the two is evidence
that the label measures something a person would recognise. Disagreement is not
evidence that either is wrong.

TWO STAGES, split at the model call.

  --stage build   deterministic, no network. Draws the sample, renders the exact
                  prompt for every item, and writes items.jsonl plus a manifest.
  --stage score   deterministic, no network. Reads judgments.jsonl produced by
                  whatever judge the author chooses, and computes the verdict.

The split exists because the model call is the one step that leaves this
machine, and separating it means the sampling, the prompts and the scoring are
all auditable without it. Ship items.jsonl and the judge's raw output and a
third party can recompute the verdict exactly.

WHAT IS SHOWN TO THE JUDGE. Titles and years, nothing else. Five works per
window, drawn uniformly at a fixed seed from the works inside that window,
listed publication date ascending, under the builder's 400-work cap. Two
windows per item: the student's late window [t0+6, t0+15] and the advisor's
early window [.., t0+5]. No names, no concepts, no discipline, no label.

THE PLACEBO. Every true item is paired with a placebo item that replaces the
advisor with a cohort-matched advisor, matched on t0 and drawn at the same fixed
seed, equal in number. A judge that says "inherits" as often for a random
cohort-matched advisor as for the true one is reading topic similarity in the
field, not inheritance.

ORDER OF REPORTING IS FIXED AND ENFORCED. The true-versus-placebo gap and the
self-consistency rate are computed and reported first. If either fails its
threshold the script prints the failure and refuses to print kappa at all,
because a kappa from an instrument that cannot separate the placebo or cannot
agree with itself is a number that will be quoted out of context.

  python code/r40_t31_adjudication.py --stage build --field chemistry
  python code/r40_t31_adjudication.py --stage score --field chemistry

Output: results/revision/T3_1_adjudication/<field>/
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline"))
SUP = ROOT / "data" / "supplement"
OUT = ROOT / "results" / "revision" / "T3_1_adjudication"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
EARLY_YEARS, LATE_YEARS = 5, 15
WORKS_PER_WINDOW = 5
SAMPLE_SEED = 0
N_PAIRS = 200                 # true items; the placebo doubles this
WORK_CAP = 400                # the builder's cap, audit finding F13

# Every judgment must carry these. The field is named decoding rather than
# temperature because claude-opus-5 rejects temperature with a 400, and a field
# whose name contradicts its content breaks anyone parsing the release.
REQUIRED_JUDGE_FIELDS = ("model", "date", "decoding")

# Gates that must pass before kappa is reported at all.
MIN_TRUE_MINUS_PLACEBO = 0.10     # the instrument must separate the placebo
MIN_SELF_CONSISTENCY = 0.80       # repeated items must get the same answer

PROMPT = """You are shown two lists of paper titles with their publication years.

List A is five papers by one researcher, published early in their career.
List B is five papers by a second researcher, published years later.

List A:
{a}

List B:
{b}

Question: do the papers in List B work on substantially the same research topics
as the papers in List A?

Answer with exactly one word, YES or NO, and nothing else."""


def render(titles):
    return "\n".join(f"  {y}  {t}" for y, t in titles)


def draw(rng, rows, k=WORKS_PER_WINDOW):
    """Uniform without replacement, then publication date ascending."""
    if len(rows) <= k:
        sel = list(range(len(rows)))
    else:
        sel = sorted(rng.choice(len(rows), k, replace=False))
    out = [rows[i] for i in sel]
    return sorted(out, key=lambda r: (r[0], r[1]))


def build(field):
    tp = SUP / f"titles_{field}.parquet"
    if not tp.exists():
        print(f"STOPPING: {tp} is absent.")
        print("  T3.1 needs the title supplement from the B1 fetch "
              "(code/r35_fetch_titles_abstracts.py, then --merge).")
        print("  Refusing to build an adjudication sample from anything else: "
              "the concept lists in the frozen tables are the very thing this "
              "instrument is supposed to be independent of.")
        return 2

    tt = pd.read_parquet(tp)
    ds = pd.read_parquet(ROOT / "data" / f"clean_dataset_{field}.parquet",
                         columns=["student_pid", "st_openalex_id",
                                  "adv_openalex_id", "t0", "y",
                                  "early_concepts"])
    ds = ds[ds.early_concepts.apply(len) > 0].reset_index(drop=True)

    def bare(x):
        return x.rsplit("/", 1)[-1] if isinstance(x, str) else x

    by = {}
    for aid, g in tt.groupby("author_id", sort=False):
        by[aid] = list(zip(g.publication_year.tolist(), g.title.tolist()))

    rng = np.random.default_rng(SAMPLE_SEED)
    # cohort-matched placebo pool: advisors of students sharing the same t0
    adv_by_t0 = {}
    for aid, t0 in zip(ds.adv_openalex_id, ds.t0):
        a = bare(aid)
        if isinstance(a, str):
            adv_by_t0.setdefault(int(t0), []).append(a)

    usable = []
    for r in ds.itertuples():
        s, a = bare(r.st_openalex_id), bare(r.adv_openalex_id)
        sw, aw = by.get(s), by.get(a)
        if not sw or not aw:
            continue
        t0 = int(r.t0)
        late = [(y, t) for y, t in sw[:WORK_CAP]
                if t and t0 + EARLY_YEARS < y <= t0 + LATE_YEARS]
        early = [(y, t) for y, t in aw[:WORK_CAP] if t and y <= t0 + EARLY_YEARS]
        if len(late) >= 2 and len(early) >= 2:
            usable.append((r, s, a, t0, late, early))
    if len(usable) < N_PAIRS:
        print(f"STOPPING: only {len(usable)} rows of {field} have at least two "
              f"titled works in both windows; {N_PAIRS} are needed. The fetch "
              f"may be incomplete.")
        return 2

    pick = rng.choice(len(usable), N_PAIRS, replace=False)
    items = []
    for n, i in enumerate(pick):
        r, s, a, t0, late, early = usable[i]
        pool = [x for x in adv_by_t0.get(t0, []) if x != a and by.get(x)]
        if not pool:
            continue
        pl = pool[int(rng.integers(0, len(pool)))]
        pl_early = [(y, t) for y, t in by[pl][:WORK_CAP]
                    if t and y <= t0 + EARLY_YEARS]
        if len(pl_early) < 2:
            continue
        for arm, adv_titles in (("true", early), ("placebo", pl_early)):
            A = draw(rng, adv_titles)
            B = draw(rng, late)
            items.append({
                "item_id": f"{field}-{n:04d}-{arm}",
                "field": field, "arm": arm,
                "student_pid": str(r.student_pid), "t0": t0,
                "frozen_label": int(r.y),
                "n_advisor_titles_in_window": len(adv_titles),
                "n_student_titles_in_window": len(late),
                "prompt": PROMPT.format(a=render(A), b=render(B)),
            })

    # self-consistency probes: 10 percent of items repeated verbatim under a
    # new id, so a judge that is not deterministic is measurable
    probe_ix = rng.choice(len(items), max(1, len(items) // 10), replace=False)
    for j in probe_ix:
        d = dict(items[j])
        d["item_id"] = d["item_id"] + "-probe"
        d["probe_of"] = items[j]["item_id"]
        items.append(d)

    d = OUT / field
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    man = {
        "task": "T3.1", "field": field,
        "n_items": len(items),
        "n_true": sum(1 for i in items if i["arm"] == "true"
                      and "probe_of" not in i),
        "n_placebo": sum(1 for i in items if i["arm"] == "placebo"
                         and "probe_of" not in i),
        "n_probes": sum(1 for i in items if "probe_of" in i),
        "sample_seed": SAMPLE_SEED, "works_per_window": WORKS_PER_WINDOW,
        "work_cap": WORK_CAP,
        "student_window": "[t0+6, t0+15]", "advisor_window": "<= t0+5",
        "placebo": "cohort-matched advisor, same t0, equal number",
        "shown_to_judge": "titles and years only",
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "prompt_template": PROMPT,
        "gates_before_kappa": {
            "min_true_minus_placebo": MIN_TRUE_MINUS_PLACEBO,
            "min_self_consistency": MIN_SELF_CONSISTENCY},
        "judge_record_required": list(REQUIRED_JUDGE_FIELDS),
        "reading": ("a third instrument, not human validation; agreement is "
                    "evidence the label tracks something a reader would "
                    "recognise, disagreement is not evidence either is wrong"),
    }
    (d / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"[{field}] {man['n_true']} true + {man['n_placebo']} placebo + "
          f"{man['n_probes']} probes = {len(items)} items -> {d/'items.jsonl'}")
    print(f"  prompt sha256 {man['prompt_sha256'][:16]}")
    print("  next: run every item's 'prompt' through one judge under one "
          "decoding configuration and write judgments.jsonl with fields "
          "{item_id, answer, model, date, decoding}")
    return 0


def score(field):
    d = OUT / field
    jp = d / "judgments.jsonl"
    if not jp.exists():
        print(f"STOPPING: {jp} is absent. Run the items through a judge first.")
        return 2
    items = {json.loads(l)["item_id"]: json.loads(l)
             for l in open(d / "items.jsonl", encoding="utf-8")}
    J = {}
    meta = set()
    refusals = []
    for l in open(jp, encoding="utf-8"):
        r = json.loads(l)
        a = str(r["answer"]).strip().upper()
        if a not in ("YES", "NO"):
            # A refusal is the API declining to answer, not a malformed answer:
            # stop_reason 'refusal' with an empty text block. It is counted as
            # NO, which holds the denominator at the number of items drawn and
            # reads a non-answer as the absence of a YES. The alternative,
            # dropping the item, shrinks the denominator while holding the
            # numerator and so raises the arm's rate. Both are computed below
            # and both are reported, because a convention that decided the
            # verdict would be a finding about the convention.
            if r.get("stop_reason") == "refusal" and a == "":
                refusals.append(r["item_id"])
                J[r["item_id"]] = 0
                meta.add(tuple(r[k] for k in REQUIRED_JUDGE_FIELDS))
                continue
            raise SystemExit(f"r40: item {r['item_id']} answered {a!r}; the "
                             f"prompt demands exactly YES or NO.")
        # judge_record_required is a promise the manifest makes to whoever
        # parses the release. Enforce it here, or a judgments file written
        # against an older field name scores silently with the field as null.
        absent = [k for k in REQUIRED_JUDGE_FIELDS
                  if r.get(k) in (None, "")]
        if absent:
            raise SystemExit(
                f"r40: item {r['item_id']} is missing {absent}; every judgment "
                f"must record {REQUIRED_JUDGE_FIELDS}. A file written against "
                f"an older field name is not scored as though the field were "
                f"blank.")
        J[r["item_id"]] = 1 if a == "YES" else 0
        meta.add(tuple(r[k] for k in REQUIRED_JUDGE_FIELDS))
    if len(meta) != 1:
        raise SystemExit(f"r40: judgments mix {len(meta)} judge configurations "
                         f"{sorted(meta)}; one run, one judge.")
    model, date, decoding = meta.pop()

    # ---- gate 1, the placebo separation, reported before anything else ----
    tr = [J[k] for k, v in items.items()
          if v["arm"] == "true" and "probe_of" not in v and k in J]
    pl = [J[k] for k, v in items.items()
          if v["arm"] == "placebo" and "probe_of" not in v and k in J]
    gap = float(np.mean(tr) - np.mean(pl)) if tr and pl else float("nan")

    # the same two rates with the refusals dropped instead of counted as NO
    rf = set(refusals)
    tr_x = [J[k] for k, v in items.items()
            if v["arm"] == "true" and "probe_of" not in v and k in J and k not in rf]
    pl_x = [J[k] for k, v in items.items()
            if v["arm"] == "placebo" and "probe_of" not in v and k in J and k not in rf]
    gap_x = float(np.mean(tr_x) - np.mean(pl_x)) if tr_x and pl_x else float("nan")

    # ---- gate 2, self-consistency ----
    pr = [(J[k], J[v["probe_of"]]) for k, v in items.items()
          if "probe_of" in v and k in J and v["probe_of"] in J]
    cons = float(np.mean([a == b for a, b in pr])) if pr else float("nan")

    g1 = bool(gap >= MIN_TRUE_MINUS_PLACEBO)
    g2 = bool(cons >= MIN_SELF_CONSISTENCY)
    out = {
        "task": "T3.1", "field": field,
        "judge": {"model": model, "date": date, "decoding": decoding},
        "gate_1_true_minus_placebo": {
            "true_yes_rate": round(float(np.mean(tr)), 4),
            "placebo_yes_rate": round(float(np.mean(pl)), 4),
            "gap": round(gap, 4), "threshold": MIN_TRUE_MINUS_PLACEBO,
            "passes": g1,
            "refusal_convention": "counted as NO; denominators held at the "
                                  "number of items drawn",
            "n_refusals": len(refusals), "refused_items": sorted(refusals),
            "if_refusals_excluded": {
                "n_true": len(tr_x), "n_placebo": len(pl_x),
                "true_yes_rate": round(float(np.mean(tr_x)), 4),
                "placebo_yes_rate": round(float(np.mean(pl_x)), 4),
                "gap": round(gap_x, 4),
                "passes": bool(gap_x >= MIN_TRUE_MINUS_PLACEBO),
                "note": "reported so the convention is visible; it moves the "
                        "gap by at most one item over the denominator"},
            "gap_ceiling": {
                "value": round(float(np.mean(tr)), 4),
                "why": "the gap is the true-arm rate minus the placebo rate "
                       "and the placebo rate cannot fall below zero, so the "
                       "true-arm rate is the largest gap this design can "
                       "produce",
                "headroom": round(float(np.mean(pl)), 4)}},
        "gate_2_self_consistency": {
            "n_probes": len(pr), "rate": round(cons, 4),
            "threshold": MIN_SELF_CONSISTENCY, "passes": g2},
    }
    print(f"[{field}] gate 1 true minus placebo "
          f"{out['gate_1_true_minus_placebo']['gap']:+.4f} "
          f"({'PASS' if g1 else 'FAIL'})")
    if refusals:
        print(f"[{field}] {len(refusals)} refusal(s) counted as NO: "
              f"{', '.join(sorted(refusals))}")
        print(f"[{field}] gap if excluded instead {gap_x:+.4f} "
              f"(reported, not used)")
    print(f"[{field}] gate 2 self-consistency "
          f"{out['gate_2_self_consistency']['rate']:.4f} "
          f"({'PASS' if g2 else 'FAIL'})")

    if not (g1 and g2):
        out["kappa"] = None
        out["verdict"] = (
            "NO VALIDITY CONCLUSION. The instrument failed a gate that must "
            "pass before its agreement with the frozen label means anything, "
            "so kappa is deliberately not computed.")
        print(f"[{field}] {out['verdict']}")
    else:
        y = np.array([items[k]["frozen_label"] for k, v in items.items()
                      if v["arm"] == "true" and "probe_of" not in v and k in J])
        j = np.array(tr)
        po = float((y == j).mean())
        pe = float((y == 1).mean() * (j == 1).mean()
                   + (y == 0).mean() * (j == 0).mean())
        k_bal = 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)
        # reweighted to the true base rate of the discipline, because the
        # sample is drawn from rows with titles in both windows and its base
        # rate need not match the cohort's
        base = float(pd.read_parquet(
            ROOT / "data" / f"clean_dataset_{field}.parquet",
            columns=["y"]).y.mean())
        w = np.where(y == 1, base / max(y.mean(), 1e-9),
                     (1 - base) / max(1 - y.mean(), 1e-9))
        po_w = float(np.average(y == j, weights=w))
        pe_w = float(base * np.average(j, weights=w)
                     + (1 - base) * (1 - np.average(j, weights=w)))
        k_rw = 1.0 if pe_w >= 1.0 else (po_w - pe_w) / (1 - pe_w)
        out["kappa"] = {"balanced": round(k_bal, 4),
                        "reweighted_to_base_rate": round(k_rw, 4),
                        "sample_base_rate": round(float(y.mean()), 4),
                        "cohort_base_rate": round(base, 4)}
        out["verdict"] = ("both gates pass, so the agreement figures may be "
                          "read; this is a third instrument and not human "
                          "validation")
        print(f"[{field}] kappa balanced {k_bal:.4f}, reweighted {k_rw:.4f}")

    (d / "verdict.json").write_text(json.dumps(out, indent=2))
    print(f"-> {d / 'verdict.json'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "score"])
    ap.add_argument("--field", default=None, choices=FIELDS)
    a = ap.parse_args()
    fs = [a.field] if a.field else FIELDS
    rc = 0
    for f in fs:
        os.environ["DATASET"] = f
        rc = (build(f) if a.stage == "build" else score(f)) or rc
    raise SystemExit(rc)
