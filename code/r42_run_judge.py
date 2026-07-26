#!/usr/bin/env python3
"""T3.1: run the adjudication items through one judge and record the answers.

This is the one step in T3.1 that leaves this machine. It sits between
r40 --stage build, which renders every prompt with no network, and
r40 --stage score, which computes the verdict from what this writes. It makes
no decisions: it sends each item's stored 'prompt' verbatim and records the
answer verbatim, including an answer that is not YES or NO, so that a judge
that misbehaves is visible in the artifact rather than silently dropped.

  python code/r42_run_judge.py --stage submit  --field chemistry [--limit 20]
  python code/r42_run_judge.py --stage collect --field chemistry

submit  creates one Message Batch, one request per item, custom_id = item_id,
        and writes batch.json. --limit N submits only the first N items, for a
        cheap smoke test that the model answers in the demanded form before
        the full sample is spent.
collect polls that batch until it ends and writes judgments.jsonl.

WHY BATCHES. The items are independent one-word classifications with no shared
prefix, so there is nothing for prompt caching to reuse and nothing gained by
streaming. The Batches API halves the price and removes the rate-limit handling
that a serial loop would need. Results arrive in any order and are keyed by
custom_id, never by position.

ON DECODING. claude-opus-5 rejects temperature, top_p and top_k with a 400; the
parameters were removed on the Opus 4.7 family. The recorded field is therefore
named decoding rather than temperature, and it states what was actually sent:
the model's default, with no sampling parameter set. A field named temperature
holding a sentence about temperature not existing would break anyone parsing
the released artifact. The determinism a temperature of 0 was standing in for
is measured directly by r40's gate 2, which repeats ten percent of the items
verbatim under a new id.

ON THINKING. Adaptive thinking is on by default on claude-opus-5 and is left
on at effort low. Disabling it is available at effort high or below, and would
be cheaper, but on this model a disabled-thinking request can leak internal
tags into the visible answer. That failure would be caught, because r40 refuses
any answer that is not exactly YES or NO, but it would cost a rerun.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "T3_1_adjudication"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]

MODEL = "claude-opus-5"
EFFORT = "low"
MAX_TOKENS = 1024      # caps thinking plus answer; a one-word answer needs room
DECODING_RECORD = "default; temperature not accepted on claude-opus-5"
POLL_SECONDS = 30


def client():
    try:
        import anthropic
    except ImportError:
        raise SystemExit(
            "r42: the anthropic SDK is not installed.\n"
            "  pip install anthropic")
    try:
        return anthropic, anthropic.Anthropic()
    except Exception as e:
        raise SystemExit(
            f"r42: could not construct a client ({e}).\n"
            "  set ANTHROPIC_API_KEY in the environment, or run `ant auth "
            "login` so a profile is on disk.")


def read_items(field):
    d = OUT / field
    p = d / "items.jsonl"
    if not p.exists():
        raise SystemExit(
            f"r42: {p} is absent. Run r40 --stage build --field {field} first.")
    man = d / "manifest.json"
    if not man.exists():
        raise SystemExit(f"r42: {man} is absent; refusing to judge items whose "
                         f"provenance is not recorded.")
    items = [json.loads(l) for l in open(p, encoding="utf-8")]
    return d, items, json.loads(man.read_text(encoding="utf-8"))


def submit(field, limit):
    anthropic, cl = client()
    from anthropic.types.message_create_params import (
        MessageCreateParamsNonStreaming)
    from anthropic.types.messages.batch_create_params import Request

    d, items, man = read_items(field)
    bp = d / "batch.json"
    if bp.exists():
        raise SystemExit(
            f"r42: {bp} already records batch "
            f"{json.loads(bp.read_text())['batch_id']}. Collect it, or delete "
            f"the file deliberately before submitting a second run.")
    if limit:
        items = items[:limit]

    reqs = [Request(custom_id=it["item_id"],
                    params=MessageCreateParamsNonStreaming(
                        model=MODEL, max_tokens=MAX_TOKENS,
                        output_config={"effort": EFFORT},
                        messages=[{"role": "user", "content": it["prompt"]}]))
            for it in items]
    batch = cl.messages.batches.create(requests=reqs)
    rec = {"task": "T3.1", "field": field, "batch_id": batch.id,
           "n_requests": len(reqs), "n_items_in_file": len(
               list(open(d / "items.jsonl", encoding="utf-8"))),
           "model": MODEL, "effort": EFFORT, "max_tokens": MAX_TOKENS,
           "decoding": DECODING_RECORD,
           "prompt_sha256": man["prompt_sha256"],
           "submitted_utc": datetime.now(timezone.utc).isoformat(),
           "limit": limit}
    bp.write_text(json.dumps(rec, indent=2))
    print(f"[{field}] submitted {len(reqs)} requests as {batch.id}")
    if limit:
        print(f"  SMOKE TEST ONLY: {limit} of {rec['n_items_in_file']} items. "
              f"Delete {bp} before the full run.")
    print(f"  -> {bp}")
    return 0


def collect(field):
    anthropic, cl = client()
    d, items, man = read_items(field)
    bp, jp = d / "batch.json", d / "judgments.jsonl"
    if not bp.exists():
        raise SystemExit(f"r42: {bp} is absent; nothing was submitted.")
    if jp.exists():
        raise SystemExit(f"r42: {jp} already exists; refusing to overwrite a "
                         f"judgment record.")
    rec = json.loads(bp.read_text(encoding="utf-8"))

    while True:
        b = cl.messages.batches.retrieve(rec["batch_id"])
        if b.processing_status == "ended":
            break
        print(f"[{field}] {b.processing_status}: "
              f"{b.request_counts.processing} processing, "
              f"{b.request_counts.succeeded} succeeded, "
              f"{b.request_counts.errored} errored")
        time.sleep(POLL_SECONDS)

    date = datetime.now(timezone.utc).date().isoformat()
    rows, bad = [], []
    for r in cl.messages.batches.results(rec["batch_id"]):
        if r.result.type != "succeeded":
            bad.append(f"{r.custom_id}: {r.result.type}")
            continue
        m = r.result.message
        txt = "".join(b.text for b in m.content if b.type == "text").strip()
        rows.append({"item_id": r.custom_id, "answer": txt,
                     "model": m.model, "date": date,
                     "decoding": DECODING_RECORD,
                     "stop_reason": m.stop_reason,
                     "usage": {"input_tokens": m.usage.input_tokens,
                               "output_tokens": m.usage.output_tokens}})
    if bad:
        print(f"[{field}] {len(bad)} requests did not succeed:")
        for b_ in bad[:10]:
            print("   ", b_)
        raise SystemExit(
            "r42: refusing to write a partial judgment record. Resubmit the "
            "failed ids as their own batch, or delete batch.json and rerun the "
            "whole field; a judgments file that is silently short would make "
            "r40's gates read on a different sample than the one built.")

    served = sorted({r["model"] for r in rows})
    if served != [MODEL]:
        raise SystemExit(f"r42: responses came from {served}, not {MODEL}. One "
                         f"run, one judge.")
    with open(jp, "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda z: z["item_id"]):
            f.write(json.dumps(r) + "\n")

    n_yes = sum(1 for r in rows if r["answer"].upper() == "YES")
    n_no = sum(1 for r in rows if r["answer"].upper() == "NO")
    other = [r["item_id"] for r in rows if r["answer"].upper() not in
             ("YES", "NO")]
    tok = sum(r["usage"]["output_tokens"] for r in rows)
    print(f"[{field}] {len(rows)} judgments: {n_yes} YES, {n_no} NO, "
          f"{len(other)} neither ({tok} output tokens)")
    if other:
        print(f"  recorded verbatim; r40 --stage score will refuse them: "
              f"{other[:5]}")
    print(f"  -> {jp}")
    print(f"  next: python code/r40_t31_adjudication.py --stage score "
          f"--field {field}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["submit", "collect"])
    ap.add_argument("--field", default=None, choices=FIELDS)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.limit and a.stage != "submit":
        raise SystemExit("r42: --limit applies to --stage submit only.")
    rc = 0
    for f in ([a.field] if a.field else FIELDS):
        rc = (submit(f, a.limit) if a.stage == "submit" else collect(f)) or rc
    sys.exit(rc)
