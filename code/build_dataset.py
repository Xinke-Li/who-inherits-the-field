#!/usr/bin/env python3
"""Field-parameterized modeling-table builder for the multidiscipline suite.

Runs the EXACT frozen protocol on a new discipline's resolved pairs. The stage
functions (fetch/table/coauth/build/outcomes) are imported VERBATIM from
build_neuro_dataset; this wrapper only redirects their I/O globals to an isolated
suite_data/<field>/build tree and relabels the discipline. Every window / label /
feature / filter parameter is therefore identical to the frozen econ + neuro
builds -- that byte-for-byte identity of protocol is the paper's claim. No frozen
artifact (data_econ/, data_neuro/, the two v1 tables) is read or written here.

PRE-REGISTERED (fixed here, before any results):
  EARLY_YEARS=5  LATE_YEARS=15  MIN_WORKS_PER_WINDOW=3  OBS_YEAR=2026
  PROFILE_TOPK=10  CONCEPT_MIN_SCORE=0.3  JACCARD_THETA=0.2
  MAX_WORKS_PER_AUTHOR=400  T0_MIN=1950  MAX_CAREER_SPAN=60
  label y = 1[ jaccard(student late-window concepts, advisor early profile) > 0.2 ].
  build stage runs the leakage guard AND an independent late_overlap recompute
  assert (recompute jaccard from the works store; require |Δ| < 1e-9 on every row).

For MIN_WORKS_PER_WINDOW=2 sensitivity builds pass --min-works 2 (the ONLY param
that may move, and only for the pre-registered E16 >=2-works robustness arm).

Stages resumable. Usage:
  export OPENALEX_API_KEY=...   OPENALEX_MAILTO=lixinke@uchicago.edu
  python build_dataset.py --field chemistry all
  python build_dataset.py --field math table coauth build outcomes
"""
import argparse, os, sys, json, hashlib, shutil
import numpy as np
import pandas as pd

sys.path.insert(0, "Neuro Data")
import build_neuro_dataset as bnd   # stage logic imported verbatim

DISCIPLINE = {"math": "mathematics", "physics": "physics", "chemistry": "chemistry",
              "neuro": "neuroscience", "econ": "economics",
              "econ_min2": "economics", "neuro_min2": "neuroscience",
              "econ_min2": "economics"}


def configure(field, base_root, pairs_file, min_works, works_store=None):
    base = os.path.join(base_root, "build")
    cache = os.path.join(base, "cache")
    os.makedirs(cache, exist_ok=True)
    bnd.BASE = base
    bnd.PAIRS_FILE = pairs_file
    bnd.CACHE_DIR = cache
    # works_store override lets a build REUSE an already-fetched store (sample
    # validation, >=2 sensitivity on an existing discipline) instead of re-fetching.
    bnd.STORE = works_store or os.path.join(cache, "works_store.jsonl")
    bnd.COAUTH_CACHE = os.path.join(cache, "coauth_early.jsonl")
    bnd.LABELS_CACHE = os.path.join(cache, "temporal_labels_v2.parquet")
    if min_works != 3:
        bnd.MIN_WORKS_PER_WINDOW = min_works   # ONLY the >=2 sensitivity arm moves this
    return base


def recompute_late_overlap_assert(base):
    """Independent gate: rebuild late_overlap from the works store with the exact
    profile/jaccard logic and require it matches the stored column within 1e-9."""
    clean = pd.read_parquet(os.path.join(base, "clean_dataset.parquet"))
    works = bnd.load_store()          # reads bnd.STORE (patched)
    EY, LY = bnd.EARLY_YEARS, bnd.LATE_YEARS
    max_abs = 0.0
    for r in clean.itertuples():
        sw = works.get(r.st_openalex_id); aw = works.get(r.adv_openalex_id)
        if sw is None or aw is None:
            raise AssertionError(f"works missing for {r.student_pid}")
        t0 = r.t0
        late, _, _ = bnd.profile(sw, t0 + EY + 1, t0 + LY)
        advp, _, _ = bnd.profile(aw, None, t0 + EY)
        # stage_table stores round(jaccard, 4); recompute on the same yardstick.
        recomputed = round(bnd.jaccard(late, advp), 4)
        max_abs = max(max_abs, abs(recomputed - float(r.late_overlap)))
    assert max_abs < 1e-9, f"late_overlap recompute mismatch: max|Δ|={max_abs}"
    print(f"[verify] late_overlap recompute assert PASSED (max|Δ|={max_abs:.2e} < 1e-9)", flush=True)


def finalize(field, base, out_data_dir):
    src = os.path.join(base, "clean_dataset.parquet")
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()
    os.makedirs(out_data_dir, exist_ok=True)
    for fn in ["clean_dataset.parquet", "clean_dataset.csv", "outcomes.parquet"]:
        p = os.path.join(base, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out_data_dir, fn))
    # relabel discipline/field in the summary and pin the sha
    s = json.load(open(os.path.join(base, "dataset_summary.json")))
    s["discipline"] = DISCIPLINE.get(field, field)
    s["field"] = field
    s["sha256_parquet"] = sha
    json.dump(s, open(os.path.join(out_data_dir, f"dataset_summary_{field}.json"), "w"), indent=2)
    print(f"[finalize] field={field} sha256={sha}", flush=True)
    print(f"[finalize] base_rate={s.get('label_base_rate')} n_students={s.get('n_students')} "
          f"coauth_early_rate={s.get('coauth_early_rate')}", flush=True)
    print(f"[finalize] frozen table + summary -> {out_data_dir}", flush=True)
    return sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True, choices=list(DISCIPLINE))
    ap.add_argument("--pairs", default=None, help="resolved *_Pairs_Openalex_Ultimate.parquet")
    ap.add_argument("--base", default=None, help="working root (default suite_data/<field>)")
    ap.add_argument("--out-data-dir", default=None,
                    help="where to place the frozen table (default artifact/code/data_<field>)")
    ap.add_argument("--min-works", type=int, default=3)
    ap.add_argument("--works-store", default=None, help="reuse an existing works_store.jsonl")
    ap.add_argument("stages", nargs="*", default=["all"])
    args = ap.parse_args()

    field = args.field
    base_root = args.base or os.path.join("suite_data", field)
    cap = {"econ": "Econ", "econ_min2": "Econ", "neuro_min2": "Neuro"}.get(field, field.capitalize())
    pairs = args.pairs or os.path.join(base_root, f"{cap}_Pairs_Openalex_Ultimate.parquet")
    out_data_dir = args.out_data_dir or os.path.join("artifact", "code", f"data_{field}")

    base = configure(field, base_root, pairs, args.min_works, args.works_store)
    print(f"[config] field={field} pairs={pairs} base={base} min_works={bnd.MIN_WORKS_PER_WINDOW}", flush=True)

    order = ["fetch", "table", "coauth", "build", "outcomes"]
    stages = order if args.stages in (["all"], []) else args.stages
    for st in stages:
        print(f"==== stage {st} (field={field}) ====", flush=True)
        bnd.STAGES[st]()

    if "build" in stages:
        recompute_late_overlap_assert(base)
        finalize(field, base, out_data_dir)


if __name__ == "__main__":
    main()
