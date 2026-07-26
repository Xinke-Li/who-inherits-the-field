#!/usr/bin/env python3
"""T2.7 - a result file behind the window-filter paragraph, and a plain
statement of which of its numbers cannot have one.

THE CLAIM UNDER AUDIT. main.tex, Appendix H: "The relaxed table adds 14 percent
more students, 24,918 against 21,846, and the certificates hold: the base rate
moves from 0.251 to 0.240, the advisor-placebo gap is 0.111 in both, and the
best tabular rung moves from 0.426 to 0.407." Four pairs of numbers, and until
now no result file behind any of them.

WHAT WAS FOUND. clean_dataset_neuro_min2.parquet exists nowhere: not in data/,
not in data/supplement/, not anywhere under the project root, not in
zenodo_archive.zip, not in colab_bundle.zip. What survives is
data/dataset_summary_neuro_min2.json, which carries n_students, the base rate,
the min-works parameter and the table's pinned sha256. So:

  min3 side, all of it     regenerable from the frozen result files, and this
                           script regenerates it rather than quoting it
  min2 n and base rate     recorded in the surviving summary, read from it here
  min2 gap and min2 rung   NOT regenerable, and not reconstructible either: the
                           relaxed table is gone and rebuilding it needs the
                           builder's works store, which is also gone

This script does not adopt the two unbacked numbers and does not delete them.
It records them as quoted-from-the-build, names the artifact that would be
required, and pins the sha the summary records so a recovered file can be
verified rather than trusted.

ALSO RUN, because it is runnable and it is the sensitivity the paragraph is
about: e16_minworks_sensitivity.measure, imported verbatim, on the >=3 frozen
table and the >=5 subset. That direction tightens the filter where the missing
min2 arm relaxes it, so it bounds the same collider concern from the other side.

Output: results/revision/T2_7_minworks/e16_minworks_sensitivity.json

Usage
  DATASET=neuro DATASET_PATH=data/clean_dataset_neuro.parquet \
      python code/r50_t27_minworks.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline"))
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline" / "experiments"))

from utils import data as D                                    # noqa: E402
from e16_minworks_sensitivity import measure                   # noqa: E402

OUT = ROOT / "results" / "revision" / "T2_7_minworks"
RES = ROOT / "results" / "results_neuro"
DATA = ROOT / "data"

# the paper's quoted values, one place, checked not carried
QUOTED = {"n_min3": 21846, "n_min2": 24918, "pct_more": 14,
          "base_min3": 0.251, "base_min2": 0.240,
          "gap_min3": 0.111, "gap_min2": 0.111,
          "rung_min3": 0.426, "rung_min2": 0.407}


def check(name, quoted, value, tol):
    ok = value is not None and abs(value - quoted) <= tol
    return {"quoted_in_paper": quoted, "regenerated": value,
            "tolerance": tol, "match": bool(ok)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e1 = json.loads((RES / "e1_baselines.json").read_text())
    e10 = json.loads((RES / "e10_advisor_placebo.json").read_text())
    s3 = json.loads((DATA / "dataset_summary_neuro.json").read_text())
    s2 = json.loads((DATA / "dataset_summary_neuro_min2.json").read_text())

    rung3 = e1["summary"]["M3_gbdt_tabular"]["auc_pr"]["mean"]
    true3 = e10["summary"]["true"]["L1_logit_overlap"]["auc_pr"]["mean"]
    plac3 = e10["summary"]["placebo_cohort"]["L1_logit_overlap"]["auc_pr"]["mean"]
    gap3 = true3 - plac3

    min2_table = DATA / "clean_dataset_neuro_min2.parquet"
    min2_present = min2_table.exists()

    out = {
        "task": "T2.7",
        "claim_under_audit": ("main.tex Appendix H, window filter paragraph: "
                              "24,918 against 21,846; base rate 0.251 to 0.240; "
                              "advisor-placebo gap 0.111 in both; best tabular "
                              "rung 0.426 to 0.407"),
        "frozen_tables_unchanged": True,
        "min3_arm_regenerated": {
            "source_files": ["results/results_neuro/e1_baselines.json",
                             "results/results_neuro/e10_advisor_placebo.json",
                             "data/dataset_summary_neuro.json"],
            "n_students": check("n_min3", QUOTED["n_min3"], s3["n_students"], 0),
            "base_rate": check("base_min3", QUOTED["base_min3"],
                               s3["label_base_rate"], 0.0005),
            "best_tabular_rung": {
                **check("rung_min3", QUOTED["rung_min3"], round(rung3, 4), 0.0005),
                "model": "M3_gbdt_tabular", "unrounded": rung3},
            "advisor_placebo_gap": {
                **check("gap_min3", QUOTED["gap_min3"], round(gap3, 4), 0.0005),
                "true_L1": round(true3, 4), "cohort_placebo_L1": round(plac3, 4),
                "unrounded": gap3}},
        "min2_arm": {
            "table_path": "data/clean_dataset_neuro_min2.parquet",
            "table_present": min2_present,
            "searched": ["data/", "data/supplement/", "the whole project root",
                         "zenodo_archive.zip", "colab_bundle.zip"],
            "surviving_record": "data/dataset_summary_neuro_min2.json",
            "pinned_sha256": s2["sha256_parquet"],
            "min_works_per_window": s2["params"]["min_works_per_window"],
            "recorded_in_summary": {
                "n_students": check("n_min2", QUOTED["n_min2"], s2["n_students"], 0),
                "base_rate": check("base_min2", QUOTED["base_min2"],
                                   s2["label_base_rate"], 0.0005),
                "pct_more_students": {
                    "quoted_in_paper": QUOTED["pct_more"],
                    "regenerated": round(100 * (s2["n_students"] / s3["n_students"] - 1), 1),
                    "match": abs(100 * (s2["n_students"] / s3["n_students"] - 1)
                                 - QUOTED["pct_more"]) < 0.5}},
            "not_regenerable": {
                "advisor_placebo_gap": QUOTED["gap_min2"],
                "best_tabular_rung": QUOTED["rung_min2"],
                "reason": ("both need the relaxed table, which is absent from "
                           "repository and archive; rebuilding it needs the "
                           "builder's per-author works store, which is also "
                           "absent. Neither number is adopted here and neither "
                           "is deleted: they stand as computed during the frozen "
                           "build, and the sha above lets a recovered table be "
                           "verified rather than trusted"),
                "status": "quoted from the frozen build, no artifact"}},
    }

    # the runnable direction: tighten the filter instead of relaxing it
    df = D.load_dataset()
    sens = {">=3_frozen": measure(df)}
    sub5 = df[(df.early_prod >= 5) & (df.late_prod >= 5)].copy()
    sens[">=5_subset"] = measure(sub5)
    sens[">=2_relaxed"] = {
        "status": "ABSENT - the relaxed table is gone; see min2_arm above"}
    g3, g5 = (sens[">=3_frozen"]["M1_minus_placebo_gap"],
              sens[">=5_subset"]["M1_minus_placebo_gap"])
    out["runnable_sensitivity"] = {
        "source": "e16_minworks_sensitivity.measure, imported verbatim",
        "discipline": "neuro",
        "thresholds": sens,
        "reading": (f"the advisor-vs-placebo gap under tightening is {g3:+.3f} "
                    f"at >=3 and {g5:+.3f} at >=5; the relaxing direction the "
                    f"appendix quotes cannot be re-measured")}

    p = OUT / "e16_minworks_sensitivity.json"
    p.write_text(json.dumps(out, indent=2))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    print(json.dumps(out["min3_arm_regenerated"], indent=2))
    print(json.dumps(out["min2_arm"]["recorded_in_summary"], indent=2))
    print(f"[r50] -> {p}\n[r50] sha256 {sha}")


if __name__ == "__main__":
    main()
