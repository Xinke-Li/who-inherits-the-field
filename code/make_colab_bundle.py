#!/usr/bin/env python3
"""Assemble colab/e12_colab_bundle.zip: one self-contained archive the author
uploads to Colab. It lays every input at the exact path config and the training
scripts expect (traced from config.py, e2_hgt.py, h_extra_gnns.py,
e12_hgt_vs_baselines.py), so there is no manual directory building.

Layout inside the zip (unzips to /content):
  paper_pipeline/                 config + experiments + utils
  paper_pipeline/results_<f>/     e1_baselines.json (the tabular ceiling) + the
                                  other certificates + e12_manifest_<f>.json
  data_<f>/clean_dataset.parquet  config.CLEAN_DATASET  (SHA pinned)
  data_<f>/outcomes.parquet       config.OUTCOMES

Before zipping, a dry check imports config with DATASET=<f> for each discipline and
asserts CLEAN_DATASET, OUTCOMES, and results_<f>/e1_baselines.json all resolve. If
any is missing the build stops rather than shipping a bundle that would crash.

Run locally:  python make_colab_bundle.py
"""
import os, shutil, subprocess, sys, tempfile

PKG = "KDD New dataset 2027"
CODE_PP = os.path.join(PKG, "code", "paper_pipeline")
DATA = os.path.join(PKG, "data")
RESULTS = os.path.join(PKG, "results")
COLAB = os.path.join(PKG, "colab")
FIELDS = ["math", "chemistry", "physics", "econ", "neuro"]
CERTS = ["e1_baselines", "e10_advisor_placebo", "e14_self_persistence",
         "e6_innovation_premium", "e7_node2vec"]


def assemble(stage):
    # code
    shutil.copytree(CODE_PP, os.path.join(stage, "paper_pipeline"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in FIELDS:
        # frozen inputs at the config-expected layout
        dd = os.path.join(stage, f"data_{f}")
        os.makedirs(dd, exist_ok=True)
        shutil.copy2(os.path.join(DATA, f"clean_dataset_{f}.parquet"),
                     os.path.join(dd, "clean_dataset.parquet"))
        shutil.copy2(os.path.join(DATA, f"outcomes_{f}.parquet"),
                     os.path.join(dd, "outcomes.parquet"))
        # baselines + certificates + manifest under results_<f>
        rd = os.path.join(stage, "paper_pipeline", f"results_{f}")
        os.makedirs(rd, exist_ok=True)
        for c in CERTS:
            src = os.path.join(RESULTS, f"results_{f}", f"{c}.json")
            if os.path.exists(src):
                shutil.copy2(src, rd)
        man = os.path.join(COLAB, f"e12_manifest_{f}.json")
        if os.path.exists(man):
            shutil.copy2(man, rd)


def dry_check(stage):
    pp = os.path.join(stage, "paper_pipeline")
    ok = True
    for f in FIELDS:
        code = ("import config as c, os, sys;"
                "miss=[n for n,p in (('clean',c.CLEAN_DATASET),('outcomes',c.OUTCOMES),"
                "('e1',c.RESULTS_DIR/'e1_baselines.json')) if not os.path.exists(p)];"
                "sys.exit('MISSING '+','.join(miss) if miss else 0)")
        r = subprocess.run([sys.executable, "-c", code], cwd=pp,
                           env={**os.environ, "DATASET": f}, capture_output=True, text=True)
        status = "OK" if r.returncode == 0 else f"FAIL ({r.stdout.strip()}{r.stderr.strip()[-200:]})"
        print(f"[dry-check] DATASET={f:9s} {status}")
        ok = ok and r.returncode == 0
    return ok


def main():
    stage = tempfile.mkdtemp(prefix="e12bundle_")
    try:
        assemble(stage)
        if not dry_check(stage):
            print("dry check failed; not building the zip. fix the layout above.")
            sys.exit(1)
        out = os.path.join(COLAB, "e12_colab_bundle")
        if os.path.exists(out + ".zip"):
            os.remove(out + ".zip")
        shutil.make_archive(out, "zip", root_dir=stage)
        mb = os.path.getsize(out + ".zip") / 1e6
        print(f"[bundle] wrote {out}.zip ({mb:.1f} MB); all five disciplines resolve")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
