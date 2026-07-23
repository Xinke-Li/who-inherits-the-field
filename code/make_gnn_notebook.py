#!/usr/bin/env python3
"""Generate colab/e12_gnn.ipynb: a single-upload GPU notebook. The author uploads
one archive, e12_colab_bundle.zip, and this notebook. The bundle already lays every
input at the path config expects, so there is no manual directory building. The
notebook delegates to the frozen scripts e2_hgt.py (HGT) and h_extra_gnns.py (RGCN
and the cohort-time GAT), then aggregates against the tabular ceiling, so the model
architectures and hyperparameters are exactly the pipeline's. Run this generator
locally; it writes the .ipynb. It touches no frozen data."""
import json, os

PKG = "KDD New dataset 2027"
_ID = [0]


def _id():
    _ID[0] += 1
    return f"cell{_ID[0]:02d}"


def md(t):
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "id": _id(), "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = [

md("""# Graph model leg (e12), single-upload, Drive-persistent

Put `e12_colab_bundle.zip` on Drive at `MyDrive/e12_colab_bundle.zip` (uploading it
to the session at `/content/` also works), open this notebook, choose a GPU runtime,
and run the cells in order.

The bundle contains the code and, for each discipline, the frozen table at the exact
path config expects, so there is nothing to arrange by hand. All five disciplines run
by default (economics, math, physics, chemistry, neuroscience). This notebook trains
the graph models by calling the frozen scripts, so the architectures and
hyperparameters match the paper's comparison exactly.

HGT tuning is two-stage to save GPU. Each of the sixteen hyperparameter configs runs
only a few seeds first (two by default), ranked by mean validation AUC-PR, and then
the single best config is run to the full ten seeds, reusing the seeds already
computed. Only the winner reaches ten seeds, which is three to five times cheaper than
a full grid at ten seeds each and gives the same tuned model.

Live persistence: every finished per-seed json is copied to `MyDrive/e12_out` the
moment it is written (override the folder with the `E12_PERSIST` env var). If the
runtime disconnects or is recycled, reconnect and run this same notebook again:
Cell 1 copies the finished results back from Drive and every training loop skips the
seeds that are already done, so the run resumes where it stopped instead of starting
over.

The notebook then compares each graph model to the tabular ceiling, the seed-wise
maximum of M2 and M3 read from `results_<field>/e1_baselines.json`, with a paired test
corrected across the model family and a bootstrap interval. A graph model exceeds the
ceiling only if its best seed-mean beats the ceiling, the corrected paired test is
significant, and the bootstrap interval on the difference excludes zero.

Run order: math and chemistry first, then physics, economics, neuroscience. The output
is `e12_full_results.zip`: the complete `results_<field>/` trees, that is every
per-seed json from `results_hgt/`, `results_hgt_grid/` and `results_extra_gnns/`
(each carrying per-student `test_scores` and `test_labels` next to the metrics) plus
the aggregates. A copy of the zip also lands in `MyDrive/e12_out`, so the full set
survives even if the browser download is skipped.
"""),

code("""# Cell 1: dependencies, unzip the one bundle, enter the code tree
import subprocess, sys, os, zipfile
if not os.environ.get("E12_SKIP_PIP"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch", "torch_geometric==2.6.1", "python-louvain",
                    "pandas==2.2.2", "pyarrow", "scikit-learn==1.5.2", "scipy"], check=True)
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    print("not on Colab; skipping Drive mount")

# the one archive you uploaded. Default is the Colab Drive path; override with E12_BUNDLE.
BUNDLE = os.environ.get("E12_BUNDLE", "/content/drive/MyDrive/e12_colab_bundle.zip")
if not os.path.exists(BUNDLE) and os.path.exists("/content/e12_colab_bundle.zip"):
    BUNDLE = "/content/e12_colab_bundle.zip"
assert os.path.exists(BUNDLE), f"upload e12_colab_bundle.zip; not found at {BUNDLE}"
RUN = os.environ.get("E12_RUN", "/content/run")
zipfile.ZipFile(BUNDLE).extractall(RUN)
PP = os.path.join(RUN, "paper_pipeline")
os.chdir(PP); sys.path.insert(0, PP)
print("unzipped bundle to", RUN, "| entered", PP, "| exists:", os.path.isdir(PP))

FIELDS = os.environ.get("E12_FIELDS", "math,chemistry,physics,econ,neuro").split(",")
SEEDS = list(range(int(os.environ.get("E12_SEEDS", "10"))))

def data_path(field):
    return f"../data_{field}/clean_dataset.parquet"

# Live persistence. Every finished result json is copied into PERSIST the moment
# it is written, so a runtime recycle loses nothing. On a fresh runtime the block
# below copies everything already in PERSIST back into results_<field>/, and the
# training loops skip seeds whose json is present: rerunning this notebook after
# a disconnect resumes instead of restarting.
import shutil
PERSIST = os.environ.get("E12_PERSIST", "/content/drive/MyDrive/e12_out")
if not os.environ.get("E12_PERSIST") and not os.path.isdir("/content/drive/MyDrive"):
    print("WARNING: Drive is not mounted and E12_PERSIST is not set; results stay")
    print("on the runtime disk only and are LOST if the runtime is recycled.")
    PERSIST = None
if PERSIST:
    os.makedirs(PERSIST, exist_ok=True)
    restored = 0
    for _root, _dirs, _fns in os.walk(PERSIST):
        for _fn in _fns:
            if not _fn.endswith(".json"):
                continue
            _src = os.path.join(_root, _fn)
            _rel = os.path.relpath(_src, PERSIST)
            _dst = os.path.join(PP, _rel)
            if not os.path.exists(_dst):
                os.makedirs(os.path.dirname(_dst), exist_ok=True)
                shutil.copy2(_src, _dst)
                restored += 1
    print(f"[persist] {PERSIST}: restored {restored} result file(s) from earlier runs")

def persist(rel):
    \"\"\"Copy a result file (path relative to paper_pipeline) into PERSIST now.\"\"\"
    if PERSIST:
        dst = os.path.join(PERSIST, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(rel, dst)
"""),

code("""# Cell 2: dry check. Every input resolves for every discipline before training.
import importlib, os
for f in FIELDS:
    os.environ["DATASET"] = f
    import config as C; importlib.reload(C)
    miss = [n for n, p in (("clean", C.CLEAN_DATASET), ("outcomes", C.OUTCOMES),
                           ("e1_baselines", C.RESULTS_DIR / "e1_baselines.json")) if not os.path.exists(p)]
    print(f"[{f:9s}] {'OK' if not miss else 'MISSING ' + ','.join(miss)}"
          f"  sha={C.EXPECTED_SHA256[:12]}  results={C.RESULTS_DIR.name}")
    assert not miss, f"{f}: missing {miss}"
"""),

code("""# Cell 3: HGT, standard configuration, 10 seeds per discipline (resumable)
import subprocess, sys, os
def hgt_standard(fields):
    for f in fields:
        env = {**os.environ, "DATASET": f}
        out = f"results_{f}/results_hgt"; os.makedirs(out, exist_ok=True)
        for s in SEEDS:
            fn = f"{out}/hgt_none_seed{s}.json"
            if os.path.exists(fn):
                print(f"[skip] {fn} already done", flush=True)
                continue
            print(f"== HGT {f} seed {s} ==", flush=True)
            subprocess.run([sys.executable, "experiments/e2_hgt.py", "--data", data_path(f),
                            "--ablate", "none", "--seed", str(s), "--out", out], env=env, check=True)
            persist(fn)
hgt_standard(FIELDS)      # or hgt_standard(["math"]) for one discipline
"""),

code("""# Cell 4: HGT tuned, efficient two-stage over lr x hidden x layers x dropout.
# Selection: each of the 16 configs runs only SELECT_SEEDS seeds (default 2), ranked
# by mean VAL AUC-PR. Decision: the single best config is run to the full 10 seeds,
# reusing the seeds already computed in selection. Only the winner reaches 10 seeds,
# tagged 'best'; losers stop at SELECT_SEEDS. This is 3-5x cheaper than 16 x 10 and
# gives the same tuned model. Resumable.
import subprocess, sys, os, glob, json, itertools, shutil
import numpy as np
GRID = list(itertools.product([1e-3, 5e-3], [64, 128], [2, 3], [0.0, 0.5]))
SELECT_SEEDS = list(range(int(os.environ.get("E12_SELECT_SEEDS", "2"))))

def _run_hgt(field, out, seed, lr, hid, lay, dr, tag, env):
    subprocess.run([sys.executable, "experiments/e2_hgt.py", "--data", data_path(field),
                    "--ablate", "none", "--seed", str(seed), "--lr", str(lr), "--hidden", str(hid),
                    "--layers", str(lay), "--dropout", str(dr), "--tag", tag, "--out", out],
                   env=env, check=True)
    persist(f"{out}/hgt_none_seed{seed}_{tag}.json")

def hgt_tuned(fields):
    for f in fields:
        env = {**os.environ, "DATASET": f}
        out = f"results_{f}/results_hgt_grid"; os.makedirs(out, exist_ok=True)
        # selection stage: SELECT_SEEDS per config
        for lr, hid, lay, dr in GRID:
            tag = f"grid_lr{lr}_h{hid}_l{lay}_d{dr}"
            for s in SELECT_SEEDS:
                if os.path.exists(f"{out}/hgt_none_seed{s}_{tag}.json"):
                    print(f"[skip] {out}/hgt_none_seed{s}_{tag}.json already done", flush=True)
                    continue
                print(f"== HGT-select {f} {tag} seed {s} ==", flush=True)
                _run_hgt(f, out, s, lr, hid, lay, dr, tag, env)
        # rank configs by mean VAL AUC-PR over the selection seeds
        best_hp, best_tag, best_val = None, None, -1.0
        for lr, hid, lay, dr in GRID:
            tag = f"grid_lr{lr}_h{hid}_l{lay}_d{dr}"
            vs = [json.load(open(x)).get("val_auc_pr", 0)
                  for x in glob.glob(f"{out}/hgt_none_seed*_{tag}.json")]
            if vs and float(np.mean(vs)) > best_val:
                best_val, best_tag, best_hp = float(np.mean(vs)), tag, (lr, hid, lay, dr)
        print(f"[{f}] best HGT config {best_tag} (mean val {best_val:.4f}); running it to full 10 seeds",
              flush=True)
        # decision stage: winner to full SEEDS, tagged 'best', reusing selection seeds
        lr, hid, lay, dr = best_hp
        for s in SEEDS:
            dst = f"{out}/hgt_none_seed{s}_best.json"
            if os.path.exists(dst):
                print(f"[skip] {dst} already done", flush=True)
                continue
            reuse = f"{out}/hgt_none_seed{s}_{best_tag}.json"
            if os.path.exists(reuse):
                shutil.copy(reuse, dst); persist(dst); continue
            print(f"== HGT-best {f} seed {s} ==", flush=True)
            _run_hgt(f, out, s, lr, hid, lay, dr, "best", env)
hgt_tuned(FIELDS)
"""),

code("""# Cell 5: RGCN and the cohort-time GAT (TGAT-lite), 10 seeds per discipline (resumable)
import subprocess, sys, os
def extra_gnns(fields):
    for f in fields:
        env = {**os.environ, "DATASET": f}
        out = f"results_{f}/results_extra_gnns"; os.makedirs(out, exist_ok=True)
        for model in ("rgcn", "tgat"):
            for s in SEEDS:
                fn = f"{out}/{model}_seed{s}.json"
                if os.path.exists(fn):
                    print(f"[skip] {fn} already done", flush=True)
                    continue
                print(f"== {model} {f} seed {s} ==", flush=True)
                subprocess.run([sys.executable, "experiments/h_extra_gnns.py",
                                "--model", model, "--seed", str(s), "--out", out], env=env, check=True)
                persist(fn)
extra_gnns(FIELDS)
"""),

code("""# Cell 6: aggregate each graph model against the tabular ceiling
# The ceiling is the seed-wise max(M2, M3) read from results_<field>/e1_baselines.json.
# For each graph model we report the seed mean and the mean gap to the ceiling; with
# ten seeds we add the paired test, its Benjamini-Hochberg corrected p, a bootstrap
# interval on the gap, and the three-gate exceeds flag. Writes
# results_<field>/e12_hgt_vs_baselines.json. Robust to seed count, so a one-seed run
# still fills the ceiling and the gaps.
import json, glob, os
import numpy as np
from scipy.stats import wilcoxon

def per_seed_ceiling(field):
    e1 = json.load(open(f"results_{field}/e1_baselines.json"))
    ps = e1.get("per_seed", {})
    m2, m3 = ps.get("M2_logit_tabular", []), ps.get("M3_gbdt_tabular", [])
    n = min(len(m2), len(m3))
    return [max(m2[i]["auc_pr"], m3[i]["auc_pr"]) for i in range(n)]

def model_seeds(field, pattern):
    d = {}
    for fn in glob.glob(f"results_{field}/{pattern}"):
        r = json.load(open(fn)); t = r.get("test", r)
        s = r.get("seed", t.get("seed"))
        if s is not None and t.get("auc_pr") is not None:
            d[int(s)] = {"auc_pr": t["auc_pr"], "auc_roc": t.get("auc_roc")}
    return d

def bh(pv):
    order = np.argsort(pv); m = len(pv); adj = [1.0] * m; prev = 1.0
    for rank, idx in enumerate(reversed(order), 1):
        prev = min(prev, pv[idx] * m / (m - rank + 1)); adj[idx] = prev
    return adj

def boot_ci(diffs, n=2000):
    rng = np.random.default_rng(0)
    ms = [rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(n)]
    return round(float(np.percentile(ms, 2.5)), 4), round(float(np.percentile(ms, 97.5)), 4)

def aggregate(field):
    if not os.path.exists(f"results_{field}/e1_baselines.json"):
        print(f"[{field}] e1_baselines.json NOT FOUND; ceiling cannot be computed"); return
    ceil = per_seed_ceiling(field)
    ceil_mean = float(np.mean(ceil)) if ceil else None
    models = {"hgt": model_seeds(field, "results_hgt/hgt_none_seed*.json"),
              "hgt_tuned": model_seeds(field, "results_hgt_grid/hgt_none_seed*_best.json"),
              "rgcn": model_seeds(field, "results_extra_gnns/rgcn_seed*.json"),
              "gat_cohort_time": model_seeds(field, "results_extra_gnns/tgat_seed*.json")}
    packs, names, pvals = {}, [], []
    for name, seeds in models.items():
        common = sorted(s for s in seeds if s < len(ceil))
        if not common:
            continue
        g = np.array([seeds[s]["auc_pr"] for s in common], float)
        c = np.array([ceil[s] for s in common], float)
        diffs = g - c
        p = {"n_seeds": len(common), "seed_mean_auc_pr": round(float(g.mean()), 4),
             "delta_vs_ceiling_mean": round(float(diffs.mean()), 4)}
        if len(common) >= 3:
            try:
                pv = float(wilcoxon(g, c).pvalue)
            except ValueError:
                pv = 1.0
            p["paired_p"] = pv; p["bootstrap_ci_diff"] = list(boot_ci(diffs))
            names.append(name); pvals.append(pv)
        else:
            p["note"] = "paired test and bootstrap need >=3 seeds; gap shown"
        packs[name] = p
    for name, pa in zip(names, (bh(pvals) if pvals else [])):
        packs[name]["paired_p_bh"] = round(float(pa), 6)
        m = packs[name]
        m["exceeds_ceiling"] = bool(m["seed_mean_auc_pr"] > (ceil_mean if ceil_mean is not None else 1e9)
                                    and m["paired_p_bh"] < 0.05 and m["bootstrap_ci_diff"][0] > 0)
    out = {"field": field,
           "tabular_ceiling_mean_max_M2_M3": round(ceil_mean, 4) if ceil_mean is not None else None,
           "n_ceiling_seeds": len(ceil), "models": packs,
           "gates": "exceeds_ceiling requires seed_mean>ceiling AND BH p<0.05 AND bootstrap CI excludes 0",
           "any_exceeds": any(v.get("exceeds_ceiling") for v in packs.values())}
    json.dump(out, open(f"results_{field}/e12_hgt_vs_baselines.json", "w"), indent=2)
    persist(f"results_{field}/e12_hgt_vs_baselines.json")
    print(f"[{field}] ceiling {out['tabular_ceiling_mean_max_M2_M3']} | "
          f"models {list(packs)} | any_exceeds {out['any_exceeds']}", flush=True)

for f in FIELDS:
    aggregate(f)
"""),

code("""# Cell 7: zip EVERYTHING the corrected aggregation needs and download it.
# The archive holds the complete results_<field>/ tree for every discipline:
# all per-seed jsons from results_hgt/, results_hgt_grid/ and results_extra_gnns/
# (each with test_scores, test_labels, val_auc_pr) plus e1_baselines.json and the
# e12 aggregate. A copy of the zip also lands in PERSIST, so the full set exists
# on Drive even if the browser download is skipped.
import shutil, os, zipfile
STAGE = os.path.join(RUN, "e12_full_out")
if os.path.exists(STAGE):
    shutil.rmtree(STAGE)
os.makedirs(STAGE)
for f in FIELDS:
    src = f"results_{f}"
    if os.path.isdir(src):
        shutil.copytree(src, os.path.join(STAGE, src))
zip_path = os.path.join(RUN, "e12_full_results")
shutil.make_archive(zip_path, "zip", STAGE)
with zipfile.ZipFile(zip_path + ".zip") as z:
    names = z.namelist()
per_seed = [n for n in names if any(k in n for k in
            ("results_hgt/", "results_hgt_grid/", "results_extra_gnns/"))]
print(f"e12_full_results.zip: {len(names)} files, {len(per_seed)} per-seed jsons")
if PERSIST:
    shutil.copy2(zip_path + ".zip", os.path.join(PERSIST, "e12_full_results.zip"))
    print("[persist] zip copied to", PERSIST)
try:
    from google.colab import files
    files.download(zip_path + ".zip")
except Exception:
    print("not on Colab; e12_full_results.zip is at", zip_path + ".zip")
"""),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(PKG, "colab", "e12_gnn.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print(f"wrote {out} with {len(CELLS)} cells")
