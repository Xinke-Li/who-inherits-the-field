#!/usr/bin/env python3
"""Build the self-contained T2.2b + T3.4 bundle and its Colab notebook.

One bundle, one notebook, two tasks:

  T2.2b  the graph arm under the strict LINEAGE contract, RGCN and GAT, both
         protocols, five disciplines, ten seeds  (r32_lineage_contract.py)
  T3.4   TabPFN on GPU without the 1,000-row CPU context  (r33_tabpfn_gpu.py)

The bundle carries the lineage tables r31 precomputed, so the GPU leg needs
neither the 1.2 GB OpenAlex cache nor the resolver tables, and the 39
strict-contract summaries, so every lineage cell prints its own
lineage-minus-strict difference at the moment it finishes.

Notebook ordering is deliberate. TabPFN's dependency floor moves numpy and
scipy, which breaks the frozen sklearn the ceiling is computed with. So the
lineage sweep runs and is downloaded first, TabPFN is installed only after
that, and the cell that installs it re-runs gate 1 and aborts if the ceiling
stops reproducing.

  python code/r34_make_lineage_bundle.py            # build
  python code/r34_make_lineage_bundle.py --smoke    # build, then run the
                                                    # bundle-only smoke test
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision"
NB_PATH = ROOT / "colab" / "t2_2b_lineage_tabpfn.ipynb"
ZIP_PATH = OUT / "T2_2b_bundle.zip"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
# Bumped when a notebook change requires a bundle change, or the reverse. Cell 1
# asserts it, so a mismatched pair fails before anything trains.
#   1  first build, three-process sweep
#   2  single-process sweep (--stage all), mandatory two-runtime split
BUNDLE_REVISION = 2
STRICT_TREE = (ROOT / "results" / "revision" / "T2_1_strict_contract" /
               "T2_1_final" / "T2_1_strict_contract")
CELL_RE = re.compile(r"^(rgcn|gat|hgt)_(prereg|tuned)_(strict|legacy)$")

CODE = [
    "code/r31_lineage_table.py",          # provenance; not run on Colab
    "code/r32_lineage_contract.py",
    "code/r33_tabpfn_gpu.py",
    "code/r25_strict_contract.py",        # gate 1 and PREREG_HP
    "code/r3_rgcn_symmetric.py",
    "code/r_eval_util.py",
    "code/e12_corrected_aggregation.py",
    "code/paper_pipeline/config.py",
    "code/paper_pipeline/utils/__init__.py",
    "code/paper_pipeline/utils/data.py",
    "code/paper_pipeline/utils/stats.py",
    "code/paper_pipeline/experiments/e2_hgt.py",
    "code/paper_pipeline/experiments/h_extra_gnns.py",
]


def strict_cells():
    """The 39 verdict summaries, at depth <field>/<cell>/summary.json."""
    out = []
    for p in sorted(STRICT_TREE.glob("*/*/summary.json")):
        if CELL_RE.match(p.parent.name):
            out.append(p)
    return out


def capabilities(strict_n):
    """Read what the packed code actually supports, so a stale bundle is
    visible at cell 1 rather than only after the results come back wrong."""
    e2 = (ROOT / "code" / "paper_pipeline" / "experiments"
          / "e2_hgt.py").read_text(encoding="utf-8")
    contracts = sorted({c for c in ("legacy", "strict", "f1a", "f1b",
                                    "strict_lineage") if f'"{c}"' in e2})
    r3 = (ROOT / "code" / "r3_rgcn_symmetric.py").read_text(encoding="utf-8")
    archs = sorted({a for a in ("rgcn", "gat", "hgt") if f'arch == "{a}"' in r3})
    r32 = (ROOT / "code" / "r32_lineage_contract.py").read_text(encoding="utf-8")
    tables = sorted(set(re.findall(r'\("Table (\d+[a-z])",', r32)))
    r33 = (ROOT / "code" / "r33_tabpfn_gpu.py").read_text(encoding="utf-8")
    man = json.loads((ROOT / "data" / "supplement"
                      / "lineage_manifest.json").read_text())
    nb = NB_PATH.read_text(encoding="utf-8") if NB_PATH.exists() else ""
    return {
        "graph_contracts": contracts,
        "architectures": archs,
        "lineage_target_tables": [f"Table {t}" for t in tables],
        "tabpfn_full_context": "PRETRAIN_LIMIT" in r33 and "--max-train" in r33,
        "lineage_tables": {f: man[f]["n_rows"] for f in FIELDS if f in man},
        "lineage_ancestry_coverage": {
            f: man[f]["grand_advisor_coverage"] for f in FIELDS if f in man},
        "lineage_works_coverage": {
            f: man[f]["grand_advisor_with_cached_works"] for f in FIELDS if f in man},
        "strict_summaries_packed": strict_n,
        "notebook_multi_discipline_sweep": "SWEEP_FIELDS" in nb,
        "notebook_defers_tabpfn_install": "TABPFN_INSTALL_GUARD" in nb,
        "notebook_single_process_sweep": "'--stage', 'all'" in nb,
        "notebook_enforces_runtime_split": "LINEAGE_ALREADY_DOWNLOADED" in nb,
        # An explicit pairing token. The notebook asserts this exact value, so
        # an older bundle uploaded against a newer notebook fails at cell 1
        # instead of running twenty cells under the wrong assumptions. Bump it
        # whenever a notebook change requires a bundle change or the reverse.
        "bundle_revision": BUNDLE_REVISION,
    }


def manifest_hash(paths):
    h = hashlib.sha256()
    for rel in sorted(paths):
        p = ROOT / rel
        if p.exists():
            h.update(rel.encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def validate(zip_paths, nb_path):
    """Compile every packed .py and every notebook code cell. Refuse on
    failure: a broken notebook costs seconds here and hours on an A100."""
    bad = []
    for rel in zip_paths:
        p = ROOT / rel
        if p.suffix != ".py":
            continue
        try:
            compile(p.read_text(encoding="utf-8"), rel, "exec")
        except SyntaxError as e:
            bad.append(f"{rel}: line {e.lineno}: {e.msg}")
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            compile(src, f"{Path(nb_path).name}:cell{i}", "exec")
        except SyntaxError as e:
            bad.append(f"notebook cell {i}: line {e.lineno}: {e.msg}")
        if "\r" in src:
            bad.append(f"notebook cell {i}: contains a carriage return")
    if bad:
        raise SystemExit("REFUSING TO WRITE BUNDLE, source does not parse:\n  "
                         + "\n  ".join(bad))
    print(f"  validated: {sum(1 for r in zip_paths if r.endswith('.py'))} modules, "
          f"{sum(1 for c in nb['cells'] if c['cell_type'] == 'code')} notebook cells")


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [p for p in CODE if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"missing from the bundle manifest: {missing}")
    lin = [ROOT / "data" / "supplement" / f"lineage_{f}.parquet" for f in FIELDS]
    absent = [str(p) for p in lin if not p.exists()]
    if absent:
        raise SystemExit(f"lineage tables absent: {absent}\n"
                         f"Build them first: python code/r31_lineage_table.py")
    sc = strict_cells()
    if len(sc) != 39:
        raise SystemExit(
            f"expected 39 strict verdict summaries under {STRICT_TREE}, "
            f"found {len(sc)}. Refusing to ship a bundle whose lineage cells "
            f"cannot all name their matching strict cell.")

    validate(CODE, NB_PATH)

    packed = list(CODE)
    packed += [f"data/clean_dataset_{f}.parquet" for f in FIELDS]
    packed += [f"data/supplement/lineage_{f}.parquet" for f in FIELDS]

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in CODE:
            z.write(ROOT / p, p)
        for f in FIELDS:
            z.write(ROOT / "data" / f"clean_dataset_{f}.parquet",
                    f"data/clean_dataset_{f}.parquet")
            z.write(ROOT / "data" / "supplement" / f"lineage_{f}.parquet",
                    f"data/supplement/lineage_{f}.parquet")
            for name in ("e1_baselines.json", "e12_corrected_vs_m5.json"):
                src = ROOT / "results" / f"results_{f}" / name
                if not src.exists():
                    raise SystemExit(f"gate 1 input absent: {src}")
                z.write(src, f"results/results_{f}/{name}")
            cpu = (ROOT / "results" / "robustness" / "extra_rungs_partial"
                   / f"{f}_tabpfn.json")
            if cpu.exists():
                z.write(cpu, f"results/robustness/extra_rungs_partial/{f}_tabpfn.json")
        z.write(ROOT / "data" / "supplement" / "lineage_manifest.json",
                "data/supplement/lineage_manifest.json")
        # the 39 strict summaries, at the path r32.STRICT_ROOTS[1] resolves
        for p in sc:
            rel = p.relative_to(STRICT_TREE)
            z.write(p, f"results/revision/T2_1_strict_contract/{rel.as_posix()}")

        caps = capabilities(len(sc))
        z.writestr("BUNDLE.json", json.dumps({
            "task": "T2.2b strict lineage contract + T3.4 TabPFN full context",
            "build_timestamp_utc": dt.datetime.now(dt.timezone.utc)
                                     .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "manifest_sha256": manifest_hash(packed),
            "capabilities": caps,
            "fields": FIELDS,
            "entrypoints": {"lineage": "code/r32_lineage_contract.py",
                            "tabpfn": "code/r33_tabpfn_gpu.py"},
            "gate1": "reproduces M5 and M5' and blocks the run on mismatch",
            "changes_vs_r25": [
                "build_graph_v2(contract='strict_lineage'): the strict "
                "construction plus advisor--mentored_by-->advisor from the AFT "
                "parent map and advisor--studies_lineage-->concept from "
                "grand-advisor works dated at or before the focal student's "
                "t0+5, both keyed to the focal cohort, both with reverse edges",
                "direct-advisor features standardized on the direct block "
                "alone, so they are bit-identical to the strict arm"],
            "tabpfn_note": ("installed only after the lineage sweep is "
                            "downloaded; the install moves numpy and scipy and "
                            "the notebook re-runs gate 1 afterwards"),
            "no_remote_access": True,
        }, indent=2))

    mb = ZIP_PATH.stat().st_size / 1e6
    md5 = hashlib.md5(ZIP_PATH.read_bytes()).hexdigest()
    b = json.loads(zipfile.ZipFile(ZIP_PATH).read("BUNDLE.json"))
    print(f"  bundle: {ZIP_PATH}  ({mb:.1f} MB)")
    print(f"  strict summaries packed: {len(sc)}")
    print()
    print(f"BUNDLE  build {b['build_timestamp_utc']}  manifest "
          f"{b['manifest_sha256']}  md5 {md5}  ({mb:.1f} MB)")
    return ZIP_PATH


# --------------------------------------------------------------------------
# bundle-only smoke test: extract to a sandbox with no access to the repo tree
# --------------------------------------------------------------------------
def smoke():
    sand = Path(tempfile.mkdtemp(prefix="t22b_sandbox_"))
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(sand)
    print(f"\nbundle-only sandbox: {sand}")
    fails = []

    def run(argv, env_field, label, expect=0):
        import os
        env = dict(os.environ)
        env.update(DATASET=env_field,
                   DATASET_PATH=str(sand / "data" / f"clean_dataset_{env_field}.parquet"),
                   RESULTS_DIR=str(sand / "results" / f"results_{env_field}"))
        env.pop("NEURO_DATASET", None)
        r = subprocess.run([sys.executable] + argv, cwd=sand, env=env,
                           capture_output=True, text=True)
        ok = r.returncode == expect
        print(f"  [{'ok ' if ok else 'FAIL'}] {label} (exit {r.returncode})")
        if not ok:
            fails.append(label)
            print("\n".join(r.stdout.splitlines()[-25:]))
            print("\n".join(r.stderr.splitlines()[-25:]))
        return r

    for arch in ("rgcn", "gat"):
        run(["code/r32_lineage_contract.py", "--smoke", "--stage", "all",
             "--arch", arch, "--protocol", "tuned"], "econ",
            f"lineage {arch} smoke cell")
        p = (sand / "results" / "revision" / "T2_2b_smoke" / "econ"
             / f"{arch}_tuned_lineage" / "summary.json")
        if not p.exists():
            fails.append(f"lineage {arch}: no summary.json")
            continue
        d = json.loads(p.read_text())
        need = ["target_table", "target_row", "delta_vs_M5prime",
                "student_ci95_vs_M5prime", "p_BH", "exceeds_fair",
                "lineage_graph", "matching_strict_cell",
                "lineage_minus_strict_delta"]
        miss = [k for k in need if k not in d]
        if miss:
            fails.append(f"lineage {arch}: summary missing {miss}")
        if not d["matching_strict_cell"].get("found"):
            fails.append(f"lineage {arch}: matching strict cell not resolved "
                         f"inside the bundle")
        if d["lineage_graph"]["ancestry_edges"] < 1:
            fails.append(f"lineage {arch}: graph carries no ancestry edges")
        print(f"        table {d['target_table']} row {d['target_row']!r}, "
              f"ancestry {d['lineage_graph']['ancestry_edges']}, "
              f"strict cell {d['matching_strict_cell'].get('cell')}, "
              f"lineage-strict {d['lineage_minus_strict_delta']}")

    # the lineage refusal: a bundle whose lineage table went missing must stop
    hidden = sand / "data" / "supplement" / "lineage_econ.parquet"
    tmp = hidden.with_suffix(".hidden")
    hidden.rename(tmp)
    run(["code/r32_lineage_contract.py", "--smoke", "--stage", "gate1",
         "--arch", "rgcn"], "econ", "lineage refuses without its table",
        expect=12)
    tmp.rename(hidden)

    # TabPFN: the GPU path cannot be exercised on this machine, so the sandbox
    # checks everything reachable without a GPU and the notebook runs the real
    # one-seed smoke on Colab before the sweep.
    r = run(["code/r33_tabpfn_gpu.py", "--stage", "fit"], "econ",
            "tabpfn refuses to run its GPU task on CPU", expect=1)
    if "Refusing" not in (r.stdout + r.stderr):
        fails.append("tabpfn: CPU refusal did not print the intended reason")
    r = run(["-c",
             "import sys; sys.path[:0]=['code','code/paper_pipeline',"
             "'code/paper_pipeline/experiments']; import r33_tabpfn_gpu as R;"
             "import pandas as pd;"
             "d=pd.read_parquet('data/clean_dataset_econ.parquet');"
             "miss=[c for c in R.FEATURES if c not in d.columns];"
             "assert not miss, miss;"
             "assert R.PRETRAIN_LIMIT==10000;"
             "print('tabpfn features present:', len(R.FEATURES))"],
            "econ", "tabpfn feature columns present in the packed parquet")

    print()
    if fails:
        print("SMOKE FAILURES:")
        for f in fails:
            print("  ", f)
        return 1
    shutil.rmtree(sand, ignore_errors=True)
    print("bundle-only smoke test passed")
    return 0


NB_CELLS = [
    ("markdown", """# T2.2b lineage contract + T3.4 TabPFN full context

Upload `T2_2b_bundle.zip` when the first cell asks. Nothing is downloaded from
any remote except the two pip packages the cells name, and nothing is
published.

**Gate 1 runs first and blocks.** If the frozen ceiling does not reproduce in
any of the five disciplines, the notebook stops instead of producing a verdict.

**Run the cells in order.** TabPFN's dependency floor moves numpy and scipy,
which breaks the sklearn the ceiling is computed with. So the lineage sweep
finishes and is downloaded before TabPFN is installed, and the install cell
re-runs gate 1 afterwards and aborts if the ceiling has moved."""),
    ("code", """from google.colab import files
import zipfile, os, pathlib, json
up = files.upload()                     # choose T2_2b_bundle.zip
name = next(iter(up))
pathlib.Path('/content/t22b').mkdir(exist_ok=True)
with zipfile.ZipFile(name) as z:
    z.extractall('/content/t22b')
os.chdir('/content/t22b')

print('RUN THIS NOTEBOOK IN TWO RUNTIMES. The split is enforced, not advised.')
print('  runtime 1 : cells 1 to 9. Cell 9 downloads the lineage results.')
print('  then      : Runtime > Restart runtime.')
print('  runtime 2 : cells 1 to 3, then 10 to 14.')
print('Why: installing TabPFN moves numpy and scipy, and the ceiling is')
print('computed with sklearn. Cell 10 refuses to run until cell 9 has written')
print('its archive, and it re-checks gate 1 afterwards on economics only, so')
print('the other four disciplines are protected by the restart and nothing')
print('else. Do not re-run the lineage sweep after cell 10 in the same runtime.')
print()

B = json.load(open('BUNDLE.json'))
print('build   :', B.get('build_timestamp_utc'))
print('manifest:', B.get('manifest_sha256', '')[:16])
caps = B.get('capabilities', {})
for k, v in caps.items():
    print(f'{k:34}', v)

# What THIS notebook intends to use. A stale bundle fails here, loudly, rather
# than producing a short table hours later.
assert 'strict_lineage' in caps.get('graph_contracts', []), (
    'STALE BUNDLE: e2_hgt has no strict_lineage contract. Re-upload.')
missing_arch = {'rgcn', 'gat'} - set(caps.get('architectures', []))
assert not missing_arch, f'STALE BUNDLE: architectures missing {missing_arch}'
missing_tbl = {'Table 12c', 'Table 13c'} - set(caps.get('lineage_target_tables', []))
assert not missing_tbl, f'STALE BUNDLE: table routing missing {missing_tbl}'
assert caps.get('tabpfn_full_context'), 'STALE BUNDLE: no full-context TabPFN'
assert caps.get('strict_summaries_packed') == 39, (
    f"STALE BUNDLE: {caps.get('strict_summaries_packed')} strict summaries "
    f"packed, need 39, so some lineage cell could not name its strict match.")
assert len(caps.get('lineage_tables', {})) == 5, (
    'STALE BUNDLE: lineage tables missing for some discipline.')
assert caps.get('notebook_multi_discipline_sweep')
assert caps.get('notebook_defers_tabpfn_install'), (
    'STALE BUNDLE: built before the deferred TabPFN install ordering.')

# Pairing token. This notebook and its bundle are built together by r34 and are
# not interchangeable across revisions: revision 2 runs one process per cell
# with --stage all and enforces the two-runtime split, and a revision 1 bundle
# carries a BUNDLE.json that predates both.
NEED_BUNDLE_REVISION = 2
got_rev = caps.get('bundle_revision')
assert got_rev == NEED_BUNDLE_REVISION, (
    f'BUNDLE AND NOTEBOOK DO NOT MATCH: this notebook needs bundle revision '
    f'{NEED_BUNDLE_REVISION}, the uploaded bundle is revision {got_rev!r}. '
    f'Rebuild with "python code/r34_make_lineage_bundle.py" and upload the '
    f'T2_2b_bundle.zip it writes, or use the notebook that shipped with this '
    f'bundle. Do not proceed: the sweep cell would call --stage all against a '
    f'bundle built for the three-stage split.')
assert caps.get('notebook_single_process_sweep'), (
    'STALE BUNDLE: built before the single-process sweep.')
assert caps.get('notebook_enforces_runtime_split'), (
    'STALE BUNDLE: built before the mandatory two-runtime split.')
print()
print(f'bundle capability check passed (revision {got_rev})')"""),
    ("code", """# Version drift is the most common way a long Colab run dies. Assert, do not hope.
REQ_PYG = '2.8.0'
REQ_TORCH_MAJOR = 2

import subprocess, sys, torch
try:
    import torch_geometric
except ImportError:
    subprocess.run([sys.executable,'-m','pip','-q','install',
                    f'torch-geometric=={REQ_PYG}'], check=True)
    import torch_geometric

import numpy, scipy, sklearn
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print('GPU  :', gpu)
print('torch:', torch.__version__, 'cuda', torch.version.cuda)
print('pyg  :', torch_geometric.__version__)
print('numpy/scipy/sklearn:', numpy.__version__, scipy.__version__, sklearn.__version__)

assert gpu is not None, 'FAIL: no GPU. Runtime > Change runtime type > A100.'
assert torch_geometric.__version__ == REQ_PYG, (
    f'FAIL: PyG {torch_geometric.__version__}, need {REQ_PYG}. '
    f'Run: pip install torch-geometric=={REQ_PYG} then restart the runtime.')
assert int(torch.__version__.split('.')[0]) >= REQ_TORCH_MAJOR, (
    f'FAIL: torch {torch.__version__}, need >= {REQ_TORCH_MAJOR}.x')
BASELINE_VERSIONS = (numpy.__version__, scipy.__version__, sklearn.__version__)
print('version assertions passed')"""),
    ("code", """SWEEP_FIELDS = ['econ', 'math', 'physics', 'neuro', 'chemistry']  # smallest first
FIELD = 'econ'

import os
def set_field(f):
    os.environ['DATASET'] = f
    os.environ['DATASET_PATH'] = f'/content/t22b/data/clean_dataset_{f}.parquet'
    os.environ['RESULTS_DIR'] = f'/content/t22b/results/results_{f}'
    return f
set_field(FIELD)
print('sweep:', SWEEP_FIELDS)"""),
    ("code", """# ---- GATE 1 for every discipline. Blocks on any failure.
import subprocess
for f in SWEEP_FIELDS:
    set_field(f)
    rc = subprocess.run(['python', 'code/r32_lineage_contract.py',
                         '--stage', 'gate1', '--arch', 'rgcn']).returncode
    assert rc == 0, f'GATE 1 FAILED for {f} - do not trust any verdict from this session'
set_field(FIELD)
print('gate 1 passed for all', len(SWEEP_FIELDS), 'disciplines')"""),
    ("code", """# ---- SMOKE: one lineage RGCN cell and one lineage GAT cell, 1 config,
# ---- 1 seed, 3 epochs, on the smallest discipline. Numbers are meaningless;
# ---- the point is that both paths execute and emit a schema-valid summary
# ---- that names its matching strict cell. Blocks the sweep on failure.
import subprocess, json, pathlib
set_field('econ')
for arch in ('rgcn', 'gat'):
    rc = subprocess.run(['python', 'code/r32_lineage_contract.py', '--smoke',
                         '--stage', 'all', '--arch', arch,
                         '--protocol', 'tuned']).returncode
    assert rc == 0, f'SMOKE FAILED for lineage {arch} (exit {rc})'
    p = pathlib.Path(f'/content/t22b/results/revision/T2_2b_smoke/econ/'
                     f'{arch}_tuned_lineage/summary.json')
    d = json.loads(p.read_text())
    assert d['matching_strict_cell'].get('found'), (
        f'{arch}: the matching strict cell was not resolved from the bundle')
    assert d['lineage_graph']['ancestry_edges'] > 0, (
        f'{arch}: the graph carries no ancestry edges, so it is not a lineage graph')
    print(f"  {arch}: {d['target_table']} / {d['target_row']}, ancestry "
          f"{d['lineage_graph']['ancestry_edges']}, strict match "
          f"{d['matching_strict_cell']['cell']}")
set_field(FIELD)
print('lineage smoke passed')"""),
    ("code", """# ---- OPTIONAL: restore a previously downloaded partial tree.
# ---- Run on a resumed session; skip on a fresh run. This cell VERIFIES its
# ---- own work: it asserts the exact absolute paths the sweep cell checks, and
# ---- fails loudly if they are not there. A restore that lands one directory
# ---- off is silent otherwise, and the completed cells get recomputed.
import zipfile, pathlib, shutil, tempfile
from google.colab import files

ROOT_OUT = pathlib.Path('/content/t22b/results/revision/T2_2b_lineage_contract')
ROOT_OUT.mkdir(parents=True, exist_ok=True)

up = files.upload()          # the T2_2b_*_out.zip downloaded earlier
staged = 0
for name in up:
    tmp = pathlib.Path(tempfile.mkdtemp())
    with zipfile.ZipFile(name) as z:
        z.extractall(tmp)
    for sm in tmp.rglob('summary.json'):
        cell_dir = sm.parent
        field = cell_dir.parent.name
        if field not in SWEEP_FIELDS:
            print('skipped, unrecognised discipline:', sm)
            continue
        dest = ROOT_OUT / field / cell_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in cell_dir.iterdir():
            shutil.copy2(f, dest / f.name)
        staged += 1
        print('staged', dest)
    shutil.rmtree(tmp, ignore_errors=True)

found = sorted(ROOT_OUT.rglob('summary.json'))
print()
print(f'{staged} cell(s) staged; {len(found)} summary.json now present:')
for p in found:
    print('  ', p)
assert found, 'RESTORE FAILED: no summary.json under ' + str(ROOT_OUT)
for p in found:
    assert p.parent.parent.name in SWEEP_FIELDS, (
        f'RESTORE FAILED: {p} is not at ROOT_OUT/<field>/<cell>/summary.json, '
        f'so the sweep cell will not find it')
print('restore verified: the sweep cell will skip these cells')"""),
    ("code", """# ---- FULL LINEAGE SWEEP: 5 disciplines x {rgcn, gat} x {prereg, tuned}
# ---- = 20 cells, unattended. Run and walk away.
# ---- ONE process per cell, --stage all. The earlier three-process split
# ---- (grid, seeds, aggregate) paid a full gate 1 in each, 60 refits of the
# ---- ceiling for 20 cells, about 1.2 h of the run. It also put the training
# ---- and the ceiling refit in different processes. One process per cell does
# ---- both once and keeps them together.
# ---- Resume is unchanged: grid.json, winner.json and seed*.json all persist
# ---- and are skipped on re-entry, so --stage all after a disconnect picks up
# ---- exactly where it stopped.
import subprocess, time, pathlib

EXIT_OK, EXIT_GATE1_FAILED, EXIT_NO_INPUT = 0, 2, 12
ROOT_OUT = pathlib.Path('/content/t22b/results/revision/T2_2b_lineage_contract')
PLAN = [('rgcn', 'prereg'), ('gat', 'prereg'),
        ('rgcn', 'tuned'),  ('gat', 'tuned')]

t_all = time.time()
done, skipped, failed = 0, 0, []
for field in SWEEP_FIELDS:
    set_field(field)
    for arch, proto in PLAN:
        cell = f'{arch}_{proto}_lineage'
        if (ROOT_OUT / field / cell / 'summary.json').exists():
            print(f'[{field}/{cell}] already complete'); done += 1; continue
        print('=' * 72); print(f'[{field}/{cell}] running', flush=True)
        t = time.time()
        rc = subprocess.run(['python', 'code/r32_lineage_contract.py',
                             '--stage', 'all',
                             '--arch', arch, '--protocol', proto]).returncode
        el = time.time() - t
        if rc == EXIT_OK:
            print(f'  ok ({el:.0f}s)'); done += 1
        elif rc == EXIT_NO_INPUT:
            print(f'  the lineage table for {field} is not in this bundle, '
                  f'skipping the cell ({el:.0f}s)'); skipped += 1
        elif rc == EXIT_GATE1_FAILED:
            # the ceiling stopped reproducing: every later cell is suspect too
            print(f'  GATE 1 FAILED ({el:.0f}s). Stopping the sweep: a verdict '
                  f'against a moved ceiling is worse than no verdict.')
            failed.append(f'{field}/{cell}@gate1')
            break
        else:
            # keep going: one bad cell must not cost the whole run
            print(f'  FAILED exit {rc} ({el:.0f}s)')
            failed.append(f'{field}/{cell}@exit{rc}')
    else:
        continue
    break

set_field(FIELD)
print()
print(f'LINEAGE SWEEP FINISHED in {time.time()-t_all:.0f}s: {done} cells done, '
      f'{skipped} skipped for missing input, {len(failed)} failed')
for f in failed: print('  FAILED:', f)
if done < 20 and not failed:
    print('NOTE: fewer than 20 cells present. Re-run this cell; it resumes.')"""),
    ("code", """# ---- the lineage answer, all disciplines, without downloading anything
import json, pathlib
ROOT_OUT = pathlib.Path('/content/t22b/results/revision/T2_2b_lineage_contract')
print(f"{'discipline':11} {'cell':22} {'lineage d':>10} {'strict d':>9} "
      f"{'lin-str':>8} {'CI (student)':>20} {'p_BH':>8}  exceeds")
print('-' * 104)
for field in SWEEP_FIELDS:
    for p in sorted((ROOT_OUT / field).glob('*/summary.json')):
        d = json.loads(p.read_text())
        s = d.get('matching_strict_cell', {})
        sd = s.get('delta_vs_M5prime')
        ld = d.get('lineage_minus_strict_delta')
        print(f"{field:11} {p.parent.name:22} {d['delta_vs_M5prime']:+10.4f} "
              f"{(f'{sd:+.4f}' if sd is not None else 'n/a'):>9} "
              f"{(f'{ld:+.4f}' if ld is not None else 'n/a'):>8} "
              f"{str(d['student_ci95_vs_M5prime']):>20} {d['p_BH']:>8.4f}  "
              f"{d['exceeds_fair']}")
print()
print('read the lin-str column, not the lineage column: it is the only one that '
      'isolates the two lineage relations from everything else')"""),
    ("code", """# ---- DOWNLOAD THE LINEAGE RESULTS NOW, BEFORE TabPFN TOUCHES THE ENVIRONMENT.
import shutil
shutil.make_archive('/content/T2_2b_lineage_out', 'zip',
                    '/content/t22b/results/revision', 'T2_2b_lineage_contract')
from google.colab import files
files.download('/content/T2_2b_lineage_out.zip')"""),
    ("code", """# ---- TABPFN_INSTALL_GUARD
# ---- TabPFN's dependency floor moves numpy and scipy. On this machine that
# ---- broke the frozen sklearn outright. So: install, then re-import, then
# ---- re-run gate 1 and refuse to continue if the ceiling has moved.
import subprocess, sys, importlib, pathlib
before = BASELINE_VERSIONS

# The lineage results must already be off this machine before the environment
# moves under them. Cell 9 writes this archive; if it is absent or empty, cell 9
# did not run in this runtime and the sweep results are either missing or about
# to be invalidated by the install below.
# Set this to True ONLY if the lineage archive is already downloaded to your
# own machine and Colab has since given you a fresh VM, which wipes /content.
LINEAGE_ALREADY_DOWNLOADED = False

ARCHIVE = pathlib.Path('/content/T2_2b_lineage_out.zip')
have = ARCHIVE.exists() and ARCHIVE.stat().st_size > 0
assert have or LINEAGE_ALREADY_DOWNLOADED, (
    'REFUSING to install TabPFN: /content/T2_2b_lineage_out.zip is missing or '
    'empty, so cell 9 has not run on this filesystem. Installing TabPFN now '
    'would move numpy and scipy under a lineage sweep whose results are not '
    'saved. Two cases. If you have not run the sweep yet, run cells 1 to 9. '
    'If you ran it in an earlier session and already have the archive on your '
    'own machine, Colab has given you a fresh VM; set '
    'LINEAGE_ALREADY_DOWNLOADED = True at the top of this cell and re-run it.')
print(f'lineage archive: '
      f'{ARCHIVE.stat().st_size} bytes' if have
      else 'lineage archive: absent, proceeding on your explicit override')

subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'tabpfn'], check=True)

import numpy, scipy, sklearn
for m in (numpy, scipy, sklearn):
    importlib.reload(m)
import tabpfn
after = (numpy.__version__, scipy.__version__, sklearn.__version__)
print('numpy/scipy/sklearn before:', before)
print('numpy/scipy/sklearn after :', after)
print('tabpfn:', getattr(tabpfn, '__version__', 'unknown'))
if before != after:
    print('WARNING: the install moved the analysis stack. Gate 1 decides.')

rc = subprocess.run(['python', 'code/r32_lineage_contract.py',
                     '--stage', 'gate1', '--arch', 'rgcn']).returncode
assert rc == 0, (
    'GATE 1 NO LONGER PASSES after installing TabPFN. The ceiling is computed '
    'with sklearn, so every number from here would be against a moved ceiling. '
    'Restart the runtime, re-upload the bundle, and run only the TabPFN cells.')
print('gate 1 still passes after the TabPFN install')"""),
    ("code", """# ---- TabPFN smoke: one discipline, whatever the fit stage does for seed 0.
# ---- This is the first time the GPU TabPFN path runs anywhere, so it runs
# ---- alone before the sweep.
import subprocess, json, pathlib
set_field('econ')
rc = subprocess.run(['python', 'code/r33_tabpfn_gpu.py', '--stage', 'all']).returncode
assert rc == 0, f'TabPFN smoke failed with exit {rc} (13 = out of memory)'
d = json.loads(pathlib.Path('/content/t22b/results/revision/T3_4_tabpfn_gpu/'
                            'econ/summary.json').read_text())
print(f"econ: {d['train_rows_used']}/{d['train_rows_available']} training rows, "
      f"capped={d['context_capped']}, over pretraining limit="
      f"{d['exceeds_pretraining_limit']}, mean {d['tabpfn_mean_auc_pr']:.4f}, "
      f"delta vs M5' {d['delta_vs_M5prime']:+.4f}")
set_field(FIELD)
print('tabpfn smoke passed')"""),
    ("code", """# ---- TabPFN sweep: the remaining four disciplines, resumable per seed.
import subprocess, time
t_all = time.time()
done, failed = 0, []
for field in SWEEP_FIELDS:
    set_field(field)
    print('=' * 72); print(f'[tabpfn/{field}] running', flush=True)
    rc = subprocess.run(['python', 'code/r33_tabpfn_gpu.py',
                         '--stage', 'all']).returncode
    if rc == 0:
        done += 1
    elif rc == 13:
        print(f'  {field}: OUT OF MEMORY at full context. Re-run this one field '
              f'with an explicit cap, which is recorded as a handicap:')
        print(f"    os.environ['DATASET']='{field}'; "
              f"!python code/r33_tabpfn_gpu.py --stage all --max-train 8000")
        failed.append(f'{field}@oom')
    else:
        failed.append(f'{field}@exit{rc}')
set_field(FIELD)
print()
print(f'TABPFN SWEEP FINISHED in {time.time()-t_all:.0f}s: {done} done, '
      f'{len(failed)} failed')
for f in failed: print('  FAILED:', f)"""),
    ("code", """# ---- the TabPFN answer
import json, pathlib
R = pathlib.Path('/content/t22b/results/revision/T3_4_tabpfn_gpu')
print(f"{'discipline':11} {'rows':>7} {'over lim':>8} {'GPU mean':>9} "
      f"{'CPU 1k':>7} {'vs M5prime':>11} {'CI (student)':>20} exceeds")
print('-' * 92)
for f in SWEEP_FIELDS:
    p = R / f / 'summary.json'
    if not p.exists():
        print(f'{f:11} (absent)'); continue
    d = json.loads(p.read_text())
    cpu = d.get('cpu_1000row_mean_auc_pr')
    print(f"{f:11} {d['train_rows_used']:>7} {str(d['exceeds_pretraining_limit']):>8} "
          f"{d['tabpfn_mean_auc_pr']:>9.4f} "
          f"{(f'{cpu:.4f}' if cpu is not None else 'n/a'):>7} "
          f"{d['delta_vs_M5prime']:>+11.4f} "
          f"{str(d['student_ci95_vs_M5prime']):>20} {d['exceeds_fair']}")"""),
    ("code", """import shutil
shutil.make_archive('/content/T2_2b_all_out', 'zip',
                    '/content/t22b/results', 'revision')
from google.colab import files
files.download('/content/T2_2b_all_out.zip')"""),
]


def notebook():
    cells = []
    for kind, src in NB_CELLS:
        c = {"cell_type": kind, "metadata": {},
             "source": src.splitlines(keepends=True)}
        if kind == "code":
            c["outputs"] = []
            c["execution_count"] = None
        cells.append(c)
    nb = {"cells": cells, "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 0}
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"  notebook: {NB_PATH}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--smoke", action="store_true",
                   help="after building, extract the zip into a sandbox with "
                        "no access to this repository and run the cells that "
                        "can run without a GPU")
    args = a.parse_args()
    notebook()
    build()
    raise SystemExit(smoke() if args.smoke else 0)
