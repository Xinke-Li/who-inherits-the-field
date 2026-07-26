#!/usr/bin/env python3
"""Build the self-contained T2.1 bundle and the Colab notebook.

The bundle holds the fixed code and one discipline's inputs, so the notebook
needs nothing but the upload. Nothing is fetched from or pushed to any remote.

  python code/r26_make_t2_1_bundle.py --field chemistry
"""
import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision"
FIELDS = ["chemistry", "physics", "neuro", "econ", "math"]

CODE = [
    "code/r25_strict_contract.py",
    "code/r27_attribution.py",
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




def _capabilities():
    """Read what the packed code actually supports, so a stale bundle is
    visible at cell 1 rather than only after the results come back wrong."""
    import re
    src = (ROOT / "code" / "r27_attribution.py").read_text(encoding="utf-8")
    cells = sorted(set(re.findall(r'^\s{4}"([A-Z]_[a-z0-9_]+)":', src, re.M)))
    e2 = (ROOT / "code" / "paper_pipeline" / "experiments" /
          "e2_hgt.py").read_text(encoding="utf-8")
    contracts = sorted({c for c in ("legacy", "strict", "f1a", "f1b")
                        if f'"{c}"' in e2})
    r3 = (ROOT / "code" / "r3_rgcn_symmetric.py").read_text(encoding="utf-8")
    archs = sorted({a for a in ("rgcn", "gat", "hgt")
                    if f'arch == "{a}"' in r3})
    nb = ROOT / "colab" / "t2_1_strict_contract.ipynb"
    sweep = "SWEEP_FIELDS" in nb.read_text(encoding="utf-8") if nb.exists() else False
    return {"attribution_cells": cells, "graph_contracts": contracts,
            "architectures": archs, "notebook_multi_discipline_sweep": sweep}


def _manifest_hash(paths):
    """SHA-256 over the packed file CONTENTS, order-independent of zip metadata."""
    import hashlib
    h = hashlib.sha256()
    for rel in sorted(paths):
        p = ROOT / rel
        if p.exists():
            h.update(rel.encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()

def _validate(zip_paths, nb_path):
    """Compile every .py packed and every notebook code cell. Refuse on failure.

    A broken notebook fails in seconds; a broken module inside the bundle fails
    hours into an A100 run. Both are cheap to rule out here.
    """
    import json as _json
    bad = []
    for rel in zip_paths:
        p = ROOT / rel
        if p.suffix != ".py":
            continue
        try:
            compile(p.read_text(encoding="utf-8"), rel, "exec")
        except SyntaxError as e:
            bad.append(f"{rel}: line {e.lineno}: {e.msg}")
    nb = _json.loads(Path(nb_path).read_text(encoding="utf-8"))
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

def build(field):
    zf = OUT / "T2_1_bundle.zip"
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [p for p in CODE if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"missing from the bundle manifest: {missing}")

    nb_path = ROOT / "colab" / "t2_1_strict_contract.ipynb"
    if nb_path.exists():
        _validate(CODE, nb_path)

    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in CODE:
            z.write(ROOT / p, p)
        for fld in FIELDS:
            z.write(ROOT / "data" / f"clean_dataset_{fld}.parquet",
                    f"data/clean_dataset_{fld}.parquet")
            for f in ("e1_baselines.json", "e12_corrected_vs_m5.json"):
                src = ROOT / "results" / f"results_{fld}" / f
                if src.exists():
                    z.write(src, f"results/results_{fld}/{f}")
            # B(i) pins the legacy grid winner, read from results/robustness/.
            # Omitting these made --config legacy fail on Colab while passing
            # locally, because the full repo has them and the bundle did not.
            for arch in ("rgcn", "gat", "hgt"):
                g = (ROOT / "results" / "robustness" / "full_symmetric_grid" /
                     f"{fld}_{arch}_sym_seed0.json")
                if g.exists():
                    z.write(g, f"results/robustness/full_symmetric_grid/{g.name}")
        import datetime as _dt
        caps = _capabilities()
        packed = list(CODE) + [f"data/clean_dataset_{f}.parquet" for f in FIELDS]
        z.writestr("BUNDLE.json", json.dumps({
            "task": "T2.1 strict temporal contract",
            "build_timestamp_utc": _dt.datetime.now(_dt.timezone.utc)
                                      .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "manifest_sha256": _manifest_hash(packed),
            "capabilities": caps,
            "fields": FIELDS,
            "entrypoint": "code/r25_strict_contract.py",
            "gate1": "reproduces M5 and M5' and blocks the run on mismatch",
            "changes_vs_r3": [
                "build_graph_v2(contract='strict'): advisor nodes keyed by "
                "(advisor_pid, t0), prior-cohort sibling edges only (F1a, F1b)",
                "class weight from the training split alone (F2)"],
            "no_remote_access": True,
        }, indent=2))
    mb = zf.stat().st_size / 1e6
    print(f"  bundle: {zf}  ({mb:.1f} MB)")
    return zf, mb


NB_CELLS = [
    ("markdown", """# T2.1 - the graph arm under the strict temporal contract

Upload `T2_1_bundle.zip` when the first cell asks. Nothing is downloaded and
nothing is published; the notebook reads the bundle and writes local files you
download at the end.

**Gate 1 runs first and blocks.** If the frozen ceiling does not reproduce, the
notebook stops instead of producing a verdict."""),
    ("code", """from google.colab import files
import zipfile, os, pathlib, json
up = files.upload()                     # choose T2_1_bundle.zip
name = next(iter(up))
pathlib.Path('/content/t21').mkdir(exist_ok=True)
with zipfile.ZipFile(name) as z:
    z.extractall('/content/t21')
os.chdir('/content/t21')

B = json.load(open('BUNDLE.json'))
print('build   :', B.get('build_timestamp_utc'))
print('manifest:', B.get('manifest_sha256', '')[:16])
caps = B.get('capabilities', {})
for k, v in caps.items():
    print(f'{k:34}', v)

# What THIS notebook intends to use. A stale bundle fails here, loudly, instead
# of silently producing a short attribution table hours later.
NEED_CELLS = {'A_legacy_globalw', 'B_legacy_trainw', 'C_strict_globalw',
              'E_f1a_only_trainw', 'F_f1b_only_trainw'}
NEED_CONTRACTS = {'legacy', 'strict', 'f1a', 'f1b'}
missing_cells = NEED_CELLS - set(caps.get('attribution_cells', []))
missing_contr = NEED_CONTRACTS - set(caps.get('graph_contracts', []))
assert not missing_cells, (
    f'STALE BUNDLE: attribution cells missing {sorted(missing_cells)}. '
    f'Re-upload the current T2_1_bundle.zip.')
assert not missing_contr, (
    f'STALE BUNDLE: graph contracts missing {sorted(missing_contr)}. '
    f'Re-upload the current T2_1_bundle.zip.')
assert caps.get('notebook_multi_discipline_sweep'), (
    'STALE BUNDLE: built before the five-discipline sweep. Re-upload.')
print()
print('bundle capability check passed')"""),
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

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print('GPU  :', gpu)
print('torch:', torch.__version__, 'cuda', torch.version.cuda)
print('pyg  :', torch_geometric.__version__)

assert gpu is not None, 'FAIL: no GPU. Runtime > Change runtime type > A100.'
assert torch_geometric.__version__ == REQ_PYG, (
    f'FAIL: PyG {torch_geometric.__version__}, need {REQ_PYG}. '
    f'Run: pip install torch-geometric=={REQ_PYG} then restart the runtime.')
assert int(torch.__version__.split('.')[0]) >= REQ_TORCH_MAJOR, (
    f'FAIL: torch {torch.__version__}, need >= {REQ_TORCH_MAJOR}.x')
print('version assertions passed')"""),
    ("code", """# ---- disciplines. The RGCN sweep runs all of these; the
# ---- attribution 2x2 is chemistry-only and uses FIELD.
SWEEP_FIELDS = ['chemistry', 'physics', 'neuro', 'econ', 'math']
FIELD = 'chemistry'          # attribution / restore / summary target

import os
def set_field(f):
    os.environ['DATASET'] = f
    os.environ['DATASET_PATH'] = f'/content/t21/data/clean_dataset_{f}.parquet'
    os.environ['RESULTS_DIR'] = f'/content/t21/results/results_{f}'
    return f
set_field(FIELD)
print('sweep:', SWEEP_FIELDS)"""),
    ("code", """# ---- GATE 1 for every discipline in the sweep. Blocks on any failure.
import subprocess
for f in SWEEP_FIELDS:
    set_field(f)
    rc = subprocess.run(['python', 'code/r25_strict_contract.py',
                         '--stage', 'gate1']).returncode
    assert rc == 0, f'GATE 1 FAILED for {f} - do not trust any verdict from this session'
set_field(FIELD)
print('gate 1 passed for all', len(SWEEP_FIELDS), 'disciplines')"""),
    ("code", """# ---- OPTIONAL: restore a previously downloaded partial tree.
# ---- Run on a resumed session; skip on a fresh run. This cell VERIFIES its own
# ---- work: it asserts the exact absolute paths the run cell will check, and
# ---- fails loudly if they are not there. A restore that lands one directory
# ---- off is silent otherwise, and the completed cells get recomputed.
import zipfile, pathlib, shutil, tempfile
from google.colab import files

ROOT_OUT = pathlib.Path('/content/t21/results/revision/T2_1_strict_contract')
ROOT_OUT.mkdir(parents=True, exist_ok=True)

up = files.upload()          # the T2_1_<field>_out.zip downloaded earlier
staged = 0
for name in up:
    tmp = pathlib.Path(tempfile.mkdtemp())
    with zipfile.ZipFile(name) as z:
        z.extractall(tmp)
    # Accept any nesting: locate every summary.json and place its cell
    # directory at ROOT_OUT/<field>/<cell>/, which is what the run cell checks.
    for sm in tmp.rglob('summary.json'):
        cell_dir = sm.parent
        dest = ROOT_OUT / FIELD / cell_dir.name
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
    print('  ', p)                      # absolute, not relative
assert found, 'RESTORE FAILED: no summary.json under ' + str(ROOT_OUT)
for p in found:
    assert p.parent.parent.name == FIELD, (
        f'RESTORE FAILED: {p} is not at ROOT_OUT/{FIELD}/<cell>/summary.json, '
        f'so the run cell will not find it')
print('restore verified: the run cell will skip these cells')"""),
    ("code", """# ---- FULL SWEEP: every remaining cell, unattended. Run and walk away.
# ---- Resumable: a cell with summary.json is skipped whole; within a cell each
# ---- completed seed is skipped. Safe to re-run after a disconnect.
import subprocess, time, json, pathlib

EXIT_OK, EXIT_SKIPPED, EXIT_NO_CONFIG = 0, 10, 11
ROOT_OUT = pathlib.Path('/content/t21/results/revision/T2_1_strict_contract')

# (arch, protocol) -> the table row it populates. Written into each summary.json
# by r25 so assembly never parses cell names.
PLAN = [('rgcn', 'prereg'), ('rgcn', 'tuned'),
        ('gat',  'prereg'), ('gat',  'tuned'),
        ('hgt',  'prereg'), ('hgt',  'tuned')]

def configs_for(field, arch, proto):
    c = ['strict']
    # B(i) pins the legacy grid winner. Only meaningful for the tuned protocol,
    # and omitted for chemistry RGCN where both grids chose the same config.
    if proto == 'tuned' and not (field == 'chemistry' and arch == 'rgcn'):
        c.append('legacy')
    return c

t_all = time.time()
done, skipped, failed = 0, 0, []
for field in SWEEP_FIELDS:
    set_field(field)
    for arch, proto in PLAN:
        for cfg in configs_for(field, arch, proto):
            cell = f'{arch}_{proto}_{cfg}'
            if (ROOT_OUT / field / cell / 'summary.json').exists():
                print(f'[{field}/{cell}] already complete'); done += 1; continue
            print('=' * 72); print(f'[{field}/{cell}] running', flush=True)
            flags = ['--arch', arch, '--protocol', proto, '--config', cfg]
            ok = True
            for stage in ('grid', 'seeds', 'aggregate'):
                t = time.time()
                rc = subprocess.run(['python', 'code/r25_strict_contract.py',
                                     '--stage', stage] + flags).returncode
                el = time.time() - t
                if rc == EXIT_SKIPPED:
                    print(f'  {stage}: skipped by design ({el:.0f}s)')
                elif rc == EXIT_OK:
                    print(f'  {stage}: ok ({el:.0f}s)')
                elif rc == EXIT_NO_CONFIG:
                    print(f'  {stage}: no legacy config for {field}/{arch}, '
                          f'skipping this cell'); ok = False; skipped += 1; break
                else:
                    # keep going: one bad cell must not cost the whole hour
                    print(f'  {stage}: FAILED exit {rc}'); ok = False
                    failed.append(f'{field}/{cell}@{stage}'); break
            if ok: done += 1

# forced single-session attribution recompute, chemistry only
set_field('chemistry')
print('=' * 72); print('[attribution] forced single-session recompute', flush=True)
rc = subprocess.run(['python', 'code/r27_attribution.py', '--force']).returncode
if rc != 0: failed.append(f'attribution@exit{rc}')

set_field(FIELD)
print()
print(f'SWEEP FINISHED in {time.time()-t_all:.0f}s: {done} cells done, '
      f'{skipped} skipped for missing config, {len(failed)} failed')
for f in failed: print('  FAILED:', f)"""),
    ("code", """# ---- attribution 2x2: splits the strict-contract increase into its causes.
# ---- Cell D (strict + train-only) is the verdict run and is read, not rerun.
# ---- Cell A deliberately uses the UNFIXED build_graph; every file it writes
# ---- says so and is marked IS_VERDICT_CELL false.
import subprocess, time, json, pathlib

t0 = time.time()
rc = subprocess.run(['python', 'code/r27_attribution.py']).returncode
print(f'--- attribution: exit {rc} ({time.time()-t0:.0f}s) ---')
assert rc == 0, f'attribution run failed with exit {rc}'

A = pathlib.Path(f'/content/t21/results/revision/T2_1_strict_contract/{FIELD}')
rows = [('A_legacy_globalw', 'legacy construction, global weight'),
        ('B_legacy_trainw', 'legacy construction, train-only weight'),
        ('C_strict_globalw', 'strict construction, global weight')]
print()
print(f"{'cell':22} {'mean AUC-PR':>12}  what it isolates")
print('-' * 72)
vals = {}
for cell, label in rows:
    f = A / 'attribution' / cell / 'summary.json'
    if f.exists():
        d = json.loads(f.read_text())
        vals[cell] = d['mean_auc_pr']
        print(f"{cell:22} {d['mean_auc_pr']:>12.4f}  {label}")
d = A / 'rgcn_tuned_strict' / 'summary.json'
if d.exists():
    v = json.loads(d.read_text())
    vals['D'] = v['seed_mean_auc_pr']
    print(f"{'D_strict_trainw':22} {v['seed_mean_auc_pr']:>12.4f}  "
          f"the verdict run (delta vs M5' {v['delta_vs_M5prime']:+.4f})")
print()
if 'A_legacy_globalw' in vals:
    print(f"anchor: cell A reproduces the legacy stack at "
          f"{vals['A_legacy_globalw']:.4f}; compare against r3's legacy mean to "
          f"read off the environment offset.")
if {'A_legacy_globalw', 'B_legacy_trainw', 'C_strict_globalw'} <= set(vals):
    print(f"F2 alone (B - A): {vals['B_legacy_trainw']-vals['A_legacy_globalw']:+.4f}")
    print(f"F1 alone (C - A): {vals['C_strict_globalw']-vals['A_legacy_globalw']:+.4f}")"""),
    ("code", """# ---- the answer, all disciplines, without downloading anything
import json, pathlib
ROOT_OUT = pathlib.Path('/content/t21/results/revision/T2_1_strict_contract')
LEGACY = {'prereg': 0.022, 'tuned': 0.035}
print(f"{'discipline':11} {'cell':22} {'strict d':>9} {'CI (student)':>22} "
      f"{'p_BH':>8}  verdict")
print('-' * 92)
for field in SWEEP_FIELDS:
    for p in sorted((ROOT_OUT / field).glob('rgcn_*/summary.json')):
        d = json.loads(p.read_text())
        print(f"{field:11} {p.parent.name:22} {d['delta_vs_M5prime']:+9.4f} "
              f"{str(d['student_ci95_vs_M5prime']):>22} {d['p_BH']:>8.4f}  "
              f"exceeds={d['exceeds_fair']}")
print()
print('chemistry legacy readings for reference: prereg +0.022, tuned +0.035')
print('stop condition (b): any cell that was null and now shows exceeds=True')"""),
    ("code", """import shutil
shutil.make_archive('/content/T2_1_all_out', 'zip',
                    '/content/t21/results/revision', 'T2_1_strict_contract')
from google.colab import files
files.download('/content/T2_1_all_out.zip')"""),
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
    p = ROOT / "colab" / "t2_1_strict_contract.ipynb"
    p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"  notebook: {p}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--field", default="chemistry")
    args = a.parse_args()
    notebook()
    build(args.field)
