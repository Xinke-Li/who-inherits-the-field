#!/usr/bin/env python3
"""Assemble the two strict-contract tables (tab:full20b, tab:fullgridb) from the strict-contract cells.

"Table 12b" and "Table 13b" below are the project's internal names for the two,
and the keys the assembled JSON uses; the paper refers to them only by label.

The assembly aborts on any integrity violation. A sweep bug wrote economics'
RGCN outputs into chemistry's directory -- the completion check used the global
FIELD rather than the loop variable -- and three chemistry cells carried econ's
data, econ's ceiling and econ's 495-row score arrays. Nothing detected it until
the numbers were read by eye. These three checks find that class of fault in
seconds, so no table is assembled without them:

  * every summary's `field` equals its parent directory name
  * its M5prime matches that discipline's frozen ceiling to 1e-3
  * its per-seed test_scores length equals that discipline's test cohort size

Rows are routed by the `target_table` and `target_row` fields each cell declares,
never by parsing cell names, because --protocol tuned serves two different tables
depending on architecture.

  python code/r28_assemble_tables.py [--repair]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "revision" / "T2_1_strict_contract"
# the sweep download nests one level; the authoritative tree is named explicitly
TREE = BASE / "T2_1_final" / "T2_1_strict_contract"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
M5P = {"chemistry": 0.5374, "econ": 0.3613, "math": 0.4357,
       "neuro": 0.4249, "physics": 0.6485}
NTEST = {"econ": 495, "math": 935, "neuro": 3463, "physics": 2218,
         "chemistry": 4617}
TABLE_MAP = {("hgt", "prereg"): ("Table 12b", "HGT"),
             ("hgt", "tuned"): ("Table 12b", "HGT tuned"),
             ("rgcn", "prereg"): ("Table 12b", "RGCN"),
             ("gat", "prereg"): ("Table 12b", "GAT"),
             ("rgcn", "tuned"): ("Table 13b", "RGCN symmetric"),
             ("gat", "tuned"): ("Table 13b", "GAT symmetric")}


def canonical_cells():
    """The authoritative cell directories: <tree>/<field>/<cell>/summary.json."""
    out = {}
    for f in FIELDS:
        d = TREE / f
        if not d.is_dir():
            continue
        for c in sorted(d.iterdir()):
            if not (c.is_dir() and (c / "summary.json").exists()):
                continue
            # attribution cells declare themselves; they carry no ceiling and
            # must not be routed into a table or integrity-checked as verdicts
            j = json.loads((c / "summary.json").read_text(encoding="utf-8"))
            if j.get("IS_VERDICT_CELL") is False or c.name == "attribution":
                continue
            out[(f, c.name)] = c
    return out


def check(cells):
    """Three integrity checks. Returns the list of violations."""
    bad = []
    for (field, cell), d in sorted(cells.items()):
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        if s.get("field") != field:
            bad.append(f"{field}/{cell}: field={s.get('field')!r} but path says {field!r}")
        m = s.get("M5prime_mean")
        if m is None or abs(m - M5P[field]) > 1e-3:
            bad.append(f"{field}/{cell}: M5prime={m} but {field}'s ceiling is {M5P[field]}")
        seed0 = d / "seed0.json"
        if seed0.exists():
            n = len(json.loads(seed0.read_text(encoding="utf-8")).get("test_scores", []))
            if n != NTEST[field]:
                bad.append(f"{field}/{cell}: seed0 has {n} test scores, "
                           f"{field}'s cohort is {NTEST[field]}")
    return bad


def repair():
    """Restore the three corrupt chemistry cells from the earlier download."""
    done = []
    # the earlier, verified tree
    srcs = sorted(BASE.glob("**/chemistry/rgcn_prereg_strict/summary.json"))
    good = None
    for p in srcs:
        s = json.loads(p.read_text(encoding="utf-8"))
        if s.get("field") == "chemistry" and abs(
                (s.get("M5prime_mean") or 0) - M5P["chemistry"]) < 1e-3:
            good = p.parent.parent
            break
    if good is None:
        raise SystemExit("repair: no verified chemistry tree found to restore from")
    print(f"  restoring from {good}")
    for cell in ("rgcn_prereg_strict", "rgcn_tuned_strict"):
        src, dst = good / cell, TREE / "chemistry" / cell
        if not src.is_dir():
            print(f"  WARNING: {src} missing, skipped")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        done.append(f"restored chemistry/{cell} ({len(list(dst.glob('seed*.json')))} seeds)")
    legacy = TREE / "chemistry" / "rgcn_tuned_legacy"
    if legacy.exists():
        shutil.rmtree(legacy)
        done.append("deleted chemistry/rgcn_tuned_legacy (redundant; the strict "
                    "grid selected the legacy config)")
    # backfill routing
    for (field, cell), d in canonical_cells().items():
        p = d / "summary.json"
        s = json.loads(p.read_text(encoding="utf-8"))
        parts = cell.split("_")
        key = (parts[0], parts[1])
        if key in TABLE_MAP and not s.get("target_table"):
            s["target_table"], s["target_row"] = TABLE_MAP[key]
            s["target_backfilled"] = True
            p.write_text(json.dumps(s, indent=2), encoding="utf-8", newline="\n")
            done.append(f"backfilled routing into {field}/{cell}")
    for x in done:
        print("  " + x)


def check_env_homogeneous(cells):
    """Every cell feeding one table must have run on the same stack.

    A sweep long enough for Colab to disconnect is a sweep that can finish on a
    different GPU or a different torch build, and the resulting table would mix
    them without saying so. The measured stack-to-stack offset on this project
    is 0.0014, larger than the 0.0013 run-to-run floor and larger in magnitude
    than mathematics GAT's interval bound, so a mixed table can move a verdict.
    Compare the three strings that identify the stack and refuse rather than
    average across them.
    """
    seen, bad = {}, []
    for (field, cell), d in sorted(cells.items()):
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        env = s.get("env") or {}
        key = (env.get("gpu"), env.get("torch"), env.get("torch_geometric"))
        seen.setdefault(s.get("target_table"), {}).setdefault(key, []).append(
            f"{field}/{cell}")
    for table, groups in seen.items():
        if len(groups) > 1:
            bad.append(f"{table} mixes {len(groups)} environments: " + "; ".join(
                f"{k} <- {v[:3]}{'...' if len(v) > 3 else ''}"
                for k, v in groups.items()))
    return bad


def assemble(cells):
    tables = {"Table 12b": {}, "Table 13b": {}}
    for (field, cell), d in sorted(cells.items()):
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        t, row = s.get("target_table"), s.get("target_row")
        if not t:
            raise SystemExit(f"{field}/{cell}: no target_table; cannot route")
        cfg = cell.rsplit("_", 1)[-1]
        tables.setdefault(t, {}).setdefault(field, []).append({
            "row": row, "config": cfg,
            "delta_vs_M5prime": s.get("delta_vs_M5prime"),
            "ci": s.get("student_ci95_vs_M5prime"),
            "p_BH": s.get("p_BH"), "exceeds": s.get("exceeds_fair"),
            "cell": cell,
        })
    return tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair", action="store_true")
    a = ap.parse_args()

    if a.repair:
        print("REPAIR")
        repair()
        print()

    cells = canonical_cells()
    bad = check(cells)
    print(f"INTEGRITY: {len(cells)} cells, {len(bad)} violations")
    for b in bad:
        print("  VIOLATION:", b)
    if bad:
        raise SystemExit("ABORTING ASSEMBLY: a cell's content disagrees with its "
                         "path. Fix the tree before any table is built.")
    print(f"  clean {len(cells)} of {len(cells)}")

    env_bad = check_env_homogeneous(cells)
    for b in env_bad:
        print("  ENV VIOLATION:", b)
    if env_bad:
        raise SystemExit("ABORTING ASSEMBLY: cells feeding one table ran on "
                         "different stacks. The stack-to-stack offset measured "
                         "on this project (0.0014) exceeds the run-to-run floor "
                         "(0.0013), so a mixed table can move a verdict. Re-run "
                         "the odd cells on one stack.")
    print(f"  one environment per table")

    tables = assemble(cells)
    out = {"integrity": {"cells": len(cells), "violations": 0}, "tables": {}}
    for t in ("Table 12b", "Table 13b"):
        print(f"\n{t}")
        print(f"  {'discipline':11} {'row':16} {'cfg':7} {'delta':>8} "
              f"{'CI':>20} {'p_BH':>7}  verdict")
        n_ex = 0
        for field in FIELDS:
            for r in sorted(tables.get(t, {}).get(field, []), key=lambda x: x["row"]):
                ex = "EXCEEDS" if r["exceeds"] else "null"
                n_ex += bool(r["exceeds"])
                print(f"  {field:11} {r['row']:16} {r['config']:7} "
                      f"{r['delta_vs_M5prime']:+8.4f} {str(r['ci']):>20} "
                      f"{r['p_BH']:>7.4f}  {ex}")
        total = sum(len(v) for v in tables.get(t, {}).values())
        print(f"  -> {n_ex} of {total} exceed")
        out["tables"][t] = {"rows": tables.get(t, {}), "n_exceeds": n_ex,
                            "n_rows": total}
    (TREE / "assembled_tables.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(f"\n-> {TREE / 'assembled_tables.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
