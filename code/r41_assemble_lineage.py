#!/usr/bin/env python3
"""C9: assemble Tables 12c and 13c from the lineage cells, in one command.

The Colab output lands at
results/revision/T2_2b_lineage_contract/<field>/<cell>/summary.json and this
turns it into tables with no further decisions. It is r28_assemble_tables for
the lineage arm, with the same refusals and two more that the lineage arm needs.

Five things it refuses on, in this order, because a later check is meaningless
if an earlier one fails:

  1  fewer than twenty cells, and it names exactly which are missing
  2  a cell whose recorded field disagrees with the directory it sits in, or
     whose per-seed arrays do not match that discipline's cohort size
  3  a cell whose M5 prime disagrees with the frozen ceiling by more than
     1e-3, which means the in-process refit that produced it drifted
  4  cells feeding one table that ran on different stacks, the check
     r28.check_env_homogeneous applies to 12b and 13b
  5  a cell that does not declare a target_table and target_row

Rows are routed by each cell's own target_table and target_row, never by
parsing its directory name, because --protocol tuned serves two tables
depending on architecture.

The lineage minus strict column is printed and stored. It is the only column
that isolates the two lineage relations from everything else, because a lineage
cell and its matching strict cell differ in nothing but those relations.

  python code/r41_assemble_lineage.py
  python code/r41_assemble_lineage.py --tree <dir>   # a synthetic or staged tree

Output: <tree>/assembled_lineage_tables.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from r28_assemble_tables import FIELDS, M5P, NTEST  # noqa: E402

DEFAULT_TREE = ROOT / "results" / "revision" / "T2_2b_lineage_contract"
ARCHS = ("rgcn", "gat")
PROTOCOLS = ("prereg", "tuned")
EXPECT = {(f, f"{a}_{p}_lineage") for f in FIELDS
          for a in ARCHS for p in PROTOCOLS}
TABLES = ("Table 12c", "Table 13c")
DISP = {"econ": "economics", "math": "mathematics", "neuro": "neuroscience",
        "physics": "physics", "chemistry": "chemistry"}


def load(tree):
    out = {}
    for f in FIELDS:
        d = tree / f
        if not d.is_dir():
            continue
        for c in sorted(d.iterdir()):
            p = c / "summary.json"
            if c.is_dir() and p.exists():
                out[(f, c.name)] = json.loads(p.read_text(encoding="utf-8"))
    return out


def check_env_homogeneous(cells):
    """r28.check_env_homogeneous, over dicts already loaded rather than paths."""
    seen, bad = {}, []
    for (field, cell), s in sorted(cells.items()):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=None)
    a = ap.parse_args()
    tree = Path(a.tree) if a.tree else DEFAULT_TREE

    if not tree.is_dir():
        print(f"STOPPING: {tree} does not exist. The lineage arm has not landed.")
        return 2
    cells = load(tree)

    # ---- 1. completeness, naming what is absent ----
    got = set(cells)
    missing = sorted(EXPECT - got)
    extra = sorted(got - EXPECT)
    print(f"[r41] {len(got)} of {len(EXPECT)} lineage cells present in {tree}")
    if extra:
        print(f"  unexpected cells, not routed: {extra}")
    if missing:
        print(f"  MISSING {len(missing)}:")
        for f, c in missing:
            print(f"    {f}/{c}")
        print("STOPPING: the lineage tables would be silently short. Re-run "
              "those cells; the sweep resumes and skips what is complete.")
        return 3

    # ---- 2. integrity, r28's three checks ----
    bad = []
    for (f, c), s in sorted(cells.items()):
        if s.get("field") != f:
            bad.append(f"{f}/{c}: records field {s.get('field')!r}")
        n = len(s.get("per_seed_auc_pr") or [])
        if n != 10:
            bad.append(f"{f}/{c}: {n} seeds, expected 10")
        st = (s.get("lineage_graph") or {}).get("n_rows")
        if st is not None and st < NTEST[f]:
            bad.append(f"{f}/{c}: graph has {st} rows, fewer than {f}'s "
                       f"{NTEST[f]} test students")
    for b in bad:
        print("  VIOLATION:", b)
    if bad:
        print("STOPPING: a cell's content disagrees with its path.")
        return 4

    # ---- 3. the ceiling each cell was actually scored against ----
    drift = [f"{f}/{c}: M5' {s.get('M5prime_mean')} against frozen {M5P[f]}"
             for (f, c), s in sorted(cells.items())
             if abs(float(s.get("M5prime_mean", -9)) - M5P[f]) > 1e-3]
    for d in drift:
        print("  CEILING DRIFT:", d)
    if drift:
        print("STOPPING: a cell was scored against a ceiling that is not the "
              "frozen one. Its delta is not comparable to any other cell's.")
        return 5

    # ---- 4. one stack per table ----
    env_bad = check_env_homogeneous(cells)
    for b in env_bad:
        print("  ENV VIOLATION:", b)
    if env_bad:
        print("STOPPING: cells feeding one table ran on different stacks. The "
              "stack-to-stack offset measured on this project (0.0014) exceeds "
              "the run-to-run floor (0.0013), so a mixed table can move a "
              "verdict.")
        return 6
    print("  clean: 20 cells, one stack per table, ceilings match")

    # ---- 5. route and emit ----
    tables = {t: {} for t in TABLES}
    for (f, c), s in sorted(cells.items()):
        t, row = s.get("target_table"), s.get("target_row")
        if t not in TABLES or not row:
            print(f"STOPPING: {f}/{c} declares target_table {t!r}; cannot route.")
            return 7
        strict = s.get("matching_strict_cell") or {}
        tables[t].setdefault(f, []).append({
            "row": row, "config": "lineage",
            "delta_vs_M5prime": s.get("delta_vs_M5prime"),
            "ci": s.get("student_ci95_vs_M5prime"),
            "p_BH": s.get("p_BH"), "exceeds": s.get("exceeds_fair"),
            "strict_delta": strict.get("delta_vs_M5prime"),
            "strict_cell": strict.get("cell"),
            "lineage_minus_strict": s.get("lineage_minus_strict_delta"),
            "ancestry_edges": (s.get("lineage_graph") or {}).get(
                "ancestry_edges"),
            "rows_with_ancestry_share": (s.get("lineage_coverage") or {}).get(
                "rows_with_ancestry_share"),
            "cell": c,
        })

    out = {
        "integrity": {"cells": len(cells), "violations": 0,
                      "one_stack_per_table": True},
        "tables": {t: {"rows": tables[t],
                       "n_rows": sum(len(v) for v in tables[t].values()),
                       "n_exceeds": sum(1 for v in tables[t].values()
                                        for r in v if r["exceeds"])}
                   for t in TABLES},
    }
    (tree / "assembled_lineage_tables.json").write_text(json.dumps(out, indent=2))

    print()
    hdr = (f"{'discipline':12} {'row':26} {'lineage':>9} {'strict':>9} "
           f"{'lin-str':>9} {'CI (student)':>20} {'p_BH':>8}  verdict  cover")
    print(hdr); print("-" * len(hdr))
    for t in TABLES:
        print(f"[{t}]")
        for f in FIELDS:
            for r in sorted(tables[t].get(f, []), key=lambda z: z["row"]):
                sd = r["strict_delta"]
                lm = r["lineage_minus_strict"]
                cov = r["rows_with_ancestry_share"]
                print(f"{DISP[f]:12} {r['row']:26} "
                      f"{r['delta_vs_M5prime']:+9.4f} "
                      f"{(f'{sd:+.4f}' if sd is not None else 'n/a'):>9} "
                      f"{(f'{lm:+.4f}' if lm is not None else 'n/a'):>9} "
                      f"{str(r['ci']):>20} {r['p_BH']:>8.4f}  "
                      f"{'EXCEEDS' if r['exceeds'] else 'null   '}  "
                      f"{(f'{cov:.2f}' if cov is not None else 'n/a')}")
    print()
    for t in TABLES:
        print(f"  {t}: {out['tables'][t]['n_rows']} rows, "
              f"{out['tables'][t]['n_exceeds']} exceed")
    print("\nread the lin-str column: it is the only one that isolates the two "
          "lineage relations. Read it beside the coverage column, because a "
          "null where three rows in four carry no ancestry edge is a null "
          "about coverage, not about lineage.")
    print(f"\n-> {tree / 'assembled_lineage_tables.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
