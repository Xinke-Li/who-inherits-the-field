#!/usr/bin/env python3
"""Emit Table 4, the body's verdict table, from the assembled strict cells.

Table 4 used to list the seven superseded crossings of the unrepaired construction,
with a superseded gap beside a corrected one. That comparison is the audit's
narrative and it is kept, in Table~\\ref{tab:full20} and Figure~\\ref{fig:audit},
where all twenty pre-registered rows still sit. The body table is now the eight
cells a reader has to see to know what the graph arm did under the contract the
paper actually states.

The eight are chosen, not thresholded, and each earns its line:

  chemistry RGCN, GAT            the two pre-registered cells that exceed
  chemistry RGCN-sym, GAT-sym    the two symmetric cells that exceed
  chemistry HGT tuned            fails the second gate alone, interval excludes
                                 zero, so three of four chemistry architectures
                                 sit at or over the line
  neuroscience RGCN-sym          the boundary case, lower bound $-$0.0006
  physics HGT tuned              the largest point estimate outside chemistry
  mathematics GAT                the one below-ceiling cell whose student
                                 interval excludes zero

Every number is read from assembled_tables.json. Nothing is typed here.

  python code/r37_emit_table4.py

Output: paper/table4_verdicts.tex
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from r28_assemble_tables import TREE  # noqa: E402

OUT = ROOT / "paper" / "table4_verdicts.tex"
# Short forms, matching Table~\ref{tab:fullgrid}, so the body table fits one
# column. The caption spells out which discipline "neuro" is.
DISP = {"econ": "economics", "math": "math", "neuro": "neuro",
        "physics": "physics", "chemistry": "chemistry"}
LONG = {"econ": "economics", "math": "mathematics", "neuro": "neuroscience",
        "physics": "physics", "chemistry": "chemistry"}

# (field, table, row label, config) -> the protocol name the body column shows
SELECT = [
    ("chemistry", "Table 12b", "RGCN",           "strict", "pre-reg."),
    ("chemistry", "Table 12b", "GAT",            "strict", "pre-reg."),
    ("chemistry", "Table 13b", "RGCN symmetric", "strict", "sym.\\ grid"),
    ("chemistry", "Table 13b", "GAT symmetric",  "strict", "sym.\\ grid"),
    ("chemistry", "Table 12b", "HGT tuned",      "strict", "tuned"),
    ("neuro",     "Table 13b", "RGCN symmetric", "strict", "sym.\\ grid"),
    ("physics",   "Table 12b", "HGT tuned",      "strict", "tuned"),
    ("math",      "Table 12b", "GAT",            "strict", "pre-reg."),
]


def num(x):
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.4f}"


def verdict(r):
    """Three gates decide. A cell that fails only the p gate, and a cell below
    the ceiling whose interval excludes zero, are both distinct from a null and
    are named rather than collapsed into one word."""
    if r["exceeds"]:
        return "exceeds"
    lo, hi = r["ci"]
    if hi < 0:
        return "below"
    if lo > 0:
        return "null (gate 2)"
    return "null"


def main():
    data = json.loads((TREE / "assembled_tables.json").read_text(encoding="utf-8"))
    rows, chosen = [], []
    for field, tab, row, cfg, proto in SELECT:
        cand = [r for r in data["tables"][tab]["rows"].get(field, [])
                if r["row"] == row and r["config"] == cfg]
        if len(cand) != 1:
            raise SystemExit(
                f"r37: {field}/{tab}/{row}/{cfg} matched {len(cand)} cells, "
                f"expected exactly one. Refusing to emit a body table from an "
                f"ambiguous or missing cell.")
        r = cand[0]
        chosen.append((field, row, r))
        lo, hi = r["ci"]
        rows.append(
            f"{DISP[field]} & {row.replace(' symmetric', '-sym')} & {proto} & "
            f"{num(r['delta_vs_M5prime'])} & {num(lo)}, {num(hi)} & "
            f"{r['p_BH']:.4f} & {verdict(r)} \\\\")

    n_exceed = sum(1 for _, _, r in chosen if r["exceeds"])
    total_exceed = sum(data["tables"][t]["n_exceeds"] for t in data["tables"])
    total_rows = sum(data["tables"][t]["n_rows"] for t in data["tables"])

    # Any exceeding cell left out must be named. "the remainder are null" would
    # be false otherwise, and a reader counting exceedances would come up short.
    picked = {(f, r["cell"]) for f, _, r in chosen}
    omitted = []
    for tab in data["tables"]:
        for f, rs in data["tables"][tab]["rows"].items():
            for r in rs:
                if r["exceeds"] and (f, r["cell"]) not in picked:
                    omitted.append(f"{DISP[f]} {r['row']} at the pinned "
                                   f"{r['config']} configuration "
                                   f"({num(r['delta_vs_M5prime'])})")
    if omitted:
        omit_tex = ("The " + ("one" if len(omitted) == 1 else str(len(omitted)))
                    + " exceeding cell" + ("" if len(omitted) == 1 else "s")
                    + " not listed here " + ("is " if len(omitted) == 1 else "are ")
                    + "; ".join(omitted) + ". ")
    else:
        omit_tex = ""

    tex = r"""\begin{table}[t]
\caption{\textbf{The graph arm under the strict contract: the eight cells that
carry the result.} $\Delta$ is the ten-seed mean gap to the validation-symmetric
ceiling M5$'$, the interval is the paired student-level bootstrap of
eq.~(\ref{eq:gates}) over 2000 draws, and the verdict applies all three gates.
Of __TOTALROWS__ cells across the two protocols, __TOTALEXCEED__ exceed and
__NEXCEED__ of those are listed here. __OMITTED__Every remaining cell is null,
and Appendix~\ref{app:strict} gives all __TOTALROWS__. The last three rows are
listed because they are not nulls of the same kind: neuroscience turns on the
third gate alone, physics carries the largest point estimate outside chemistry,
and mathematics is the one cell below the ceiling whose student interval
excludes zero. ``null (gate 2)'' marks a cell whose interval excludes zero but
whose corrected $p$ does not clear 0.05, and ``below'' a cell whose interval
lies wholly under the ceiling. Table~\ref{tab:full20} retains all twenty
comparisons under the unrepaired construction as the pre-specified audit
trail. Cells under
\texttt{results/revision/T2\_1\_strict\_contract/T2\_1\_final/}.}
\label{tab:gnn}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{lllrlrl}
\toprule
discipline & arch. & protocol & $\Delta$ vs M5$'$ & 95\% CI (student) &
$p_{\mathrm{BH}}$ & verdict \\
\midrule
__ROWS__
\bottomrule
\end{tabular}
\end{table}
"""
    tex = (tex.replace("__ROWS__", "\n".join(rows))
              .replace("__TOTALROWS__", str(total_rows))
              .replace("__TOTALEXCEED__", str(total_exceed))
              .replace("__NEXCEED__", str(n_exceed))
              .replace("__OMITTED__", omit_tex))
    OUT.write_text(tex, encoding="utf-8", newline="\n")
    print(f"  wrote {OUT}")
    print(f"  {len(rows)} rows, {n_exceed} of the {total_exceed} exceeding "
          f"cells among {total_rows} total")
    for field, row, r in chosen:
        print(f"    {DISP[field]:13} {row:16} {r['delta_vs_M5prime']:+.4f} "
              f"{r['ci']}  p_BH {r['p_BH']:.4f}  {verdict(r)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
