#!/usr/bin/env python3
"""Emit the strict-contract appendix: Tables 12b and 13b plus the attribution
subsection, as a LaTeX fragment built from the assembled cells.

Rows are routed by each summary's own target_table and target_row fields, never
by parsing cell names, because --protocol tuned serves two different tables
depending on architecture. r28_assemble_tables.py has already refused to route
anything whose content disagrees with its path.

  python code/r29_emit_strict_appendix.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from r28_assemble_tables import FIELDS, TREE  # noqa: E402

OUT = ROOT / "paper" / "strict_appendix.tex"
DISP = {"econ": "economics", "math": "mathematics", "neuro": "neuroscience",
        "physics": "physics", "chemistry": "chemistry"}


def num(x):
    s = f"{abs(x):.4f}"
    return ("$+$" if x >= 0 else "$-$") + s


def rows_for(tab, data):
    out = []
    for f in FIELDS:
        for r in sorted(data["tables"][tab]["rows"].get(f, []),
                        key=lambda z: (z["row"], z["config"])):
            lo, hi = r["ci"]
            out.append(
                f"{DISP[f]} & {r['row']} & {r['config']} & "
                f"{num(r['delta_vs_M5prime'])} & [{num(lo)}, {num(hi)}] & "
                f"{r['p_BH']:.4f} & "
                f"{'exceeds' if r['exceeds'] else 'null'} \\\\")
    return "\n".join(out)


def main():
    data = json.loads((TREE / "assembled_tables.json").read_text(encoding="utf-8"))
    # The attribution numbers come from the artifact r27's verify step writes,
    # not from constants typed here. An earlier draft hardcoded the environment
    # offset at -0.0006; recomputed from r3's ten per-seed files it is -0.0014,
    # which sits at the determinism floor rather than inside it.
    ap = TREE / "chemistry" / "attribution" / "attribution.json"
    if not ap.exists():
        raise SystemExit(
            f"r29: {ap} is absent. Run\n"
            f"  python code/r27_attribution.py --verify-only --dir "
            f"{ap.parent}\n"
            f"first. Refusing to emit an attribution appendix from constants.")
    A = json.loads(ap.read_text(encoding="utf-8"))
    if not A.get("same_session"):
        raise SystemExit("r29: the attribution cells do not share one session; "
                         "refusing to emit the additivity claim.")
    if A.get("environment_offset_B_minus_r3") is None:
        raise SystemExit("r29: the environment offset was not computed (r3's "
                         "per-seed files were absent when r27 verified). "
                         "Re-run the verify step in the full repository.")

    # The determinism floor and how it was derived come from the measurement,
    # not from a number typed here. The offset is compared against it in the
    # text below, and 0.0014 against 0.0013 is a comparison a reader must be
    # able to check, so both sides are printed.
    # Two copies exist on purpose. The measurement was made during the T2.1
    # session and landed beside the local gate1 outputs; a duplicate now sits
    # in the authoritative tree so a reader who is told that tree is canonical
    # finds it there. Prefer the canonical one, and refuse if they diverge.
    dm_canon = TREE / "chemistry" / "DETERMINISM_MEASURED.json"
    dm_all = sorted((ROOT / "results" / "revision")
                    .rglob("DETERMINISM_MEASURED.json"))
    if not dm_all:
        raise SystemExit(
            "r29: DETERMINISM_MEASURED.json not found under results/revision/. "
            "The appendix compares the environment offset against the "
            "determinism floor and will not state that comparison from a "
            "constant.")
    dm = dm_canon if dm_canon.exists() else dm_all[0]
    bodies = {p.read_bytes() for p in dm_all}
    if len(bodies) > 1:
        raise SystemExit(
            f"r29: the {len(dm_all)} copies of DETERMINISM_MEASURED.json under "
            f"results/revision/ are not identical: "
            f"{[str(p) for p in dm_all]}. Refusing to quote a floor whose "
            f"source disagrees with itself.")
    DET = json.loads(dm.read_text(encoding="utf-8"))
    floor = float(DET["max_drift_point_estimate"])
    if abs(floor - float(A["determinism_floor"])) > 1e-12:
        raise SystemExit(
            f"r29: the floor recorded by the attribution verifier "
            f"({A['determinism_floor']}) disagrees with the measurement "
            f"({floor}) in {dm[0]}. Refusing to emit either.")
    n12 = data["tables"]["Table 12b"]["n_rows"]
    e12 = data["tables"]["Table 12b"]["n_exceeds"]
    n13 = data["tables"]["Table 13b"]["n_rows"]
    e13 = data["tables"]["Table 13b"]["n_exceeds"]

    tex = r"""
\section{The Graph Arm Under the Strict Contract}
\label{app:strict}

The construction of Appendix~\ref{app:gnn} violates the time contract of
Section~\ref{sec:data} in two ways that Section~\ref{sec:limits} states. This
appendix reruns the whole arm with both repaired, and reports the attribution of
the resulting change. Tables~\ref{tab:full20} and~\ref{tab:fullgrid} are retained
unchanged as the pre-specified audit trail.

Two things change and nothing else. Advisor nodes are keyed by
$(\mathrm{advisor}, t_0)$ rather than by advisor, so each student reads advisor
features measured at that student's own freeze date instead of at a donor's;
sibling reachability is restricted to prior cohorts by an explicit
\path{student--sibling--student} relation, which is the rule the neighbour-feature
ceiling already obeys. The training loss's class weight is computed from the
training split alone. The grid, the selection seeds, the ten evaluation seeds and
the budgets are those of Appendix~\ref{app:gnn}.

A blocking check runs before any graph model trains: M5 and M5$'$ are recomputed
in the rerun harness and compared against the frozen values of
Table~\ref{tab:ladder} and Appendix~\ref{app:full20}. All five disciplines
reproduce, with a worst per-seed deviation of $4.4\times10^{-5}$. Every cell
below persists its per-seed test scores and labels, so any verdict here can be
re-tested against a different ceiling or a different uncertainty model without
retraining.

All thirty-nine cells live under
\protect\path{results/revision/T2_1_strict_contract/T2_1_final/T2_1_strict_contract/},
which is where the Colab download archive unpacks; the repeated directory name
is the archive's own nesting and not a typographical error. Paths quoted below
are relative to it.

\begin{table*}[t]
\caption{\textbf{The pre-specified protocol under the strict
contract.} __N12__ cells. $\Delta$ is against the validation-symmetric ceiling
M5$'$; the interval is the paired student-level bootstrap of
eq.~(\ref{eq:gates}) and the verdict applies all three of its gates. Compare
Table~\ref{tab:full20}, which is the same protocol on the unrepaired
construction. __E12__ cells exceed. One cell directory each, under the tree
named above.}
\label{tab:full20b}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llllllr@{\hspace{4pt}}l}
\toprule
discipline & architecture & config & $\Delta$ vs M5$'$ & 95\% CI (student) &
$p_{\mathrm{BH}}$ & \multicolumn{2}{l}{verdict} \\
\midrule
__ROWS12__
\bottomrule
\end{tabular}
\end{table*}

\begin{table*}[t]
\caption{\textbf{The symmetric protocol under the strict contract.}
__N13__ cells, not twenty: chemistry's relational cell has no separate
legacy-configuration row because the strict grid selected the same configuration
the legacy grid had chosen, so the two coincide and one reading covers both.
Columns as in Table~\ref{tab:full20b}. __E13__ cells exceed, all in chemistry.
One cell directory each, under the tree named above.}
\label{tab:fullgridb}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llllllr@{\hspace{4pt}}l}
\toprule
discipline & architecture & config & $\Delta$ vs M5$'$ & 95\% CI (student) &
$p_{\mathrm{BH}}$ & \multicolumn{2}{l}{verdict} \\
\midrule
__ROWS13__
\bottomrule
\end{tabular}
\end{table*}

\paragraph{Which repair moved the number, and by how much.}
The strict reading is larger than the pre-specified one, and three causes are
confounded in that comparison: the advisor-keying repair, the class-weight
repair, and the fact that the two runs used different library and hardware
stacks. Five cells at the shared winning configuration, all ten seeds, all
computed in one session so that no cross-session drift enters, separate them.
Every cell records the session it ran in and the five are checked to agree
before these differences are reported, because the quantities being compared
are the same size as the drift between sessions.
Cell~B is the anchor, because the published symmetric run used a train-only class
weight and so corresponds to the legacy construction with that weight, not to the
global-weight cell; its mean of __B__ against the ten-seed mean of __R3__ that
the published run recorded puts the environment offset at __OFF__. In magnitude
that is __OFFABS__ against the run-to-run floor of __FLOOR__ reported below, so
the offset sits slightly above the floor rather than inside it. The two stacks
therefore differ by an amount that is resolvable against run-to-run noise, if
only just, and the anchor reproduces the published cell to a little worse than
the precision repeated runs on one stack give.

\begin{table}[t]
\caption{\textbf{Attribution of the strict-contract change.} Mean test AUC-PR
over ten seeds, chemistry, relational network, at the configuration both grids
selected. Cells A and B use the unrepaired construction deliberately, for
attribution only; every file they write records that and is marked as not a
verdict cell. From
\protect\path{chemistry/attribution/} under the tree named above.}
\label{tab:attribution}
\small
\begin{tabular}{llr}
\toprule
cell & construction and class weight & mean AUC-PR \\
\midrule
A & unrepaired, weight over all splits & __MA__ \\
B & unrepaired, train-only weight (anchor) & __MB__ \\
C & repaired, weight over all splits & __MC__ \\
E & advisor keying repaired only & __ME__ \\
F & sibling masking repaired only & __MF__ \\
\bottomrule
\end{tabular}
\end{table}

The class weight contributes nothing: $B-A$ is __D_F2__. The advisor-keying
repair carries the whole effect, $E-B$ of __D_F1A__, and the sibling masking
contributes nothing on its own, $F-B$ of __D_F1B__. The two halves are additive:
they sum to __D_SUM__ against a $C-B$ of __D_F1__, a gap of __D_GAP__ that sits inside
the run-to-run floor, so they are separable rather than interacting. An earlier
reading of these cells put the interaction at 0.0027, but that comparison mixed
cells computed in different sessions and the apparent term did not survive
recomputation in one.

Two caveats bound how precisely these numbers should be read. First, run-to-run
variation on identical hardware, library versions and seeds is __FLOOR__, the
widest drift across __NCELLS__ cells each run __NREPEATS__, so the
advisor-keying effect is a range of roughly $+$0.009 to $+$0.011 rather than a
point estimate. That measurement is
\protect\path{chemistry/DETERMINISM_MEASURED.json}, carried both in the tree
named above and beside the local gate outputs at
\protect\path{results/revision/T2_1_strict_contract/chemistry/}; the two copies
are byte-identical and this appendix refuses to build if they diverge. Second, cell~E
reinstates legacy sibling reachability with explicit one-hop sibling edges,
because keying advisor nodes by cohort splits the node that previously supplied
two-hop reachability. Explicit one-hop edges and two-hop paths through a shared
advisor are not equivalent message passing, so cell~E isolates the repaired
advisor features together with an approximation of legacy reachability rather
than legacy reachability itself.
"""
    m = A["cell_mean_auc_pr"]
    subs = {
        "__ROWS12__": rows_for("Table 12b", data),
        "__ROWS13__": rows_for("Table 13b", data),
        "__N12__": str(n12), "__E12__": str(e12),
        "__N13__": str(n13), "__E13__": str(e13),
        "__MA__": f"{m['A']:.4f}", "__MB__": f"{m['B']:.4f}",
        "__MC__": f"{m['C']:.4f}", "__ME__": f"{m['E']:.4f}",
        "__MF__": f"{m['F']:.4f}",
        "__B__": f"{m['B']:.4f}",
        "__R3__": f"{A['r3_legacy_mean_auc_pr']:.4f}",
        "__OFF__": num(A["environment_offset_B_minus_r3"]),
        "__OFFABS__": f"{abs(A['environment_offset_B_minus_r3']):.4f}",
        "__FLOOR__": f"{floor:.4f}",
        "__NCELLS__": {5: "five"}.get(int(DET["n_cells_repeated"]),
                                      str(DET["n_cells_repeated"])),
        "__NREPEATS__": {2: "twice", 3: "three times"}.get(
            int(DET["n_repeats_per_cell"]), f"{DET['n_repeats_per_cell']} times"),
        "__D_F2__": num(A["F2_class_weight_B_minus_A"]),
        "__D_F1A__": num(A["F1a_advisor_keying_E_minus_B"]),
        "__D_F1B__": num(A["F1b_sibling_masking_F_minus_B"]),
        "__D_F1__": num(A["F1_both_C_minus_B"]),
        "__D_SUM__": num(A["additivity_sum"]),
        "__D_GAP__": f"{A['additivity_gap']:.4f}",
    }
    for k, v in subs.items():
        tex = tex.replace(k, v)
    left = [k for k in subs if k in tex]
    if left:
        raise SystemExit(f"r29: placeholders survived substitution: {left}")
    OUT.write_text(tex, encoding="utf-8", newline="\n")
    print(f"  wrote {OUT}")
    print(f"  Table 12b: {n12} rows, {e12} exceed")
    print(f"  Table 13b: {n13} rows, {e13} exceed")
    print(f"  attribution: session {A['session_id']}, offset "
          f"{A['environment_offset_B_minus_r3']:+.4f}, additivity gap "
          f"{A['additivity_gap']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
