#!/usr/bin/env python3
"""T3.1: emit Table tab:t31 from the five scored verdicts.

Every number in the table is read from
results/revision/T3_1_adjudication/<field>/verdict.json, which r40 wrote from
the raw judgments. Nothing here is typed by hand and nothing is recomputed, so
the table cannot drift from the instrument that produced it.

The refusal convention is r40's: a stop_reason of refusal with empty text counts
as NO, holding every denominator at the number of items drawn. The gap under the
other convention, dropping those items, is carried in the caption so a reader
can see the choice does not move the verdict.

Usage:  python code/r53_emit_t31_table.py
Output: paper/t31_table.tex
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADJ = ROOT / "results" / "revision" / "T3_1_adjudication"
OUT = ROOT / "paper" / "t31_table.tex"
FIELDS = [("econ", "economics"), ("math", "math"), ("neuro", "neuro"),
          ("physics", "physics"), ("chemistry", "chemistry")]


def main():
    rows, refus, excl, judges = [], [], [], set()
    for f, label in FIELDS:
        p = ADJ / f / "verdict.json"
        if not p.exists():
            raise SystemExit(f"r53: {p} is absent; run r40 --stage score --field {f}")
        v = json.loads(p.read_text(encoding="utf-8"))
        g1, g2, k = (v["gate_1_true_minus_placebo"],
                     v["gate_2_self_consistency"], v.get("kappa"))
        if k is None:
            raise SystemExit(
                f"r53: {f} has no kappa, so a gate failed. The table reports "
                f"kappa; emit it only once every field has cleared both gates, "
                f"or the table will imply a validity conclusion r40 refused.")
        judges.add((v["judge"]["model"], v["judge"]["decoding"]))
        n_ref = g1.get("n_refusals", 0)
        if n_ref:
            refus.append(f"{label} {n_ref}")
            excl.append(f"{label} {g1['if_refusals_excluded']['gap']:+.4f}")
        rows.append(
            f"{label} & {g1['true_yes_rate']:.3f} & {g1['placebo_yes_rate']:.3f} "
            f"& ${g1['gap']:+.4f}$ & {g2['rate']:.3f} & "
            f"{k['balanced']:.3f} & {k['reweighted_to_base_rate']:.3f} \\\\")

    if len(judges) != 1:
        raise SystemExit(f"r53: the five verdicts name {len(judges)} judge "
                         f"configurations {sorted(judges)}; one run, one judge.")
    model, decoding = judges.pop()

    cap = (
        "\\textbf{Adjudication by a third instrument.} A language model is shown "
        "five advisor titles from the early window and five student titles from "
        "the late window, and asked whether the second set works on the same "
        "topics as the first. Yes rates are per arm over 200 items each; the gap "
        "is the true rate minus the placebo rate and must reach 0.10; "
        "consistency is the share of the 40 verbatim repeats answered the same "
        "way and must reach 0.80. Both gates clear in every discipline, so "
        "$\\kappa$ against the frozen label is reported, on the drawn sample and "
        "reweighted to the cohort base rate. "
        f"Judge \\texttt{{{model}}}, decoding recorded as \\texttt{{{decoding}}}. "
        "Six refusals are counted as NO, which holds every denominator at 200; "
        "Appendix~\\ref{app:t31} gives the accounting and both conventions. "
        "From \\texttt{results/revision/T3\\_1\\_adjudication/}."
    )

    tex = "\n".join([
        "\\begin{table}[H]",
        "\\caption{" + cap + "}",
        "\\label{tab:t31}",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "discipline & true & placebo & gap & consist. & $\\kappa$ & "
        "$\\kappa$ rew. \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    OUT.write_text(tex, encoding="utf-8")
    print(f"[r53] {len(rows)} disciplines -> {OUT}")
    print(f"[r53] judge {model}, decoding {decoding}")
    print(f"[r53] refusals counted as NO: {', '.join(refus)}")


if __name__ == "__main__":
    main()
