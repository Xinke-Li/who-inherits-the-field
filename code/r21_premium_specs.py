#!/usr/bin/env python3
"""T2.9 (audit finding W6.3): report all four innovation-premium specifications
for all five disciplines, instead of quoting specification 3 alone.

Reads the frozen e6 result files and writes a single table. Nothing is re-fitted;
the numbers already exist and were simply not all reported.

  python code/r21_premium_specs.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "T2_9_premium_specs"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
SPECS = ["1_raw", "2_controls", "3_controls_FE", "4_structural_FE"]
ALPHA = 0.05


def sign(c, p):
    if p >= ALPHA:
        return "ns"
    return "pos" if c > 0 else "neg"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, out = [], {"task": "T2.9", "finding": "W6.3", "alpha": ALPHA,
                     "outcome": "late_cite_pct_mean", "fields": {}}

    for f in FIELDS:
        src = ROOT / "results" / f"results_{f}" / "e6_innovation_premium.json"
        d = json.load(open(src))
        po = d["primary_outcome"]["late_cite_pct_mean"]
        rec = {s: {"coef": po[s]["coef"], "se": po[s]["se"], "p": po[s]["p"],
                   "n": po[s]["n"], "reading": sign(po[s]["coef"], po[s]["p"])}
               for s in SPECS}
        rec["frozen_verdict"] = d.get("verdict", "")
        rec["source_file"] = f"results/results_{f}/e6_innovation_premium.json"
        rec["spec3_vs_spec4_reading_changes"] = (
            rec["3_controls_FE"]["reading"] != rec["4_structural_FE"]["reading"])
        out["fields"][f] = rec
        rows.append((f, rec))

    changed = [f for f, r in rows if r["spec3_vs_spec4_reading_changes"]]
    out["disciplines_whose_reading_changes_between_spec3_and_spec4"] = changed
    out["n_changed"] = len(changed)
    out["regressors"] = {
        "1_raw": "divergence (concept space)",
        "2_controls": "divergence (concept space)",
        "3_controls_FE": "divergence (concept space)",
        "4_structural_FE": "n2v_divergence (structural Node2Vec)",
    }
    out["interpretation"] = (
        "Specifications 1 to 3 are a nested ladder on one regressor, "
        "concept-space divergence: raw, then controls, then cohort and "
        "institution fixed effects. Specification 4 keeps those controls and "
        "fixed effects but substitutes a different regressor, structural "
        "Node2Vec divergence (e6_innovation_premium.py:68-78). It is therefore "
        "not a stricter test of specification 3; it estimates the same "
        "specification for a different construct. A reading that differs "
        "between them is a difference between two notions of divergence, not a "
        "robustness failure of one of them. Specification 3 is the terminal "
        "rung of the ladder for the construct the paper defines, which is why "
        "it is the headline; no pre-registration document fixes that choice in "
        "advance, so the paper states it as the ladder's endpoint rather than "
        "as a pre-specified selection.")
    out["summary"] = (
        f"The reported reading differs between the two constructs in "
        f"{len(changed)} of five disciplines ({', '.join(changed)}). Both are "
        "now reported.")
    json.dump(out, open(OUT / "premium_specs.json", "w"), indent=2)

    hdr = f"  {'field':10} " + " ".join(f"{s:>22}" for s in SPECS)
    print(hdr)
    for f, r in rows:
        cells = " ".join(f"{r[s]['coef']:+.4f} p{r[s]['p']:<6.4g}{r[s]['reading']:>4}"
                         for s in SPECS)
        print(f"  {f:10} {cells}")
    print(f"\n[T2.9] reading changes between spec 3 and spec 4 in "
          f"{len(changed)}/5: {', '.join(changed)}")
    print(f"[T2.9] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
