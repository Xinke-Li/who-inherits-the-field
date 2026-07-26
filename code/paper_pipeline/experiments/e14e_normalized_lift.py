"""E14e - branch geometry of the E14 decision rule, plus normalized lift.

POST-HOC AND DESCRIPTIVE. Nothing here is pre-registered, and nothing here
changes a verdict. The E14 decision rule (e14_self_persistence.py) stands exactly
as written and its tags are reported as it produced them; this script only adds
the quantities needed to read those tags honestly.

WHY IT EXISTS (the part the paper uses). The rule's two substantive branches are
fixed 0.05 margins around two reference points: Branch A accepts
`best <= band + 0.05`, opening upward from the E10 placebo band; Branch B accepts
`best >= M1 - 0.05`, opening downward from M1. Their acceptance regions therefore
INTERSECT exactly when the references lie closer than twice the margin. In
neuroscience band=0.298 and M1=0.387 are 0.089 apart (< 0.10), so [0.337, 0.348]
satisfies both branches - and the neuroscience best rung, 0.344, lands inside it.
(These are neuroscience acceptance-interval values; the 0.348 here is that interval's
upper bound, a neuroscience CI bound, not an economics co-authorship figure.)
The script returns A only because it evaluates A before B; B first would have
returned the opposite reading. In economics the references are 0.165 apart and
the regions are disjoint, so its tag is unambiguous. This is a property of the
rule's construction, checkable from the reference points alone before any data
exist, and it is what the paper reports.

SECONDARY (computed, not used in the paper's prose). The normalized shares below
express best and M1 as fractions of the M1-over-base range. They are a
descriptive ratio of two AUC-PR gaps, NOT a decomposition: AUC-PR is a rank-based
summary and its differences do not partition into shares the way variance does.
They are retained because they answer a reviewer who asks whether the tags track
signal strength (they do not: the discipline tagged advisor-information-required
has the LARGER student_recovered_share), and because the pre-registered rule's own
`best <= band + 0.05` already treats an AUC-PR gap as an absolute quantity - the
ratio introduces no assumption the rule had not already made. Kept out of the
prose deliberately; the branch-geometry argument above needs no such caveat.

READ-ONLY. Consumes results_{econ,neuro}/e14_self_persistence.json. Trains
nothing, touches no frozen table, draws no randomness: a deterministic
recomputation over existing result files. Re-running it cannot change any number
in the paper unless an upstream result file itself changed.

Definitions (all from a_student_only_ladder):
    base    summary.M0_prior.auc_pr.mean               (test-split prior)
    M1      summary.M1_ref_logit_overlap.auc_pr.mean   (advisor-informed reference)
    best    max over the four student-only rungs of auc_pr.mean
    band    frozen_refs.E10_placebo_L1_band_max        (E10 placebo ceiling)

    signal_range             = M1 - base
    gap_M1_minus_best        = M1 - best                (absolute, not comparable
                                                         across disciplines - the
                                                         reason the shares exist)
    student_recovered_share  = (best - base) / signal_range
    advisor_residual_share   = (M1 - best) / signal_range
    margin_as_share_of_range = 0.05 / signal_range
    branch_A_slack           = (band + 0.05) - best     (>0 passes, <0 fails)

Guard: branch_A_slack's sign must agree with the tag the upstream json recorded
(econ negative -> C_INTERMEDIATE, neuro positive -> A_...). If an upstream file is
re-run and its verdict moves, this assert fires rather than letting the paper's
prose drift away from its result files.

Output: results_{econ,neuro}/e14e_normalized_lift.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

MARGIN = 0.05          # Branch A's pre-registered absolute margin
RUNGS = ("M1s_typicality", "M1s_scalars", "M4s_tfidf", "M3s_gbdt")


def analyse(path):
    a = json.loads(Path(path).read_text())["a_student_only_ladder"]
    s, fr, v = a["summary"], a["frozen_refs"], a["verdict"]

    base = s["M0_prior"]["auc_pr"]["mean"]
    m1 = s["M1_ref_logit_overlap"]["auc_pr"]["mean"]
    band = fr["E10_placebo_L1_band_max"]
    best_name = max(RUNGS, key=lambda n: s[n]["auc_pr"]["mean"])
    best = s[best_name]["auc_pr"]["mean"]

    rng = m1 - base
    assert rng > 0, f"non-positive signal range in {path}"

    slack = (band + MARGIN) - best

    # Branch geometry: the two acceptance regions and whether they intersect.
    a_hi = band + MARGIN          # Branch A accepts best <= a_hi
    b_lo = m1 - MARGIN            # Branch B accepts best >= b_lo
    refs_gap = m1 - band          # separation of the two reference points
    overlaps = a_hi >= b_lo       # equivalently: refs_gap < 2 * MARGIN

    out = {
        "source": Path(path).name,
        "note": "post-hoc descriptive; not part of the pre-registered rule",
        "inputs": {"base_rate": round(base, 4), "placebo_band": round(band, 4),
                   "best_student_rung": best_name, "best_student_auc_pr": round(best, 4),
                   "M1_auc_pr": round(m1, 4), "margin": MARGIN},
        "branch_geometry": {
            "branch_A_accepts_at_or_below": round(a_hi, 4),
            "branch_B_accepts_at_or_above": round(b_lo, 4),
            "reference_gap_M1_minus_band": round(refs_gap, 4),
            "twice_margin": round(2 * MARGIN, 4),
            "regions_overlap": bool(overlaps),
            "overlap_interval": [round(b_lo, 4), round(a_hi, 4)] if overlaps else None,
            "best_satisfies_A": bool(best <= a_hi),
            "best_satisfies_B": bool(best >= b_lo),
            "tag_decided_by_evaluation_order": bool(best <= a_hi and best >= b_lo),
        },
        # 5 dp, not 4: neuro's range is 0.12254, and storing it as 0.1225 would
        # re-round to 0.122 while the true value gives 0.123 - a double-rounding
        # artifact that would put the json and the paper's prose one ulp apart.
        "signal_range": round(rng, 5),
        "gap_M1_minus_best": round(m1 - best, 5),
        "student_recovered_share": round((best - base) / rng, 4),
        "advisor_residual_share": round((m1 - best) / rng, 4),
        "margin_as_share_of_range": round(MARGIN / rng, 4),
        "branch_A_slack": round(slack, 4),
        "branch_A_satisfied": bool(slack >= 0),
        "upstream_verdict_branch": v["branch"],
    }

    # The tag the upstream json recorded must agree with the slack we recompute.
    tagged_A = v["branch"].startswith("A_")
    assert tagged_A == out["branch_A_satisfied"], (
        f"{path}: upstream verdict {v['branch']} disagrees with recomputed "
        f"branch_A_slack {slack:+.4f} - the result file moved under the paper")
    return out


def main():
    econ = C.REPO_ROOT / "results_econ" / "e14_self_persistence.json"
    neuro = C.REPO_ROOT / "results_neuro" / "e14_self_persistence.json"
    for src, dest in ((econ, C.REPO_ROOT / "results_econ" / "e14e_normalized_lift.json"),
                      (neuro, C.REPO_ROOT / "results_neuro" / "e14e_normalized_lift.json")):
        assert src.exists(), f"missing {src}"
        r = analyse(src)
        dest.write_text(json.dumps(r, indent=2))
        pct = lambda x: f"{x * 100:.1f}%"
        g = r["branch_geometry"]
        print(f"--- {dest.parent.name}")
        print(f"    base {r['inputs']['base_rate']}  band {r['inputs']['placebo_band']}  "
              f"best {r['inputs']['best_student_auc_pr']} ({r['inputs']['best_student_rung']})  "
              f"M1 {r['inputs']['M1_auc_pr']}")
        print(f"    BRANCH GEOMETRY")
        print(f"      A accepts best <= {g['branch_A_accepts_at_or_below']}   "
              f"B accepts best >= {g['branch_B_accepts_at_or_above']}")
        print(f"      reference gap (M1-band) {g['reference_gap_M1_minus_band']} vs "
              f"2*margin {g['twice_margin']}  -> overlap: {g['regions_overlap']}"
              f"{' ' + str(g['overlap_interval']) if g['regions_overlap'] else ''}")
        print(f"      best satisfies A={g['best_satisfies_A']} B={g['best_satisfies_B']}"
              f"  -> decided by evaluation order: {g['tag_decided_by_evaluation_order']}")
        print(f"      branch_A_slack {r['branch_A_slack']:+.4f}  tag={r['upstream_verdict_branch']}")
        print(f"    NORMALIZED (not used in prose)")
        print(f"      signal_range {r['signal_range']}  recovered "
              f"{pct(r['student_recovered_share'])}  residual "
              f"{pct(r['advisor_residual_share'])}  margin/range "
              f"{pct(r['margin_as_share_of_range'])}")
        print(f"    -> {dest}")


if __name__ == "__main__":
    main()
