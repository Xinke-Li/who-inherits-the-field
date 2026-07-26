#!/usr/bin/env python3
"""T2.1 attribution: split the strict-contract increase into its three causes.

Chemistry RGCN's crossing rose from a legacy +0.035 (tuned symmetric) to a strict
+0.0448. Three causes are confounded in that comparison:

  (1) F1 repaired   advisor nodes keyed by (advisor_pid, t0), prior-cohort
                    siblings only
  (2) F2 repaired   class weight from the training split alone (chemistry:
                    3.5832 global -> 3.7390 train-only; a higher positive-class
                    weight can raise AUC-PR by itself)
  (3) environment   the legacy number was produced on a different torch / PyG /
                    GPU stack

A 2x2 on ONE stack separates them, at the shared winning configuration, tuned
protocol, ten seeds each:

                 global class weight        train-only class weight
  legacy         cell A (attribution anchor) cell B
  strict         cell C                      cell D  (already run, the verdict)

CELL B IS THE ANCHOR, not cell A: rgcn_symmetric_verdict.json records that r3
used a train-only class weight, so r3's legacy mean of 0.5724 corresponds to
legacy construction WITH a train-only weight. B minus r3 is the environment
offset; B minus A and D minus C isolate the class weight (F2); D minus B is the
construction effect (F1). Cells E and F then split F1 into its two halves, F1a
(advisor keying) and F1b (sibling masking).

CELL A DELIBERATELY USES THE UNFIXED e2_hgt.build_graph. That is the one
sanctioned use of the legacy path and every output file says so, so that no
reader can mistake an attribution cell for a verdict.

Outputs go to results/revision/T2_1_strict_contract/<field>/attribution/, which
the verdict aggregator does not read.

  DATASET=chemistry python code/r27_attribution.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline"))
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline" / "experiments"))

FIELD = os.environ.get("DATASET", "chemistry")
OUT = ROOT / "results" / "revision" / "T2_1_strict_contract" / FIELD / "attribution"
SEEDS = list(range(10))

# the configuration both grids selected for chemistry RGCN
SHARED_HP = {"lr": 0.005, "hidden": 64, "layers": 3, "dropout": 0.5}
FORCE = False
R3_LEGACY_MEAN = 0.5724   # rgcn_symmetric_verdict.json, train-only weight

CELLS = {
    "A_legacy_globalw": {
        "construction": "legacy", "class_weight": "all",
        "role": ("legacy construction with a GLOBAL class weight. NOT the "
                 "anchor: r3 used a train-only weight, so cell B is the anchor. "
                 "This cell exists to isolate the class-weight effect."),
        "WARNING": ("uses the UNFIXED e2_hgt.build_graph (audit finding F1) on "
                    "purpose, for attribution only. This is NOT a verdict cell "
                    "and must never be read as one."),
    },
    "B_legacy_trainw": {
        "construction": "legacy", "class_weight": "train",
        "role": ("THE ANCHOR. r3's legacy run used a train-only class weight "
                 "(see rgcn_symmetric_verdict.json), so this cell, not cell A, "
                 "is what reproduces r3's 0.5724 on this stack. The difference "
                 "is the environment offset."),
        "WARNING": ("uses the UNFIXED e2_hgt.build_graph for attribution only. "
                    "NOT a verdict cell."),
    },
    "E_f1a_only_trainw": {
        "construction": "f1a", "class_weight": "train",
        "role": ("F1a alone: advisor nodes keyed by (advisor_pid, t0) so each "
                 "student gets correctly-timed advisor features, with sibling "
                 "reachability restored to all cohorts."),
        "APPROXIMATION": ("This is NOT exact. In the legacy graph, sibling "
                 "reachability is two-hop through a shared advisor node. Keying "
                 "advisors by (advisor_pid, t0) splits that node, so legacy "
                 "reachability is reinstated here with 133,492 explicit one-hop "
                 "sibling edges. Explicit one-hop edges and two-hop-through-a-"
                 "shared-advisor are not equivalent message passing: different "
                 "relation type, hop count and normalisation. This cell "
                 "therefore isolates repaired advisor features plus an "
                 "APPROXIMATION of legacy reachability, not legacy reachability "
                 "itself, so the F1a/F1b split is informative but not an exact "
                 "decomposition."),
        "WARNING": "attribution cell, not a verdict cell.",
    },
    "F_f1b_only_trainw": {
        "construction": "f1b", "class_weight": "train",
        "role": ("F1b alone: sibling reachability restricted to prior cohorts, "
                 "with legacy .first() advisor features retained."),
        "WARNING": "attribution cell, not a verdict cell.",
    },
    "C_strict_globalw": {
        "construction": "strict", "class_weight": "all",
        "role": "strict construction, F2 not repaired: isolates the contract effect",
        "WARNING": "attribution cell, not a verdict cell.",
    },
}


def verify(out_dir=None):
    """Check the property the attribution arithmetic depends on, and emit it.

    The five cells are only comparable if they were computed in one process:
    cross-repeat drift is 0.0013 while the additivity gap being tested is
    0.0008, so a mixed set would decide the question by accident. Every cell
    records a session_id for exactly this reason, and until this function
    existed nothing read it. A field nobody reads is not a guard.

    Runs on CPU with no torch, so it can be applied to a downloaded tree.
    Returns 0 only if all five cells are present and share one session.
    """
    d = Path(out_dir) if out_dir else OUT
    means, sessions, missing = {}, {}, []
    for name in CELLS:
        p = d / name / "summary.json"
        if not p.exists():
            missing.append(name)
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        means[name[0]] = j["mean_auc_pr"]
        sessions[name] = (j.get("session") or {}).get("session_id")
    if missing:
        print(f"[attr:verify] FAIL: cells absent from {d}: {missing}")
        return 1

    ids = set(sessions.values())
    same = len(ids) == 1 and None not in ids
    for name, sid in sessions.items():
        print(f"[attr:verify] {name:22} session {sid}")
    if not same:
        print(f"[attr:verify] FAIL: the five cells do not share one session "
              f"({sorted(str(i) for i in ids)}). The additivity test compares "
              f"differences near 0.001 against a cross-session drift of the "
              f"same size, so this set cannot answer it. Re-run with --force.")
        return 1

    A, B, C, E, F = (means[k] for k in "ABCEF")
    rep = {
        "field": FIELD, "session_id": ids.pop(), "same_session": True,
        "cell_mean_auc_pr": {k: means[k] for k in "ABCEF"},
        "F2_class_weight_B_minus_A": round(B - A, 4),
        "F1a_advisor_keying_E_minus_B": round(E - B, 4),
        "F1b_sibling_masking_F_minus_B": round(F - B, 4),
        "F1_both_C_minus_B": round(C - B, 4),
        "additivity_sum": round((E - B) + (F - B), 4),
        "additivity_gap": round(abs((E - B) + (F - B) - (C - B)), 4),
        "determinism_floor": 0.0013,
    }

    # The environment offset needs r3's published legacy run, which lives under
    # results/robustness/ and is not packed into the Colab bundle. Absent, the
    # field is null and names what it looked for; it gates nothing.
    r3p = ROOT / "results" / "robustness" / "rgcn_symmetric"
    seeds = sorted(r3p.glob("rgcn_sym_seed[0-9].json"))
    if len(seeds) == 10:
        r3_mean = float(np.mean([json.loads(p.read_text())["auc_pr"]
                                 for p in seeds]))
        rep["r3_legacy_mean_auc_pr"] = round(r3_mean, 6)
        rep["environment_offset_B_minus_r3"] = round(B - r3_mean, 4)
        rep["offset_note"] = (
            "computed from r3's ten per-seed files, not from its rounded "
            "published mean; it sits at the determinism floor rather than "
            "inside it")
    else:
        rep["r3_legacy_mean_auc_pr"] = None
        rep["environment_offset_B_minus_r3"] = None
        rep["offset_note"] = f"r3 per-seed files not present under {r3p}"

    (d / "attribution.json").write_text(json.dumps(rep, indent=2))
    print(f"[attr:verify] one session, five cells")
    print(f"[attr:verify] F2  B-A = {rep['F2_class_weight_B_minus_A']:+.4f}")
    print(f"[attr:verify] F1a E-B = {rep['F1a_advisor_keying_E_minus_B']:+.4f}")
    print(f"[attr:verify] F1b F-B = {rep['F1b_sibling_masking_F_minus_B']:+.4f}")
    print(f"[attr:verify] F1  C-B = {rep['F1_both_C_minus_B']:+.4f}  "
          f"additivity gap {rep['additivity_gap']:.4f} against a floor of "
          f"{rep['determinism_floor']}")
    off = rep["environment_offset_B_minus_r3"]
    if off is None:
        print("[attr:verify] environment offset: r3 per-seed files absent here")
    else:
        print(f"[attr:verify] environment offset B - r3 = {off:+.4f}")
    print(f"[attr:verify] -> {d / 'attribution.json'}")
    return 0


def main():
    import argparse
    import platform
    import uuid
    import e2_hgt as E2
    import r3_rgcn_symmetric as R3
    import torch
    from utils import data as D

    global FORCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="recompute every cell in ONE session. The additivity "
                         "test compares differences of ~0.003 against a "
                         "cross-repeat drift of ~0.0013, so cells restored from "
                         "an earlier session cannot be compared with fresh ones.")
    FORCE = ap.parse_args().force

    # One id for every cell written by this process, so no later reader can mix
    # runs again without it being visible in the JSON.
    SESSION = uuid.uuid4().hex[:12]
    SESSION_ENV = {
        "session_id": SESSION,
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None),
        "torch": torch.__version__, "cuda": getattr(torch.version, "cuda", None),
        "platform": platform.platform(), "forced_recompute": FORCE,
    }
    globals()["SESSION_ENV"] = SESSION_ENV
    print(f"[attr] session {SESSION} force={FORCE} gpu={SESSION_ENV['gpu']}")

    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = D.temporal_split(D.load_dataset())

    # budgets: the tuned protocol, differing only in class-weight scope
    R3.BUDGETS["attr_trainw"] = {"epochs": 200, "patience": 15,
                                 "weight_decay": 1e-4,
                                 "class_weight_scope": "train"}
    R3.BUDGETS["attr_globalw"] = {"epochs": 200, "patience": 15,
                                  "weight_decay": 1e-4,
                                  "class_weight_scope": "all"}

    graphs = {}
    for name, spec in CELLS.items():
        cdir = OUT / name
        cdir.mkdir(parents=True, exist_ok=True)
        if (cdir / "summary.json").exists() and not FORCE:
            print(f"[attr] {name}: already complete, skipping")
            continue
        if FORCE:
            # Cross-session drift confounds the additivity test: B drifts 0.0004
            # and C drifts 0.0013 between repeats, while the additivity gap is
            # only 0.0027. Restoring some cells and computing others mixes GPU
            # sessions, so --force recomputes all five together.
            for stale in cdir.glob("*.json"):
                stale.unlink()

        con = spec["construction"]
        if con not in graphs:
            graphs[con] = (E2.build_graph(df, "none") if con == "legacy"
                           else E2.build_graph_v2(df, "none", contract=con))
            print(f"[attr] built {con} graph: {len(graphs[con].edge_types)} "
                  f"edge types, {graphs[con]['advisor'].num_nodes} advisor nodes")
        data = graphs[con]
        budget = ("attr_trainw" if spec["class_weight"] == "train"
                  else "attr_globalw")

        aps = []
        for s in SEEDS:
            f = cdir / f"seed{s}.json"
            if f.exists():
                aps.append(json.loads(f.read_text())["test_auc_pr"])
                continue
            t = time.time()
            r = R3.train_eval(data, s, dev, SHARED_HP, budget, arch="rgcn")
            assert r["model_class"] == "RGCNSym", r["model_class"]
            f.write_text(json.dumps({
                "cell": name, "seed": s, "hp": SHARED_HP,
                "construction": con, "class_weight_scope": spec["class_weight"],
                "budget": budget, "model_class": r["model_class"],
                "test_auc_pr": r["auc_pr"], "test_auc_roc": r["auc_roc"],
                "test_scores": r["test_scores"], "test_labels": r["test_labels"],
                "seconds": round(time.time() - t, 1),
                "ROLE": spec["role"], "WARNING": spec["WARNING"],
                "session": SESSION_ENV,
            }, indent=2))
            aps.append(r["auc_pr"])
            print(f"[attr] {name} seed {s}: AUC-PR {r['auc_pr']:.4f} "
                  f"({time.time()-t:.0f}s)", flush=True)

        (cdir / "summary.json").write_text(json.dumps({
            "cell": name, "field": FIELD, "hp": SHARED_HP,
            "construction": con, "class_weight_scope": spec["class_weight"],
            "mean_auc_pr": round(float(np.mean(aps)), 4),
            "std_auc_pr": round(float(np.std(aps)), 4),
            "per_seed_auc_pr": [round(v, 4) for v in aps],
            "ROLE": spec["role"], "WARNING": spec["WARNING"],
            "session": SESSION_ENV,
            "IS_VERDICT_CELL": False,
        }, indent=2))
        print(f"[attr] {name}: mean AUC-PR {np.mean(aps):.4f}")

    print("\n[attr] cells written to", OUT)
    print("[attr] cell D (strict + train-only) is the verdict run; read it from "
          "../rgcn_tuned_strict/summary.json")
    return verify()


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(add_help=False)
    _p.add_argument("--verify-only", action="store_true")
    _p.add_argument("--dir", default=None,
                    help="verify a downloaded attribution tree instead of OUT")
    _known, _ = _p.parse_known_args()
    if _known.verify_only:
        raise SystemExit(verify(_known.dir))
    raise SystemExit(main())
