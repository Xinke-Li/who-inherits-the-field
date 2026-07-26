#!/usr/bin/env python3
"""T2.12 (audit finding F7): regenerate the e12 split manifests so they declare
the graph that build_graph actually constructs, and verify by rebuilding.

The frozen manifests under results/results_<field>/ declare three edge types
while e2_hgt.build_graph constructs five, omitting both concept relations, which
carry most of the graph's information. Anyone reconstructing the graph from the
manifest built the wrong graph.

Nothing under results/results_*/ is modified. Regenerated manifests go to
results/revision/T2_12_manifest/ and the frozen ones stay as the audit trail.
Verification does not compare strings against the source: the graph is rebuilt on
CPU and its edge types are read off the built object, then diffed against what
the regenerated manifest declares.

One subprocess per discipline, because config resolves the dataset at import.

  python code/r18_verify_manifest.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "T2_12_manifest"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]

FIVE = [("student", "studies", "concept"),
        ("advisor", "studies", "concept"),
        ("advisor", "advises", "student"),
        ("student", "at", "institution"),
        ("student", "coauth", "advisor")]


def run_one(field):
    """Rebuild the graph for one discipline and emit its manifest + check."""
    sys.path.insert(0, str(ROOT / "code" / "paper_pipeline"))
    sys.path.insert(0, str(ROOT / "code" / "paper_pipeline" / "experiments"))
    from utils import data as D
    import e2_hgt as E2

    # build_graph reads df.split, which temporal_split adds; same call order as
    # the frozen manifest writer, so the split is identical by construction.
    df = D.temporal_split(D.load_dataset())
    data = E2.build_graph(df, "none")

    built = sorted(t for t in data.edge_types if not t[1].startswith("rev_"))
    reverses = sorted(t for t in data.edge_types if t[1].startswith("rev_"))
    decl = sorted(FIVE)
    ok = built == decl

    frozen = ROOT / "results" / f"results_{field}" / f"e12_manifest_{field}.json"
    old = json.load(open(frozen)) if frozen.exists() else {}

    man = dict(old)
    man["edge_types"] = [f"{s} -{r}-> {d}" for s, r, d in decl]
    man["reverse_edges"] = ("every relation above also carries a rev_<rel> edge, "
                            "so the graph has ten edge types")
    man["gnn_node_features"] = {
        "student": list(E2.TABULAR_ST),
        "advisor": list(E2.TABULAR_ADV),
        "institution": "learnable embedding, no input features",
        "concept": "learnable embedding, no input features",
    }
    man["regenerated_by"] = "code/r18_verify_manifest.py (T2.12, finding F7)"
    man["supersedes"] = (f"results/results_{field}/e12_manifest_{field}.json, "
                         "which declared three edge types; retained unchanged")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(man, open(OUT / f"e12_manifest_{field}.json", "w"), indent=2)

    check = {
        "field": field,
        "built_forward_edge_types": [list(t) for t in built],
        "declared_in_regenerated_manifest": [list(t) for t in decl],
        "match": ok,
        "n_forward": len(built), "n_reverse": len(reverses),
        "n_total": len(data.edge_types),
        "frozen_manifest_declared": old.get("edge_types"),
        "frozen_manifest_n": len(old.get("edge_types", [])),
        "n_students": int(data["student"].num_nodes),
    }
    json.dump(check, open(OUT / f"_check_{field}.json", "w"), indent=2)
    print(f"[{'OK' if ok else 'MISMATCH'}] {field}: built {len(built)} forward + "
          f"{len(reverses)} reverse = {len(data.edge_types)}; frozen manifest "
          f"declared {len(old.get('edge_types', []))}")
    return 0 if ok else 1


def main():
    if os.environ.get("R18_FIELD"):
        return run_one(os.environ["R18_FIELD"])

    OUT.mkdir(parents=True, exist_ok=True)
    rc = 0
    for f in FIELDS:
        env = dict(os.environ)
        env["R18_FIELD"] = f
        env["DATASET"] = f
        env["DATASET_PATH"] = str(ROOT / "data" / f"clean_dataset_{f}.parquet")
        env.pop("NEURO_DATASET", None)
        rc |= subprocess.call([sys.executable, str(Path(__file__).resolve())], env=env)

    checks = [json.load(open(OUT / f"_check_{f}.json")) for f in FIELDS
              if (OUT / f"_check_{f}.json").exists()]
    report = {"task": "T2.12", "finding": "F7",
              "claim": ("the released manifest declares every edge type that "
                        "build_graph constructs"),
              "verification": ("graph rebuilt on CPU per discipline; edge types "
                               "read off the built object and diffed against the "
                               "regenerated manifest"),
              "all_fields_match": bool(checks) and all(c["match"] for c in checks),
              "fields": {c["field"]: c for c in checks}}
    json.dump(report, open(OUT / "verification.json", "w"), indent=2)
    for f in FIELDS:
        p = OUT / f"_check_{f}.json"
        if p.exists():
            p.unlink()
    print(f"\n[T2.12] all five fields match: {report['all_fields_match']}")
    print(f"[T2.12] -> {OUT}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
