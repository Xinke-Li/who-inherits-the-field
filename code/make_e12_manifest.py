#!/usr/bin/env python3
"""Write the GNN split manifest for one discipline, so the graph models on GPU
train on the SAME temporal split and the SAME pre-window features as the tabular
ladder. Reads the frozen table through the DATASET override and the pipeline's own
temporal_split, so the split is identical by construction. Read-only.

  DATASET=math python make_e12_manifest.py --out "KDD New dataset 2027/colab/e12_manifest_math.json"
"""
import argparse, json, os, sys, hashlib
from pathlib import Path

sys.path.insert(0, "artifact/code/paper_pipeline")
import config as C
from utils import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    field = os.environ.get("DATASET", "econ")

    df = D.load_dataset()
    dsp = D.temporal_split(df)
    split = {}
    for name in ("train", "val", "test"):
        sub = dsp[dsp["split"] == name] if "split" in dsp.columns else None
        if sub is None:
            # temporal_split may return a dict or a column; handle the column case
            raise SystemExit("temporal_split did not add a 'split' column; adapt loader")
        split[name] = sub["student_pid"].astype(str).tolist()

    sha = hashlib.sha256(open(C.CLEAN_DATASET, "rb").read()).hexdigest()
    manifest = {
        "field": field,
        "sha256_table": sha,
        "split_quantiles": list(C.SPLIT_QUANTILES),
        "n_train": len(split["train"]), "n_val": len(split["val"]), "n_test": len(split["test"]),
        "feature_cols": C.TABULAR_FEATURES + C.BOOL_FEATURES,
        "label": C.LABEL,
        "seeds": C.SEEDS,
        "table_file": f"clean_dataset_{field}.parquet",
        "pairs_file": f"pairs_resolved_{field}.parquet",
        # F7: this list declared three relations while build_graph constructs five.
        # A reader reconstructing the graph from the manifest built the wrong graph,
        # missing the two concept relations that carry most of its information.
        # Kept in sync with e2_hgt.ABLATIONS and e2_hgt.build_graph.
        "edge_types": ["student -studies-> concept",
                       "advisor -studies-> concept",
                       "advisor -advises-> student",
                       "student -at-> institution",
                       "student -coauth-> advisor"],
        "reverse_edges": ("every relation above also carries a rev_<rel> edge, so "
                          "the graph has ten edge types and two message-passing "
                          "layers traverse them in both directions"),
        # F7, second half: feature_cols below is the tabular ladder's feature set,
        # which is not what the graph's nodes carry. State both.
        "gnn_node_features": {
            "student": ["early_prod", "early_breadth", "early_overlap"],
            "advisor": ["adv_early_prod", "adv_early_breadth", "adv_career_age_at_t0"],
            "institution": "learnable embedding, no input features",
            "concept": "learnable embedding, no input features",
        },
        "train_student_pids": split["train"],
        "val_student_pids": split["val"],
        "test_student_pids": split["test"],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(manifest, open(args.out, "w"), indent=2)
    print(f"[manifest] {field}: train {manifest['n_train']} val {manifest['n_val']} "
          f"test {manifest['n_test']} -> {args.out}")


if __name__ == "__main__":
    main()
