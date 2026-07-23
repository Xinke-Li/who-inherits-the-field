#!/usr/bin/env python3
"""Concept-graph density per discipline, for the paper's data-characteristics
table. A pure data statistic (no model, no training): the mean degree of a
concept node in the e12 heterogeneous graph, computed exactly as
e2_hgt.build_graph defines the two concept edge groups. Student-side edges are
the per-row early_concepts sets; advisor-side edges are the deduplicated
(advisor, concept) pairs over adv_profile; the concept vocabulary is the union
of both. Rows with empty early_concepts are dropped, mirroring the loaders.

Output: results/concept_density.json
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]

out = {}
for f in FIELDS:
    df = pd.read_parquet(ROOT / "data" / f"clean_dataset_{f}.parquet")
    df = df[df.early_concepts.apply(len) > 0]
    concepts = ({c for l in df.early_concepts for c in l}
                | {c for l in df.adv_profile for c in l})
    st_edges = sum(len(set(l)) for l in df.early_concepts)
    adv_pairs = {(a, c) for a, l in zip(df.advisor_pid, df.adv_profile) for c in l}
    out[f] = {"n_concepts": len(concepts),
              "student_concept_edges": st_edges,
              "advisor_concept_edges": len(adv_pairs),
              "mean_concept_degree": round((st_edges + len(adv_pairs)) / len(concepts), 2)}

path = ROOT / "results" / "concept_density.json"
path.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
