#!/usr/bin/env python3
"""Assemble the five-discipline coauthorship axis: OpenAlex early-window rate
(paper metric, authoritative) beside the AFT ever-coauthored proxy (ranking only,
coverage-confounded). Reads each discipline's coauth_early_rate from its frozen
dataset summary; falls back to 'PENDING' when a build has not finished. Writes
coauthorship_axis.json. Read-only.
"""
import json, os

AFT = json.load(open("coauthorship_by_tree.json"))   # ever-coauth proxy per tree

# Primary axis: all five disciplines resolved by the SAME pub-ID anchoring on the
# same 2026 snapshot (one identical protocol, econ read from econ).
SUMMARIES = {
    "econ": "artifact/code/data_econ/dataset_summary_econ.json",
    "math":    "artifact/code/data_math/dataset_summary_math.json",
    "neuro":   "Neuro Data/build_neuro/dataset_summary.json",
    "physics": "artifact/code/data_physics/dataset_summary_physics.json",
    "chemistry": "artifact/code/data_chemistry/dataset_summary_chemistry.json",
}

# map primary discipline -> AFT tree token for the proxy column
AFT_TREE = {"econ": "econ", "math": "math", "neuro": "neuro",
            "physics": "physics", "chemistry": "chemistry"}


def row(disc, path, aft_key):
    s = json.load(open(path)) if os.path.exists(path) else {}
    return {
        "early_window_coauth_openalex": s.get("coauth_early_rate", "PENDING"),
        "aft_ever_coauth_proxy": AFT.get(aft_key, {}).get("rate"),
        "base_rate": s.get("label_base_rate", "PENDING"),
        "n_students": s.get("n_students", "PENDING"),
    }


axis = {d: row(d, p, AFT_TREE.get(d, d)) for d, p in SUMMARIES.items()}
known = {d: v["early_window_coauth_openalex"] for d, v in axis.items()
         if isinstance(v["early_window_coauth_openalex"], (int, float))}
out = {
    "note": ("early_window_coauth_openalex is the reported axis. All five disciplines share "
             "one resolver (pub-ID anchoring) on one 2026 snapshot, so the column is internally "
             "comparable. That resolver resolves a student only through indexed publications, "
             "and students who publish co-author with their advisor more often, so each rate "
             "sits above the rate in the full cohort. The bias runs one way in every field, so "
             "the rates are read only as an ordering of the disciplines, never as field-level "
             "population statistics. aft_ever_coauth_proxy is ranking-only and additionally "
             "coverage-confounded (see coauth_coverage_check)."),
    "axis": axis,
    "early_window_known": known,
    "early_window_range_ratio": (round(max(known.values()) / min(known.values()), 2)
                                 if len(known) >= 2 else None),
}
json.dump(out, open("coauthorship_axis.json", "w"), indent=2)
print(json.dumps(out, indent=2))
print("wrote coauthorship_axis.json")
