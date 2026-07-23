# Who Inherits the Field? A Leakage-Audited Benchmark for Intellectual Inheritance in Doctoral Training

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21501632-blue.svg)](https://doi.org/10.5281/zenodo.21501632)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0001--0403--3606-a6ce39.svg)](https://orcid.org/0009-0001-0403-3606)

This repository holds the code, documentation, and reproduction materials for a
forward-prediction benchmark of intellectual inheritance in doctoral training,
spanning five scientific genealogies built from a single 2026 snapshot under one
byte-identical protocol: economics, mathematics, physics, chemistry, and
neuroscience, 330,282 raw student-advisor pairs distilled into 68,235 modeled
student-advisor pairs, one student per pair. The funnel is dominated by two
requirements: both members of a pair must resolve to OpenAlex authors, and the
label must be observable by 2026. Every modeling table is frozen
and pinned by SHA-256. The benchmark is designed as a negative control for graph
learning: a strong tabular baseline sets the ceiling, four graph architectures
are compared against it under pre-registered rules, and the evaluation itself is
audited. The accompanying paper is under review at the KDD 2027 Datasets and
Benchmarks track. Author: Xinke Li, University of Chicago
([ORCID 0009-0001-0403-3606](https://orcid.org/0009-0001-0403-3606)).

## The task

Let t0 be the year of a student's first publication. Every input feature is
measured on or before t0+5 (early concepts, early productivity and breadth, the
advisor's early profile and career age, early co-authorship), and the label is
read at t0+15: y = 1 when the Jaccard overlap between the student's late-window
concepts and the advisor's early profile exceeds 0.2. Nothing dated after t0+5
enters a feature, so the task is forward prediction under a temporal contract.
The paper defines the contract, the certificates that audit it, and the
pre-registered comparison rules.

## Repository layout

| Directory | Contents |
|---|---|
| `code/` | Dataset builders and the frozen experiment pipeline (`code/paper_pipeline/experiments/`) |
| `data/` | Attrition funnels, dataset summaries, the co-authorship axis, and `SHA256SUMS`; the parquet tables ship through Zenodo, not git |
| `colab/` | The GPU notebook and bundle for the graph-model leg (e12) |
| `datasheet/` | The datasheet for the dataset |
| `reproduction/` | An assertion-style script that recomputes the headline numbers from the frozen tables |
| `results/` | Per-discipline result files, including the per-seed graph-model artifacts and the corrected e12 aggregation |

The five frozen modeling tables (`clean_dataset_<field>.parquet`), the resolved
pairs, and the outcomes tables are archived at
https://doi.org/10.5281/zenodo.21501632. Download the archive, place the
`data/` files into this repository's `data/` directory, and verify them against
the pinned hashes before running anything.

## Quickstart

Fetch the data from Zenodo and verify the hashes:

```
(cd data && sha256sum -c SHA256SUMS)
```

Run the assertion-style reproduction, 26 checks against the frozen tables
(hashes, sample sizes, base rates, the label definition, censoring, and the
co-authorship ordering):

```
python reproduction/reproduce_assertions.py
```

Run the tabular ladder on one discipline. The `DATASET` override selects the
field and pins its SHA-256; `DATASET` takes the values `econ`, `math`, `neuro`,
`physics`, `chemistry`. Run from the repository root:

```
DATASET=math DATASET_PATH=data/clean_dataset_math.parquet python code/paper_pipeline/experiments/e1_baselines.py
```

The certificates run the same way: `e9a_placebo.py` (y-scrambling placebo,
30 seeds), `e9b_advisor_disjoint.py` (advisor-disjoint split),
`e10_advisor_placebo.py` (cohort-matched advisor placebo). The student-only
control `e14_self_persistence.py` also reads the per-author works cache, which
is larger than the archive; its outputs are committed under `results/`.

The graph-model leg runs on GPU: see `colab/README_GNN.md` for the single-upload
bundle and the run order. Its per-seed artifacts from the canonical run are
already committed under `results/results_<field>/`, so the corrected aggregation
reruns on CPU without any GPU work:

```
python code/e12_corrected_aggregation.py --all
```

An OpenAlex key is needed only to rebuild the data from scratch
(`export OPENALEX_API_KEY=...`); reproducing the experiments from the frozen
tables needs no key.

## The certificates

Four certificates audit the contract and the advisor signal, and the paper
reports all four over the five disciplines. The y-scrambling placebo retrains
the ceiling model on permuted labels and lands on chance everywhere, so no
individual-level information leaks through the features. The advisor-disjoint
split confines every advisor's students to one fold and moves the ceiling by at
most 0.044 in four disciplines; in chemistry it trips its pre-stated threshold,
and the paper reports that as produced. The advisor-placebo control swaps the
true advisor for a cohort-matched placebo and loses 0.111 to 0.161 AUC-PR in
every discipline. The student-only control removes every advisor-derived input
and sets the floor. Beyond the certificates, the evaluation of the graph leg is
itself audited: a naive aggregation reported seven ceiling crossings, and three
protocol corrections (comparator, bootstrap level, validation access) removed
six; `LEAKAGE_AUDIT_e12.md` documents the audit and
`results/results_<field>/e12_corrected_vs_m5.json` holds the corrected numbers.

## Results map

Every number in the paper traces to a script and a result file. The mapping is
the provenance table in `BUILD_REPORT.md`, Section 7. In brief: the ladder table
reads `results/results_<field>/e1_baselines.json`; the graph comparison table
and the ladder figure read `e12_corrected_vs_m5.json` (with
`e12_hgt_vs_baselines.json` kept as the naive contrast); the certificate table
reads `e9a_placebo.json`, `e9b_advisor_disjoint.json`,
`e10_advisor_placebo.json`, and `e14_self_persistence.json`; the mechanism
table reads `e10_advisor_placebo.json`, `e14_self_persistence.json`, and
`e6_innovation_premium.json`.

## License, citation, contact

The modeling tables derive from OpenAlex records, which are CC0, joined to the
Academic Family Tree genealogy, which is CC BY 4.0; the derived tables are
released under CC BY 4.0. The code and the datasheet are MIT, in `LICENSE`.
Citation metadata is in `CITATION.cff`. Questions and issues: open a GitHub
issue on this repository. Author ORCID:
[0009-0001-0403-3606](https://orcid.org/0009-0001-0403-3606).
