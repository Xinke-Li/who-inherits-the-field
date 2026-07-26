# A Leakage-Audited Forward-Prediction Benchmark for Research-Area Persistence in Doctoral Training

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21501632-blue.svg)](https://doi.org/10.5281/zenodo.21501632)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0001--0403--3606-a6ce39.svg)](https://orcid.org/0009-0001-0403-3606)

This repository holds the code, documentation, and reproduction materials for a
forward-prediction benchmark of research-area persistence in doctoral training,
spanning five scientific genealogies built from a single 2026 snapshot under one
byte-identical protocol: economics, mathematics, physics, chemistry, and
neuroscience, 330,282 raw student-advisor pairs distilled into 68,235 modeled
student-advisor pairs, one student per pair. The funnel is dominated by two
requirements: both members of a pair must resolve to OpenAlex authors, and the
label must be observable by 2026. Every modeling table is frozen
and pinned by SHA-256. A strong tabular model sets the ceiling in every
discipline; of four graph architectures only two clear that ceiling, in one
discipline of the five, every exceeding cell in chemistry, flagged and audited.
Five construction choices are sized against a 0.0013 determinism floor, and the
evaluation's own history, from the superseded first aggregation's seven crossings to
the corrected protocol, ships with the artifact.
The accompanying paper, "A Leakage-Audited Forward-Prediction
Benchmark for Research-Area Persistence in Doctoral Training", is under
review at the KDD 2027 Datasets and
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
pre-specified comparison rules.

## Repository layout

| Directory | Contents |
|---|---|
| `code/` | Dataset builders and the frozen experiment pipeline (`code/paper_pipeline/experiments/`). The post hoc robustness scripts sit in two places: `code/paper_pipeline/experiments/r1_*, r2_*, r6_*, r11_*, r12_*, r13_*, r14_*, r15_*, r16_*` and `code/r3_*, r5_*, r7_*, r8_*, r17_*` |
| `data/` | Attrition funnels, dataset summaries, the co-authorship axis, and `SHA256SUMS`; the parquet tables ship through Zenodo, not git |
| `colab/` | The GPU notebook and bundle for the graph-model leg (e12) |
| `datasheet/` | The datasheet for the dataset |
| `reproduction/` | An assertion-style script that recomputes the headline numbers from the frozen tables |
| `results/` | Per-discipline result files, including the per-seed graph-model artifacts and the corrected e12 aggregation |
| `results/robustness/` | The post hoc robustness layer, 1,006 files: threshold and profile-size sweeps, the continuous-label check, live-API drift, the symmetric RGCN and GAT grid over all five disciplines, the swap-label control, the extra ladder rungs (CatBoost, TabPFN), the parallel-taxonomy labels, the power and TOST table, and the resolver ORCID cross-check. `SHA256SUMS.robustness` pins every one of them |
| `results/revision/` | The post-review revision layer: the strict-contract graph cells with their per-seed score arrays, the tabular genealogy arm, the composition bound, the attribution set, the co-authorship strata, and the time-contract verification |

Three groups of files, split by where they live.

Every result file the paper cites is in this repository, including the full
symmetric grid, the strict-contract cells and both hash manifests. Cloning is
not sufficient to check every number, and the earlier README said it was.
`reproduction/reproduce_assertions.py` runs the subset of its checks that need
only committed files in a bare clone and reports how many it skipped; the rest
need the five frozen modeling tables, which are on Zenodo rather than here for
size reasons. Put those in `data/` and the whole set runs.

The five frozen modeling tables (`clean_dataset_<field>.parquet`), the resolved
pairs, and the outcomes tables are archived at
https://doi.org/10.5281/zenodo.21501632. Download the archive, place the
`data/` files into this repository's `data/` directory, and verify them against
the pinned hashes before running anything. They are needed to rerun the
experiments and to rebuild the frozen tables. The deposit also carries the
five per-author concept event tables, `concept_events_<field>.parquet`; place
them in `data/supplement/` so the time-contract verifier can rebuild the label
from events.

The two OpenAlex caches, `results/robustness/openalex_cache/` (per-author
concept works, about 1.2 GB) and `results/robustness/openalex_topics_cache/`
(per-author topics, about 270 MB), are neither in this repository nor in the
Zenodo archive. They are needed only to rebuild the post hoc robustness
families from scratch; every result those families produced is already in the
clone. They are regenerable with `code/r5_fetch_author_works.py` and
`code/r8_fetch_topics.py`, which read the OpenAlex API and need a key. A
regenerated cache will not reproduce `SHA256SUMS.caches` byte for byte, because
the live API drifts between snapshots; Appendix G of the paper measures that
drift against the frozen label and finds Cohen's kappa of 0.995 to 1.000.

The record is cited by its version DOI, 10.5281/zenodo.21501632.
Reproducibility does not rest on the identifier: the exact bytes a reader
needs are pinned by SHA-256 in `data/SHA256SUMS` and
`results/robustness/SHA256SUMS.robustness`, and both manifests verify on a
clone with the archive in place.

## Quickstart

Fetch the data from Zenodo and verify the hashes:

```
(cd data && sha256sum -c SHA256SUMS)
```

Verify the robustness layer. This one needs no Zenodo download, because every
file it pins is in the clone:

```
(cd results/robustness && sha256sum -c SHA256SUMS.robustness)
```

Run the assertion-style reproduction. **Unpack the archive into `data/` first**;
with it in place the harness runs **90 checks**: hashes, sample sizes, base
rates, the label definition, censoring, the co-authorship ordering, the
strict-contract and lineage verdict cells, and the completeness and hashes of
the robustness layer. In a bare clone before that step it runs 59 and skips the
31 that read the modeling tables, so a smaller number means the archive is
missing rather than that a check failed. It names every skipped group and why,
so a smaller total is never silent:

```
python reproduction/reproduce_assertions.py
```

Thirty-one of those checks read the five modeling tables, which are in
`zenodo_archive.zip` rather than in git. Run straight after cloning, the script
says so and runs the other 59; unpack the archive into `data/` for all 90.

It needs Python 3 with `pandas` and `pyarrow` installed (`pip install pandas
pyarrow`); `pyarrow` is the parquet engine, and without it the table checks stop
on an ImportError.

Two manifests pin this repository, and they have different provenance.
`results/robustness/SHA256SUMS.robustness` pins the 1,006 result files that
ship in the git clone, so it verifies immediately after cloning.
`results/robustness/SHA256SUMS.caches` pins the 182 OpenAlex cache shards,
which are gitignored and are not distributed with this release; it records the
exact shards the published results were computed from, so it verifies a copy of
those shards rather than a freshly fetched one. `data/SHA256SUMS` pins the 10 parquet tables that also come from
Zenodo. Line endings for the pinned paths are fixed by `.gitattributes`;
editing that file silently breaks both manifests on Windows checkouts.

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
and sets the floor.

A fifth instrument, added post hoc, bounds what the third certificate can
claim. The advisor-placebo control above is feature-only: it holds the label
fixed and swaps only the feature's advisor, so feature and label still share
the true advisor's profile and part of the measured gap is that mechanical
alignment. The swap-label control swaps the label's advisor too, at a
base-rate-calibrated threshold, and it does not confirm advisor specificity in
any well-calibrated discipline: the excess is indistinguishable from zero in
economics and physics and reverses in mathematics, neuroscience, and chemistry.
The feature-only gap is therefore an upper bound, not a measurement of advisor
specificity. It runs from
`code/paper_pipeline/experiments/e17_swap_label_cache.py` and lands in
`results/robustness/e17_swap_label.json`.

Beyond the certificates, the evaluation of the graph leg is
itself audited: the superseded first aggregation reported seven ceiling crossings, and three
protocol corrections (comparator, bootstrap level, validation access) removed
six; `LEAKAGE_AUDIT_e12.md` documents the audit and
`results/results_<field>/e12_corrected_vs_m5.json` holds the corrected numbers.

## Results map

Every number in the paper traces to a script and a result file. The mapping is
the provenance table in `BUILD_REPORT.md`, Section 7. In brief: the ladder table
reads `results/results_<field>/e1_baselines.json`; the graph comparison table
and the ladder figure read `e12_corrected_vs_m5.json` (with
`e12_hgt_vs_baselines.json` kept as the superseded contrast); the certificate table
reads `e9a_placebo.json`, `e9b_advisor_disjoint.json`,
`e10_advisor_placebo.json`, and `e14_self_persistence.json`; the mechanism
table reads `e10_advisor_placebo.json`, `e14_self_persistence.json`, and
`e6_innovation_premium.json`. The robustness layer has its own rows in the
same provenance table, one per `r*` script. The paper's audit-trail appendices
record every claim that changed since the submitted draft, in both directions.

## License, citation, contact

The modeling tables derive from OpenAlex records, which are CC0, joined to the
Academic Family Tree genealogy, which is CC BY 4.0; the derived tables are
released under CC BY 4.0. The code and the datasheet are MIT, in `LICENSE`.
Citation metadata is in `CITATION.cff`. Questions and issues: open a GitHub
issue on this repository. Author ORCID:
[0009-0001-0403-3606](https://orcid.org/0009-0001-0403-3606).
