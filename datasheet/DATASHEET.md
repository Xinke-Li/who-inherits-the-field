# Datasheet: A Leakage-Audited Forward-Prediction Benchmark for Research-Area Persistence in Doctoral Training

This datasheet follows the datasheets-for-datasets template (Gebru et al., 2021).
It describes version 1.0.0, a new dataset that does not continue any earlier version
history. The accompanying paper is the KDD 2027 Datasets and Benchmarks
submission "A Leakage-Audited Forward-Prediction Benchmark for Research-Area
Persistence in Doctoral Training".

## Motivation

The dataset supports one task: predict whether a doctoral student inherits an
advisor's research area, framed as forward prediction under a time contract. Every
input feature is measured on or before the student's fifth career year (t0 plus 5),
and the label is read at t0 plus 15. The purpose is a leakage-audited benchmark that
lets a reader test whether graph or text models beat a strong tabular baseline, and
whether inheritance signal is advisor-specific. It was assembled by the paper's
authors for that benchmark. No external body funded a separate data-collection
effort; the records come from public sources described below.

## Composition

Each instance is one student-advisor pair, represented by the student's early-window
research profile, the advisor's early-window profile, their early co-authorship
count, and administrative fields. There are five tables, one per discipline, from a
single 2026 snapshot: economics (2,698 students), math (5,051), physics (11,810),
chemistry (26,830), and neuroscience (21,846). The label y is 1 when the Jaccard
overlap between the student's late-window concepts and the advisor's early profile
exceeds 0.2. Base rates are 0.202, 0.201, 0.352, 0.218, and 0.251 respectively.

The features are the pre-window quantities listed in the pipeline config:
early_overlap, early_prod, early_breadth, the advisor's early_prod, early_breadth,
and career age at t0, and the early co-authorship count and indicator. Columns that
would leak the label, such as late-window quantities and any full-career citation
count, are held out of the feature set and named in a banned-columns list the build
asserts against.

The tables do not contain the raw text of publications. They contain OpenAlex
concept labels and counts. The genealogy edges (advising relations) come from the
Academic Family Tree. The dataset is a sample in one sense: it covers the students
whose publications resolve to OpenAlex author identifiers by pub-ID anchoring, which
is 31 to 70 percent of pairs depending on the discipline. This selection is reported
as a limitation, since it keeps students with indexed publications.

## Collection process

Student-advisor pairs come from the Academic Family Tree, filtered to graduate
advising edges where both people carry the discipline token in their major-area
field, on the 2026 snapshot. Each person is resolved to an OpenAlex author
identifier by anchoring on their known publication identifiers, which is
homonym-proof. Resolver precision is measured by the ORCID cross-check in
`results/robustness/resolver_audit.json`, which ships with this release: 3,548
unique pairs, raw agreement 0.981 with a Wilson interval of 0.976 to 0.985, and
bias-corrected precision between 0.989 and 0.996. An economics ground-truth run
reported 86.7 percent precision; that figure is historical, because the
validation set it was measured on is not distributed here. Per-author
publication histories, concepts, years, and citation counts are read from OpenAlex.
The window and label parameters are fixed and identical across disciplines.

## Preprocessing, cleaning, labeling

The pipeline builds each table in resumable stages: fetch author works, form the
temporal feature table, count early co-authorship, apply the leakage guard, and
export. A pair enters the modeling table only if both people resolve, both have at
least three works in each window, the career span is at most 60 years, and t0 falls
in 1950 to 2011 so the fifteen-year label is observed by 2026. The build recomputes
late_overlap independently and asserts it matches the stored column within 1e-9.
Each table is frozen and pinned by SHA-256 in `data/SHA256SUMS`.

## Uses

The intended use is a leakage-audited benchmark that holds graph learning to a
strong tabular ceiling, and a leakage-audit protocol.
Graph and text models are compared to a tabular ceiling under a fixed temporal
split; the co-authorship axis and the four certificates travel with the tables so a
user can re-run the audit. A fifth instrument, the post hoc swap-label control,
bounds what the advisor-placebo certificate can claim: it swaps the label's
advisor as well as the feature's and does not confirm advisor specificity in any
well-calibrated discipline, so the feature-only gap is an upper bound rather than
a measurement of advisor specificity. The dataset should not be read as a causal statement
about advising: the resolution step selects students who publish, so the sample is
not a random draw of doctoral students, and inheritance here is a predictive
relation, not a treatment effect.

## Distribution and licensing

The tables derive from OpenAlex records, which are CC0, joined to Academic Family
Tree genealogy data, which is CC BY 4.0. Redistribution of the derived tables
follows CC BY 4.0. The code is MIT. The Zenodo archive carries the five frozen
modeling tables with the resolved pairs and outcomes. The two OpenAlex caches,
`results/robustness/openalex_cache/` and
`results/robustness/openalex_topics_cache/`, are needed only to rebuild the post
hoc robustness families from scratch; they are not distributed with this release
and are regenerable with `code/r5_fetch_author_works.py` and
`code/r8_fetch_topics.py` from the OpenAlex API.
`results/robustness/SHA256SUMS.caches` records the exact shards the published
results were computed from. Every result file
the paper cites is in the public git repository and needs no download. The
repository is https://github.com/Xinke-Li/who-inherits-the-field and the archive
DOI is 10.5281/zenodo.21501632, the current version DOI; the identifier the
paper prints resolves to the same record and always to the latest
version and matches the paper's Availability section. Reproducibility does not
rest on the DOI: the exact bytes a reader needs are pinned by SHA-256 in
`data/SHA256SUMS` and `results/robustness/SHA256SUMS.robustness`.

## Maintenance

The author maintains the dataset. Corrections and version notes are recorded in
the repository's revision records, and the paper's audit-trail appendices state
every claim that changed since the submitted draft, in both
directions. The snapshot is fixed at 2026, so the tables are static; a future
snapshot would be a new dataset version, not an edit of this one.

OpenAlex has deprecated Concepts in favor of Topics. The primary label reads
the concept assignments pinned in the 2026 cache, so the deprecation does not
move the released tables; a future version that migrates the primary label to
topics constitutes a new dataset version, and the kappa table of the paper's
parallel-taxonomy appendix (Appendix L) is the conversion baseline between the
two label generations.
