# Datasheet: five-discipline benchmark of intellectual inheritance

This datasheet follows the datasheets-for-datasets template (Gebru et al., 2021).
It describes version 1.0.0, a new dataset that does not continue any earlier version
history.

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
homonym-proof and held 86.7 percent precision on economics ground truth. Per-author
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

The intended use is a negative-control benchmark and a leakage-audit protocol.
Graph and text models are compared to a tabular ceiling under a fixed temporal
split; the co-authorship axis and the four certificates travel with the tables so a
user can re-run the audit. The dataset should not be read as a causal statement
about advising: the resolution step selects students who publish, so the sample is
not a random draw of doctoral students, and inheritance here is a predictive
relation, not a treatment effect.

## Distribution and licensing

The tables derive from OpenAlex records, which are CC0, joined to Academic Family
Tree genealogy data, which is CC BY 4.0. Redistribution of the derived tables
follows CC BY 4.0. The code is MIT. The large tables are distributed through a
Zenodo archive with a DOI; the code and this datasheet are in the public git
repository. The repository is https://github.com/Xinke-Li/who-inherits-the-field
and the archive DOI is 10.5281/zenodo.21501632, matching the paper's
Availability section.

## Maintenance

The authors maintain the dataset. Corrections and version notes will be recorded in
the repository. The snapshot is fixed at 2026, so the tables are static; a future
snapshot would be a new dataset version, not an edit of this one.
