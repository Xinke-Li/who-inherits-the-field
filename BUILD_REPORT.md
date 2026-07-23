# Build report: five-discipline suite on the 2026 snapshot

This report records the numbers the build actually produced, the points where the
data departs from the v9 mainline expectation, and the actions that remain for the
author. Every figure here is written by a script to a result file; the provenance
table at the end maps each one. Nothing in this report is carried over from an
earlier draft.

## 1. What is built

Five frozen modeling tables share one protocol on one 2026 AFT snapshot: economics,
math, physics, chemistry, and neuro. The window, label, and filter parameters are
identical across all five (early 5, late 15, minimum 3 works per window, top 10
concepts, theta 0.2, t0 in 1950 to 2011, career span at most 60, at most 400
works per author). Each table passed a leakage guard and an independent
late_overlap recompute at tolerance 1e-9. The SHA-256 of each is pinned in
`config._PINNED_SHA` and listed in `data/SHA256SUMS`.

| discipline | modeled student-advisor pairs (one student each) | base rate | tabular ceiling (E1) | SHA-256 head |
|---|---|---|---|---|
| economics | 2,698 | 0.202 | 0.352 | 5ef1eb6f4c06 |
| math | 5,051 | 0.201 | 0.405 | aa5cc66ba9d2 |
| neuro | 21,846 | 0.251 | 0.426 | 9e9dabb80c9a |
| physics | 11,810 | 0.352 | 0.644 | 9d3a4460c8f6 |
| chemistry | 26,830 | 0.218 | 0.535 | 0b4d21bf7b16 |

The tabular ceiling is the best of M2 (logistic on tabular features), M3 (gradient
boosting on the same), and M5 (gradient boosting with neighbor-feature
aggregation), at 10 seeds. neuro reuses its frozen 2026 table; the other four were
resolved and built in this session by pub-ID anchoring, which held 86.7 percent
precision on econ ground truth with the 0.3 anchor threshold left in place.

## 2. The co-authorship axis, and its main caveat

The paper-metric co-authorship rate is the early-window rate from each build's
coauth stage. On the shared resolver it reads:

| economics | math | neuro | physics | chemistry |
|---|---|---|---|---|
| 0.583 | 0.678 | 0.763 | 0.788 | 0.791 |

That span is 1.36x. The AFT ever-coauthored proxy computed before the builds
ranked the same order but at lower, coverage-confounded levels (econ 0.163, math
0.293, neuro 0.374, physics 0.445, chemistry 0.552); it belongs only in an
appendix note, and `coauth_coverage_check.json` documents why (physics and
chemistry students carry three to four times as many indexed links as econ and
math students, so the proxy inflates them).

The caveat that matters: pub-ID anchoring resolves a student only through indexed
publications, and students who publish co-author with their advisor more often, so a
rate on the resolved sample sits above the rate in the full cohort. The same resolver
produces every field, so the bias runs one way everywhere and the econ below sciences
ranking holds, but the levels are selection inflated rather than population rates. We
report the rates only as an ordering of the disciplines. `coauthorship_axis.json`
carries this note, and the paper states it in the Limitations.

## 3. The mechanism certificates, reported per discipline

The advisor-versus-cohort-placebo gap (E10) is positive in all five, so a true
advisor's early profile predicts a student's late profile better than a
cohort-matched placebo does everywhere.

| economics | math | neuro | physics | chemistry |
|---|---|---|---|---|
| +0.137 | +0.161 | +0.111 | +0.157 | +0.114 |

The self-persistence branch (E14) and the innovation premium (E6) vary by field,
and the paper reports them as discipline-specific rather than forcing a single
reading. E14 places neuro in branch A (advisor information required), math in
branch C (advisor adds beyond the student's own signal), and economics, physics,
and chemistry in branch B (a student-only model matches the advisor feature). E6
supports the innovation premium in math and physics, reads a significant penalty
in neuro, and reads a compositional, fixed-effect-fragile correlation in economics
and chemistry.

## 4. Where the data departs from the mainline expectation

Three departures are worth stating plainly, because v9 asks that the conclusions
follow the data.

First, the co-authorship axis is descriptive, not a wide quantitative span. On the
shared resolver economics reads 0.583 and the sciences run 0.678 to 0.791, a 1.36x
range. We report the axis only as an ordering of the disciplines, and Section 2
explains why the resolved rates run above the full-cohort rates. What the axis
supports is narrow and still worth stating: the tabular ceiling does not track
co-authorship, since chemistry at 0.791 and math at 0.678 sit far apart in ceiling
(0.535 versus 0.405) for reasons of base rate, not collaboration.

Second, the economics mechanism certificates are the pub-ID resolved readings.
economics sits in E14 branch B, where a student-only model matches the advisor
feature, and its innovation premium is compositional under fixed effects. The modeled
sample is 2,698 students, the smallest in the suite, which the funnel reports openly.

Third, the layer-one headline is now evidence, with one exception and one audit
attached. The GNN leg ran on GPU three times; the third run is canonical and its
per-seed artifacts (10 seeds x 4 architectures x 5 disciplines, with per-student
test scores) live under `results/results_<field>/`. A naive aggregation of those
runs reported seven ceiling crossings (four in physics, two in chemistry, one in
neuro). The graph-construction audit (`LEAKAGE_AUDIT_e12.md`) found the
construction CLEAN and located three evaluation-protocol deviations instead
(comparator drift off M5, seed-level bootstrap, asymmetric validation access).
Re-aggregating under the pre-registered protocol
(`code/e12_corrected_aggregation.py`) removes six of the seven crossings; the
survivor is chemistry RGCN, +0.022 over the validation-symmetric ceiling M5', with
a student-level bootstrap interval of 0.007 to 0.036 and corrected p 0.0078. The
paper reports the negative control with that exception and the audit as a result.

Fourth, the certificate suite is now complete on all five disciplines. The
y-scrambling placebo (E9a, 30 seeds, global and within-cohort variants) passes
everywhere: global-shuffle AUC-ROC 0.489 to 0.501, placebo AUC-PR within 0.015 of
each base rate. The advisor-disjoint split (E9b, 10 seeds, rule fixed in the
docstring before the run: PASS if the ceiling drops by no more than 0.05 AUC-PR)
passes in economics (+0.044), math (+0.005), neuro (+0.009), and physics (+0.013)
and FAILS in chemistry (-0.0596, of which about a third is the random split's
lower base rate, 0.225 vs 0.245, and the rest a real AUC-ROC drop of 0.019). The
chemistry failure is consistent with the e12 finding that chemistry is the one
discipline with a surviving relational-graph gain: part of its ceiling rides on
advisor identity. The paper reports the verdict as produced.

## 5. Robustness: two works per window

The collider concern is answered for neuro. Relaxing the minimum from three works
to two adds 14 percent more students and leaves the certificates intact: 24,918
students against 21,846, base rate 0.240 against 0.251, advisor-placebo gap 0.111
against 0.111 to three decimals, tabular ceiling 0.407 against 0.426. The economics
arm builds from the economics works store by the same command with the minimum set to
two, and is the one remaining appendix table to run.

## 6. Actions that remain for the author

The author list in `CITATION.cff` is filled (Xinke Li, University of Chicago),
and the public repository and archive are linked: code at
https://github.com/Xinke-Li/who-inherits-the-field, data at
https://doi.org/10.5281/zenodo.21501632. One optional table remains: run the
economics two-works appendix table with the same builder to complete the pair
with neuroscience.

The GNN colab has been run for the five disciplines (third run canonical; the
full per-seed trees are unpacked under `results/results_<field>/` and verified
5 x 4 x 10 complete, labels bitwise-matching the local temporal split). The
corrected aggregation and the paper's layer one are final. The aggregation jsons
of the first two GPU runs were never committed and are superseded by the third
run's artifacts; nothing in the paper cites them.

The publication artifacts are prepared: the public branch is a single-commit
history authored by Xinke Li alone, and the Zenodo deposit uploads
`zenodo_archive.zip` (data, datasheet, reproduction; the manifest is in the
archive itself) under the reserved DOI above. `PUBLISH_STEPS.md` records the
remaining click-through steps.

## 7. Provenance

| number | script | file |
|---|---|---|
| co-authorship axis (early-window, AFT proxy) | `coauthorship_axis.py` | `data/coauthorship_axis.json` |
| AFT proxy and coverage confound | `measure_coauthorship.py`, `coauth_coverage_check.py` | `data/coauthorship_by_tree.json`, `data/coauth_coverage_check.json` |
| frozen tables, base rates, coauth rates, SHA | `build_dataset.py` | `artifact/code/data_<field>/dataset_summary_<field>.json`, `data/SHA256SUMS` |
| funnels | `funnel_table.py` | `suite_data/<field>/funnel_<field>.json` |
| abstract totals (330,282 raw student-advisor pairs, 68,235 modeled pairs, one student per pair) | column sums of the funnel table | paper Table 1 (`data/funnel_<field>.json`) |
| E1, E10, E14, E6, E7 (seed-level) | `experiments/*.py` under the DATASET override | `results_<field>/*.json` |
| two-works robustness | `build_dataset.py --min-works 2` | `artifact/code/data_neuro_min2/dataset_summary_neuro_min2.json` |
| resolver precision | `resolve_by_pubid.py validate` | econ ground-truth run, 86.7 percent |
| GNN per-seed artifacts (third run, canonical) | `colab/e12_gnn.ipynb` driving `e2_hgt.py` / `h_extra_gnns.py` | `results/results_<field>/results_hgt/`, `results_hgt_grid/`, `results_extra_gnns/`; archive `results/e12_full_results.zip` |
| naive e12 aggregation (audit contrast only) | `colab/e12_gnn.ipynb` cell 6 | `results/results_<field>/e12_hgt_vs_baselines.json` |
| corrected e12 aggregation (paper numbers: Table gnn, audit subsection, ladder GNN slot, M5' ceilings) | `code/e12_corrected_aggregation.py` | `results/results_<field>/e12_corrected_vs_m5.json`, `results/e12_corrected_summary.json` |
| graph-construction and protocol audit | manual audit | `LEAKAGE_AUDIT_e12.md` |
| y-scrambling placebo certificate, five disciplines (paper Table certs: shuffle ROC and AP columns) | `experiments/e9a_placebo.py` (verdict computed from the docstring rule) | `results/results_<field>/e9a_placebo.json` |
| advisor-disjoint certificate, five disciplines (paper Table certs: disjoint shift column; chemistry trips the -0.05 rule at -0.0596) | `experiments/e9b_advisor_disjoint.py` (decision rule in docstring, fixed before the run) | `results/results_<field>/e9b_advisor_disjoint.json` |
| advisor-placebo gap and student-only floor columns of the certificate table | `experiments/e10_advisor_placebo.py`, `experiments/e14_self_persistence.py` | `results/results_<field>/e10_advisor_placebo.json`, `results/results_<field>/e14_self_persistence.json` |
| data-characteristics table (paper Table 2): base rate and coauth | `build_dataset.py` summaries | `data/dataset_summary_<field>.json` |
| data-characteristics table: mean concept-node degree | `code/concept_density.py` (mirrors `e2_hgt.build_graph` edge definitions) | `results/concept_density.json` |
| data-characteristics table: n_test and resolution coverage | temporal split / funnels | `results/results_<field>/e12_corrected_vs_m5.json` (`n_test`), `data/funnel_<field>.json` (`st_id_coverage`, `adv_id_coverage`) |
| certificate prose: within-cohort shuffle AUC-ROC (0.487 to 0.525) | `experiments/e9a_placebo.py` | `results/results_<field>/e9a_placebo.json` (`variants.cohort`) |
| certificate prose: disjoint AUC-ROC decomposition and base rates | `experiments/e9b_advisor_disjoint.py` | `results/results_<field>/e9b_advisor_disjoint.json` |
| student-only floors and M1-minus-floor bootstrap CIs (paper Section 5) | `experiments/e14_self_persistence.py` | `results/results_<field>/e14_self_persistence.json` (`a_student_only_ladder.verdict`, `.comparisons`) |
| innovation-premium coefficients and p values (paper Section 5) | `experiments/e6_innovation_premium.py` | `results/results_<field>/e6_innovation_premium.json` (`primary_outcome.late_cite_pct_mean.3_controls_FE`) |
| two-row genealogy networks figure and Q/k panel titles | `code/make_network_figure.py` | `paper/figures/F12_five_discipline_networks.pdf`, `data/network_modularity.json` |
| framework overview figure (paper Figure 1; drawn TikZ, no computation: headline values from the funnel and data-characteristics tables, the e12 audit, and the e10 placebo; visual language carried over from the prior submission's Figure 1) | `paper/figures/F10_framework.tex` (pdflatex standalone) | `paper/figures/F10_framework.pdf` |
| certificate geometry dot plot (paper Figure 6: prior, cohort placebo, student-only floor, M1, best tabular per discipline) | `code/make_certificate_geometry.py` | `paper/figures/F13_certificate_geometry.pdf`; reads `results/results_<field>/e1_baselines.json`, `e10_advisor_placebo.json`, `e14_self_persistence.json` |
| audit dumbbell figure (paper Figure 4: the seven naive crossings, naive vs corrected gaps and intervals) | `code/make_audit_dumbbell.py` | `paper/figures/F14_audit_dumbbell.pdf`; reads `results/results_<field>/e12_hgt_vs_baselines.json` (naive), `e12_corrected_vs_m5.json` (corrected) |
| y-scrambling certificate boxplots (paper Figure 5: per-seed test AUC-ROC, global and within-cohort, 30 seeds) | `code/make_shuffle_certificate.py` | `paper/figures/F15_shuffle_certificate.pdf`; reads `results/results_<field>/e9a_global_perseed.jsonl`, `e9a_perseed.jsonl` |
