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
`config._PINNED_SHA` and listed in `data/SHA256SUMS`. The post hoc robustness
layer is pinned separately: `results/robustness/SHA256SUMS.robustness` covers
the 1,006 result files that ship in the git clone, and
`results/robustness/SHA256SUMS.caches` covers the 182 OpenAlex cache shards
distributed only through Zenodo. A `.gitattributes` marks those paths and
`data/SHA256SUMS` as `-text`, so the checked-out bytes equal the committed
bytes on every platform and both manifests verify; editing it silently
breaks them.

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
resolved and built in this session by pub-ID anchoring with the 0.3 anchor
threshold left in place. Resolver precision is the ORCID cross-check in
`results/robustness/resolver_audit.json`: 3,548 pairs, raw agreement 0.981
(Wilson 0.976 to 0.985), bias-corrected 0.989 to 0.996. The older econ
ground-truth figure of 86.7 percent is historical; its validation set is not
distributed with this release.

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

These branch assignments predate the post hoc sweeps and are superseded as a
reading, though not as numbers. Across the five thresholds of
`results/robustness/theta_sweep_summary.json`, the four profile sizes of
`topk_sweep_summary.json`, and the two taxonomies of `topics_parallel.json`,
only physics and chemistry hold one branch throughout; economics, math, and
neuro sit within 0.02 of a branch boundary and move with the construction, so
the paper reads their labels at the pre-specified threshold only. The neuro
branch A label is the weakest of the three: its floor satisfies both branch
conditions at once and the label follows the rule's evaluation order.

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
test scores) live under `results/results_<field>/`. The superseded first aggregation of those
runs reported seven ceiling crossings (four in physics, two in chemistry, one in
neuro). The graph-construction audit (`LEAKAGE_AUDIT_e12.md`) found the
construction CLEAN and located three evaluation-protocol deviations instead
(comparator drift off M5, seed-level bootstrap, asymmetric validation access).
Re-aggregating under the pre-specified protocol
(`code/e12_corrected_aggregation.py`) removes six of the seven crossings; the
survivor is chemistry RGCN. That crossing has two readings and the paper keeps
both. The pre-specified asymmetric-budget reading is +0.022 over the
validation-symmetric ceiling M5', student-level interval 0.007 to 0.036,
corrected p 0.0078; it is retained as the superseded audit trail, because the
relational and attention models trained on a larger budget than the transformer
and that asymmetry was not resolved at the time. The post hoc symmetric
re-tuning of Appendix G gives the relational network the transformer's own
16-configuration grid and unified budget, and the crossing strengthens rather
than dissolving: +0.035 over M5', student-level interval +0.020 to +0.049,
corrected p 0.0078
(`results/robustness/rgcn_symmetric_verdict.json`). The paper's Section 5
quotes the symmetric numbers. The paper reports the ceiling comparison with
that exception and the audit as a result.

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

Fifth, the symmetric protocol was extended to the whole relational arm and
changed no verdict. RGCN and GAT were re-tuned under the transformer's grid and
unified budget in all five disciplines, ten cells, at ten seeds each
(`code/r17_full_symmetric_grid.py`,
`results/robustness/full_symmetric_grid/`). All nine new cells are null against
M5'; economics GAT and the physics and neuro relational networks screen
positive at the seed level but every student-level interval includes zero.
Chemistry RGCN remains the only crossing, so symmetric evaluation of the entire
graph arm adds no survivor.

## 5. Robustness: two works per window

The collider concern is answered for neuro. Relaxing the minimum from three works
to two adds 14 percent more students and leaves the certificates intact: 24,918
students against 21,846, base rate 0.240 against 0.251, advisor-placebo gap 0.111
against 0.111 to three decimals, tabular ceiling 0.407 against 0.426. The relaxation is
reported for neuroscience only. The economics arm would build from the economics
works store by the same command with the minimum set to two, but it was not run
and the paper claims the relaxation for neuroscience alone; no economics
two-works table exists in `data/` or `results/`. This is a stated limitation of
the release rather than pending work.

## 6. Status

The release is published. Code and documentation are at
https://github.com/Xinke-Li/who-inherits-the-field on the public
branch `main`; the data archive is at https://doi.org/10.5281/zenodo.21501632,
which is the Zenodo concept DOI and always resolves to the latest version.
`CITATION.cff` carries the author (Xinke Li, University of Chicago) and the same
concept DOI. The economics two-works table named in Section 5 was not run and is
recorded there as a limitation, not as pending work.

The GNN colab has been run for the five disciplines (third run canonical; the
full per-seed trees are unpacked under `results/results_<field>/` and verified
5 x 4 x 10 complete, labels bitwise-matching the local temporal split). The
corrected aggregation and the paper's layer one are final. The aggregation jsons
of the first two GPU runs were never committed and are superseded by the third
run's artifacts; nothing in the paper cites them.

The Zenodo archive carries `zenodo_archive.zip` (data,
datasheet, reproduction); its five frozen tables were verified byte-identical to
`data/SHA256SUMS` and to the hashes printed in the paper's Table 8. Depositing
the two OpenAlex caches that rebuild the post hoc robustness families is a
pending author action.

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
| resolver precision (primary) | `paper_pipeline/experiments/r13_resolver_audit.py` | `results/robustness/resolver_audit.json`, 3,548 ORCID pairs, 0.981 raw, 0.989 to 0.996 bias-corrected |
| resolver precision (historical) | `resolve_by_pubid.py validate` | econ ground-truth run, 86.7 percent; validation set not distributed |
| GNN per-seed artifacts (third run, canonical) | `colab/e12_gnn.ipynb` driving `e2_hgt.py` / `h_extra_gnns.py` | `results/results_<field>/results_hgt/`, `results_hgt_grid/`, `results_extra_gnns/`; archive `results/e12_full_results.zip` |
| superseded e12 aggregation (audit contrast only) | `colab/e12_gnn.ipynb` cell 6 | `results/results_<field>/e12_hgt_vs_baselines.json` |
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
| audit dumbbell figure (paper Figure 4: the seven superseded crossings, superseded vs corrected gaps and intervals) | `code/make_audit_dumbbell.py` | `paper/figures/F14_audit_dumbbell.pdf`; reads `results/results_<field>/e12_hgt_vs_baselines.json` (superseded), `e12_corrected_vs_m5.json` (corrected) |
| y-scrambling certificate boxplots (paper Figure 5: per-seed test AUC-ROC, global and within-cohort, 30 seeds) | `code/make_shuffle_certificate.py` | `paper/figures/F15_shuffle_certificate.pdf`; reads `results/results_<field>/e9a_global_perseed.jsonl`, `e9a_perseed.jsonl` |

### The post hoc robustness layer

Every row below is post hoc: it was computed after the pre-specified results
were in, and the paper labels it so wherever it is used. Two directories are
retained superseded outputs, kept deliberately as audit trail rather than left
behind as clutter: `results/robustness/e17_partial_gridcal/` holds the
first-pass swap-label calibration that missed its target base rate, and
`results/robustness/rgcn_symmetric/cpu_superseded/` holds the partial CPU grid
that the device-homogeneous GPU run replaced. Neither feeds a number in the
paper.

| number | script | file |
|---|---|---|
| threshold sweep (paper Appendix G, Tables 19 and 20) | `code/paper_pipeline/experiments/r1_theta_sweep.py` | `results/robustness/theta_sweep_summary.json`, `theta_0.10.json`, `theta_0.15.json`, `theta_0.20.json`, `theta_0.25.json`, `theta_0.30.json`, `theta_partial/` |
| continuous-label check, Spearman rho (paper Table 21) | `code/paper_pipeline/experiments/r2_continuous_label.py` | `results/robustness/continuous_label_summary.json`, `continuous_partial/` |
| symmetric RGCN re-tuning of the one crossing (paper Appendix G: +0.035 over M5') | `code/r3_rgcn_symmetric.py` | `results/robustness/rgcn_symmetric_verdict.json`, `rgcn_symmetric/` |
| per-author concept works fetch, 89,374 authors | `code/r5_fetch_author_works.py` | `results/robustness/openalex_cache/` (gitignored; not distributed with this release, regenerable from the OpenAlex API, pinned by `SHA256SUMS.caches`) |
| profile-size sweep and live-API drift calibration (paper Tables 22 and 23) | `code/paper_pipeline/experiments/r6_topk_sweep.py` | `results/robustness/topk_sweep_summary.json`, `topk_5.json`, `topk_10.json`, `topk_15.json`, `topk_20.json`, `topk_partial/` |
| both hash manifests | `code/r7_merge_colab.py` (`write_manifest`) | `results/robustness/SHA256SUMS.robustness` (1,006 files in the clone), `results/robustness/SHA256SUMS.caches` (182 undeposited OpenAlex cache shards) |
| per-author topics fetch, 89,374 authors | `code/r8_fetch_topics.py` | `results/robustness/openalex_topics_cache/` (gitignored; not distributed with this release, regenerable from the OpenAlex API, pinned by `SHA256SUMS.caches`) |
| power and equivalence, minimum detectable effect and TOST (paper Table 14) | `code/paper_pipeline/experiments/r11_power_tost.py` | `results/robustness/power_tost.json` |
| corpus-internal vocabulary bound, null by construction (paper Section 7) | `code/paper_pipeline/experiments/r12_vocab_bound.py` | `results/robustness/vocab_bound.json` |
| resolver ORCID cross-check, 3,548 pairs (paper Appendix B) | `code/paper_pipeline/experiments/r13_resolver_audit.py` | `results/robustness/resolver_audit.json` |
| concept-level composition of the top-10 profiles (paper Appendix G) | `code/paper_pipeline/experiments/r14_concept_levels.py` | `results/robustness/concept_level_analysis.json`, `concept_levels.json` |
| extra ladder rungs, CatBoost and TabPFN (paper Table 25) | `code/paper_pipeline/experiments/r15_extra_rungs.py`, `r15_tabpfn_standalone.py` | `results/robustness/extra_rungs.json`, `extra_rungs_partial/` |
| CatBoost under the corrected eq. (2) protocol (paper Appendix I) | `code/paper_pipeline/experiments/r15c_catboost_eq2.py` | `results/robustness/catboost_eq2_neuro.json` |
| TabPFN under the corrected eq. (2) protocol (paper Appendix I) | `code/paper_pipeline/experiments/r15f_tabpfn_eq2.py` | `results/robustness/tabpfn_eq2_econ.json`, `tabpfn_eq2_math.json` |
| parallel-taxonomy labels, subfield and topic (paper Table 26, Appendix J) | `code/paper_pipeline/experiments/r16_topics_parallel.py` | `results/robustness/topics_parallel.json`, `topics_partial/` |
| full symmetric grid, RGCN and GAT over five disciplines (paper Table 13) | `code/r17_full_symmetric_grid.py` | `results/robustness/full_symmetric_grid/` (810 files, including the five per-discipline verdicts and `DONE_fullgrid.flag`) |
| swap-label control (paper Table 24, Appendix H) | `code/paper_pipeline/experiments/e17_swap_label_cache.py` | `results/robustness/e17_swap_label.json`, `e17_partial/`; superseded first pass in `e17_partial_gridcal/` |
| Colab merge-back verification and the three decision tables | `code/r7_merge_colab.py` | verification only; writes the two manifests above |
