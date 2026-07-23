# Graph-construction leakage audit — e12 GNN leg, five disciplines

**Scope:** audit only. No code changed, no GPU re-run, no frozen data / paper / results touched.
**Date:** 2026-07-22
**Trigger:** e12 reports four-of-four graph architectures crossing the tabular ceiling in
physics (p_bh ≤ 0.0039), two in chemistry (RGCN, GAT+cohort-time), one in neuroscience
(RGCN), none in economics or mathematics. By the paper's own negative-control logic, a
structural gain is first assumed to be a construction artifact.

---

## 0. Verdict

**No leakage was found in the graph construction.** All five disciplines are **CLEAN** on
the temporal contract and on `BANNED_COLUMNS`. The head suspicion — that student↔concept
edges might carry late-window concepts — is **refuted by independent recomputation from the
raw OpenAlex works stores**: 0 mismatches and 0 contaminating terms across all five
disciplines, on every row (§2).

Real defects were found, none of which is the mechanism behind the physics result:

| # | Defect | Class | Severity |
|---|---|---|---|
| F1 | Advisor route carries a few years of future information — `.first()` collapse of advisor features, and 2-hop aggregation of later-cohort siblings that M5 is forbidden | time-window violation | minor (bounded: ≤4 yr, 4–22 % of test rows) |
| F2 | `h_extra_gnns.py` computes the class weight over train+val+test labels | label leakage into the loss | minor (2–4 % weight shift) |
| F3 | e12's ceiling is `max(M2, M3)`, dropping **M5**, which `e1_baselines.py:13-15` names as the *pre-registered* comparator | protocol deviation | **major** |
| F4 | e12's bootstrap resamples **10 seeds**, not test students, so the CI omits test-set sampling error entirely | protocol deviation | **major** |
| F9 | `h_extra_gnns` runs 300/30 epochs with no weight decay while claiming to be "identical to E2" (200/15, wd 1e-4) | protocol deviation | major |
| F14 | Graph models early-stop on the temporal VAL cohort's AUC-PR (100s of checkpoints); M3/M5 early-stop on a random 15 % slice of TRAIN and never see VAL | protocol deviation | **major** |
| F5, F6, F10–F13 | closure exemption, GAT-only `t0` channel, stale docstring, post-freeze sample selection, three ceiling definitions, 400-work cap | see §4 | minor / informational |

**The physics crossing is not a leakage artifact. It is an artifact of the comparison —
F3 (a comparator that drops the pre-registered baseline M5), F4 (an uncertainty interval ~5×
too narrow because it resamples seeds instead of students), and F14 (a model-selection budget
the ceiling is not given).** Under the paper's own original e12 protocol
(`e12_hgt_vs_baselines.py`, which compares against M5 *and* bootstraps over test students) the
physics gaps are of the same order as the noise. See §6 and §7.

---

## 1. What was audited, and that it is the code that ran

The e12 five-discipline results were produced by `colab/e12_gnn.ipynb` driving the scripts
packed in `colab/e12_colab_bundle.zip`. The bundle's copies were verified **byte-identical**
(SHA-256) to the repo copies audited here:

| file | sha256[:16] | bytes |
|---|---|---|
| `paper_pipeline/config.py` | `1d3923a2e7433941` | 6584 |
| `paper_pipeline/experiments/e2_hgt.py` | `af268a93f3e175ac` | 11997 |
| `paper_pipeline/experiments/h_extra_gnns.py` | `c3722f1b5ff61bb9` | 9175 |
| `paper_pipeline/utils/data.py` | `af9787aebeceb0f1` | 8265 |

The notebook contains **no graph construction of its own**: cells 3–5 shell out to
`e2_hgt.py` (`hgt`, `hgt_tuned`) and `h_extra_gnns.py` (`rgcn`, `tgat` → reported as
`gat_cohort_time`). **All four architectures build their graph through the single function
`e2_hgt.build_graph`** (`h_extra_gnns.py:181` calls `E2.build_graph(ds, args.ablate)`), so
there is exactly one graph-construction code path for all four models and all five
disciplines.

The split was reproduced locally and matches the frozen manifests exactly for all five
disciplines (train/val/test = econ 1646/556/495, math 3356/760/935, physics 7645/1946/2218,
chemistry 16378/5835/4617, neuro 14225/4156/3463), so every diagnostic below is computed on
precisely the graphs that were trained.

---

## 2. Head suspicion — student↔concept edges: **CLEAN**

### 2.1 Code path

`e2_hgt.py:105-108` — student→concept edges:

```python
if ("student", "studies", "concept") not in skip:
    add_edges("student", "studies", "concept",
              [(s_ix[r.student_pid], c_ix[c]) for r in df.itertuples()
               for c in r.early_concepts])
```

`e2_hgt.py:109-112` — advisor→concept edges use `r.adv_profile`.
`e2_hgt.py:76-77` — the concept node set is built from `early_concepts ∪ adv_profile` only.

`late_overlap`, `late_prod` and any late-window concept list appear **nowhere** in
`build_graph`. `df.y` is attached at `e2_hgt.py:125` as `data["student"].y`, but neither
`HGTNet.Net.forward` (`e2_hgt.py:153-165`) nor `RGCN.forward` / `TGATLite.forward`
(`h_extra_gnns.py:80-84`, `118-128`) ever reads it; it is consumed only as masked
supervision in `train_eval`. So `y` is not an input.

### 2.2 Builder windows

`build_neuro_dataset.py:129-138` (`profile`) filters `y0 <= year <= y1`, inclusive on both
ends. `stage_table` (`:165-167`):

```python
early, n_e, br_e = profile(sw, t0, t0 + EARLY_YEARS)          # [t0, t0+5]
late,  n_l, _    = profile(sw, t0 + EARLY_YEARS + 1, t0 + LATE_YEARS)   # [t0+6, t0+15]
advp,  n_a, br_a = profile(aw, None, t0 + EARLY_YEARS)        # (-inf, t0+5]
```

The two student windows are disjoint by construction. `adv_profile` is exactly the set `A`
in the label — legitimately, since it is `≤ t0+5`.

### 2.3 Independent recomputation from raw sources — the decisive test

Rather than trust the builder, `early_concepts`, `adv_profile` and `y` were recomputed from
the per-author `works_store.jsonl` caches using `profile()`/`jaccard()` verbatim, for **every
row of all five frozen tables**, and additionally tested for contamination: does the stored
`early_concepts` ever contain a concept that appears **only** in the student's late-window
works?

| discipline | rows | authors matched | `early_concepts` mismatches | `adv_profile` mismatches | `y` mismatches | rows with a **late-only** concept in `early_concepts` |
|---|---:|---:|---:|---:|---:|---:|
| econ | 2 697 | 3 975 / 3 975 | **0** | **0** | **0** | **0** |
| math | 5 051 | 7 380 / 7 380 | **0** | **0** | **0** | **0** |
| physics | 11 809 | 15 628 / 15 628 | **0** | **0** | **0** | **0** |
| chemistry | 26 830 | 33 528 / 33 528 | **0** | **0** | **0** | **0** |
| neuro | 21 844 | 29 379 / 29 379 | **0** | **0** | **0** | **0** |

**Conclusion: the student-side concept edges use `early_concepts` (≤ t0+5) and nothing else,
verified against raw source, in all five disciplines. `C_late` is not encoded in the graph,
directly or by proxy.** The head suspicion is closed.

---

## 3. Full provenance table

The code path is identical for all five disciplines, so one table covers all of them
(per-discipline density numbers are in §5).

### 3.1 Node types

| node | count source | features | source columns | window | banned? | verdict |
|---|---|---|---|---|---|---|
| `student` | `df.student_pid` (`e2_hgt.py:74`) | `TABULAR_ST` = `early_prod`, `early_breadth`, `early_overlap` (`:36`, `:86-88`) | `profile(sw, t0, t0+5)` counts + `J(C_early, A)` | ≤ t0+5 | no | **clean** |
| `advisor` | `sorted(df.advisor_pid.unique())` (`:75`) | `TABULAR_ADV` = `adv_early_prod`, `adv_early_breadth`, `adv_career_age_at_t0` (`:37`, `:90-93`) | `profile(aw, None, t0+5)` counts, `t0 − min(adv years)` | ≤ t0+5 **per-row**, but collapsed by `.first()` → see **F1** | no | **F1** |
| `institution` | `df.institution_name` (`:76`) | none — learnable `Embedding` (`:148`) | AFT genealogy record (PhD institution), carried through `pairs_resolved_<field>.parquet` | predetermined by design; **carries no year stamp**, so ≤ t0+5 is assumed, not verifiable from the frozen table | no | clean (assumption noted) |
| `concept` | `early_concepts ∪ adv_profile` (`:76-77`) | none — learnable `Embedding` (`:148`) | as above | ≤ t0+5 | no | clean |

### 3.2 Edge types

| edge | line | source column | builder window | banned-column check | verdict |
|---|---|---|---|---|---|
| `student –studies→ concept` | `:105-108` | `early_concepts` | `profile(sw, t0, t0+5)` | — | **clean** (§2.3) |
| `advisor –studies→ concept` | `:109-112` | `adv_profile` | `profile(aw, None, t0+5)` | — | **clean** |
| `advisor –advises→ student` | `:113-115` | `advisor_pid` | genealogy, predetermined | — | **clean** |
| `student –at→ institution` | `:116-119` | `institution_name` | genealogy, predetermined | — | **clean** |
| `student –coauth→ advisor` | `:120-123` | `coauth_early` | OpenAlex count with `publication_year:{t0}-{t0+5}` (`build_neuro_dataset.py:213-215`) | `co_authored` (banned, full-career) is **not** used; the allowed `coauth_early` is | **clean** |
| all `rev_*` mirrors | `:102` | mirrors of the above | — | — | clean |

**No citation edges exist in the graph at all**, so `st_cites_adv_broad` /
`adv_cites_st_broad` cannot enter. Confirmed by exhaustive enumeration of `add_edges` calls
(`e2_hgt.py:105-123` — five calls, no others anywhere in the bundle).

**Banned columns:** none of `st_h_index`, `adv_h_index`, `adv_total_citations`,
`st_h_recomp`, `co_authored`, `st_cites_adv_broad`, `adv_cites_st_broad`, `career_len`,
`late_overlap`, `late_prod` is referenced anywhere in `build_graph`. The last three of these
plus `y` are physically present in the frozen parquet but are never read by the graph code.
`utils/data.load_dataset:30-36` re-asserts the invariant set on every load (`h_extra_gnns`
goes through it; `e2_hgt.py:248` reads the parquet directly and skips the hash check — see
F8b).

**`coauth_early` missingness:** the `-1` sentinel path (`build_neuro_dataset.py:242-244`)
would silently turn "not fetched" into `coauth_early = False`. Checked: **0 missing rows in
all five frozen tables**, so the concern is moot here.

---

## 4. Findings

### F1 — advisor node features are as-of the wrong student's freeze date  ·  time-window violation  ·  minor

`e2_hgt.py:90`:

```python
adv_tab = (df.groupby("advisor_pid")[TABULAR_ADV].first().reindex(advisors))
```

`adv_early_prod`, `adv_early_breadth` and `adv_career_age_at_t0` are computed **per
student-row**, as-of that student's `t0+5` (`build_neuro_dataset.py:167,176-177`). An advisor
with *k* students has *k* different values. `.first()` collapses them to one — the value from
whichever of their students appears first in parquet row order, which `stage_table:182` sorts
by `early_prod` **descending**, i.e. essentially at random with respect to `t0`.

When the donor student's `t0` is later than student *i*'s, student *i*'s advisor node carries
advisor information from as late as `t0_donor + 5 > t0_i + 5` — after *i*'s freeze, and inside
*i*'s label window. Measured on the frozen tables:

| discipline | students whose advisor node is as-of a later `t0` | max years ahead | **test-split** affected | max years ahead (test) |
|---|---:|---:|---:|---:|
| econ | 650 / 2 697 (24.1 %) | 34 | **3.6 %** | 3 |
| math | 1 389 / 5 051 (27.5 %) | 43 | **4.5 %** | 3 |
| physics | 3 474 / 11 809 (29.4 %) | 44 | **7.1 %** | 4 |
| chemistry | 8 873 / 26 830 (33.1 %) | 48 | **7.0 %** | 3 |
| neuro | 6 007 / 21 844 (27.5 %) | 59 | **4.0 %** | 2 |

(Verified: parquet row order is `early_prod`-descending in every table, and the `.first()`
donor equals the group-maximum-`early_prod` student for every advisor.)

**The same advisor node also opens a second, wider route.** Both `e2_hgt` and `h_extra_gnns`
run `LAYERS = 2`, so message passing reaches `student → advisor → sibling-student`: a test
student aggregates the *early-window features* of same-advisor siblings, including siblings
whose `t0` is **later** than its own, whose features are measured over `[t0_j, t0_j+5]` with
`t0_j + 5 > t0_i + 5`. `utils/data.py:162-163` forbids exactly this for M5
(`prior = (t0 <= t0[a])`), so the graph arm again gets what the tabular neighbour baseline is
denied:

| discipline | test students reaching a **future** sibling | sibling links that are future | max years ahead |
|---|---:|---:|---:|
| econ | 9.3 % | 6.6 % | 3 |
| math | 12.2 % | 6.6 % | 3 |
| physics | **20.3 %** | 10.6 % | 4 |
| chemistry | **21.7 %** | 8.6 % | 3 |
| neuro | 11.5 % | 6.9 % | 2 |

This is a genuine breach of the stated contract ("every input must come from records ≤
`t0+5`") and it is **specific to the graph models** — M2/M3/M5 read each row's own correct
`adv_early_prod` and aggregate only prior siblings, so the ceiling is unaffected.

It is nonetheless too small to be the mechanism, on both routes. The `.first()` route touches
3.6–7.1 % of test rows by at most 2–4 years on three weak advisor covariates, and its ranking
(physics 7.1 % ≈ chemistry 7.0 % > math 4.5 % > neuro 4.0 % > econ 3.6 %) puts math above
neuro. The 2-hop route ranks chemistry 21.7 % > physics 20.3 % > math 12.2 % > neuro 11.5 % >
econ 9.3 % — again with math above neuro, and carrying at most 2–4 years of drift in three
early-window scalars that are then averaged over the whole sibling set. Neither ordering
reproduces physics-4 > chemistry-2 > neuro-1 > {econ, math}-0.

*Minimal fix (not applied):* key advisor nodes by `(advisor_pid, t0)` — or, more simply,
give each student's advisor a row-specific feature vector and mask advisor→student edges to
siblings with `t0_j <= t0_i`, which makes the graph obey M5's own rule. Taking the
**minimum-`t0`** donor for `.first()` is a one-line conservative patch for the feature half.

### F2 — `h_extra_gnns` computes the class weight over train+val+test  ·  label leakage  ·  minor

`h_extra_gnns.py:136-137`:

```python
w = torch.tensor([1.0, float((y == 0).sum()) / max(1, int((y == 1).sum()))], device=device)
```

`y` here is `data["student"].y` — **all** nodes. `e2_hgt.py:182-183` does the same computation
restricted to `y[masks["train"]]`. Test-set labels therefore enter the RGCN / GAT training
loss, as a scalar. Measured shift:

| discipline | global weight (h_extra_gnns) | train-only weight (e2_hgt) | ratio |
|---|---:|---:|---:|
| econ | 3.9396 | 3.8129 | 1.033 |
| math | 3.9764 | 3.9572 | 1.005 |
| physics | 1.8408 | 1.8925 | 0.973 |
| chemistry | 3.5832 | 3.7390 | 0.958 |
| neuro | 2.9840 | 3.1184 | 0.957 |

It is real leakage by definition, and it affects exactly the two models that cross in
chemistry and neuro. But it is a 2–4 % change in a single loss scalar, on a rank metric, and
it *reduces* the positive-class weight in physics/chemistry/neuro (ratio < 1) — the direction
that makes the minority class *less* emphasised. It cannot plausibly produce +0.010 to
+0.025 AUC-PR.

*Minimal fix (not applied):* `y[masks["train"]]`, matching `e2_hgt.py:182`.

### F3 — the ceiling drops M5, contradicting the pre-registered decision rule  ·  protocol deviation  ·  **major**

`e1_baselines.py:13-15` states the rule verbatim:

> Decision rule (pre-registered, updated 2026-07-05): any GNN contribution in the paper is
> claimed RELATIVE TO M5 (the strongest graph-aware non-GNN baseline), with M3/M4 reported
> for the ladder.

The repo's own `e12_hgt_vs_baselines.py:68` implements exactly that
(`for base in ("M5_gbdt_nfa", "M3_gbdt_tabular", "M2_logit_tabular")`). The five-discipline
results, however, came from **notebook cell 6**, whose `per_seed_ceiling()` reads only
`M2_logit_tabular` and `M3_gbdt_tabular` — **M5 is not in the comparator set**.

This matters most in physics, where M5 is the *only* discipline in which the neighbourhood
baseline itself beats the declared ceiling:

| discipline | max(M2,M3) = ceiling | **M5_gbdt_nfa** | M5 − ceiling |
|---|---:|---:|---:|
| econ | 0.3559 | 0.3390 | −0.0169 |
| math | 0.4079 | 0.3980 | −0.0099 |
| **physics** | **0.6345** | **0.6437** | **+0.0092** |
| chemistry | 0.5349 | 0.5307 | −0.0042 |
| neuro | 0.4257 | 0.4231 | −0.0026 |

So in physics roughly **45 % of the HGT gap (+0.0206) and 77 % of the GAT gap (+0.0120) is
already achieved by a purely tabular model** that aggregates temporally-closed advisor
siblings. Re-expressed against the pre-registered comparator:

| discipline | model | vs ceiling | **vs M5** | e12 verdict |
|---|---|---:|---:|---|
| physics | hgt | +0.0206 | **+0.0114** | exceeds |
| physics | hgt_tuned | +0.0260 | **+0.0168** | exceeds |
| physics | rgcn | +0.0193 | **+0.0101** | exceeds |
| physics | gat_cohort_time | +0.0120 | **+0.0028** | exceeds |
| chemistry | rgcn | +0.0249 | +0.0291 | exceeds |
| chemistry | gat_cohort_time | +0.0124 | +0.0165 | exceeds |
| neuro | rgcn | +0.0096 | +0.0123 | exceeds |
| econ | gat_cohort_time | +0.0163 | +0.0332 | *not* exceeding |

Note the last row. Against M5 — the comparator the paper says it uses — econ's GAT shows the
**largest point gap in the whole suite** (+0.0332), yet econ is reported as a discipline where
the graph adds nothing. That comparison was simply never made: against `max(M2, M3)` the same
model reaches +0.0163 and fails only the p-gate (`p_bh = 0.168`, on n_test = 495 with large
seed variance; its seed-level CI, `[0.0001, 0.0313]`, does exclude zero). So the comparator
choice, not the graph, is deciding which disciplines look clean — in both directions.

*Minimal fix (not applied):* restore M5 to `per_seed_ceiling` — `max(M2, M3, M5)` — or drop
back to `e12_hgt_vs_baselines.py`, which already does this.

### F4 — the bootstrap resamples seeds, not students  ·  protocol deviation  ·  **major**

Notebook cell 6:

```python
def boot_ci(diffs, n=2000):
    rng = np.random.default_rng(0)
    ms = [rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(n)]
```

`diffs` is the vector of **10 seed-level** gaps on a **fixed** test set. The resulting interval
measures seed-to-seed jitter of model initialisation and contains **no test-set sampling
error at all**. The repo's own `e12_hgt_vs_baselines.py:104-117` does the correct thing —
a score-level paired bootstrap over test students — and the frozen econ artifact shows what
that scale looks like: HGT − M5 point `+0.0014`, CI `[−0.0566, +0.0608]`, width **0.117**.

Measured on the exact five test sets (paired student-level bootstrap of an M3 − M2 gap, as a
calibration of the noise floor; 2000 draws):

| discipline | n_test | student-level paired CI width | e12 seed-level CI width (mean) | ratio | largest graph gap |
|---|---:|---:|---:|---:|---:|
| econ | 495 | 0.1205 | 0.0282 | 4.3× | +0.0163 |
| math | 935 | 0.1092 | 0.0180 | 6.1× | −0.0028 |
| **physics** | 2 218 | **0.0422** | **0.0084** | **5.0×** | **+0.0260** |
| chemistry | 4 617 | 0.0304 | 0.0235 | 1.3× | +0.0249 |
| neuro | 3 463 | 0.0355 | 0.0187 | 1.9× | +0.0096 |

Read as half-widths: physics carries ≈ **±0.021** of student-level sampling noise on a
model-vs-model gap. The physics gaps are +0.0120 … +0.0260 — i.e. **at or below the noise
floor**, while the reported CI (`[0.016, 0.0245]` for HGT) is five times too narrow to see
it. On the same reading, only **chemistry RGCN (+0.0249 vs ±0.015)** clearly clears
student-level noise; physics `hgt_tuned` (+0.0260 vs ±0.021) is marginal; neuro RGCN
(+0.0096 vs ±0.018) does not clear it.

*This calibration uses an M3 − M2 gap as a proxy because the per-seed GNN score files were
never returned from Colab (F8a). It fixes the order of magnitude, not the exact interval.*

**The paired Wilcoxon has the same defect.** The split is seed-independent, and `M2` is
bit-identical across all 10 seeds (`std ≈ 3e-17` in every `e1_baselines.json`), so seed *k* of
a GNN and seed *k* of the ceiling share nothing but an index — the "pairing" removes no
common variance. What the test actually asks is whether the gap has a consistent sign across
10 initialisations of a model evaluated on one fixed test set. With near-zero variance on both
sides, any consistently-signed gap, however small, reaches the two-sided floor of `1/512 =
0.001953` — which is exactly the value reported for three of the four physics models. The
p-value is therefore a statement about seed stability, not about evidence against the null.

*Minimal fix (not applied):* bootstrap over test students, paired at the score level, as
`e12_hgt_vs_baselines.py:104-117` already does. This needs the per-seed `test_scores` /
`test_labels` arrays that `e2_hgt.py:222-225` writes — i.e. it needs those JSONs back, not a
GPU re-run, **if** they still exist in the Colab session.

### F5 — the GNN is exempt from the closure rule M5 obeys  ·  structural asymmetry  ·  informational

`utils/data.py:123-177` (`build_nfa_features`) states and enforces the paper's temporal
contract for neighbour information: a same-advisor sibling *j*'s **label** may be used for
student *i* only if `t0_j + 15 <= t0_i + 5` (line 169), i.e. only if *j*'s 15-year window had
already closed at *i*'s freeze.

The GNN honours no such rule. Advisor, institution and concept nodes are shared across the
temporal split; their representations (learnable `Embedding` for institution/concept,
message-passed state for advisor) are fit by backprop against **train** labels and then read
by **test** students through `advises` / `at` / `studies` edges. Measured exposure:

| discipline | test students with ≥1 train sibling | of those, sibling links **not closed** at the test freeze | AUC of *train-sibling mean y* → test y |
|---|---:|---:|---:|
| econ | 43.6 % | 65.6 % | 0.581 |
| math | 49.4 % | 62.8 % | 0.604 |
| physics | 49.2 % | 45.5 % | **0.661** |
| chemistry | 53.6 % | 38.6 % | 0.613 |
| neuro | 46.7 % | 53.2 % | 0.588 |

The same exemption applies to **concept** nodes, which is the larger channel: a concept
embedding fit on train labels is in effect a target encoder over the concept vocabulary, and
§5 shows it is the most predictive of the shared channels (concept-prior → test y, AUC-ROC
0.703 in physics).

Three observations. First, this is **not leakage in the ordinary sense** — no post-`t0+5`
*record* enters; it is ordinary supervised learning in which a node-level parameter is fit on
training labels. Standard transductive GNN practice permits it, and the paper's ceiling
models learn the same statistics in aggregate form.

Second, and decisively for the physics question, the *unclosed* fraction runs **opposite** to
the observed pattern: econ (65.6 %) and math (62.8 %) — the two clean disciplines — are the
most exposed, chemistry (38.6 %) the least.

Third, the concept channel was tested directly against the closure rule: rebuild the
concept-level label prior using **only** train rows whose 15-year window had already closed at
each test student's freeze (`t0_j + 15 <= t0_i + 5`), and see how much predictive power
survives.

| discipline | concept prior, all train labels | **closure-restricted only** | AUC lost | train rows usable under closure |
|---|---:|---:|---:|---:|
| econ | 0.6457 | 0.5790 | −0.0667 | 48.2 % |
| math | 0.5912 | 0.5598 | −0.0314 | 39.2 % |
| **physics** | 0.7032 | **0.6874** | **−0.0157** | 65.1 % |
| chemistry | 0.6770 | 0.6512 | −0.0258 | 69.0 % |
| neuro | 0.5968 | 0.5790 | −0.0178 | 57.2 % |

In physics the channel is almost entirely intact under closure (−0.0157 of 0.703, on 65 % of
train rows), i.e. it is built on labels that *were* observable at the test students' freeze
dates. In econ — a clean discipline — it degrades four times as much. **So the concept
channel, which is the graph's main advantage over the ceiling, is legitimate information in
exactly the disciplines where the graph wins.** That is evidence for, not against, the gain
being real.

The exemption remains an internal inconsistency worth stating in the paper: M5 is held to a
closure rule that the GNNs are not, which makes the M5-vs-GNN comparison slightly
GNN-favourable.

### F6 — the GAT receives `t0`, which no ceiling model receives  ·  protocol deviation  ·  minor

`h_extra_gnns.py:183-184` attaches `data["student"].t0_norm`, consumed only by `TGATLite`
(`:124-127`). `t0` is admissible under the contract (it is the window anchor, not a post-freeze
record), but `config.TABULAR_FEATURES` contains no `t0`, so M2/M3/M5 never see it — and the
test split is *defined by* `t0`, making it a direct cohort indicator. This affects
`gat_cohort_time` only. It does not explain physics (all four models cross there, and only
one has `t0`), and in math the same model is the worst performer (−0.0493).

*Minimal fix (not applied):* either drop the channel or add `t0` to the ceiling models' feature set.

### F7 — the e12 manifest under-declares the graph  ·  documentation defect  ·  minor

`code/make_e12_manifest.py:44-45` hardcodes

```python
"edge_types": ["advising (student-advisor)", "coauth_early (student-advisor)",
               "student-institution"],
```

omitting `student–studies→concept` and `advisor–studies→concept`, the two edge groups that
carry most of the graph's information (§5). The manifests shipped with the results therefore
describe a three-relation graph while the trained graph has five (ten with reverses). The
same manifest's `feature_cols` lists the eight M2/M3 tabular features, not the GNN's actual
3 student + 3 advisor node features. Cosmetic for correctness, material for an artifact
reviewer trying to reproduce the graph from the manifest.

### F8 — reproducibility gaps  ·  minor

**F8a.** `e12_results.zip` returned only `e12_hgt_vs_baselines.json` and the manifests — 15
entries, no per-seed model JSONs. `e2_hgt.py:222-225` writes per-seed `test_scores` and
`test_labels`, exactly what a student-level paired bootstrap (F4) needs, and cell 7 copies
only `e12_*.json` and `h_extra_gnns_*.json` from the results root, missing
`results_hgt/`, `results_hgt_grid/` and `results_extra_gnns/`. Consequence: **no e12 verdict
can currently be re-tested against a different ceiling or a different uncertainty model
without re-running the GPU leg.**

**F8b.** `e2_hgt.py:248` reads the parquet directly (`pd.read_parquet(args.data)`), bypassing
`utils/data.load_dataset`'s SHA-256 pin and leakage-invariant asserts. `h_extra_gnns.py:179`
does go through `load_dataset`. The notebook's cell 2 checks the SHA independently, so the
frozen tables *were* pinned for this run — but the HGT script alone offers no guarantee.

### F9 — `h_extra_gnns` is not "identical to E2", contrary to its own docstring  ·  protocol deviation  ·  major

`h_extra_gnns.py:15-22` asserts a "PRE-REGISTERED PROTOCOL (identical to E2, fixed before any
run)", and `:192` writes `"protocol": "E2-identical, standard config only"` into every result
file. Three concrete differences falsify that:

| | `e2_hgt.py` | `h_extra_gnns.py` |
|---|---|---|
| epochs / patience | 200 / 15 (`:43`) | **300 / 30** (`:56`) |
| optimizer | `Adam(lr, weight_decay=1e-4)` (`:184`) | `Adam(lr)` — **no weight decay** (`:138`) |
| class weight | train mask only (`:182`) | **all splits** (`:136`, = F2) |

A 1.5× larger training budget with 2× the patience and no L2 is a materially different — and
strictly more permissive — search. `rgcn` and `gat_cohort_time` carry the chemistry and neuro
verdicts, so the claim of protocol identity is load-bearing and currently false.

### F10 — `gat_cohort_time` does not implement the temporal encoding its docstring describes  ·  documentation defect  ·  minor

`h_extra_gnns.py:11-13` describes "the same graph with each edge carrying a time channel
`dt = t0(student) − year(edge event)`, passed through a sinusoidal time encoding concatenated
to messages (TGAT-style)". The implementation (`:118-128`) applies `TimeEnc` to a single
**node-level** scalar, `data["student"].t0_norm`, *after* both convolution layers. There is no
per-edge `dt` anywhere in the file, and edges carry no year. The reported label
`gat_cohort_time` correctly describes what was run; the docstring does not, and would
mislead anyone reading the artifact.

### F11 — sample inclusion is conditioned on post-freeze records  ·  design-level caveat  ·  informational

`build_neuro_dataset.py:163-168` drops a pair when
`max(years) − min(years) > MAX_CAREER_SPAN` (the student's **last** publication year, i.e.
whole career) or when `n_l < MIN_WORKS_PER_WINDOW` (**late-window** productivity). Both are
post-`t0+5` quantities used for cohort selection — for physics, `span_gt_max` removes 5 139
pairs and `sparse_windows` 6 480, out of 12 290 surviving the window filters
(`data/funnel_physics.json`). A model trained on the surviving rows is therefore conditioned
on the future in the selection sense.

This is arguably unavoidable — a label cannot be computed for a student with no late-window
works — and it is disclosed as counts in the funnel files. It is listed here for completeness
because it is a genuine post-freeze dependency, and because the funnel does not separate
early-window from late-window sparsity. **It cannot explain the graph result: every model,
graph and tabular alike, sees exactly the same surviving rows.**

### F12 — three incompatible ceiling definitions coexist in the codebase  ·  protocol deviation  ·  minor

1. `e12_hgt_vs_baselines.py:68` — compare against **M5, M3, M2** separately, student-level paired bootstrap. *(The pre-registered one.)*
2. notebook cell 6 `per_seed_ceiling` — **mean of per-seed `max(M2, M3)`**, seed-level bootstrap. *(The one that produced the five-discipline results.)*
3. `h_extra_gnns_summary.py:41-42` — **`max` of the seed-means** of M2 and M3.

(2) and (3) coincide whenever one component dominates on every seed, which holds in all five
tables here, so they are not in practical conflict — but they are not the same quantity, and
neither is (1). Relatedly, `h_extra_gnns_paired.py:51` hardcodes
`CEILING_COMPONENT = {"econ": "M2_logit_tabular", "neuro": "M3_gbdt_tabular"}`, which is
**wrong for econ** in these tables (M2 = 0.3430 < M3 = 0.3523, so the ceiling component is
M3). That script was not used for the suite results, so nothing published is affected, but it
would mislabel econ's ceiling if run.

### F13 — the 400-work fetch cap truncates the latest works  ·  data-quality caveat  ·  minor

`build_neuro_dataset.py:47,79` caps each author at `MAX_WORKS_PER_AUTHOR = 400`, fetched
`sort=publication_date:asc`, so the works that go missing are an author's **most recent**
ones. This cannot corrupt `early_concepts` or `adv_profile` (both are early-window sets and
the ascending sort keeps them), but it can truncate a hyper-prolific student's **late** window
and therefore their label. Physics is where this bites hardest — its top rows sit at the cap
(`early_prod` 396, `adv_early_prod` 400). The effect is label *noise*, which depresses every
model equally; it is noted so it is not mistaken later for a graph-side artifact.

### F14 — the graph arm selects on the temporal VAL cohort; the ceiling never does  ·  protocol deviation  ·  **major**

This is the second-largest comparability problem after F3/F4, and it is easy to miss.

The graph models early-stop on **VAL AUC-PR — the reported metric — evaluated on the true
temporal validation cohort**, once per epoch, keeping the best checkpoint:
`e2_hgt.py:201-209` (up to 200 checkpoints), `h_extra_gnns.py:155-158` (up to 300).
`hgt_tuned` adds a 16-config grid selected on mean VAL AUC-PR (notebook cell 4).

The ceiling models do **not**. `e1_baselines.py:68-70` (M3) and `:90-92` (M5) early-stop with
`HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.15)` — a *random
15 % subset of TRAIN*, on log-loss, never touching the VAL cohort. The VAL cohort enters M2/M3/M5
only through `S.best_f1_threshold` (`e1_baselines.py:71`), and `utils/stats.py:16-26` shows the
threshold affects `f1` alone — **`auc_pr` is threshold-free**, so the reported metric of every
ceiling model is chosen with zero exposure to VAL. M2 has no early stopping at all.

So the graph arm gets hundreds of checkpoint selections (and, for `hgt_tuned`, a grid) directly
optimising the scored metric on the cohort *temporally adjacent to test*, while the ceiling gets
its stopping point from a random slice of the *oldest* cohorts. Under temporal drift that is a
systematic advantage to the graph arm that has nothing to do with graph structure. It is
consistent with `hgt_tuned` being physics's best model (+0.0260, the largest gap in the suite),
and it applies to all four architectures in all five disciplines.

*Minimal fix (not applied):* give the tabular ceiling the same budget — early-stop M3/M5 on the
temporal VAL cohort's AUC-PR rather than an internal random split — before any "graph exceeds
ceiling" claim is made. This is a **CPU-only** change to `e1_baselines.py` and needs no GPU.

### Clean confirmations

- `profile()` bounds inclusive; early and late windows disjoint (`build_neuro_dataset.py:129-138, 165-166`).
- `early_concepts`, `adv_profile`, `y` reproduce exactly from raw works stores, 5/5 disciplines, 0 mismatches (§2.3).
- No late-window concept ever appears in a stored `early_concepts` list, 5/5 disciplines, 0 rows (§2.3).
- No citation edges of any kind in the graph; `st_cites_adv_broad` / `adv_cites_st_broad` unreachable.
- `institution_name` is the one graph input whose time-window compliance rests on the data
  model rather than on a check: the genealogy record carries no year (`link_dateadded` is the
  date the AFT *link* was entered, not the degree year). A PhD institution necessarily precedes
  the career, so the assumption is sound, but it is an assumption. Empirically the channel is
  near-useless anyway — institution bridging covers 87–96 % of test students but predicts test
  `y` at AUC-ROC 0.51–0.55 — so nothing in the verdict depends on it.
- Co-authorship edge uses `coauth_early` (year-filtered `[t0, t0+5]`), never the banned full-career `co_authored`; 0 missing values in all five frozen tables.
- `y` is attached to the graph but never read by any `forward()`; supervision only, under masks.
- Node-feature standardisation over train+val+test (`e2_hgt.py:87, 92`) is transductive but label-free — it uses feature moments, not labels. Standard practice, harmless here.
- HGT tuning selects on `val_auc_pr` only (notebook cell 4); test is never consulted during selection.
- The five-discipline builds share `build_neuro_dataset`'s stage functions verbatim; `build_dataset.py` moves only paths and labels. `MIN_WORKS_PER_WINDOW` is the sole movable parameter and was 3 for all five (confirmed against the frozen `dataset_summary_<field>.json` params blocks).
- BH correction is applied across the 4 models within a discipline, not across the 20 model×discipline hypotheses. Recomputing BH over all 20 raw p-values changes **no** verdict (every current "exceeds" survives at q < 0.05; physics 0.00651–0.00868, chemistry 0.00651/0.01953, neuro 0.00868). Multiple comparison is *not* the problem here — F4 is.

---

## 5. Per-discipline audit tables

The node/edge/feature provenance is identical across disciplines (§3) — one code path, one
builder, verified. What differs is graph *density*, which is the honest lever behind the
outcome pattern.

| | econ | math | **physics** | **chemistry** | **neuro** |
|---|---:|---:|---:|---:|---:|
| students | 2 697 | 5 051 | 11 809 | 26 830 | 21 844 |
| test students | 495 | 935 | 2 218 | 4 617 | 3 463 |
| advisors | 1 478 | 2 521 | 5 325 | 10 437 | 10 464 |
| concept vocabulary | 6 305 | 12 590 | 14 152 | 22 352 | 21 432 |
| student→concept edges | 26 128 | 48 514 | 114 900 | 259 011 | 209 604 |
| advisor→concept edges | 14 764 | 25 176 | 53 197 | 104 194 | 104 468 |
| **mean concept-node degree** | **6.49** | **5.85** | **11.88** | **16.25** | **14.65** |
| test concept vocab covered by train | 63.5 % | 66.0 % | 70.8 % | 78.0 % | 79.8 % |
| coauth edge rate | 58.3 % | 67.8 % | 78.8 % | 79.1 % | 76.3 % |
| P(c ∈ C_late \| c ∈ C_early) | 0.264 | 0.255 | **0.293** | 0.240 | 0.249 |
| concept-prior → test y (AUC-ROC) | 0.646 | 0.591 | **0.703** | 0.677 | 0.597 |
| … same, closure-restricted (F5) | 0.579 | 0.560 | **0.687** | 0.651 | 0.579 |
| advisor-profile-prior → test y (AUC-ROC) | 0.653 | 0.618 | **0.728** | 0.708 | 0.624 |
| train-sibling-mean-y → test y (AUC-ROC) | 0.581 | 0.604 | **0.661** | 0.613 | 0.588 |
| test students reaching a future sibling (F1b) | 9.3 % | 12.2 % | 20.3 % | 21.7 % | 11.5 % |
| test rows with an as-of-late advisor node (F1a) | 3.6 % | 4.5 % | 7.1 % | 7.0 % | 4.0 % |
| **graph-construction verdict** | **CLEAN** | **CLEAN** | **CLEAN** | **CLEAN** | **CLEAN** |
| defects present (all shared, none discipline-specific) | F1,F2,F5,F6,F9–F13 | ← | ← | ← | ← |
| e12 models crossing | 0 | 0 | 4 | 2 | 1 |

Every discipline is CLEAN on the temporal contract and on `BANNED_COLUMNS`. F1/F2/F5/F6 are
code-level defects present in all five identically; none is a per-discipline leak.

---

## 6. Why physics? — and is it the same construction defect feeding all four architectures?

**Yes, all four architectures share one `build_graph`** (`h_extra_gnns.py:181`), so a
construction defect would necessarily hit all four — which is precisely the pattern observed
in physics. That is why the suspicion was well-posed. But no such defect exists (§2, §3), and
the defects that do exist (F1, F2, F5) are ruled out individually because their
per-discipline ordering contradicts the outcome:

| candidate mechanism | discipline ordering | matches physics > chem > neuro > {econ, math}? |
|---|---|---|
| F1a advisor-feature as-of error (test %) | phys 7.1 ≈ chem 7.0 > math 4.5 > neuro 4.0 > econ 3.6 | **no** (math above neuro) |
| F1b 2-hop future-sibling reach (test %) | chem 21.7 > phys 20.3 > math 12.2 > neuro 11.5 > econ 9.3 | **no** (math above neuro) |
| F2 class-weight deviation (\|1 − ratio\|) | neuro .043 ≈ chem .042 > econ .033 > phys .027 > math .005 | **no** (physics near the bottom) |
| F5 unclosed sibling exposure | econ 65.6 % > math 62.8 % > neuro 53.2 % > phys 45.5 % > chem 38.6 % | **no** (exactly inverted) |
| F6 GAT `t0` channel | one model only | **no** (four models cross in physics) |
| F9 h_extra_gnns training budget | `rgcn`/`tgat` only | **no** (HGT crosses in physics by *more* than RGCN/GAT) |

The last row is worth stating on its own: in physics, **`hgt_tuned` (+0.0260) and `hgt`
(+0.0206) exceed by more than `rgcn` (+0.0193) and `gat_cohort_time` (+0.0120)**. HGT runs
through `e2_hgt.py`, which has none of F2, F6 or F9. That single ordering rules out every
`h_extra_gnns`-specific defect as the driver of the physics result.

What *does* order correctly is graph density and test-set size — i.e. how much a graph model
has to learn from, and how much statistical resolution the split affords:

- **Mean concept-node degree** separates the two groups cleanly: physics 11.88, chemistry
  16.25, neuro 14.65 (all crossing) vs math 5.85, econ 6.49 (both clean). At degree ≈ 6, most
  concept nodes are near-singletons and their learnable embeddings are essentially untrained
  noise for test students; at degree ≈ 12–16 they carry a usable, train-fit concept prior.
- **Concept-prior predictiveness** ranks physics first on both measures (0.703 / 0.728) — and
  physics also has the highest concept persistence, P(c ∈ C_late | c ∈ C_early) = 0.293. In
  physics, what you publish on early you keep publishing on, so concept identity — which the
  graph carries as edges and the ceiling models do not carry at all — is worth more there
  than anywhere else.
- **Test-set size** decides whether a small real gap clears the gates. econ's GAT has the
  suite's largest point gap against the pre-registered comparator (+0.0332 vs M5, §F3) and
  is still declared clean, because n_test = 495. physics has n_test = 2218 and a seed-level
  CI 5× narrower than the true sampling noise (§F4) — the combination is what makes four
  small, real, same-signed gaps all read as "significant".

So the honest reading of physics is: **a small genuine advantage from information the ceiling
was not given (concept identity + advisor neighbourhood), amplified into four "significant"
crossings by a comparator that excludes the one baseline holding that information (F3) and an
uncertainty interval that ignores test-set sampling error (F4).** Not leakage.

**A caveat on that reading, stated against interest.** One attempt was made to close the loop
by building a strictly leakage-free *tabular* model with the graph's own information channels
— M2/M3 features plus train-fit concept-identity priors (mean/max/min/coverage over
`early_concepts` and over `adv_profile`) plus advisor-neighbourhood aggregates under M5's
closure rule — and asking whether it reaches the GNN's number. **It did not.** Ten-seed test
AUC-PR:

| discipline | M6 (ad-hoc legal tabular) | vs ceiling | vs best graph model |
|---|---:|---:|---:|
| econ | 0.3193 ± 0.0150 | −0.0366 | −0.0529 |
| math | 0.3279 ± 0.0071 | −0.0800 | −0.0772 |
| physics | 0.6192 ± 0.0024 | **−0.0153** | −0.0413 |
| chemistry | 0.4758 ± 0.0038 | −0.0591 | −0.0840 |
| neuro | 0.3716 ± 0.0034 | −0.0541 | −0.0638 |

This is a single unoptimised attempt — eleven extra noisy columns handed to a GBDT will dilute
a strong tabular signal, and that is the likely reason it lands below even M2/M3. It therefore
does **not** establish that no legal tabular model can reproduce the graph gain; it only means
this audit did not produce one. The one weakly suggestive detail is that physics has by far the
smallest deficit (−0.0153 vs −0.037…−0.080 elsewhere), which is the direction the
concept-identity story predicts. Treat it as inconclusive, not as support.

---

## 7. Verdicts

| discipline | graph-construction verdict | basis |
|---|---|---|
| econ | **CLEAN** | §2.3 raw recomputation 0/0/0; §3 provenance |
| math | **CLEAN** | §2.3 raw recomputation 0/0/0; §3 provenance |
| physics | **CLEAN** | §2.3 raw recomputation 0/0/0; §3 provenance |
| chemistry | **CLEAN** | §2.3 raw recomputation 0/0/0; §3 provenance |
| neuro | **CLEAN** | §2.3 raw recomputation 0/0/0; §3 provenance |

**Top-level conclusion.** The physics (and chemistry, and neuro) crossings are **not a
leakage artifact**. There is no construction leak to remove. But they are also **not yet
established as a real graph signal**: they rest on a ceiling that omits the pre-registered
comparator M5 (F3), a confidence interval that omits test-set sampling error and is about 5×
too narrow in physics (F4), and a model-selection budget the ceiling models are not given
(F14 — the graph arm tunes on the temporal VAL cohort's AUC-PR, M3/M5 never see it).
Correcting any one weakens the claim; correcting all three leaves, on the present evidence,
**chemistry RGCN as the only crossing that clearly survives**, with physics `hgt_tuned`
marginal.

**This does not yet support branch B as stated.** It supports a narrower and defensible
version: *the graph adds no measurable gain in the two sparse-concept disciplines, and in the
dense-concept disciplines any gain is small, comparable to test-set sampling noise, and
largely reproduced by a tabular neighbourhood baseline (M5) that the e12 ceiling omitted.*

---

## 8. Recommended next steps (nothing applied; author's call)

No code was changed and no GPU job was run. In priority order:

0. **F14 — cheapest and highest value, CPU only, no GPU and no re-aggregation.** Early-stop
   M3 and M5 on the temporal VAL cohort's AUC-PR instead of `validation_fraction=0.15`
   (`e1_baselines.py:68-70, 90-92`). This gives the ceiling the selection budget the graph arm
   already has, and it moves the ceiling *upward*, tightening every e12 gap at once. Rerunning
   E1 is minutes of CPU per discipline.
1. **F3 / F4 — re-aggregate, no GPU needed if the per-seed JSONs can be recovered.** Restore
   M5 to the ceiling and switch to the student-level paired bootstrap already implemented in
   `e12_hgt_vs_baselines.py:104-117`. If the Colab session still holds
   `results_<field>/results_hgt/`, `results_hgt_grid/`, `results_extra_gnns/`, this is a pure
   re-aggregation. Otherwise it needs a GPU re-run purely to regenerate the score arrays.
   *Expected effect:* the physics crossings shrink to ~+0.003…+0.017 against M5 and most
   likely stop clearing a student-level interval; chemistry RGCN probably survives.
2. **F2 + F9 — one-line and three-constant fixes** in `h_extra_gnns.py`: `y[masks["train"]]`
   at `:136`, and `300/30` → `200/15` with `weight_decay=1e-4` at `:56`/`:138`, so the
   "identical to E2" claim becomes true. Removes a real if tiny label leak and a materially
   more permissive training budget from the two models that carry the chemistry and neuro
   verdicts. Requires a GPU re-run of `rgcn` / `tgat` to take effect.
3. **F1 — advisor route.** Key advisor nodes by `(advisor_pid, t0)`, or give each student a
   row-specific advisor feature vector and mask `advises` edges to siblings with
   `t0_j <= t0_i`, so the graph obeys M5's own rule. Requires a GPU re-run of all four models.
   *Expected effect:* small — both routes are bounded at 2–4 years and neither ordering tracks
   the outcome.
4. **F7, F10, F12 — documentation, no re-run.** Add the two concept edge types and the real
   GNN feature list to `make_e12_manifest.py:44-45`; correct the `h_extra_gnns.py:11-13`
   docstring to describe the node-level cohort channel actually implemented; fix
   `h_extra_gnns_paired.py:51` (econ's ceiling component is M3, not M2) or delete the hardcoded
   map in favour of a computed argmax.
5. **F5, F6, F11, F13 — disclose in the paper** rather than fix: the GNNs are not held to
   M5's sibling-closure rule; `gat_cohort_time` receives a cohort channel the ceiling models do
   not; cohort inclusion conditions on late-window productivity and full-career span; the
   400-work cap truncates the latest works of the most prolific authors.
6. **F8a — change cell 7** to archive the per-seed model directories, so the next run is
   re-analysable without GPU.

## 9. What this audit did not do

- Did not re-run any GPU training, and did not re-derive the GNN AUC-PR numbers.
- All five disciplines *were* recomputed from raw sources in §2.3, including neuroscience
  (its works store is not under `suite_data/` but at
  `Neuro Data/build_neuro/cache/works_store.jsonl`; all 29 379 required author ids resolved).
  No discipline was sampled — every row was checked.
- The F4 noise-floor figures are a **calibration** from an M3 − M2 paired student bootstrap on
  the same test sets, not the actual GNN-vs-ceiling interval, which cannot be computed without
  the per-seed score files (F8a).
- Did not touch `data/`, `paper/`, `results/`, or any frozen artifact.
