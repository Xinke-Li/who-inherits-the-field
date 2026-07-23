# Graph-model leg on GPU

The graph models run on a GPU through Colab. This notebook trains them and writes,
per discipline, a comparison against the tabular ceiling. The canonical run is
complete: its per-seed artifacts are committed under `results/results_<field>/`,
the corrected aggregation (`code/e12_corrected_aggregation.py`) reruns on CPU from
those artifacts, and the paper reports the resulting verdicts. Rerunning this
notebook is needed only to regenerate the per-seed artifacts themselves.

## One upload, no directory building

The old flow asked for sixteen files placed by hand. This flow asks for one archive.
Build it once from the repository, then put it on Drive.

```
python code/make_colab_bundle.py
```

That writes `colab/e12_colab_bundle.zip`, which already contains the code and, for
each discipline, the frozen table at the exact path `config` expects
(`data_<field>/clean_dataset.parquet`), its outcomes, the tabular baselines in
`paper_pipeline/results_<field>/e1_baselines.json`, and the split manifest. The
builder runs a dry check first: it imports `config` for every discipline and refuses
to write the zip unless every input resolves, so a broken bundle never reaches Colab.

Place `e12_colab_bundle.zip` on Drive at `MyDrive/e12_colab_bundle.zip` (uploading it
to the session at `/content/` also works), open `e12_gnn.ipynb` in Colab, and choose
a GPU runtime. The notebook unzips the bundle, enters the code tree, and reads
everything by discipline through the `DATASET` environment variable. There is nothing
to arrange by hand.

## Live persistence and resume

Colab's runtime disk is temporary; a recycled runtime deletes everything on it. The
notebook therefore copies every finished per-seed json to `MyDrive/e12_out` the
moment it is written (the folder can be overridden with the `E12_PERSIST` env var),
and the aggregation and final zip land there too. If the session disconnects or the
runtime is recycled, reconnect and run the same notebook again from the top: Cell 1
copies the finished results back from Drive, and every training loop skips seeds
whose json already exists, so the run resumes where it stopped instead of starting
over. Nothing is lost with a lost runtime.

## Models

For each discipline the notebook trains HGT in a standard and a tuned configuration,
RGCN, and a cohort-time GAT, at 10 seeds, by calling the frozen scripts `e2_hgt.py`
and `h_extra_gnns.py`. The architectures and hyperparameters are the pipeline's, not
a reimplementation. HGT tuning is two-stage: each of the sixteen grid configs runs
two selection seeds, ranked by mean validation AUC-PR, then the single winner runs to
the full ten seeds, reusing the seeds already computed. Each graph model is compared
to the tabular ceiling, the seed-wise maximum of M2 and M3 from `e1_baselines.json`,
by a paired test with Benjamini-Hochberg correction across the model family. A
discipline's ceiling stands unless a graph model clears all three pre-registered
gates: its best seed-mean beats the ceiling, the paired test is significant after
correction, and the bootstrap interval on the difference excludes zero.

## Run order

Run math and chemistry first: math is smallest and confirms the pipeline end to end,
chemistry is largest and slowest so starting it early overlaps its runtime with the
rest. Physics, economics, and neuroscience follow.

## Output and backfill

The last cell writes `e12_full_results.zip` and downloads it; a copy also lands in
`MyDrive/e12_out`, so the full set exists on Drive even if the download is skipped.
The archive holds the complete `results_<field>/` tree for every discipline: all
per-seed jsons from `results_hgt/`, `results_hgt_grid/`, and `results_extra_gnns/`,
plus `e1_baselines.json` and the `e12_hgt_vs_baselines.json` aggregate. Every
per-seed file carries, next to the test AUC-PR and AUC-ROC and the validation
AUC-PR, the per-student `test_scores` and `test_labels` arrays. Those arrays are
what a score-level paired bootstrap over test students, or a re-judgment against a
different ceiling (for example one that includes M5), needs, so any such
re-aggregation can run later on CPU without another GPU pass. Unzip the archive over
the repository's `results/` layout to backfill.

## Note

All five disciplines, economics included, use the same 2026 snapshot and the same
pub-ID resolver, so the five columns are one protocol. If you cannot run the tuned
HGT grid, the standard HGT, RGCN, and GAT comparisons still complete on their own.
