#!/usr/bin/env bash
# Local experiment battery for one suite discipline (the CPU-runnable ladder +
# certificates; the GNN leg e12 is GPU/Colab). Requires the field's frozen table
# built AND its SHA pinned in config._PINNED_SHA (else config raises by design).
# Outputs land in results_<field>/ via the DATASET override. 10 seeds + BH are
# baked into each experiment; decision rules live in each script's docstring.
#   usage: bash run_experiments.sh <field>   e.g. run_experiments.sh chemistry
field="$1"
export DATASET="$field"
cd artifact/code/paper_pipeline
# e7 (node2vec) before e6 (which optionally merges e7's embeddings). Continue on
# error so one experiment's failure does not block the rest; report a summary.
fail=""
for e in e1_baselines e10_advisor_placebo e14_self_persistence e7_node2vec e6_innovation_premium; do
  echo "=== [$field] $e ($(date)) ==="
  if ! python experiments/$e.py; then echo "!!! [$field] $e FAILED"; fail="$fail $e"; fi
done
if [ -n "$fail" ]; then echo "=== [$field] EXPERIMENTS_DONE_WITH_FAILURES:$fail ($(date)) ==="; else echo "=== [$field] EXPERIMENTS_DONE ($(date)) ==="; fi
