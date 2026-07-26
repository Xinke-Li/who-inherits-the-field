#!/usr/bin/env bash
# Full modeling-table build for one suite discipline: fetch->table->coauth->build
# ->outcomes, then late_overlap recompute assert + freeze to artifact/code/data_<field>.
# Key read from the UNTRACKED .openalex_key into env only. Resumable (rerun continues).
#   usage: bash build_field.sh <field>   e.g. build_field.sh chemistry
set -e
field="$1"
export OPENALEX_API_KEY="$(tr -d '[:space:]' < .openalex_key)"
export OPENALEX_MAILTO="lixinke@uchicago.edu"
echo "=== [$field] budget before run ==="
python oa_budget.py || true
echo "=== [$field] BUILD all ($(date)) ==="
python -u build_dataset.py --field "$field" all
echo "=== [$field] BUILD_DONE ($(date)) ==="
