#!/usr/bin/env bash
# Resolve one suite discipline's AFT persons to OpenAlex ids via pub-ID anchoring.
# Reads the OpenAlex key from the UNTRACKED .openalex_key into the env only (never
# written anywhere tracked). Uses the pre-extracted slim authorPub parquet, so no
# 1.6 GB re-scan. Resumable: rerun the same command and the wcache continues.
#   usage: bash resolve_field.sh <field> <Cap>   e.g. resolve_field.sh chemistry Chemistry
set -e
field="$1"; Cap="$2"
export OPENALEX_API_KEY="$(tr -d '[:space:]' < .openalex_key)"
export OPENALEX_MAILTO="lixinke@uchicago.edu"
slim="suite_data/aft_authorpub_slim.parquet"
det="suite_data/${field}/${Cap}_Pairs_Detailed.parquet"
wc="suite_data/${field}/works_${field}.jsonl"
out="suite_data/${field}/${Cap}_Pairs_Openalex_Ultimate.parquet"

echo "=== [$field] budget before run ==="
python oa_budget.py || true
echo "=== [$field] resolve FETCH ($(date)) ==="
python -u "Neuro Data/resolve_by_pubid.py" fetch --detailed "$det" --pubs "$slim" --wcache "$wc"
echo "=== [$field] resolve APPLY ($(date)) ==="
python -u "Neuro Data/resolve_by_pubid.py" apply --detailed "$det" --pubs "$slim" --wcache "$wc" --out "$out"
echo "=== [$field] RESOLVE_DONE ($(date)) ==="
