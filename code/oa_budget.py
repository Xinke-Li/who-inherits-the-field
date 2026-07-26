#!/usr/bin/env python3
"""Print the OpenAlex daily budget once, before a fetch stage (P0 requirement).

Reads OPENALEX_API_KEY from the ENVIRONMENT ONLY (never a tracked file). Prints
daily_budget_usd / daily_remaining_usd; saves the full /rate-limit response to
cache/openalex_budget_YYYYMMDD.json with the api_key field REDACTED. This only
PRINTS — it never gates or pauses a run (that decision is the operator's).
"""
import os, sys, json, datetime
import requests

KEY = os.environ.get("OPENALEX_API_KEY")
if not KEY:
    sys.exit("OPENALEX_API_KEY not set - export it before running.")
MAILTO = os.environ.get("OPENALEX_MAILTO", "lixinke@uchicago.edu")

r = requests.get("https://api.openalex.org/rate-limit",
                 params={"api_key": KEY, "mailto": MAILTO}, timeout=30)
r.raise_for_status()
j = r.json()
rl = j.get("rate_limit", {})
print(f"[budget] daily_budget_usd={rl.get('daily_budget_usd')} | "
      f"daily_used_usd={rl.get('daily_used_usd')} | "
      f"daily_remaining_usd={rl.get('daily_remaining_usd')} | "
      f"credits_remaining={rl.get('credits_remaining')} | "
      f"resets_in_seconds={rl.get('resets_in_seconds')}", flush=True)

j["api_key"] = "<redacted>"   # never persist the key
os.makedirs("cache", exist_ok=True)
day = datetime.datetime.utcnow().strftime("%Y%m%d")
out = os.path.join("cache", f"openalex_budget_{day}.json")
json.dump(j, open(out, "w"), indent=2)
print(f"[budget] wrote {out} (api_key redacted)", flush=True)
