"""E13 - OpenAlex economics population marginals for Appendix B (Q16).

Fills the AFT-vs-OpenAlex coverage comparison with two one-call aggregate
queries (run OUTSIDE the sandbox; needs api.openalex.org):

  B.1  economics works per publication decade, 1950-2011 (field-activity
       benchmark for the sample's t0-decade distribution);
  B.2  top institutions by economics-work count (institutional concentration
       benchmark for the sample's PhD-institution shares).

Usage: python experiments/e13_openalex_population.py --email you@uni.edu
Output: results_econ/e13_openalex_population.json
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

ECON = "C162324750"  # OpenAlex concept: Economics


def q(params):
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    args = ap.parse_args()

    by_year = q({"filter": f"concepts.id:{ECON},publication_year:1950-2011",
                 "group_by": "publication_year", "mailto": args.email})
    decade = {}
    for g in by_year["group_by"]:
        d = int(g["key"]) // 10 * 10
        decade[f"{d}s"] = decade.get(f"{d}s", 0) + g["count"]
    total = sum(decade.values())
    decade_share = {k: round(v / total, 4) for k, v in sorted(decade.items())}

    by_inst = q({"filter": f"concepts.id:{ECON},publication_year:1950-2011",
                 "group_by": "authorships.institutions.lineage",
                 "per-page": "100", "mailto": args.email})
    inst = [{"institution": g["key_display_name"], "count": g["count"]}
            for g in by_inst["group_by"][:25]]

    out = {"experiment": "E13_openalex_population",
           "note": ("Works-level field marginals. The sample-side comparison in "
                    "Appendix B uses author first-publication cohorts; these "
                    "works-per-decade shares benchmark overall field activity, "
                    "which is the closest population marginal OpenAlex exposes "
                    "in aggregate queries."),
           "econ_works_share_by_decade": decade_share,
           "top_institutions_by_econ_works": inst}
    (C.RESULTS_DIR / "e13_openalex_population.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
