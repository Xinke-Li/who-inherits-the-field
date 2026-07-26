#!/usr/bin/env python3
"""AFT genealogy -> <Field>_Pairs_Detailed.parquet  (field-parameterized).

Extracts, for one AFT major-area tree, the advisor->student training pairs and
their person metadata, in the SAME 28-column schema as Neuro_Pairs_Detailed.
This is the AFT-assembly step that precedes OpenAlex resolution
(resolve_by_pubid.py) and the modeling-table build (build_dataset.py).

PRE-REGISTERED ASSEMBLY RULE (identical to the neuroscience build; fixed before
any resolution/modeling so the cross-discipline design stays byte-comparable):
  * a pair is kept iff  connect.relation == 1  (a graduate advisor->student edge)
    AND the field token is a member of BOTH people's comma-split majorarea list
    ("both-ends" criterion). This is exactly what produced the frozen
    Neuro_Pairs_Detailed (120,347 rows); see --validate.
  * advisor_pid = connect.pid2, student_pid = connect.pid1,
    institution_name = connect.location (the degree-granting institution on the
    edge, not the person's current affiliation).
  * NO window/label/threshold choices happen here; those live downstream and are
    identical across disciplines. This script only selects rows and copies columns.

Field tokens are the literal AFT majorarea strings: econ, physics, chemistry,
math, neuro (confirmed against the 2026 snapshot).

Usage:
  python build_pairs_detailed.py --field math \
      --snap aft_2026_snapshot --out suite_data/math/Math_Pairs_Detailed.parquet
  # reproduce-check against the frozen neuro table (row/connection_id overlap):
  python build_pairs_detailed.py --field neuro --snap aft_2026_snapshot \
      --out suite_data/neuro/Neuro_Pairs_Detailed_2026.parquet \
      --validate "Neuro Data/Neuro_Pairs_Detailed.parquet"
"""
import argparse, csv, gzip, json, os
import pandas as pd

csv.field_size_limit(10**7)

# 28-column schema, order identical to Neuro_Pairs_Detailed.parquet
SCHEMA = ["connection_id", "advisor_pid", "student_pid", "dissertation_title",
          "link_dateadded", "institution_name",
          "adv_firstname", "adv_middlename", "adv_lastname", "adv_degrees",
          "adv_area", "adv_majorarea", "adv_hindex", "adv_orcid", "adv_s2id",
          "adv_homepage", "adv_location",
          "stu_firstname", "stu_middlename", "stu_lastname", "stu_degrees",
          "stu_area", "stu_majorarea", "stu_hindex", "stu_orcid", "stu_s2id",
          "stu_homepage", "stu_location"]

# people.majorarea token members for the filter
def _members(s):
    return {t.strip() for t in (s or "").split(",") if t.strip()}


def load_people(snap):
    """pid -> dict of the person fields the Detailed schema needs, + majorarea set."""
    path = os.path.join(snap, "people.csv.gz")
    people, members = {}, {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f); h = next(r)
        idx = {c: h.index(c) for c in
               ["pid", "firstname", "middlename", "lastname", "degrees", "area",
                "majorarea", "hindex", "orcid_id", "s2id", "homepage", "location"]}
        for row in r:
            pid = row[idx["pid"]]
            people[pid] = {
                "firstname": row[idx["firstname"]], "middlename": row[idx["middlename"]],
                "lastname": row[idx["lastname"]], "degrees": row[idx["degrees"]],
                "area": row[idx["area"]], "majorarea": row[idx["majorarea"]],
                "hindex": row[idx["hindex"]], "orcid": row[idx["orcid_id"]],
                "s2id": row[idx["s2id"]], "homepage": row[idx["homepage"]],
                "location": row[idx["location"]],
            }
            members[pid] = _members(row[idx["majorarea"]])
    return people, members


def build(field, snap):
    people, members = load_people(snap)
    print(f"[people] {len(people):,} persons loaded", flush=True)
    path = os.path.join(snap, "connect.csv.gz")
    rows = []
    n_rel1 = trainee_first = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f); h = next(r)
        i = {c: h.index(c) for c in
             ["cid", "pid1", "pid2", "relation", "location", "dissertation_title", "dateadded"]}
        for row in r:
            if row[i["relation"]] != "1":
                continue
            stu_pid, adv_pid = row[i["pid1"]], row[i["pid2"]]
            if not stu_pid or not adv_pid or adv_pid == "0" or stu_pid == "0":
                continue
            n_rel1 += 1
            sm, am = members.get(stu_pid, set()), members.get(adv_pid, set())
            # trainee-main-tree diagnostic (first token) — matches prompt's raw counts
            stu_first = next(iter([t.strip() for t in
                            (people.get(stu_pid, {}).get("majorarea") or "").split(",") if t.strip()]), None)
            if stu_first == field:
                trainee_first += 1
            if field not in sm or field not in am:   # both-ends criterion
                continue
            a = people.get(adv_pid, {}); s = people.get(stu_pid, {})
            rows.append({
                "connection_id": row[i["cid"]], "advisor_pid": adv_pid, "student_pid": stu_pid,
                "dissertation_title": row[i["dissertation_title"]],
                "link_dateadded": row[i["dateadded"]], "institution_name": row[i["location"]],
                "adv_firstname": a.get("firstname"), "adv_middlename": a.get("middlename"),
                "adv_lastname": a.get("lastname"), "adv_degrees": a.get("degrees"),
                "adv_area": a.get("area"), "adv_majorarea": a.get("majorarea"),
                "adv_hindex": a.get("hindex"), "adv_orcid": a.get("orcid"),
                "adv_s2id": a.get("s2id"), "adv_homepage": a.get("homepage"),
                "adv_location": a.get("location"),
                "stu_firstname": s.get("firstname"), "stu_middlename": s.get("middlename"),
                "stu_lastname": s.get("lastname"), "stu_degrees": s.get("degrees"),
                "stu_area": s.get("area"), "stu_majorarea": s.get("majorarea"),
                "stu_hindex": s.get("hindex"), "stu_orcid": s.get("orcid"),
                "stu_s2id": s.get("s2id"), "stu_homepage": s.get("homepage"),
                "stu_location": s.get("location"),
            })
    df = pd.DataFrame(rows, columns=SCHEMA)
    # The AFT connect dump carries a handful of exact-duplicate cid rows; the frozen
    # Neuro_Pairs_Detailed is unique on connection_id, so dedupe to match (per-
    # connection uniqueness; a student may still appear under several advisors).
    n_pre = len(df)
    df = df.drop_duplicates("connection_id", keep="first").reset_index(drop=True)
    return df, {"field": field, "relation1_pairs_total": n_rel1,
                "trainee_first_pairs": trainee_first,
                "both_ends_rows_pre_dedup": n_pre, "both_ends_pairs": len(df)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--snap", default="aft_2026_snapshot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate", default=None,
                    help="path to an existing *_Pairs_Detailed.parquet to compare against")
    args = ap.parse_args()

    df, meta = build(args.field, args.snap)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out, index=False)
    meta_path = os.path.splitext(args.out)[0] + "_counts.json"
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"[build] field={args.field}  relation1={meta['relation1_pairs_total']:,}  "
          f"trainee_first={meta['trainee_first_pairs']:,}  both_ends(rows)={meta['both_ends_pairs']:,}", flush=True)
    print(f"[build] wrote {args.out}  (+ {os.path.basename(meta_path)})", flush=True)

    if args.validate:
        ref = pd.read_parquet(args.validate, columns=["connection_id"])
        ref_ids = set(ref["connection_id"].astype(str))
        new_ids = set(df["connection_id"].astype(str))
        inter = ref_ids & new_ids
        print(f"[validate] reference rows={len(ref_ids):,}  new rows={len(new_ids):,}  "
              f"overlap={len(inter):,}  ref_only={len(ref_ids-new_ids):,}  new_only={len(new_ids-ref_ids):,}")
        print(f"[validate] overlap/reference = {len(inter)/max(len(ref_ids),1):.4f} "
              f"(1.0 minus a small AFT snapshot-drift delta is the pass condition)")


if __name__ == "__main__":
    main()
