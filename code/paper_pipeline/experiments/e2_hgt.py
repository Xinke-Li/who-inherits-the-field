"""E2/E3 - HGT on the leakage-free heterogeneous graph. GPU script (Colab).

STANDALONE by design: no repo imports, so it can be dropped into Colab as-is.

Usage (one run = one seed x one graph variant):
    python e2_hgt.py --data /content/drive/MyDrive/who-inherits-the-field/clean_dataset.parquet \
                     --out  /content/drive/MyDrive/who-inherits-the-field/results_hgt \
                     --seed 0 --ablate none
    --ablate: none | social | student_concept | advisor_concept | coauth | institution | advising

Graph schema (ALL edges observable at t0+5 - no label-window information):
    student  --studies-->   concept       (early_concepts, W_E)
    advisor  --studies-->   concept       (adv_profile, works <= t0+5)
    advisor  --advises-->   student       (genealogy, predetermined)
    student  --at-->        institution   (PhD institution)
    student  --coauth-->    advisor       (coauth_early only, W_E)
Ablation groups:
    social          = advises + at + coauth        (the IC2S2 headline claim, redone properly)
    student_concept / advisor_concept / coauth / institution / advising = single groups

Protocol: temporal cohort split (60/20/20 by t0 quantiles), early stopping on VAL
AUC-PR only, test evaluated once after training. One JSON per run ->
aggregate locally with e3_aggregate.py (paired Wilcoxon + BH across seeds).
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd


# ---------------- config (mirrors paper_pipeline/config.py) ----------------
SPLIT_QUANTILES = (0.6, 0.8)
TABULAR_ST = ["early_prod", "early_breadth", "early_overlap"]
TABULAR_ADV = ["adv_early_prod", "adv_early_breadth", "adv_career_age_at_t0"]
# defaults = the "standard configuration"; a val-selected grid over
# HP_GRID (see colab_hgt_runner.md) answers the under-tuning critique:
# lr x hidden x layers x dropout, best config chosen on VAL AUC-PR only,
# then re-run over the full seed grid. Test is never touched during selection.
HID, HEADS, LAYERS = 64, 4, 2
LR, EPOCHS, PATIENCE = 1e-3, 200, 15
DROPOUT = 0.0
HP_GRID = {"lr": [1e-3, 5e-3], "hidden": [64, 128],
           "layers": [2, 3], "dropout": [0.0, 0.5]}

ABLATIONS = {
    "none": set(),
    "social": {("advisor", "advises", "student"), ("student", "at", "institution"),
               ("student", "coauth", "advisor")},
    "advising": {("advisor", "advises", "student")},
    "institution": {("student", "at", "institution")},
    "coauth": {("student", "coauth", "advisor")},
    "student_concept": {("student", "studies", "concept")},
    "advisor_concept": {("advisor", "studies", "concept")},
    # only meaningful under contract="strict_lineage"; a no-op elsewhere
    "lineage": {("advisor", "mentored_by", "advisor"),
                ("advisor", "studies_lineage", "concept")},
    "lineage_concept": {("advisor", "studies_lineage", "concept")},
    "lineage_ancestry": {("advisor", "mentored_by", "advisor")},
}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_graph(df, ablate: str):
    """HeteroData with train/val/test masks on student nodes."""
    import torch
    from torch_geometric.data import HeteroData

    students = df.student_pid.tolist()
    advisors = sorted(df.advisor_pid.unique())
    institutions = sorted(df.institution_name.dropna().unique())
    concepts = sorted({c for l in df.early_concepts for c in l}
                      | {c for l in df.adv_profile for c in l})
    s_ix = {p: i for i, p in enumerate(students)}
    a_ix = {p: i for i, p in enumerate(advisors)}
    i_ix = {p: i for i, p in enumerate(institutions)}
    c_ix = {p: i for i, p in enumerate(concepts)}

    data = HeteroData()

    # node features: standardized tabular for student/advisor; learnable for inst/concept
    Xs = df[TABULAR_ST].astype(float).values
    Xs = (Xs - Xs.mean(0)) / np.clip(Xs.std(0), 1e-9, None)
    data["student"].x = torch.tensor(Xs, dtype=torch.float)

    adv_tab = (df.groupby("advisor_pid")[TABULAR_ADV].first().reindex(advisors))
    Xa = adv_tab.astype(float).values
    Xa = (Xa - Xa.mean(0)) / np.clip(Xa.std(0), 1e-9, None)
    data["advisor"].x = torch.tensor(Xa, dtype=torch.float)
    data["institution"].num_nodes = len(institutions)
    data["concept"].num_nodes = len(concepts)

    def add_edges(src_t, rel, dst_t, pairs):
        if not pairs:
            return
        e = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        data[(src_t, rel, dst_t)].edge_index = e
        data[(dst_t, f"rev_{rel}", src_t)].edge_index = e.flip(0)

    skip = ABLATIONS[ablate]
    if ("student", "studies", "concept") not in skip:
        add_edges("student", "studies", "concept",
                  [(s_ix[r.student_pid], c_ix[c]) for r in df.itertuples()
                   for c in r.early_concepts])
    if ("advisor", "studies", "concept") not in skip:
        pairs = {(a_ix[r.advisor_pid], c_ix[c]) for r in df.itertuples()
                 for c in r.adv_profile}
        add_edges("advisor", "studies", "concept", sorted(pairs))
    if ("advisor", "advises", "student") not in skip:
        add_edges("advisor", "advises", "student",
                  [(a_ix[r.advisor_pid], s_ix[r.student_pid]) for r in df.itertuples()])
    if ("student", "at", "institution") not in skip:
        add_edges("student", "at", "institution",
                  [(s_ix[r.student_pid], i_ix[r.institution_name]) for r in df.itertuples()
                   if pd.notna(r.institution_name)])
    if ("student", "coauth", "advisor") not in skip:
        add_edges("student", "coauth", "advisor",
                  [(s_ix[r.student_pid], a_ix[r.advisor_pid]) for r in df.itertuples()
                   if r.coauth_early])

    data["student"].y = torch.tensor(df.y.values, dtype=torch.long)
    for name in ("train", "val", "test"):
        data["student"][f"{name}_mask"] = torch.tensor((df.split == name).values)
    return data


def build_graph_v2(df, ablate: str, contract: str = "strict", lineage=None):
    """Graph construction closing audit findings F1a and F1b.

    contract="legacy" delegates to build_graph, byte-for-byte the frozen path,
    retained so the two can be compared in one run.

    contract="strict" changes exactly two things about the advisor route.

    F1a, the .first() collapse. build_graph keys advisor nodes by advisor_pid and
    takes one feature vector per advisor via groupby.first(), whose donor is the
    highest-early_prod student and is unrelated to the focal student's freeze
    date. The frozen table already carries adv_early_prod, adv_early_breadth and
    adv_career_age_at_t0 computed per row, as-of that row's own t0+5, so the
    correct values are present and were being discarded. Here advisor nodes are
    keyed by (advisor_pid, t0) and take that cohort's own values.

    F1b, the future-sibling route. With two message-passing layers, keying by
    advisor alone lets a test student reach same-advisor siblings whose windows
    close later than its own, for 9.3 to 21.7 percent of test students depending
    on discipline. utils/data.build_nfa_features forbids M5 exactly this via
    prior = (t0_j <= t0_i). Keying by (advisor_pid, t0) confines the two-hop
    neighbourhood to the student's own cohort, and an explicit
    student--sibling--student relation restores earlier cohorts only, edges
    directed from the prior sibling to the focal student, so the graph obeys M5's
    own rule instead of being exempt from it.

    Everything else, node sets, concept and institution edges, features and
    masks, is identical to build_graph.

    contract="strict_lineage" is the strict construction plus the multi-
    generation genealogy the graph arm has never consumed (reviewer B1, task
    T2.2b). It requires the `lineage` table written by r31_lineage_table.py and
    adds exactly two relations:

      advisor --mentored_by--> advisor      the AFT parent of the focal
                                            student's advisor, keyed to the
                                            focal cohort
      advisor --studies_lineage--> concept  that person's top-10 concepts from
                                            works dated at or before the FOCAL
                                            student's t0+5

    Grand-advisors enter the existing advisor node type, so the ancestry
    relation really is advisor--advisor, and they are keyed by
    (grand_adv_pid, t0) on the focal student's t0, the same cohort keying F1a
    imposes on direct advisors. A grand-advisor therefore contributes a
    different node, with a different concept profile, to each cohort they sit
    above. Nothing a cohort can reach is dated after that cohort's own t0+5.

    The direct-advisor feature matrix is left numerically identical to the
    strict arm: the standardization mean and standard deviation are computed
    over the direct-advisor keys alone and then applied to the grand-advisor
    rows, so a difference between the two arms cannot be a renormalization
    artifact. Grand-advisor-only nodes take the cache-derived analogues of
    TABULAR_ADV, and zeros where no works are cached.

    The ancestry edge carries no year stamp. Its compliance with the time
    contract rests on the assumption that a person's own doctoral advisor
    predates them, which is a data-model assumption rather than a checkable
    constraint, the same standing student --at--> institution has.
    """
    import torch
    from torch_geometric.data import HeteroData

    if contract == "legacy":
        return build_graph(df, ablate)
    if contract not in ("strict", "f1a", "f1b", "strict_lineage"):
        raise ValueError(
            f"contract must be one of 'legacy', 'strict', 'f1a', 'f1b', "
            f"'strict_lineage', got {contract!r}")

    # Rule: a missing input is an error, never something to adapt around. A
    # lineage run that silently produced the strict graph would be reported as
    # "lineage changes nothing", which is a false negative dressed as a result.
    lineage_on = contract == "strict_lineage"
    if lineage_on:
        if lineage is None:
            raise ValueError(
                "contract='strict_lineage' requires lineage=<DataFrame from "
                "r31_lineage_table.py>. Refusing to build the strict graph "
                "under a lineage label.")
        need = {"student_pid", "grand_adv_pid", "grand_adv_concepts",
                "grand_adv_early_prod", "grand_adv_early_breadth",
                "grand_adv_career_age_at_t0"}
        miss = need - set(lineage.columns)
        if miss:
            raise ValueError(f"lineage table is missing columns {sorted(miss)}")
        lin = lineage.set_index("student_pid")
        if not lin.index.is_unique:
            raise ValueError(
                "lineage table has duplicate student_pid rows; the row-to-node "
                "mapping would be ambiguous.")
        absent = set(df.student_pid) - set(lin.index)
        if absent:
            raise ValueError(
                f"lineage table covers {len(lin)} students but {len(absent)} of "
                f"this split's students are absent from it, first few "
                f"{sorted(absent)[:3]}. Rebuild with r31_lineage_table.py.")
    elif lineage is not None:
        raise ValueError(
            f"lineage= was supplied but contract={contract!r} would ignore it. "
            f"Refusing to run: pass contract='strict_lineage' or drop the table.")

    # F1 has two halves and the strict construction fixes both at once:
    #   F1a  advisor nodes keyed by (advisor_pid, t0) instead of .first()
    #   F1b  sibling reachability restricted to prior cohorts
    # "f1a" and "f1b" isolate one half each, so the +0.0105 attributed to F1 can
    # be split into an information gain and a masking loss.
    #   f1a: correct per-cohort advisor features, but sibling reachability
    #        deliberately restored to ALL cohorts, matching legacy exposure.
    #   f1b: legacy advisor keying and .first() features, but sibling
    #        reachability restricted to prior cohorts only.
    fix_features = contract in ("strict", "f1a", "strict_lineage")
    prior_only = contract in ("strict", "f1b", "strict_lineage")

    students = df.student_pid.tolist()
    # F1a: an advisor becomes one node per cohort they supervised.
    if fix_features:
        adv_keys = sorted({(a, int(t)) for a, t in zip(df.advisor_pid, df.t0)})
    else:
        adv_keys = sorted({(a, 0) for a in df.advisor_pid})
    n_direct_adv = len(adv_keys)

    # T2.2b. Grand-advisors join the advisor node type under the same cohort
    # keying, appended after the direct keys so rows 0..n_direct_adv-1 of the
    # feature matrix stay in the strict arm's order.
    g_pid = g_con = None
    g_tab = None
    if lineage_on:
        g_pid = list(lin["grand_adv_pid"].reindex(df.student_pid))
        g_con = list(lin["grand_adv_concepts"].reindex(df.student_pid))
        g_tab = lin[["grand_adv_early_prod", "grand_adv_early_breadth",
                     "grand_adv_career_age_at_t0"]] \
            .reindex(df.student_pid).values.astype(float)
        lin_keys = sorted({(gp, int(t)) for gp, t in zip(g_pid, df.t0)
                           if isinstance(gp, str)})
        seen = set(adv_keys)
        adv_keys = adv_keys + [k for k in lin_keys if k not in seen]

    institutions = sorted(df.institution_name.dropna().unique())
    concept_set = ({c for l in df.early_concepts for c in l}
                   | {c for l in df.adv_profile for c in l})
    concepts = sorted(concept_set)
    if lineage_on:
        # Append lineage-only concepts rather than merging and re-sorting, so
        # every concept the strict arm has keeps its index and the two arms'
        # student and advisor concept edges are literally the same tensors.
        extra = {c for l in g_con if isinstance(l, (list, np.ndarray))
                 for c in l} - concept_set
        concepts = concepts + sorted(extra)
    s_ix = {p: i for i, p in enumerate(students)}
    a_ix = {k: i for i, k in enumerate(adv_keys)}
    i_ix = {p: i for i, p in enumerate(institutions)}
    c_ix = {p: i for i, p in enumerate(concepts)}

    data = HeteroData()

    Xs = df[TABULAR_ST].astype(float).values
    Xs = (Xs - Xs.mean(0)) / np.clip(Xs.std(0), 1e-9, None)
    data["student"].x = torch.tensor(Xs, dtype=torch.float)

    # F1a: per-(advisor, cohort) features, taken from the rows of that cohort.
    def akey(r):
        return (r.advisor_pid, int(r.t0)) if fix_features else (r.advisor_pid, 0)

    rows_by_key = {}
    for r in df.itertuples():
        # legacy keying reproduces .first(): the donor is the first row in
        # parquet order for that advisor, which is early_prod-descending
        if fix_features:
            rows_by_key.setdefault(akey(r), []).append(
                [float(getattr(r, c)) for c in TABULAR_ADV])
        else:
            rows_by_key.setdefault(akey(r), [
                [float(getattr(r, c)) for c in TABULAR_ADV]])
    if lineage_on:
        # Grand-advisor-only keys have no row in the frozen table, so their
        # three features come from the cache-derived analogues r31 computed
        # as-of the focal cohort's t0+5. A key that is ALSO a direct advisor
        # keeps the frozen table's values untouched: appending the cache
        # triple to an existing key would move that node's mean and change the
        # strict arm's own features, which is what this branch must not do.
        direct = set(adv_keys[:n_direct_adv])
        for gp, t, tab in zip(g_pid, df.t0, g_tab):
            if isinstance(gp, str) and (gp, int(t)) not in direct:
                rows_by_key.setdefault((gp, int(t)), []).append(list(tab))
    Xa = np.array([np.mean(rows_by_key[k], axis=0) for k in adv_keys], dtype=float)
    # Standardize on the direct-advisor block alone and apply that transform to
    # the grand-advisor rows, so the direct-advisor features are numerically
    # identical to the strict arm and a lineage difference cannot be a
    # renormalization artifact.
    ref = Xa[:n_direct_adv]
    Xa = (Xa - ref.mean(0)) / np.clip(ref.std(0), 1e-9, None)
    data["advisor"].x = torch.tensor(Xa, dtype=torch.float)
    data["institution"].num_nodes = len(institutions)
    data["concept"].num_nodes = len(concepts)

    def add_edges(src_t, rel, dst_t, pairs):
        if not pairs:
            return
        e = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        data[(src_t, rel, dst_t)].edge_index = e
        data[(dst_t, f"rev_{rel}", src_t)].edge_index = e.flip(0)

    skip = ABLATIONS[ablate]
    if ("student", "studies", "concept") not in skip:
        add_edges("student", "studies", "concept",
                  [(s_ix[r.student_pid], c_ix[c]) for r in df.itertuples()
                   for c in r.early_concepts])
    if ("advisor", "studies", "concept") not in skip:
        pairs = {(a_ix[akey(r)], c_ix[c])
                 for r in df.itertuples() for c in r.adv_profile}
        add_edges("advisor", "studies", "concept", sorted(pairs))
    if ("advisor", "advises", "student") not in skip:
        add_edges("advisor", "advises", "student",
                  [(a_ix[akey(r)], s_ix[r.student_pid])
                   for r in df.itertuples()])
    if ("student", "at", "institution") not in skip:
        add_edges("student", "at", "institution",
                  [(s_ix[r.student_pid], i_ix[r.institution_name])
                   for r in df.itertuples() if pd.notna(r.institution_name)])
    if ("student", "coauth", "advisor") not in skip:
        add_edges("student", "coauth", "advisor",
                  [(s_ix[r.student_pid], a_ix[akey(r)])
                   for r in df.itertuples() if r.coauth_early])

    # F1b: prior-cohort siblings only, directed prior -> focal.
    by_adv = {}
    for r in df.itertuples():
        by_adv.setdefault(r.advisor_pid, []).append((int(r.t0), r.student_pid))
    sib = []
    for adv, lst in by_adv.items():
        lst.sort()
        for t_i, p_i in lst:
            for t_j, p_j in lst:
                if p_j == p_i:
                    continue
                # prior_only is F1b; without it, restore legacy exposure so that
                # f1a isolates the feature-timing fix alone
                if prior_only and t_j > t_i:
                    continue
                sib.append((s_ix[p_j], s_ix[p_i]))
    if sib:
        e = torch.tensor(sorted(set(sib)), dtype=torch.long).t().contiguous()
        data[("student", "sibling", "student")].edge_index = e

    # T2.2b: the two lineage relations. Both are keyed to the focal cohort, so
    # neither can reach a work dated after that cohort's t0+5.
    if lineage_on:
        anc, lcon, self_anc = set(), set(), 0
        for r, gp, cs in zip(df.itertuples(), g_pid, g_con):
            if not isinstance(gp, str):
                continue
            gk = (gp, int(r.t0))
            ak = akey(r)
            if gk == ak:
                self_anc += 1          # advisor is their own recorded parent
            elif ("advisor", "mentored_by", "advisor") not in skip:
                anc.add((a_ix[ak], a_ix[gk]))
            if ("advisor", "studies_lineage", "concept") not in skip:
                for c in (cs if isinstance(cs, (list, np.ndarray)) else []):
                    lcon.add((a_ix[gk], c_ix[c]))
        add_edges("advisor", "mentored_by", "advisor", sorted(anc))
        add_edges("advisor", "studies_lineage", "concept", sorted(lcon))
        data.lineage_stats = {
            "direct_advisor_nodes": int(n_direct_adv),
            "grand_advisor_only_nodes": int(len(adv_keys) - n_direct_adv),
            "ancestry_edges": int(len(anc)),
            "lineage_concept_edges": int(len(lcon)),
            "self_ancestry_rows_dropped": int(self_anc),
            "rows_with_ancestry": int(sum(1 for gp in g_pid
                                          if isinstance(gp, str))),
            "n_rows": int(len(df)),
        }

    data["student"].y = torch.tensor(df.y.values, dtype=torch.long)
    for name in ("train", "val", "test"):
        data["student"][f"{name}_mask"] = torch.tensor((df.split == name).values)

    # F6's cohort-time channel. h_extra_gnns.py:183-184 attaches it in the
    # frozen path, so every construction here must too, or the attention model
    # silently loses the input that makes it gat_cohort_time. Only that
    # architecture reads it; every other model ignores the attribute. The
    # frozen build_graph above is deliberately left untouched.
    # Shape must match h_extra_gnns.py:199 exactly: a 1-D tensor. TimeEnc does
    # its own unsqueeze(-1), so a (n, 1) tensor here yields a 3-D encoding and
    # the concat in GATSym.forward fails.
    t0 = torch.tensor(df.t0.values, dtype=torch.float)
    data["student"].t0_norm = ((t0 - t0.min()) /
                               max(float(t0.max() - t0.min()), 1.0))
    return data


class HGTNet:
    def __init__(self, data, device, hid=HID, heads=HEADS, layers=LAYERS,
                 dropout=DROPOUT):
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import HGTConv, Linear

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = dropout
                self.emb = torch.nn.ModuleDict()
                self.lin_in = torch.nn.ModuleDict()
                for nt in data.node_types:
                    if "x" in data[nt]:
                        self.lin_in[nt] = Linear(data[nt].x.shape[1], hid)
                    else:
                        self.emb[nt] = torch.nn.Embedding(data[nt].num_nodes, hid)
                self.convs = torch.nn.ModuleList(
                    [HGTConv(hid, hid, data.metadata(), heads) for _ in range(layers)])
                self.out = Linear(hid, 2)

            def forward(self, data):
                x = {}
                for nt in data.node_types:
                    if nt in self.lin_in:
                        x[nt] = self.lin_in[nt](data[nt].x).relu()
                    else:
                        x[nt] = self.emb[nt].weight
                for conv in self.convs:
                    x = conv(x, data.edge_index_dict)
                    if self.dropout > 0:
                        x = {k: F.dropout(v, p=self.dropout, training=self.training)
                             for k, v in x.items()}
                return self.out(x["student"])

        self.model = Net().to(device)


def train_eval(data, seed, device, hp=None):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, roc_auc_score

    hp = {**{"lr": LR, "hidden": HID, "layers": LAYERS, "heads": HEADS,
             "dropout": DROPOUT}, **(hp or {})}
    net = HGTNet(data, device, hid=hp["hidden"], heads=hp["heads"],
                 layers=hp["layers"], dropout=hp["dropout"]).model
    data = data.to(device)
    y = data["student"].y
    masks = {n: data["student"][f"{n}_mask"] for n in ("train", "val", "test")}
    w = torch.tensor([1.0, float((y[masks["train"]] == 0).sum())
                      / max(float((y[masks["train"]] == 1).sum()), 1.0)], device=device)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=1e-4)

    def scores(mask):
        net.eval()
        with torch.no_grad():
            p = F.softmax(net(data), dim=1)[:, 1]
        return p[mask].cpu().numpy(), y[mask].cpu().numpy()

    from sklearn.metrics import f1_score
    history = {"epoch": [], "train_loss": [], "val_auc_pr": [], "val_f1": []}
    best_val, best_state, patience = -1, None, 0
    for epoch in range(EPOCHS):
        net.train(); opt.zero_grad()
        logits = net(data)
        loss = F.cross_entropy(logits[masks["train"]], y[masks["train"]], weight=w)
        loss.backward(); opt.step()
        p_val, y_val = scores(masks["val"])
        val_ap = average_precision_score(y_val, p_val)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(loss.item()))
        history["val_auc_pr"].append(float(val_ap))
        history["val_f1"].append(float(f1_score(y_val, p_val >= 0.5)))
        if val_ap > best_val:
            best_val, patience = val_ap, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    net.load_state_dict(best_state)               # early stopping on VAL only
    p_te, y_te = scores(masks["test"])            # test touched exactly once
    p_va, y_va = scores(masks["val"])
    return {"auc_pr": float(average_precision_score(y_te, p_te)),
            "auc_roc": float(roc_auc_score(y_te, p_te)),
            "val_auc_pr": float(average_precision_score(y_va, p_va)),
            "epochs_run": epoch + 1, "seed": seed, "hp": hp,
            "history": history,
            "test_scores": [round(float(x), 5) for x in p_te],
            "test_labels": [int(v) for v in y_te],
            "val_scores": [round(float(x), 5) for x in p_va],
            "val_labels": [int(v) for v in y_va]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", default="none", choices=sorted(ABLATIONS))
    # hyperparameters (A1 tuning support; defaults = standard configuration)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--hidden", type=int, default=HID)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--heads", type=int, default=HEADS)
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--tag", default="",
                    help="filename suffix, e.g. 'grid_lr5e3_h128_l3_d05' for tuning runs")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    df = pd.read_parquet(args.data)
    df = df[df.early_concepts.apply(len) > 0].reset_index(drop=True)
    q1, q2 = np.quantile(df.t0, SPLIT_QUANTILES)
    df["split"] = np.where(df.t0 <= q1, "train", np.where(df.t0 <= q2, "val", "test"))

    hp = {"lr": args.lr, "hidden": args.hidden, "layers": args.layers,
          "heads": args.heads, "dropout": args.dropout}
    data = build_graph(df, args.ablate)
    res = train_eval(data, args.seed, device, hp=hp)
    res.update({"ablate": args.ablate, "device": device,
                "split_bounds": [int(q1), int(q2)], "n": len(df)})

    os.makedirs(args.out, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    path = os.path.join(args.out, f"hgt_{args.ablate}_seed{args.seed}{suffix}.json")
    json.dump(res, open(path, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
