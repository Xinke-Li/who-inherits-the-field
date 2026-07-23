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
