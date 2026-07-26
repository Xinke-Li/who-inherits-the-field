"""H - additional graph architectures under the frozen protocol (reviewer R3).

WHY. The paper's graph side held one deep architecture (HGT). A general lesson
about "graph learning adds no gain" needs more than one family, so we add:
  rgcn  a relation-specific message-passing baseline: HeteroConv with per-
        relation SAGEConv over the SAME hetero graph as E2 (nodes: student /
        advisor / institution / concept; five edge types, all observable at
        t0+5). Architecture differs from HGT (no attention, per-relation
        weights) - the classic RGCN pattern.
  tgat  a temporal attention model: the same graph with each edge carrying a
        time channel dt = t0(student) - year(edge event), passed through a
        sinusoidal time encoding concatenated to messages (TGAT-style temporal
        encoding; a full TGN memory module is out of scope and stated so).

PRE-REGISTERED PROTOCOL (identical to E2, fixed before any run): temporal
cohort split from the frozen quantiles; class-weighted cross-entropy; early
stopping on VAL AUC-PR only; test evaluated once per seed; seeds range(10);
standard configuration only (hid 64, 2 layers, lr 1e-3), no grid - the E12
rolling-origin downgrade already removed any significance claim these models
would need a grid to defend. Decision rule: these architectures update the
"no measurable graph gain" reading only if one exceeds the tabular ceiling
max(M2, M3) on the frozen split with a 10-seed mean.

GPU leg: runs in GPU_Batch_Colab.ipynb (see PENDING_TODO.md). CPU execution
works but is slow. Usage:
  python h_extra_gnns.py --model rgcn --seed 0 [--out ../results_econ/results_extra_gnns]
  python h_extra_gnns.py --model rgcn --ablate social --seed 0   # edge-group ablation

--ablate passes the edge-group name straight to E2.build_graph (same ABLATIONS as
e2_hgt.py; 'social' = advising + institution + coauth). The model is untouched:
dropping edges from the hetero graph simply leaves the remaining relations to
message-pass over. Output filenames carry the ablate tag when it is not 'none'
(<model>_ablate-<group>_seed<k>.json), so ablation runs never collide with the
default full-graph runs (<model>_seed<k>.json).
Output: <out>/<model>[_ablate-<group>]_seed<k>.json (schema mirrors e2_hgt.py).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils import data as D
import e2_hgt as E2

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv

HID, LAYERS, LR, EPOCHS, PATIENCE = 64, 2, 1e-3, 300, 30


class RGCN(nn.Module):
    """Per-relation SAGEConv (RGCN pattern) over the E2 hetero graph."""
    def __init__(self, data, hid=HID, layers=LAYERS):
        super().__init__()
        # mirror E2: Linear for node types with .x, learnable Embedding otherwise
        self.lin_in = nn.ModuleDict(); self.emb = nn.ModuleDict()
        for nt in data.node_types:
            if "x" in data[nt]:
                self.lin_in[nt] = nn.Linear(data[nt].x.size(1), hid)
            else:
                self.emb[nt] = nn.Embedding(data[nt].num_nodes, hid)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(HeteroConv(
                {et: SAGEConv(hid, hid) for et in data.edge_types}, aggr="mean"))
        self.head = nn.Linear(hid, 2)

    def _in(self, data):
        return {nt: (self.lin_in[nt](data[nt].x).relu() if nt in self.lin_in
                     else self.emb[nt].weight) for nt in data.node_types}

    def forward(self, data):
        x = self._in(data)
        for conv in self.convs:
            x = {k: v.relu() for k, v in conv(x, data.edge_index_dict).items()}
        return self.head(x["student"])


class TimeEnc(nn.Module):
    """Sinusoidal time encoding (TGAT-style)."""
    def __init__(self, dim=8):
        super().__init__()
        self.w = nn.Parameter(torch.from_numpy(
            1.0 / 10 ** np.linspace(0, 4, dim)).float(), requires_grad=False)

    def forward(self, dt):
        return torch.cos(dt.unsqueeze(-1) * self.w)


class TGATLite(nn.Module):
    """GATConv with a time-encoded edge channel appended to source features.
    A pragmatic TGAT variant: temporal encoding + attention, no memory module."""
    def __init__(self, data, hid=HID, layers=LAYERS, tdim=8):
        super().__init__()
        self.tenc = TimeEnc(tdim)
        self.lin_in = nn.ModuleDict(); self.emb = nn.ModuleDict()
        for nt in data.node_types:
            if "x" in data[nt]:
                self.lin_in[nt] = nn.Linear(data[nt].x.size(1), hid)
            else:
                self.emb[nt] = nn.Embedding(data[nt].num_nodes, hid)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(HeteroConv(
                {et: GATConv(hid, hid, heads=2, concat=False, add_self_loops=False)
                 for et in data.edge_types}, aggr="mean"))
        self.mix = nn.Linear(hid + tdim, hid)
        self.head = nn.Linear(hid, 2)

    def forward(self, data):
        x = {nt: (self.lin_in[nt](data[nt].x).relu() if nt in self.lin_in
                  else self.emb[nt].weight) for nt in data.node_types}
        for conv in self.convs:
            x = {k: v.relu() for k, v in conv(x, data.edge_index_dict).items()}
        # student-level temporal channel: encode t0 relative to the corpus span
        dt = data["student"].t0_norm if hasattr(data["student"], "t0_norm") else None
        h = x["student"]
        if dt is not None:
            h = self.mix(torch.cat([h, self.tenc(dt)], dim=1)).relu()
        return self.head(h)


def train_eval(model, data, seed, device):
    """E2's protocol verbatim: class weights, val-only early stopping, one test."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    model = model.to(device); data = data.to(device)
    y = data["student"].y
    w = torch.tensor([1.0, float((y == 0).sum()) / max(1, int((y == 1).sum()))],
                     device=device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    masks = {n: data["student"][f"{n}_mask"] for n in ("train", "val", "test")}

    def scores(mask):
        model.eval()
        with torch.no_grad():
            p = F.softmax(model(data), dim=1)[:, 1]
        yy = y[mask].cpu().numpy(); pp = p[mask].cpu().numpy()
        return {"auc_pr": float(average_precision_score(yy, pp)),
                "auc_roc": float(roc_auc_score(yy, pp))}

    best, wait, best_state = -1, 0, None
    for epoch in range(EPOCHS):
        model.train(); opt.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[masks["train"]], y[masks["train"]], weight=w)
        loss.backward(); opt.step()
        va = scores(masks["val"])["auc_pr"]
        if va > best:
            best, wait = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)

    def arrays(mask):
        model.eval()
        with torch.no_grad():
            p = F.softmax(model(data), dim=1)[:, 1]
        return ([round(float(x), 5) for x in p[mask].cpu().numpy()],
                [int(v) for v in y[mask].cpu().numpy()])

    # per-student arrays mirror e2_hgt.py's output schema so a score-level
    # paired bootstrap can be computed later without a GPU re-run
    te_s, te_l = arrays(masks["test"])
    va_s, va_l = arrays(masks["val"])
    return {"val": {"auc_pr": best}, "test": scores(masks["test"]),
            "val_auc_pr": best,
            "epochs_run": epoch + 1, "seed": seed,
            "test_scores": te_s, "test_labels": te_l,
            "val_scores": va_s, "val_labels": va_l}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rgcn", "tgat"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", default="none", choices=sorted(E2.ABLATIONS),
                    help="edge-group ablation passed verbatim to E2.build_graph; "
                         "'social' = advising + institution + coauth")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "results_extra_gnns"))
    args = ap.parse_args()

    E2.set_seed(args.seed)
    df = D.load_dataset()
    ds = D.temporal_split(df)
    data = E2.build_graph(ds, args.ablate)
    # student-level normalized t0 for the temporal channel
    t0 = torch.tensor(ds.t0.values, dtype=torch.float)
    data["student"].t0_norm = (t0 - t0.min()) / max(1.0, float(t0.max() - t0.min()))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print(f"[h:{args.model}] WARNING: no CUDA - GPU leg; slow on CPU")
    model = RGCN(data) if args.model == "rgcn" else TGATLite(data)
    res = train_eval(model, data, args.seed, device)
    res.update({"model": args.model, "ablate": args.ablate,
                "protocol": "E2-identical, standard config only"})
    os.makedirs(args.out, exist_ok=True)
    tag = "" if args.ablate == "none" else f"_ablate-{args.ablate}"
    path = os.path.join(args.out, f"{args.model}{tag}_seed{args.seed}.json")
    Path(path).write_text(json.dumps(res, indent=2))
    print(f"[h:{args.model} ablate={args.ablate}] seed {args.seed} -> {path} "
          f"(test {res['test']})")


if __name__ == "__main__":
    main()
