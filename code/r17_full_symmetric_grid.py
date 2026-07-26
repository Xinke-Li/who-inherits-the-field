#!/usr/bin/env python3
"""R17 - full symmetric grid: RGCN and GAT, all five disciplines (task 10).

POST-HOC ROBUSTNESS. Generalizes r3_rgcn_symmetric.py from chemistry-RGCN to
both relational architectures across every discipline, under the HGT tuned
protocol so the whole graph arm is evaluated at one symmetric operating
point. Reuses the frozen graph construction (e2_hgt.build_graph) and the two
architectures (h_extra_gnns.RGCN, h_extra_gnns.TGATLite = the cohort-time
GAT), and the corrected eq. (2) machinery
(e12_corrected_aggregation.fit_val_symmetric / fast_auc_pr).

Protocol per (discipline, architecture), identical to the HGT tuned arm:
  grid    16 configs = lr {1e-3, 5e-3} x hidden {64, 128} x layers {2, 3}
          x dropout {0.0, 0.5}  (e2_hgt.HP_GRID), grid seeds 0 and 1,
          selection on validation AUC-PR only, unified budget 200 epochs /
          patience 15 / Adam weight decay 1e-4, train-only class weight;
  winner  re-run at all 10 seeds under the same unified budget.
Then eq. (2) vs M5' (refit identically to the stored e12 run): paired
Wilcoxon on the ten seed pairs, Benjamini-Hochberg across the symmetric-grid
family per discipline {rgcn_sym, gat_sym}, and the 2000-draw paired
student-level bootstrap. Three gates: mean > 0, p_adj < 0.05, student CI
lower > 0.

The training loop mirrors e2_hgt.train_eval (train-only class weight,
val-AUC-PR early stopping, single test evaluation, per-student score arrays);
each architecture gets the HGT-style dropout placement (after each conv).
Chemistry RGCN is already done (r3): its per-seed files are copied into the
output folder by the notebook and this script's aggregate reads them.

Usage (run from the code tree that has paper_pipeline/ on the path):
  DATASET=<field> DATASET_PATH=data_<field>/clean_dataset.parquet \
      python r17_full_symmetric_grid.py train --model rgcn
  DATASET=<field> ... python r17_full_symmetric_grid.py train --model gat
  DATASET=<field> ... python r17_full_symmetric_grid.py aggregate
Output: results/robustness/full_symmetric_grid/<field>_<model>_sym_seed*.json,
        <field>_<model>_grid_selection.json, <field>_verdict.json
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- locate the code tree: paper_pipeline/ and e12_corrected_aggregation.py ---
_HERE = Path(__file__).resolve().parent
for cand in (_HERE, _HERE / "paper_pipeline", _HERE / "paper_pipeline" / "experiments",
             _HERE / "code", _HERE / "code" / "paper_pipeline",
             _HERE / "code" / "paper_pipeline" / "experiments"):
    if cand.exists():
        sys.path.insert(0, str(cand))
# also honor CWD layout (notebook cd's into the unzip root)
for cand in ("paper_pipeline", "paper_pipeline/experiments", ".", "code"):
    if Path(cand).exists():
        sys.path.insert(0, str(Path(cand).resolve()))

import config as C
from utils import data as D
from utils import stats as S
import e2_hgt as E2
import h_extra_gnns as H
from e12_corrected_aggregation import N_BOOT, fast_auc_pr, fit_val_symmetric

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

OUT_DIR = Path("results/robustness/full_symmetric_grid")
HP_GRID = E2.HP_GRID
GRID_SEEDS = [0, 1]
BUDGET = {"epochs": 200, "patience": 15, "weight_decay": 1e-4}   # HGT tuned budget
FIELDS_BY_SIZE = ["econ", "math", "physics", "neuro", "chemistry"]  # smallest n_test first


class RGCNSym(H.RGCN):
    """h_extra_gnns.RGCN + HGT-style dropout after each conv (r3 verbatim)."""

    def __init__(self, data, hid, layers, dropout):
        super().__init__(data, hid=hid, layers=layers)
        self.p_drop = dropout

    def forward(self, data):
        x = self._in(data)
        for conv in self.convs:
            x = {k: v.relu() for k, v in conv(x, data.edge_index_dict).items()}
            if self.p_drop > 0:
                x = {k: F.dropout(v, p=self.p_drop, training=self.training)
                     for k, v in x.items()}
        return self.head(x["student"])


class GATSym(H.TGATLite):
    """h_extra_gnns.TGATLite (cohort-time GAT) + the same dropout placement."""

    def __init__(self, data, hid, layers, dropout, tdim=8):
        super().__init__(data, hid=hid, layers=layers, tdim=tdim)
        self.p_drop = dropout

    def forward(self, data):
        x = {nt: (self.lin_in[nt](data[nt].x).relu() if nt in self.lin_in
                  else self.emb[nt].weight) for nt in data.node_types}
        for conv in self.convs:
            x = {k: v.relu() for k, v in conv(x, data.edge_index_dict).items()}
            if self.p_drop > 0:
                x = {k: F.dropout(v, p=self.p_drop, training=self.training)
                     for k, v in x.items()}
        dt = data["student"].t0_norm if hasattr(data["student"], "t0_norm") else None
        h = x["student"]
        if dt is not None:
            h = self.mix(torch.cat([h, self.tenc(dt)], dim=1)).relu()
        return self.head(h)


MODELS = {"rgcn": RGCNSym, "gat": GATSym}


def build_graph():
    df = D.load_dataset()
    ds = D.temporal_split(df)
    data = E2.build_graph(ds, "none")
    t0 = torch.tensor(ds.t0.values, dtype=torch.float)   # GAT time channel
    data["student"].t0_norm = (t0 - t0.min()) / max(1.0, float(t0.max() - t0.min()))
    return data


def train_eval(data, seed, device, hp, model_key):
    """e2_hgt.train_eval transposed to RGCN/GAT under the unified HGT budget."""
    E2.set_seed(seed)
    net = MODELS[model_key](data, hid=hp["hidden"], layers=hp["layers"],
                            dropout=hp["dropout"]).to(device)
    data = data.to(device)
    y = data["student"].y
    masks = {n: data["student"][f"{n}_mask"] for n in ("train", "val", "test")}
    yw = y[masks["train"]]                                # train-only class weight
    w = torch.tensor([1.0, float((yw == 0).sum()) / max(float((yw == 1).sum()), 1.0)],
                     device=device)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"],
                           weight_decay=BUDGET["weight_decay"])

    def scores(mask):
        net.eval()
        with torch.no_grad():
            p = F.softmax(net(data), dim=1)[:, 1]
        return p[mask].cpu().numpy(), y[mask].cpu().numpy()

    best_val, best_state, patience = -1, None, 0
    for epoch in range(BUDGET["epochs"]):
        net.train(); opt.zero_grad()
        loss = F.cross_entropy(net(data)[masks["train"]], y[masks["train"]], weight=w)
        loss.backward(); opt.step()
        p_val, y_val = scores(masks["val"])
        val_ap = average_precision_score(y_val, p_val)
        if val_ap > best_val:
            best_val, patience = val_ap, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= BUDGET["patience"]:
                break
    net.load_state_dict(best_state)
    p_te, y_te = scores(masks["test"])
    p_va, y_va = scores(masks["val"])
    return {"auc_pr": float(average_precision_score(y_te, p_te)),
            "auc_roc": float(roc_auc_score(y_te, p_te)),
            "val_auc_pr": float(best_val), "epochs_run": epoch + 1,
            "seed": seed, "hp": hp, "model": model_key, "device": device,
            "budget": BUDGET,
            "test_scores": [round(float(x), 5) for x in p_te],
            "test_labels": [int(v) for v in y_te],
            "val_scores": [round(float(x), 5) for x in p_va],
            "val_labels": [int(v) for v in y_va]}


def hp_tag(hp):
    return f"lr{hp['lr']}_h{hp['hidden']}_l{hp['layers']}_d{hp['dropout']}"


def cmd_train(field, model_key):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[r17] WARNING: no CUDA; running the GPU leg on CPU", flush=True)
    data = build_graph()
    configs = [dict(zip(("lr", "hidden", "layers", "dropout"), v))
               for v in itertools.product(HP_GRID["lr"], HP_GRID["hidden"],
                                          HP_GRID["layers"], HP_GRID["dropout"])]
    gtag = f"{field}_{model_key}"
    # grid at seeds 0/1
    for seed in GRID_SEEDS:
        for hp in configs:
            p = OUT_DIR / f"{gtag}_grid_seed{seed}_{hp_tag(hp)}.json"
            if p.exists():
                continue
            t0 = time.time()
            res = train_eval(data, seed, device, hp, model_key)
            p.write_text(json.dumps(res, indent=2))
            print(f"[r17:grid] {gtag} s{seed} {hp_tag(hp)}: val {res['val_auc_pr']:.4f} "
                  f"test {res['auc_pr']:.4f} ({res['epochs_run']}ep {time.time()-t0:.0f}s)",
                  flush=True)
    # winner on mean val AUC-PR over the two grid seeds
    val_by = {}
    for hp in configs:
        vals = [json.loads((OUT_DIR / f"{gtag}_grid_seed{s}_{hp_tag(hp)}.json")
                           .read_text())["val_auc_pr"] for s in GRID_SEEDS]
        val_by[hp_tag(hp)] = (float(np.mean(vals)), hp)
    wtag, (wval, whp) = max(val_by.items(), key=lambda kv: kv[1][0])
    per_seed_argmax = {s: max(configs, key=lambda hp: json.loads(
        (OUT_DIR / f"{gtag}_grid_seed{s}_{hp_tag(hp)}.json").read_text())["val_auc_pr"])
        for s in GRID_SEEDS}
    (OUT_DIR / f"{gtag}_grid_selection.json").write_text(json.dumps({
        "field": field, "model": model_key, "winner": whp,
        "winner_mean_val_auc_pr": round(wval, 4),
        "seeds_agree": all(per_seed_argmax[s] == whp for s in GRID_SEEDS),
        "all_configs_mean_val": {k: round(v[0], 4) for k, v in val_by.items()}},
        indent=2))
    print(f"[r17] {gtag} winner {wtag} (mean val {wval:.4f})", flush=True)
    # winner at 10 seeds
    for seed in C.SEEDS:
        p = OUT_DIR / f"{gtag}_sym_seed{seed}.json"
        if p.exists():
            continue
        t0 = time.time()
        res = train_eval(data, seed, device, whp, model_key)
        p.write_text(json.dumps(res, indent=2))
        print(f"[r17:{gtag}] winner s{seed}: test {res['auc_pr']:.4f} "
              f"({res['epochs_run']}ep {time.time()-t0:.0f}s)", flush=True)
    print(f"[r17] {gtag} train complete", flush=True)


def _load_seed_scores(pattern, yte, seeds):
    out = {}
    for s in seeds:
        d = json.loads(Path(pattern.format(s=s)).read_text())
        t = d.get("test", d)
        assert (np.array(d["test_labels"], int) == yte).all(), \
            f"{pattern.format(s=s)}: label mismatch with local split"
        out[s] = {"auc_pr": t["auc_pr"], "scores": np.array(d["test_scores"], float)}
    return out


def cmd_aggregate(field):
    seeds = C.SEEDS
    df_split = D.temporal_split(D.load_dataset())
    Xt, _ = D.build_features(df_split, concepts="none")
    nfa = D.build_nfa_features(df_split)
    X5 = np.hstack([Xt, nfa.values.astype(float)])
    p5 = D.split_xy(df_split, X5)
    (Xtr5, ytr), (Xva5, yva), (Xte5, yte) = p5["train"], p5["val"], p5["test"]

    # M5' refit (identical to the stored e12 run)
    m5p_scores, m5p_ap = {}, []
    for s in seeds:
        p_te, _, it, _ = fit_val_symmetric(Xtr5, ytr, Xva5, yva, Xte5, s)
        m5p_scores[s] = p_te
        m5p_ap.append(fast_auc_pr(yte, p_te))

    # symmetric-grid family; chemistry RGCN comes from the copied r3 files
    rgcn_pat = (str(OUT_DIR / "chemistry_rgcn_sym_seed{s}.json") if field == "chemistry"
                else str(OUT_DIR / f"{field}_rgcn_sym_seed{{s}}.json"))
    family = {"rgcn_sym": _load_seed_scores(rgcn_pat, yte, seeds),
              "gat_sym": _load_seed_scores(
                  str(OUT_DIR / f"{field}_gat_sym_seed{{s}}.json"), yte, seeds)}

    packs, pvals = {}, []
    for name, ps in family.items():
        g = [ps[s]["auc_pr"] for s in seeds]
        packs[name] = {"g": g, "w": S.paired_wilcoxon(g, m5p_ap)}
        pvals.append(packs[name]["w"]["p"])
    p_adj = S.bh_correction(pvals)[0]

    rng = np.random.default_rng(0)
    n = len(yte)
    idx_draws = rng.integers(0, n, size=(N_BOOT, n))
    m5p_boot = np.empty((len(seeds), N_BOOT))
    for b in range(N_BOOT):
        idx = idx_draws[b]; yb = yte[idx]
        for s in seeds:
            m5p_boot[s, b] = fast_auc_pr(yb, m5p_scores[s][idx])

    def student_ci(ps):
        gap = np.empty((len(seeds), N_BOOT))
        for b in range(N_BOOT):
            idx = idx_draws[b]; yb = yte[idx]
            for s in seeds:
                gap[s, b] = fast_auc_pr(yb, ps[s]["scores"][idx])
        pooled = (gap - m5p_boot).mean(axis=0)
        return [round(float(np.percentile(pooled, 2.5)), 4),
                round(float(np.percentile(pooled, 97.5)), 4)]

    out_models = {}
    for i, (name, ps) in enumerate(family.items()):
        g = packs[name]["g"]
        ci = student_ci(ps)
        gates = {"seed_mean_gt_ceiling": bool(np.mean(g) > np.mean(m5p_ap)),
                 "p_adj_lt_0.05": bool(p_adj[i] < 0.05),
                 "student_ci_lower_gt_0": bool(ci[0] > 0)}
        out_models[name] = {
            "seed_mean_auc_pr": round(float(np.mean(g)), 4),
            "delta_vs_M5prime": round(float(packs[name]["w"]["mean_diff"]), 4),
            "wilcoxon_p_raw": packs[name]["w"]["p"],
            "p_adj_M5prime": round(float(p_adj[i]), 6),
            "student_ci95_vs_M5prime": ci,
            "gates_fair": gates, "exceeds_fair": all(gates.values())}
        print(f"[r17:agg] {field}/{name}: mean {np.mean(g):.4f} dM5' "
              f"{packs[name]['w']['mean_diff']:+.4f} p_adj {p_adj[i]:.4f} CI {ci} "
              f"exceeds={out_models[name]['exceeds_fair']}", flush=True)

    out = {"experiment": "R17_full_symmetric_grid", "field": field,
           "note": ("RGCN and GAT under the HGT tuned protocol; BH across the "
                    "two symmetric-grid architectures per discipline; chemistry "
                    "RGCN reused from r3"),
           "M5prime_mean": round(float(np.mean(m5p_ap)), 4),
           "models": out_models,
           "any_exceeds": any(v["exceeds_fair"] for v in out_models.values())}
    (OUT_DIR / f"{field}_verdict.json").write_text(json.dumps(out, indent=2))
    print(f"[r17] {field} verdict written; any_exceeds="
          f"{out['any_exceeds']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "aggregate"])
    ap.add_argument("--model", choices=["rgcn", "gat"])
    args = ap.parse_args()
    field = C.CLEAN_DATASET.stem.replace("clean_dataset_", "").replace("clean_dataset", "") \
        or os.environ.get("DATASET", "")
    field = os.environ.get("DATASET", field)
    if args.cmd == "train":
        assert args.model, "train needs --model rgcn|gat"
        cmd_train(field, args.model)
    else:
        cmd_aggregate(field)
