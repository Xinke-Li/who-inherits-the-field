#!/usr/bin/env python3
"""R3 - Chemistry RGCN under the symmetric (HGT) tuning and budget (P0-3).

POST-HOC ROBUSTNESS CHECK, layered on top of the frozen artifact; NOT part of
the pre-registered protocol. Resolves the F9 budget asymmetry that
e12_corrected_aggregation.py discloses but does not fix: the paper's one
"exceeds" verdict (chemistry RGCN) was trained by h_extra_gnns.py at 300
epochs / patience 30 / no weight decay, while the HGT arm used 200 / 15 /
weight decay 1e-4 and a 16-configuration tuned grid. Here the RGCN gets
EXACTLY the HGT-tuned grid protocol:

  grid    16 configs = lr {1e-3, 5e-3} x hidden {64, 128} x layers {2, 3}
          x dropout {0.0, 0.5}  (e2_hgt.HP_GRID verbatim; RGCN has no heads
          dimension, the rest of the protocol is identical), at seeds 0 and 1
          (the convention of the frozen results_hgt_grid artifacts), budget
          200 epochs / patience 15 / Adam weight_decay 1e-4, selection on
          validation AUC-PR only;
  winner  re-run at all 10 seeds under the same unified budget
          (rgcn_sym_seed<k>.json), and ALSO under the ORIGINAL h_extra_gnns
          budget 300 / 30 / no weight decay (rgcn_sym_orig300_seed<k>.json)
          so both budget cells exist.

The training loop mirrors e2_hgt.train_eval (class weight over TRAIN labels
only, val-AUC-PR early stopping, test evaluated once, per-student score
arrays saved); the model is h_extra_gnns.RGCN extended with the HGT-style
dropout placement (after each conv's relu). The original-budget cell keeps
h_extra_gnns.train_eval's class-weight convention (all labels, its F2 quirk)
so that cell reproduces the original protocol exactly.

Aggregation: the corrected protocol of eq. (2) via e12_corrected_aggregation
imports (fast_auc_pr, fit_val_symmetric) - paired Wilcoxon vs M5 and M5',
BH across the four-model family {hgt, hgt_tuned, rgcn_symmetric,
gat_cohort_time} with the frozen per-seed artifacts of the other three, and
the 2000-draw paired student-level bootstrap.

Usage:
  DATASET=chemistry DATASET_PATH=data/clean_dataset_chemistry.parquet \
      python code/r3_rgcn_symmetric.py train        # grid + winner runs
  DATASET=chemistry DATASET_PATH=data/clean_dataset_chemistry.parquet \
      python code/r3_rgcn_symmetric.py aggregate    # corrected verdict

Output: results/robustness/rgcn_symmetric/*.json,
        results/robustness/rgcn_symmetric_verdict.json
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline"))
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline" / "experiments"))
sys.path.insert(0, str(ROOT / "code"))

import config as C
from utils import data as D
from utils import stats as S
import e2_hgt as E2
import h_extra_gnns as H
import e12_corrected_aggregation as E12

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

OUT_DIR = ROOT / "results" / "robustness" / "rgcn_symmetric"
FIELD = "chemistry"
HP_GRID = E2.HP_GRID              # lr x hidden x layers x dropout, verbatim
GRID_SEEDS = [0, 1]               # frozen results_hgt_grid convention
BUDGETS = {
    "hgt":      {"epochs": 200, "patience": 15, "weight_decay": 1e-4,
                 "class_weight_scope": "train"},   # e2_hgt.py:43,184
    "original": {"epochs": 300, "patience": 30, "weight_decay": 0.0,
                 "class_weight_scope": "all"},     # h_extra_gnns.py:56,136
}


class RGCNSym(H.RGCN):
    """h_extra_gnns.RGCN + the HGT-style dropout placement (e2_hgt.HGTNet
    applies dropout to every node type after each conv; same here)."""

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
    """h_extra_gnns.TGATLite + the HGT-style dropout placement, exactly as
    RGCNSym does for the relational model. This is the gat_cohort_time
    architecture; it consumes data["student"].t0_norm, which no ceiling model
    receives (audit finding F6, disclosed in Section 7)."""

    def __init__(self, data, hid, layers, dropout):
        # The guard lives here, not in the caller. A caller-side check is the
        # same silent-degradation pattern as the old fit() fallback: any new
        # call site that forgets it gets a plain attention model wearing the
        # gat_cohort_time label. Constructing the class at all requires the
        # channel that defines it.
        if not hasattr(data["student"], "t0_norm"):
            raise RuntimeError(
                "GATSym requires data['student'].t0_norm, the cohort-time "
                "channel (audit finding F6). Call attach_t0_norm(data, df) "
                "after building the graph. Refusing to construct a GAT without "
                "the input that distinguishes it from a plain attention model.")
        super().__init__(data, hid=hid, layers=layers)
        self.p_drop = dropout

    def forward(self, data):
        import torch
        x = {nt: (self.lin_in[nt](data[nt].x).relu() if nt in self.lin_in
                  else self.emb[nt].weight) for nt in data.node_types}
        for conv in self.convs:
            x = {k: v.relu() for k, v in conv(x, data.edge_index_dict).items()}
            if self.p_drop > 0:
                x = {k: F.dropout(v, p=self.p_drop, training=self.training)
                     for k, v in x.items()}
        h = x["student"]
        dt = getattr(data["student"], "t0_norm", None)
        if dt is not None:
            h = self.mix(torch.cat([h, self.tenc(dt)], dim=1)).relu()
        return self.head(h)


def attach_t0_norm(data, df):
    """F6's cohort-time channel. TGATLite reads it; h_extra_gnns.py:183-184
    attaches it in the frozen path, so the strict path must too or the GAT
    silently loses its distinguishing input."""
    import torch
    t0 = torch.tensor(df.t0.values, dtype=torch.float)
    # 1-D, matching h_extra_gnns.py:199. TimeEnc unsqueezes internally.
    data["student"].t0_norm = ((t0 - t0.min()) /
                               max(float(t0.max() - t0.min()), 1.0))
    return data

def train_eval(data, seed, device, hp, budget, arch="rgcn"):
    """e2_hgt.train_eval transposed to the RGCN (source of every protocol
    detail: train-only class weight, val-AUC-PR early stopping, single test
    evaluation, per-student score arrays). budget switches the epoch/patience/
    weight-decay/class-weight cell as documented in BUDGETS."""
    b = BUDGETS[budget]
    E2.set_seed(seed)
    # arch is additive: "rgcn" is the original path, unchanged. Anything not
    # implemented raises rather than silently falling back to RGCN, which would
    # mislabel a cell.
    if arch == "rgcn":
        net = RGCNSym(data, hid=hp["hidden"], layers=hp["layers"],
                      dropout=hp["dropout"]).to(device)
    elif arch == "gat":
        # GATSym itself refuses to construct without t0_norm, so no guard is
        # needed here and none can be forgotten at a new call site.
        net = GATSym(data, hid=hp["hidden"], layers=hp["layers"],
                     dropout=hp["dropout"]).to(device)
    elif arch == "hgt":
        net = E2.HGTNet(data, device, hid=hp["hidden"], heads=E2.HEADS,
                        layers=hp["layers"], dropout=hp["dropout"]).model
    else:
        raise NotImplementedError(
            f"train_eval: architecture {arch!r} is not wired. Refusing to run "
            f"rather than produce numbers under a {arch!r} label.")
    data = data.to(device)
    y = data["student"].y
    masks = {n: data["student"][f"{n}_mask"] for n in ("train", "val", "test")}
    if b["class_weight_scope"] == "train":
        yw = y[masks["train"]]
    else:
        yw = y
    w = torch.tensor([1.0, float((yw == 0).sum()) / max(float((yw == 1).sum()), 1.0)],
                     device=device)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"],
                           weight_decay=b["weight_decay"])

    def scores(mask):
        net.eval()
        with torch.no_grad():
            p = F.softmax(net(data), dim=1)[:, 1]
        return p[mask].cpu().numpy(), y[mask].cpu().numpy()

    best_val, best_state, patience = -1, None, 0
    for epoch in range(b["epochs"]):
        net.train(); opt.zero_grad()
        logits = net(data)
        loss = F.cross_entropy(logits[masks["train"]], y[masks["train"]], weight=w)
        loss.backward(); opt.step()
        p_val, y_val = scores(masks["val"])
        val_ap = average_precision_score(y_val, p_val)
        if val_ap > best_val:
            best_val, patience = val_ap, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= b["patience"]:
                break

    net.load_state_dict(best_state)
    p_te, y_te = scores(masks["test"])
    p_va, y_va = scores(masks["val"])
    _struct = True
    if arch == "hgt":
        try:
            from torch_geometric.nn import HGTConv
            _struct = any(isinstance(m, HGTConv) for m in net.modules())
        except ImportError:
            _struct = False
    return {"model_class": type(net).__name__, "arch_requested": arch,
            "structural_ok": _struct,
            "auc_pr": float(average_precision_score(y_te, p_te)),
            "auc_roc": float(roc_auc_score(y_te, p_te)),
            "val_auc_pr": float(average_precision_score(y_va, p_va)),
            "epochs_run": epoch + 1, "seed": seed, "hp": hp, "budget": budget,
            "budget_spec": b, "device": device,
            "test_scores": [round(float(x), 5) for x in p_te],
            "test_labels": [int(v) for v in y_te],
            "val_scores": [round(float(x), 5) for x in p_va],
            "val_labels": [int(v) for v in y_va]}


def hp_tag(hp):
    return f"lr{hp['lr']}_h{hp['hidden']}_l{hp['layers']}_d{hp['dropout']}"


def build():
    df = D.load_dataset()
    ds = D.temporal_split(df)
    return E2.build_graph(ds, "none")


def cmd_train():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[r3] WARNING: no CUDA; running the GPU leg on CPU", flush=True)
    data = build()

    configs = [dict(zip(("lr", "hidden", "layers", "dropout"), v))
               for v in itertools.product(HP_GRID["lr"], HP_GRID["hidden"],
                                          HP_GRID["layers"], HP_GRID["dropout"])]
    # ---- grid at seeds 0/1, HGT budget ----
    for seed in GRID_SEEDS:
        for hp in configs:
            path = OUT_DIR / f"rgcn_sym_grid_seed{seed}_{hp_tag(hp)}.json"
            if path.exists():
                continue
            t0 = time.time()
            res = train_eval(data, seed, device, hp, "hgt")
            path.write_text(json.dumps(res, indent=2))
            print(f"[r3:grid] seed{seed} {hp_tag(hp)}: val {res['val_auc_pr']:.4f} "
                  f"test {res['auc_pr']:.4f} ({res['epochs_run']} ep, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    # ---- select winner on validation AUC-PR (mean over the two grid seeds) ----
    val_by_cfg = {}
    for hp in configs:
        vals = [json.loads((OUT_DIR / f"rgcn_sym_grid_seed{s}_{hp_tag(hp)}.json")
                           .read_text())["val_auc_pr"] for s in GRID_SEEDS]
        val_by_cfg[hp_tag(hp)] = (float(np.mean(vals)), hp)
    winner_tag, (winner_val, winner_hp) = max(val_by_cfg.items(), key=lambda kv: kv[1][0])
    per_seed_argmax = {}
    for s in GRID_SEEDS:
        per_seed_argmax[s] = max(
            configs, key=lambda hp: json.loads(
                (OUT_DIR / f"rgcn_sym_grid_seed{s}_{hp_tag(hp)}.json").read_text())["val_auc_pr"])
    (OUT_DIR / "grid_selection.json").write_text(json.dumps({
        "winner": winner_hp, "winner_mean_val_auc_pr": round(winner_val, 4),
        "per_seed_argmax": {str(s): per_seed_argmax[s] for s in GRID_SEEDS},
        "seeds_agree": all(per_seed_argmax[s] == winner_hp for s in GRID_SEEDS),
        "all_configs_mean_val": {k: round(v[0], 4) for k, v in val_by_cfg.items()},
    }, indent=2))
    print(f"[r3] grid winner: {winner_tag} (mean val {winner_val:.4f})", flush=True)

    # ---- winner at 10 seeds, both budgets ----
    for budget, prefix in (("hgt", "rgcn_sym_seed"), ("original", "rgcn_sym_orig300_seed")):
        for seed in C.SEEDS:
            path = OUT_DIR / f"{prefix}{seed}.json"
            if path.exists():
                continue
            t0 = time.time()
            res = train_eval(data, seed, device, winner_hp, budget)
            path.write_text(json.dumps(res, indent=2))
            print(f"[r3:{budget}] seed{seed}: test {res['auc_pr']:.4f} "
                  f"({res['epochs_run']} ep, {time.time() - t0:.0f}s)", flush=True)
    print("[r3] train complete", flush=True)


def cmd_aggregate():
    """Corrected protocol of eq. (2) with the symmetric RGCN substituted into
    the four-model family; everything else from the frozen artifacts."""
    res_dir = ROOT / "results" / f"results_{FIELD}"
    e1 = json.loads((res_dir / "e1_baselines.json").read_text())
    seeds = e1["seeds"]

    def load_seed_files(patt, base=res_dir):
        out = {}
        for s in seeds:
            d = json.loads((base / patt.format(s=s)).read_text())
            t = d.get("test", d)
            out[s] = {"auc_pr": t["auc_pr"],
                      "scores": np.array(d["test_scores"], float),
                      "labels": np.array(d["test_labels"], int)}
        return out

    family = {
        "hgt": load_seed_files("results_hgt/hgt_none_seed{s}.json"),
        "hgt_tuned": load_seed_files("results_hgt_grid/hgt_none_seed{s}_best.json"),
        "rgcn_symmetric": load_seed_files("rgcn_sym_seed{s}.json", base=OUT_DIR),
        "gat_cohort_time": load_seed_files("results_extra_gnns/tgat_seed{s}.json"),
    }
    rgcn_orig_frozen = load_seed_files("results_extra_gnns/rgcn_seed{s}.json")
    rgcn_orig300 = load_seed_files("rgcn_sym_orig300_seed{s}.json", base=OUT_DIR)

    # local split + ceilings (e12.run_field's construction, via its imports)
    df = D.load_dataset()
    df_split = D.temporal_split(df)
    Xt, _ = D.build_features(df_split, concepts="none")
    nfa = D.build_nfa_features(df_split)
    X5 = np.hstack([Xt, nfa.values.astype(float)])
    tab = D.split_xy(df_split, Xt)
    p5 = D.split_xy(df_split, X5)
    (Xtr5, ytr), (Xva5, yva), (Xte5, yte) = p5["train"], p5["val"], p5["test"]
    for name, ps in family.items():
        for s in seeds:
            assert (ps[s]["labels"] == yte).all(), f"{name}/seed{s} label mismatch"

    m5_pre = [r["auc_pr"] for r in e1["per_seed"]["M5_gbdt_nfa"]]
    stored = json.loads((res_dir / "e12_corrected_vs_m5.json").read_text())
    m5p_stored = stored["ceilings"]["per_seed"]["M5_prime_val_symmetric"]

    m5p_scores, m5p_ap = {}, []
    for s in seeds:
        p_te, _, it, _ = E12.fit_val_symmetric(Xtr5, ytr, Xva5, yva, Xte5, s)
        m5p_scores[s] = p_te
        ap = average_precision_score(yte, p_te)
        m5p_ap.append(ap)
        assert abs(ap - m5p_stored[s]) < 5e-4, \
            f"M5' refit deviates from stored e12 value at seed {s}"
    print(f"[r3] M5' refit matches stored e12 per-seed values "
          f"(mean {np.mean(m5p_ap):.4f})", flush=True)

    # paired Wilcoxon + BH across the four-model family, both ceilings
    packs, pvals = {}, {"M5": [], "M5prime": []}
    for name, ps in family.items():
        g = [ps[s]["auc_pr"] for s in seeds]
        packs[name] = {"g": g}
        packs[name]["w_M5"] = S.paired_wilcoxon(g, m5_pre)
        packs[name]["w_M5prime"] = S.paired_wilcoxon(g, m5p_ap)
        pvals["M5"].append(packs[name]["w_M5"]["p"])
        pvals["M5prime"].append(packs[name]["w_M5prime"]["p"])
    adj = {fam: S.bh_correction(pv)[0] for fam, pv in pvals.items()}

    # 2000-draw paired student-level bootstrap (e12's F4 design)
    rng = np.random.default_rng(0)
    n = len(yte)
    idx_draws = rng.integers(0, n, size=(E12.N_BOOT, n))
    m5p_boot = np.empty((len(seeds), E12.N_BOOT))
    for b_ in range(E12.N_BOOT):
        idx = idx_draws[b_]
        yb = yte[idx]
        for s in seeds:
            m5p_boot[s, b_] = E12.fast_auc_pr(yb, m5p_scores[s][idx])

    def student_ci(ps):
        gap = np.empty((len(seeds), E12.N_BOOT))
        for b_ in range(E12.N_BOOT):
            idx = idx_draws[b_]
            yb = yte[idx]
            for s in seeds:
                gap[s, b_] = E12.fast_auc_pr(yb, ps[s]["scores"][idx])
        pooled = (gap - m5p_boot).mean(axis=0)
        return [round(float(np.percentile(pooled, 2.5)), 4),
                round(float(np.percentile(pooled, 97.5)), 4)]

    out_models = {}
    for i, (name, ps) in enumerate(family.items()):
        g = packs[name]["g"]
        ci = student_ci(ps)
        gates = {"seed_mean_gt_ceiling": bool(np.mean(g) > np.mean(m5p_ap)),
                 "p_adj_lt_0.05": bool(adj["M5prime"][i] < 0.05),
                 "student_ci_lower_gt_0": bool(ci[0] > 0)}
        out_models[name] = {
            "seed_mean_auc_pr": round(float(np.mean(g)), 4),
            "delta_vs_M5": round(float(packs[name]["w_M5"]["mean_diff"]), 4),
            "p_adj_M5": round(float(adj["M5"][i]), 6),
            "delta_vs_M5prime": round(float(packs[name]["w_M5prime"]["mean_diff"]), 4),
            "p_adj_M5prime": round(float(adj["M5prime"][i]), 6),
            "student_ci95_vs_M5prime": ci,
            "gates_fair": gates, "exceeds_fair": all(gates.values())}
        print(f"[r3] {name}: mean {np.mean(g):.4f} dM5' "
              f"{packs[name]['w_M5prime']['mean_diff']:+.4f} "
              f"(p_adj {adj['M5prime'][i]:.4f}) CI {ci} "
              f"exceeds={out_models[name]['exceeds_fair']}", flush=True)

    sel = json.loads((OUT_DIR / "grid_selection.json").read_text())
    verdict = out_models["rgcn_symmetric"]
    out = {
        "experiment": "R3_rgcn_symmetric", "field": FIELD,
        "note": ("post-hoc robustness; RGCN under the HGT-tuned grid protocol "
                 "(16 configs, seeds 0/1, budget 200/15, weight decay 1e-4, "
                 "train-only class weight), winner at 10 seeds under both "
                 "budgets; corrected eq.(2) protocol vs M5' with BH across the "
                 "four-model family, symmetric RGCN substituted for the "
                 "original RGCN"),
        "grid_selection": sel,
        "winner_10seed": {
            "hgt_budget_mean_auc_pr": round(float(np.mean(
                [family['rgcn_symmetric'][s]['auc_pr'] for s in seeds])), 4),
            "original_budget_mean_auc_pr": round(float(np.mean(
                [rgcn_orig300[s]['auc_pr'] for s in seeds])), 4),
            "frozen_original_rgcn_mean_auc_pr": round(float(np.mean(
                [rgcn_orig_frozen[s]['auc_pr'] for s in seeds])), 4),
        },
        "ceilings": {"M5_preregistered": round(float(np.mean(m5_pre)), 4),
                     "M5_prime_val_symmetric": round(float(np.mean(m5p_ap)), 4)},
        "models": out_models,
        "rgcn_symmetric_exceeds_fair": verdict["exceeds_fair"],
    }
    path = ROOT / "results" / "robustness" / "rgcn_symmetric_verdict.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[r3] wrote {path}; rgcn_symmetric exceeds_fair = "
          f"{verdict['exceeds_fair']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "aggregate"])
    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train()
    else:
        cmd_aggregate()
