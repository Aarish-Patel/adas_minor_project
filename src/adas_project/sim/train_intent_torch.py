"""Train the v3 driver-intent (crash-risk) models on the digital-twin data (sim/twin_intent_data.py), on the laptop's GPU.

    python -m sim.train_intent_torch [--quick]   -> models/intent_v3.json, models/intent_v3_report.json,
                                                     reports/intent_v3.png, models/intent_training/ (live log for the GUI)

Candidates, all small enough for the Pi 5 in numpy (adas/intent_net.IntentNet):
  trees3  gradient-boosted trees on the tabular window features (scikit-learn, CPU)
  mlp3    multilayer perceptron on the same features (PyTorch, GPU)
  gru3    GRU over the raw 1.6 s window of tick vectors (PyTorch, GPU)
Training: runs split into train / validation by run (never ticks of one run on both sides), AdamW with a cosine
schedule, early stopping on validation average precision, then temperature scaling on the validation set so the
probabilities are calibrated (Guo et al. 2017). The chosen model is the best validation average precision.
Evaluation on held-out test runs (new rooms, new randomised cars), per situation slice, against the v2 model the car
ran until now: AUC, average precision, Brier score, expected calibration error, recall and false-alarm rate at the
relay's 0.5 threshold, and for runs that end in a crash how early the warning came.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "intent")
LOG_DIR = os.path.join(ROOT, "models", "intent_training")
THRESH = 0.5
DT = 0.05


# ---------------------------------------------------------------- data
def load(name):
    d = np.load(os.path.join(DATA, name + ".npz"), allow_pickle=False)
    return {k: d[k] for k in d.files}


def window_idx(d, rows):
    from adas.intent_net import WINDOW
    start = rows - d["tick"][rows]
    idx = rows[:, None] - np.arange(WINDOW - 1, -1, -1)[None, :]
    return np.maximum(idx, start[:, None])


def flat_all(d, rows, chunk=50_000):
    from adas.intent_net import flat_features_batch, physics_floor_batch
    X, F = [], []
    for i in range(0, len(rows), chunk):
        Wb = d["Z"][window_idx(d, rows[i:i + chunk])].astype(np.float64)
        X.append(flat_features_batch(Wb).astype(np.float32))
        F.append(physics_floor_batch(Wb).astype(np.float32))
    return np.concatenate(X), np.concatenate(F)


def split_runs(d, frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d["run_family"])
    val = rng.random(n) < frac
    return ~val[d["run"]], val[d["run"]]


# ---------------------------------------------------------------- metrics
def ece(p, y, bins=10):
    e, n = 0.0, len(p)
    for a in range(bins):
        m = (p >= a / bins) & (p < (a + 1) / bins if a < bins - 1 else p <= 1.0)
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def metrics(p, y):
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = {"n": int(len(y)), "positives": int(y.sum())}
    if len(y) == 0:
        return out
    both = 0 < y.sum() < len(y)
    out["auc"] = float(roc_auc_score(y, p)) if both else None
    out["ap"] = float(average_precision_score(y, p)) if y.sum() else None
    out["brier"] = float(np.mean((p - y) ** 2))
    out["ece"] = ece(p, y)
    out["recall"] = float((p[y == 1] >= THRESH).mean()) if y.sum() else None
    out["false_alarm"] = float((p[y == 0] >= THRESH).mean()) if (y == 0).any() else None
    return out


def slices(d, rows):
    """Named boolean masks over `rows` of a dataset."""
    from adas.intent_net import HORIZON3, Z_BACK, Z_FREE_NOW, Z_STICK, Z_THR
    Z = d["Z"][rows]
    fam = d["run_family"][d["run"][rows]]
    sty = d["run_style"][d["run"][rows]]
    v = np.abs(d["v_true"][rows])
    fwd = ~((d["v_true"][rows] < -0.05) | ((np.abs(d["v_true"][rows]) <= 0.05) & (Z[:, Z_THR] < 0)))
    free = np.where(fwd, Z[:, Z_FREE_NOW], Z[:, Z_BACK]) * HORIZON3
    tte = d["tte"][rows]
    m = {"all": np.ones(len(rows), bool)}
    for f in np.unique(fam):
        m[f"family: {f}"] = fam == f
    for f, s in sorted(set(zip(fam, sty))):
        m[f"{f} / {s}"] = (fam == f) & (sty == s)
    for a, b, name in ((0, 0.2, "slow < 0.2 m/s"), (0.2, 0.5, "medium 0.2-0.5 m/s"), (0.5, 9, "fast > 0.5 m/s")):
        m[f"speed: {name}"] = (v >= a) & (v < b)
    for a, b in ((0, 0.5), (0.5, 1.2), (1.2, 2.49), (2.49, 99)):
        m[f"free way: {a}-{b if b < 99 else 'max'} m"] = (free >= a) & (free < b)
    m["direction: reversing"] = ~fwd
    m["direction: forward"] = fwd
    m["steering: turning"] = np.abs(Z[:, Z_STICK]) * 30 > 10
    m["steering: straight"] = np.abs(Z[:, Z_STICK]) * 30 <= 10
    m["driver: lapsed / frozen"] = d["lapsed"][rows]
    m["driver: attentive"] = ~d["lapsed"][rows]
    for a, b in ((0, 0.5), (0.5, 1.0), (1.0, 2.0)):
        m[f"contact in {a}-{b} s (positives)"] = (tte >= a) & (tte < b)
    return m


def run_level(d, rows, p):
    """Crash runs: warning lead time (s before contact that the risk first stayed >= 0.5 up to the crash);
    safe runs (no event at all): fraction with a false alarm lasting >= 0.25 s."""
    run = d["run"][rows]
    leads, alarms_safe, n_safe = [], 0, 0
    for r in np.unique(run):
        m = run == r
        pr, yr = p[m], d["y"][rows][m]
        if d["run_crashed"][r]:
            above = pr >= THRESH
            k = len(pr)
            while k > 0 and above[k - 1]:
                k -= 1
            leads.append((len(pr) - k) * DT)
        elif yr.sum() == 0:
            n_safe += 1
            run_len = np.convolve((pr >= THRESH).astype(int), np.ones(5, int), "valid")
            alarms_safe += int((run_len >= 5).any())
    leads = np.array(leads)
    return {"crash_runs": int(len(leads)), "lead_median_s": float(np.median(leads)) if len(leads) else None,
            "warned_0.5s_before": float((leads >= 0.5).mean()) if len(leads) else None,
            "warned_1s_before": float((leads >= 1.0).mean()) if len(leads) else None,
            "missed": float((leads == 0).mean()) if len(leads) else None,
            "safe_runs": n_safe, "safe_runs_with_false_alarm": float(alarms_safe / max(n_safe, 1))}


def reliability(p, y, bins=10):
    """Reliability diagram: mean predicted probability vs observed frequency per bin (calibration)."""
    out = []
    for a in range(bins):
        m = (p >= a / bins) & ((p < (a + 1) / bins) if a < bins - 1 else (p <= 1.0))
        if m.sum() >= 20:
            out.append([float(p[m].mean()), float(y[m].mean()), int(m.sum())])
    return out


def save_examples(d, p_old, p_new, per_family=2):
    """A few held-out drives per situation family with both models' risk over time, for the GUI and slides:
    one that ends in contact and one that does not, where available."""
    from adas.intent_net import HORIZON3, Z_BACK, Z_FREE_NOW, Z_THR
    ex = []
    rng = np.random.default_rng(3)
    for fam in np.unique(d["run_family"]):
        for crashed in (True, False):
            cand = np.flatnonzero((d["run_family"] == fam) & (d["run_crashed"] == crashed))
            for r in rng.permutation(cand)[:per_family // 2 or 1]:
                m = d["run"] == r
                Z = d["Z"][m]
                fwd = d["v_true"][m] >= -0.05
                ex.append({"run": int(r), "family": str(fam), "style": str(d["run_style"][r]), "crashed": bool(crashed),
                           "t": (np.arange(m.sum()) * DT).round(2).tolist(), "y": d["y"][m].astype(int).tolist(),
                           "p_old": np.round(p_old[m], 3).tolist(), "p_new": np.round(p_new[m], 3).tolist(),
                           "v": np.round(d["v_true"][m], 3).tolist(), "throttle": np.round(Z[:, Z_THR] * 255).tolist(),
                           "free": np.round(np.where(fwd, Z[:, Z_FREE_NOW], Z[:, Z_BACK]) * HORIZON3, 3).tolist()})
    json.dump(ex, open(os.path.join(LOG_DIR, "examples.json"), "w"))


# ---------------------------------------------------------------- live log (the GUI's training tab reads these)
class Log:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.path = os.path.join(LOG_DIR, "log.jsonl")
        open(self.path, "w").close()
        self.status_path = os.path.join(LOG_DIR, "status.json")

    def epoch(self, **kw):
        with open(self.path, "a") as f:
            f.write(json.dumps(kw) + "\n")

    def status(self, **kw):
        tmp = self.status_path + ".tmp"
        json.dump(dict(kw, t=time.time()), open(tmp, "w"))
        os.replace(tmp, self.status_path)


# ---------------------------------------------------------------- torch models
def torch_models():
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, n_in, widths, dropout):
            super().__init__()
            layers, a = [], n_in
            for w in widths:
                layers += [nn.Linear(a, w), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
                a = w
            layers.append(nn.Linear(a, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class GRUNet(nn.Module):
        def __init__(self, n_in, hidden):
            super().__init__()
            self.gru = nn.GRU(n_in, hidden, batch_first=True)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x):
            _, h = self.gru(x)
            return self.out(h[-1]).squeeze(-1)

    return torch, nn, MLP, GRUNet


def fit_temperature(logits, y):
    import torch
    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lg / logT.exp(), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp())


def train_torch(name, kind, cfg, data, log, epochs, device):
    """kind 'mlp': data = (Xtr, ytr, Xva, yva) flat; 'gru': data = (Zt, idx_tr, ytr, idx_va, yva) windows gathered
    on the GPU. Returns (model, mu, sd, best val logits, history)."""
    torch, nn, MLP, GRUNet = torch_models()
    from sklearn.metrics import average_precision_score, roc_auc_score
    torch.manual_seed(cfg.get("seed", 0))
    if kind == "mlp":
        Xtr, ytr, Xva, yva = data
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xt = torch.tensor((Xtr - mu) / sd, device=device)
        Xv = torch.tensor((Xva - mu) / sd, device=device)
        model = MLP(Xtr.shape[1], cfg["widths"], cfg["dropout"]).to(device)
        n_tr = len(Xtr)

        def batch(ix, val=False):
            return (Xv if val else Xt)[ix]
    else:
        Z, itr, ytr, iva, yva = data
        mu, sd = Z.mean(0), Z.std(0) + 1e-6
        Zt = torch.tensor((Z - mu) / sd, device=device)
        It, Iv = torch.tensor(itr, device=device), torch.tensor(iva, device=device)
        model = GRUNet(Z.shape[1], cfg["hidden"]).to(device)
        n_tr = len(itr)

        def batch(ix, val=False):
            return Zt[(Iv if val else It)[ix]]
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    bs = cfg.get("batch", 4096)
    steps = epochs * math.ceil(n_tr / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, best_logits, bad, hist = -1.0, None, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        tot = 0.0
        for i in range(0, n_tr, bs):
            ix = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(batch(ix)), yt[ix])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(ix)
        model.eval()
        with torch.no_grad():
            lv = torch.cat([model(batch(torch.arange(i, min(i + 65536, len(yva)), device=device), True))
                            for i in range(0, len(yva), 65536)]).cpu().numpy()
        pv = 1 / (1 + np.exp(-lv))
        vloss = float(np.mean(-(yva * np.log(pv + 1e-7) + (1 - yva) * np.log(1 - pv + 1e-7))))
        ap, auc = float(average_precision_score(yva, pv)), float(roc_auc_score(yva, pv))
        row = {"model": name, "epoch": ep + 1, "train_loss": tot / n_tr, "val_loss": vloss, "val_ap": ap,
               "val_auc": auc, "lr": float(sched.get_last_lr()[0]), "s": round(time.time() - t0, 1)}
        hist.append(row)
        log.epoch(**row)
        log.status(state="training", model=name, epoch=ep + 1, epochs=epochs, val_ap=ap, val_auc=auc, best_ap=max(best, ap))
        if ap > best + 1e-4:
            best, bad, best_logits = ap, 0, lv
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.get("patience", 20):
                break
    model.load_state_dict(best_state)
    print(f"  {name}: best val AP {best:.4f} after {len(hist)} epochs, {time.time() - t0:.0f} s", flush=True)
    return model, mu, sd, best_logits, hist


def export_mlp(model, mu, sd, T):
    lin = [m for m in model.net if m.__class__.__name__ == "Linear"]
    return {"kind": "mlp3", "mu": mu.tolist(), "sd": sd.tolist(), "temperature": T,
            "W": [m.weight.detach().cpu().numpy().T.tolist() for m in lin],
            "b": [m.bias.detach().cpu().numpy().tolist() for m in lin]}


def export_gru(model, mu, sd, T):
    g = model.gru
    c = lambda t: t.detach().cpu().numpy().tolist()
    return {"kind": "gru3", "mu": mu.tolist(), "sd": sd.tolist(), "temperature": T,
            "W_ih": c(g.weight_ih_l0), "W_hh": c(g.weight_hh_l0), "b_ih": c(g.bias_ih_l0), "b_hh": c(g.bias_hh_l0),
            "W_out": c(model.out.weight)[0], "b_out": c(model.out.bias)}


def predict_torch(kind, model, mu, sd, T, d, rows, X=None, device="cpu"):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        if kind == "mlp":
            Xt = torch.tensor((X - mu) / sd, device=device)
            for i in range(0, len(X), 65536):
                out.append(model(Xt[i:i + 65536]).cpu().numpy())
        else:
            Zt = torch.tensor((d["Z"] - mu) / sd, device=device)
            idx = torch.tensor(window_idx(d, rows), device=device)
            for i in range(0, len(rows), 65536):
                out.append(model(Zt[idx[i:i + 65536]]).cpu().numpy())
    return 1 / (1 + np.exp(-np.concatenate(out) / T))


# ---------------------------------------------------------------- main
def main(quick=False):
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sim.train_intent_net import export_trees
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU only"
    print(f"device: {gpu}", flush=True)
    log = Log()
    tr, te = load("train"), load("test")
    rows_all = np.arange(len(tr["y"]))
    m_tr, m_va = split_runs(tr)
    r_tr, r_va = rows_all[m_tr], rows_all[m_va]
    t0 = time.time()
    X_all, _ = flat_all(tr, rows_all)
    Xtr, Xva = X_all[r_tr], X_all[r_va]
    ytr, yva = tr["y"][r_tr].astype(np.float32), tr["y"][r_va].astype(np.float32)
    rows_te = np.arange(len(te["y"]))
    Xte, floor_te = flat_all(te, rows_te)
    yte = te["y"].astype(np.float32)
    print(f"features: train {len(Xtr)} val {len(Xva)} test {len(Xte)} ticks, {Xtr.shape[1]} features, "
          f"positives train {ytr.mean():.3f} test {yte.mean():.3f}, {time.time() - t0:.0f} s", flush=True)
    data_info = {"train_ticks": int(len(Xtr)), "val_ticks": int(len(Xva)), "test_ticks": int(len(Xte)),
                 "train_runs": int(len(np.unique(tr["run"][r_tr]))), "val_runs": int(len(np.unique(tr["run"][r_va]))),
                 "test_runs": int(len(te["run_family"])), "positive_rate_train": float(ytr.mean()),
                 "families": {f: int((tr["run_family"] == f).sum()) for f in np.unique(tr["run_family"])},
                 "dr_keys": tr["dr_keys"].tolist(), "dr_min": tr["run_dr"].min(0).tolist(),
                 "dr_max": tr["run_dr"].max(0).tolist(), "device": gpu}
    log.status(state="starting", data=data_info)
    epochs = 6 if quick else 200
    cands = {}

    # gradient-boosted trees (CPU)
    t1 = time.time()
    gbm = HistGradientBoostingClassifier(max_iter=60 if quick else 400, max_depth=6, learning_rate=0.08,
                                         early_stopping=True, validation_fraction=0.1, random_state=0)
    gbm.fit(Xtr, ytr)
    raw_va = gbm.decision_function(Xva)
    T = fit_temperature(raw_va, yva)
    ex = export_trees(gbm)
    ex.update(kind="trees3", temperature=T)
    p_te = 1 / (1 + np.exp(-gbm.decision_function(Xte) / T))
    cands["trees3"] = {"export": ex, "p_test": p_te, "val_ap": float(metrics(1 / (1 + np.exp(-raw_va / T)), yva)["ap"]),
                       "params": int(np.sum([len(t[0].nodes) for t in gbm._predictors])), "cfg": {"trees": gbm.n_iter_}}
    log.epoch(model="trees3", epoch=gbm.n_iter_, val_ap=cands["trees3"]["val_ap"], s=round(time.time() - t1, 1))
    print(f"  trees3: {gbm.n_iter_} trees, val AP {cands['trees3']['val_ap']:.4f}, {time.time() - t1:.0f} s", flush=True)

    grid = [("mlp3-128x128", "mlp", {"widths": [128, 128], "dropout": 0.0, "lr": 2e-3, "wd": 1e-4}),
            ("mlp3-256x128", "mlp", {"widths": [256, 128], "dropout": 0.1, "lr": 2e-3, "wd": 1e-4}),
            ("mlp3-256x256x128", "mlp", {"widths": [256, 256, 128], "dropout": 0.1, "lr": 1.5e-3, "wd": 3e-4}),
            ("gru3-32", "gru", {"hidden": 32, "lr": 3e-3, "wd": 1e-4, "patience": 12}),
            ("gru3-64", "gru", {"hidden": 64, "lr": 2e-3, "wd": 1e-4, "patience": 12})]
    if quick:
        grid = [grid[0], grid[3]]
    Zf = tr["Z"].astype(np.float32)
    idx_tr, idx_va = window_idx(tr, r_tr), window_idx(tr, r_va)
    hist_all = {}
    for name, kind, cfg in grid:
        data = (Xtr, ytr, Xva, yva) if kind == "mlp" else (Zf, idx_tr, ytr, idx_va, yva)
        model, mu, sd, lv, hist = train_torch(name, kind, cfg, data, log,
                                              epochs if kind == "mlp" else max(3, epochs // 2), device)
        T = fit_temperature(lv, yva)
        ex = export_mlp(model, mu, sd, T) if kind == "mlp" else export_gru(model, mu, sd, T)
        p_te = predict_torch(kind, model, mu, sd, T, te, rows_te, Xte, device)
        cands[name] = {"export": ex, "p_test": p_te, "val_ap": float(metrics(1 / (1 + np.exp(-lv / T)), yva)["ap"]),
                       "params": int(sum(p.numel() for p in model.parameters())), "cfg": cfg, "kind": kind}
        hist_all[name] = hist

    # the v2 model the car ran until now, on the same test ticks (its features exist from tick 16 on)
    from adas.intent_net import IntentNet
    old = IntentNet(os.path.join(ROOT, "pi", "intent_net.json"))
    V2 = te["V2"]
    ok = ~np.isnan(V2).any(axis=1)
    p_old = np.full(len(V2), np.nan)
    p_old[ok] = [old.crash_probability(x) for x in V2[ok]]

    # evaluation on the same ticks for everyone (those the old model can score)
    sl = slices(te, rows_te)
    best_name = max(cands, key=lambda k: cands[k]["val_ap"])
    report = {"data": data_info, "chosen": best_name, "candidates": {}, "slices": {}, "run_level": {}}
    for k, c in cands.items():
        report["candidates"][k] = {"val_ap": c["val_ap"], "params": c["params"], "cfg": c["cfg"],
                                   "test": metrics(c["p_test"][ok], yte[ok])}
    p_new = cands[best_name]["p_test"]
    p_new_floor = np.maximum(p_new, floor_te)
    for nm, m in sl.items():
        mm = m & ok
        report["slices"][nm] = {"old_v2": metrics(p_old[mm], yte[mm]), "new": metrics(p_new[mm], yte[mm]),
                                "new_floor": metrics(p_new_floor[mm], yte[mm])}
    p_old_filled = np.where(ok, p_old, 0.0)
    report["run_level"] = {"old_v2": run_level(te, rows_te, p_old_filled), "new": run_level(te, rows_te, p_new),
                           "new_floor": run_level(te, rows_te, p_new_floor)}
    report["reliability"] = {k: reliability(p[ok], yte[ok]) for k, p in (("old_v2", p_old), ("new", p_new_floor))}
    save_examples(te, p_old_filled, p_new_floor)

    # the car's numpy evaluator must give the same numbers as training
    from adas.intent_net import IntentNet as Net
    ex = cands[best_name]["export"]
    ex["report"] = {k: report[k] for k in ("chosen", "data")}
    ex["report"]["test_all"] = report["slices"]["all"]["new"]
    # models/ only: the relay picks up pi/intent_v3.json, which is copied there after the Monte Carlo check
    json.dump(ex, open(os.path.join(ROOT, "models", "intent_v3.json"), "w"))
    net = Net(os.path.join(ROOT, "models", "intent_v3.json"))
    samp = np.random.default_rng(0).choice(len(rows_te), 300, replace=False)
    Wd = te["Z"][window_idx(te, rows_te[samp])].astype(np.float64)
    mine = np.array([net.risk(W, floor=False) for W in Wd])
    report["numpy_vs_training_max_diff"] = float(np.abs(mine - p_new[samp]).max())
    t_inf = time.perf_counter()
    for W in Wd[:200]:
        net.risk(W)
    report["numpy_ms_per_tick_laptop"] = (time.perf_counter() - t_inf) / 200 * 1000
    json.dump(report, open(os.path.join(ROOT, "models", "intent_v3_report.json"), "w"), indent=1)
    json.dump(hist_all, open(os.path.join(LOG_DIR, "history.json"), "w"))
    log.status(state="done", chosen=best_name, test=report["slices"]["all"]["new_floor"], data=data_info)
    figure(report)
    a = report["slices"]["all"]
    print(json.dumps({"chosen": best_name, "old_v2": a["old_v2"], "new": a["new"], "new_floor": a["new_floor"],
                      "run_level": report["run_level"], "numpy_diff": report["numpy_vs_training_max_diff"],
                      "numpy_ms": report["numpy_ms_per_tick_laptop"]}, indent=1))


def figure(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [k for k in report["slices"] if not k.startswith("all") and " / " not in k
             and report["slices"][k]["new"].get("positives", 0) >= 20]
    names = ["all"] + names
    fig, axes = plt.subplots(1, 3, figsize=(17, 0.34 * len(names) + 1.8), sharey=True)
    y = np.arange(len(names))
    for ax, key, title in zip(axes, ("ap", "recall", "false_alarm"),
                              ("average precision (higher = better)", "recall at 0.5 (higher = better)",
                               "false-alarm rate at 0.5 (lower = better)")):
        for off, who, col, lab in ((-0.2, "old_v2", "#9ca3af", "v2 (car until now)"),
                                   (0.2, "new_floor", "#2563eb", f"v3 {report['chosen']} + physics floor")):
            vals = [report["slices"][n][who].get(key) or 0 for n in names]
            ax.barh(y + off, vals, 0.38, color=col, label=lab)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Driver-intent crash risk on held-out digital-twin runs (randomised cars and sensors), per situation")
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "reports", "intent_v3.png"), dpi=110)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
