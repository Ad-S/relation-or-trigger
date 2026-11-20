"""Experiment 4: concept erasure with LEACE (Belrose et al. 2023), amnesic-probing style (Elazar et al. 2021).

(a) probe mode (cached features): per encoder / shape / layer / segment, inside each grouped fold
    fit a LEACE eraser for a concept (trigger / entail / label) on the training rows, erase it from train and
    test features, and re-probe the fallacy label. Control: erase one random direction (LEACE on a concept
    is rank 1 for a binary concept). Reports label AUC before / after / after-random.
(b) forward mode: erase the concept from EVERY token of the residual stream at state l (affine eraser fit on
    mean-pooled training features of that state), run the rest of the frozen encoder, and score the held-out
    main rows with the layer-L_r readout of the fold (fit on unedited features). Causal version of (a).

Usage: python erasure.py --mode probe [--encoders nli_pair,plain_pair]
       python erasure.py --mode forward --encoder nli --layers 0,4,8,12,16,20 --concepts trigger,entail
Writes results/erasure_probe.csv or results/erasure_forward_<encoder>.csv
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch
from concept_erasure import LeaceEraser
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from common import POSITIVE, RESULTS, SEGMENTS, device, encode_with_segments, load_all, load_encoder, load_feats, load_tokenizer
from probe import C_REG
from triggers_ext import has_trigger


def concept_vectors(df, shape):
    sh = df.type_candidate.values == shape
    main = sh & (df.set.values == "main")
    trig = np.array([has_trigger(b, shape) for b in df.claim_B]).astype(int)
    ent = df.entail_AB.values
    ent_bin = (ent > np.nanmedian(ent[main])).astype(int)
    lab = (df.label.values == POSITIVE[shape]).astype(int)
    # rows each concept eraser may be fit on (trigger varies most when the trigger-swap edits are included)
    return {"trigger": (trig, sh), "entail": (ent_bin, main), "label": (lab, main)}, lab, main


def lr(Z, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return LogisticRegression(C=C_REG, max_iter=2000).fit(Z, y)


def fit_eraser(X, z, dev="cpu"):
    return LeaceEraser.fit(torch.tensor(X, dtype=torch.float32, device=dev),
                           torch.tensor(z, dtype=torch.float32, device=dev).unsqueeze(1))


def random_eraser(dim, rng):
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return lambda X: X - np.outer(X @ v, v)


def probe_cell(Xl, df, shape, seeds, meta):
    C, lab, main = concept_vectors(df, shape)
    main_idx = np.where(main)[0]
    conv = df.conversation_id.values
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        probs = {k: np.zeros(len(main_idx)) for k in ["none", "random"] + list(C)}
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(main_idx, lab[main_idx], conv[main_idx]):
            tr_i, te_i = main_idx[tr], main_idx[te]
            train_convs = set(conv[tr_i])
            variants = {"none": lambda X: X, "random": random_eraser(Xl.shape[1], rng)}
            for name, (z, fit_rows) in C.items():
                rows = np.where(fit_rows & np.isin(conv, list(train_convs)))[0]
                if z[rows].min() == z[rows].max():
                    continue
                er = fit_eraser(Xl[rows], z[rows])
                variants[name] = lambda X, er=er: er(torch.tensor(X)).numpy()
            for name, f in variants.items():
                Xtr, Xte = f(Xl[tr_i]), f(Xl[te_i])
                sc = StandardScaler().fit(Xtr)
                probs[name][te] = lr(sc.transform(Xtr), lab[tr_i]).predict_proba(sc.transform(Xte))[:, 1]
        y = lab[main_idx]
        for name, p in probs.items():
            out.append(dict(**meta, seed=seed, erased=name, auc=roc_auc_score(y, p)))
    return out


def run_probe(args):
    df = load_all()
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for enc in args.encoders.split(","):
        t0 = time.time()
        X, _ = load_feats(enc, df)
        layers = range(X.shape[0]) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
        jobs = [delayed(probe_cell)(X[l, SEGMENTS.index(seg)].astype(np.float32), df, shape, seeds,
                                    dict(encoder=enc, shape=shape, segment=seg, layer=l))
                for shape in POSITIVE for seg in args.segments.split(",") for l in layers]
        rows += [r for rr in Parallel(n_jobs=args.jobs)(jobs) for r in rr]
        print(f"{enc}: {len(jobs)} cells in {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "erasure_probe.csv", index=False)
    print("wrote results/erasure_probe.csv")


class EraseHook:
    def __init__(self, model):
        self.eraser, self.state = None, None
        self.handles = [model.embeddings.register_forward_hook(self._hook(0))]
        self.handles += [l.register_forward_hook(self._hook(i + 1)) for i, l in enumerate(model.encoder.layer)]

    def _hook(self, state):
        def fn(module, inputs, output):
            if self.eraser is None or state != self.state:
                return None
            h = output[0] if isinstance(output, tuple) else output
            h2 = self.eraser(h.float()).to(h.dtype)
            return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2
        return fn


@torch.no_grad()
def run_forward(args):
    dev = device()
    df = load_all()
    X, _ = load_feats(f"{args.encoder}_pair", df)
    mean_i = SEGMENTS.index("mean")
    tok = load_tokenizer()
    model = load_encoder(args.encoder, dev)
    hook = EraseHook(model)
    layers = [int(x) for x in args.layers.split(",")]
    rows = []
    t0 = time.time()
    for shape in POSITIVE:
        C, lab, main = concept_vectors(df, shape)
        main_idx = np.where(main)[0]
        conv = df.conversation_id.values
        Xr = X[args.readout_layer, mean_i].astype(np.float32)
        for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(
                main_idx, lab[main_idx], conv[main_idx])):
            tr_i, te_i = main_idx[tr], main_idx[te]
            sc = StandardScaler().fit(Xr[tr_i])
            clf = lr(sc.transform(Xr[tr_i]), lab[tr_i])
            mu = torch.tensor(sc.mean_, dtype=torch.float32, device=dev)
            sd = torch.tensor(sc.scale_, dtype=torch.float32, device=dev)
            w = torch.tensor(clf.coef_[0], dtype=torch.float32, device=dev)
            b = float(clf.intercept_[0])
            train_convs = set(conv[tr_i])
            test = df.iloc[te_i]

            def score(eraser=None, state=None):
                hook.eraser, hook.state = eraser, state
                out = []
                for i in range(0, len(test), 16):
                    enc, _ = encode_with_segments(tok, test.iloc[i:i + 16], shape)
                    enc = enc.to(dev)
                    h = model(**enc, output_hidden_states=True).hidden_states[args.readout_layer].float()
                    m = enc["attention_mask"].unsqueeze(-1).float()
                    pooled = (h * m).sum(1) / m.sum(1)
                    out.append((((pooled - mu) / sd) @ w + b).cpu().numpy())
                hook.eraser = None
                return np.concatenate(out)

            base = score()
            rows += [dict(encoder=args.encoder, shape=shape, fold=fold, erased="none", layer=-1, row_id=r, y=int(yv), logit=float(s))
                     for r, yv, s in zip(test.row_id, lab[te_i], base)]
            rng = np.random.default_rng(fold)
            for layer in layers:
                Xl = X[layer, mean_i].astype(np.float32)
                erasers = {}
                for name in args.concepts.split(","):
                    z, fit_rows = C[name]
                    rws = np.where(fit_rows & np.isin(conv, list(train_convs)))[0]
                    er = fit_eraser(Xl[rws], z[rws], dev)
                    erasers[name] = er
                v = torch.tensor(rng.standard_normal(Xl.shape[1]), dtype=torch.float32, device=dev)
                v = v / v.norm()
                erasers["random"] = lambda h, v=v: h - (h @ v).unsqueeze(-1) * v
                for name, er in erasers.items():
                    s = score(er, layer)
                    rows += [dict(encoder=args.encoder, shape=shape, fold=fold, erased=name, layer=layer, row_id=r, y=int(yv), logit=float(x))
                             for r, yv, x in zip(test.row_id, lab[te_i], s)]
            print(f"{shape} fold {fold}: {time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"erasure_forward_{args.encoder}.csv", index=False)
    summ = out.groupby(["shape", "erased", "layer"]).apply(lambda g: roc_auc_score(g.y, g.logit)).rename("auc").reset_index()
    print(summ.pivot_table(index=["shape", "layer"], columns="erased", values="auc").round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "forward"], required=True)
    ap.add_argument("--encoders", default="nli_pair,plain_pair")
    ap.add_argument("--encoder", default="nli")
    ap.add_argument("--segments", default="mean")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--concepts", default="trigger,entail")
    ap.add_argument("--readout_layer", type=int, default=24)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    run_probe(args) if args.mode == "probe" else run_forward(args)


if __name__ == "__main__":
    main()
