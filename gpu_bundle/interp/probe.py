"""Experiment 1: layer x segment linear probing with control tasks.

For every encoder / input mode / target / shape / segment / layer: standardised logistic regression,
StratifiedGroupKFold(5) grouped by conversation_id (same grouping as the earlier fine-tuning), several seeds.
Each cell is also probed on a CONTROL task: the same labels randomly permuted (fixed permutation per seed),
which measures what the probe can fit on its own (Hewitt & Liang 2019 adapted to sentence-level targets).
selectivity = AUC(real) - AUC(control).

Targets
  label    fallacy vs NONE                                   rows: main dataset (683), per shape
  trigger  claim B contains a trigger phrase (extended list)  rows: main + minimal-pair edits (variation)
  entail   frozen NLI model says A entails B (P > median)     rows: main dataset
  narrow   A => B more than B => A (entail_AB - entail_BA > median); NLI proxy for "B narrows A"; MB only

Segment 'trig' is only probed on rows that have trigger tokens (otherwise its zero vector leaks presence),
and never for the trigger target.

Usage: python probe.py [--encoders nli_pair,plain_pair,random_pair,nli_b_only] [--seeds 0,1,2] [--jobs -1]
Writes results/probe_results.csv
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from common import load_feats, POSITIVE, RESULTS, SEGMENTS, load_all
from triggers_ext import has_trigger

C_REG = 0.1


def oof_probs(X, y, groups, seed, C=C_REG):
    prob = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        prob[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return prob


def cell(X, y, groups, seeds, meta):
    out = []
    for seed in seeds:
        p = oof_probs(X, y, groups, seed)
        y_ctrl = np.random.default_rng(1000 + seed).permutation(y)
        pc = oof_probs(X, y_ctrl, groups, seed)
        out.append(dict(**meta, seed=seed, auc=roc_auc_score(y, p), acc=float(((p > .5) == y).mean()),
                        auc_control=roc_auc_score(y_ctrl, pc), n=len(y), n_pos=int(y.sum())))
    return out


def targets(df):
    """dict target -> (row mask, y) per shape."""
    out = {}
    main = df.set.values == "main"
    trig = np.array([has_trigger(b, s) for b, s in zip(df.claim_B, df.type_candidate)])
    ent = df.entail_AB.values
    narrow = df.entail_AB.values - df.entail_BA.values
    for shape, pos in POSITIVE.items():
        sh = df.type_candidate.values == shape
        m = main & sh
        out[("label", shape)] = (m, (df.label.values == pos).astype(int))
        out[("trigger", shape)] = (sh, trig.astype(int))
        out[("entail", shape)] = (m, (ent > np.nanmedian(ent[m])).astype(int))
        if shape == "motte_bailey":
            out[("narrow", shape)] = (m, (narrow > np.nanmedian(narrow[m])).astype(int))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="nli_pair,plain_pair,random_pair,nli_b_only")
    ap.add_argument("--targets", default="label,trigger,entail,narrow")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out", default="probe_results.csv")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    df = load_all()
    T = targets(df)
    rows = []
    for enc in args.encoders.split(","):
        t0 = time.time()
        X, seg_n = load_feats(enc, df)
        layers = range(X.shape[0]) if args.layers == "all" else [int(l) for l in args.layers.split(",")]
        jobs = []
        for (target, shape), (mask, y) in T.items():
            if target not in args.targets.split(","):
                continue
            for si, seg in enumerate(SEGMENTS):
                m = mask.copy()
                if seg == "trig":
                    if target == "trigger":
                        continue
                    m &= seg_n[si] > 0
                m &= seg_n[si] > 0 if seg in ("A", "B", "brest", "cls", "mean") else True
                if seg == "C" and shape == "strawman":
                    continue
                if "b_only" in enc and seg in ("A", "C"):
                    continue
                yy = y[m]
                if m.sum() < 40 or yy.min() == yy.max() or min(yy.sum(), len(yy) - yy.sum()) < 10:
                    continue
                g = df.conversation_id.values[m]
                for layer in layers:
                    meta = dict(encoder=enc, target=target, shape=shape, segment=seg, layer=layer)
                    jobs.append(delayed(cell)(X[layer, si, m].astype(np.float32), yy, g, seeds, meta))
        res = Parallel(n_jobs=args.jobs)(jobs)
        rows += [r for rr in res for r in rr]
        print(f"{enc}: {len(jobs)} cells in {time.time() - t0:.0f}s", flush=True)
    RESULTS.mkdir(exist_ok=True)
    out = pd.DataFrame(rows)
    out["selectivity"] = out.auc - out.auc_control
    out.to_csv(RESULTS / args.out, index=False)
    print(f"wrote {RESULTS / args.out}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
