"""Experiment 2: geometry of the minimal pairs along the fallacy and trigger directions.

Per encoder / shape / segment / layer, with 5 grouped folds (by conversation; seeds 0..):
  * fit, on training conversations only, a fallacy probe (main rows) and a trigger probe (main + edit rows),
    both on one shared standardisation; record cos(w_fallacy, w_trigger) in standardised space;
  * for each held-out minimal pair (original B -> edited B, A and C unchanged), the shift of the probe logit
    along each direction, in units of that probe's held-out logit sd:
        cf (label flips, trigger kept):    signed shift toward the new label     (relational -> large)
        ts (trigger toggled, label kept):  absolute shift                        (lexical -> large)
  * probe decisions on the pair: cf "correct flip" rate and ts "prediction kept" rate.
Segment A is especially informative: A's tokens are identical in both members of a pair, so any shift at A
is information that attention moved from B into A.

Usage: python geometry.py [--encoders nli_pair,plain_pair] [--segments mean,cls,A,B]
Writes results/geometry_pairs.csv (one row per pair x cell), results/geometry_summary.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from common import load_feats, POSITIVE, RESULTS, SEGMENTS, load_all
from probe import C_REG
from triggers_ext import has_trigger


def fit(Z, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return LogisticRegression(C=C_REG, max_iter=2000).fit(Z, y)


def run_cell(Xl, df, shape, seeds, meta):
    pos = POSITIVE[shape]
    sh = df.type_candidate.values == shape
    main = sh & (df.set.values == "main")
    pairs = np.where(sh & (df.set.values == "pairs"))[0]
    idx_of = {r: i for i, r in enumerate(df.row_id.values)}
    y = (df.label.values == pos).astype(int)
    trig = np.array([has_trigger(b, shape) for b in df.claim_B.values]).astype(int)
    conv = df.conversation_id.values
    main_idx = np.where(main)[0]
    pair_rows, cell_rows = [], []
    for seed in seeds:
        skf = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(main_idx, y[main_idx], conv[main_idx])):
            tr_i, te_i = main_idx[tr], main_idx[te]
            train_convs, test_convs = set(conv[tr_i]), set(conv[te_i])
            tr_pairs = pairs[np.isin(conv[pairs], list(train_convs))]
            te_pairs = pairs[np.isin(conv[pairs], list(test_convs))]
            sc = StandardScaler().fit(Xl[tr_i])
            f_clf = fit(sc.transform(Xl[tr_i]), y[tr_i])
            t_rows = np.concatenate([tr_i, tr_pairs])
            if trig[t_rows].min() == trig[t_rows].max():
                continue
            t_clf = fit(sc.transform(Xl[t_rows]), trig[t_rows])
            wf, wt = f_clf.coef_[0], t_clf.coef_[0]
            sf = f_clf.decision_function(sc.transform(Xl[te_i])).std() + 1e-9
            st = t_clf.decision_function(np.vstack([sc.transform(Xl[te_i]), sc.transform(Xl[te_pairs])])).std() + 1e-9
            cell_rows.append(dict(**meta, seed=seed, fold=fold,
                                  cos_fallacy_trigger=float(wf @ wt / np.linalg.norm(wf) / np.linalg.norm(wt))))
            for p in te_pairs:
                o = idx_of[df.source_pair_id.values[p]]
                Zo, Ze = sc.transform(Xl[[o]]), sc.transform(Xl[[p]])
                lf_o, lf_e = f_clf.decision_function(Zo)[0], f_clf.decision_function(Ze)[0]
                lt_o, lt_e = t_clf.decision_function(Zo)[0], t_clf.decision_function(Ze)[0]
                sign = 1 if y[p] == 1 else -1          # toward the edit's label
                pair_rows.append(dict(**meta, seed=seed, mp_id=df.row_id.values[p], edit_type=df.edit_type.values[p],
                                      y_orig=int(y[o]), y_edit=int(y[p]),
                                      fallacy_shift_signed=sign * (lf_e - lf_o) / sf,
                                      fallacy_shift_abs=abs(lf_e - lf_o) / sf,
                                      trigger_shift_abs=abs(lt_e - lt_o) / st,
                                      pred_orig=int(lf_o > 0), pred_edit=int(lf_e > 0)))
    return pair_rows, cell_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="nli_pair,plain_pair,random_pair")
    ap.add_argument("--segments", default="mean,cls,A,B")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    df = load_all()
    pair_rows, cell_rows = [], []
    for enc in args.encoders.split(","):
        X, _ = load_feats(enc, df)
        jobs = []
        for shape in POSITIVE:
            for seg in args.segments.split(","):
                if seg == "C" and shape == "strawman":
                    continue
                si = SEGMENTS.index(seg)
                for layer in range(X.shape[0]):
                    meta = dict(encoder=enc, shape=shape, segment=seg, layer=layer)
                    jobs.append(delayed(run_cell)(X[layer, si].astype(np.float32), df, shape, seeds, meta))
        for pr, cr in Parallel(n_jobs=args.jobs)(jobs):
            pair_rows += pr
            cell_rows += cr
        print(f"{enc}: {len(jobs)} cells", flush=True)
    P, Cc = pd.DataFrame(pair_rows), pd.DataFrame(cell_rows)
    P.to_csv(RESULTS / "geometry_pairs.csv", index=False)
    keys = ["encoder", "shape", "segment", "layer"]
    P["correct"] = np.where(P.edit_type == "cf", (P.pred_edit == P.y_edit) & (P.pred_orig == P.y_orig),
                            P.pred_edit == P.pred_orig)
    summ = P.groupby(keys + ["edit_type"]).agg(fallacy_shift_signed=("fallacy_shift_signed", "mean"),
                                               fallacy_shift_abs=("fallacy_shift_abs", "mean"),
                                               trigger_shift_abs=("trigger_shift_abs", "mean"),
                                               pair_behaviour=("correct", "mean"), n=("mp_id", "size")).reset_index()
    summ = summ.pivot_table(index=keys, columns="edit_type").reset_index()
    summ.columns = ["_".join(c).strip("_") for c in summ.columns]
    summ = summ.merge(Cc.groupby(keys).cos_fallacy_trigger.mean().reset_index(), on=keys)
    summ.to_csv(RESULTS / "geometry_summary.csv", index=False)
    print(f"wrote results/geometry_pairs.csv ({len(P)}), results/geometry_summary.csv ({len(summ)})")


if __name__ == "__main__":
    main()
