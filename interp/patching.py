"""Experiment 3: residual-stream activation patching on the minimal pairs (frozen encoder + linear readout).

Classifier under study: frozen encoder -> mean-pooled hidden state of layer L_r -> logistic-regression readout
fit on the training folds (grouped by conversation, seed 0). Pairs are only evaluated by the readout of the
fold in which their conversation is held out. Metric: readout logit LD (fallacy minus NONE).

clean = original example, corrupt = edited example (A and C identical, so every position before B -- [CLS],
A, [SEP], C, [SEP] -- has the same token in both runs and the same index).
  denoise S in {cls, A, C, ctx}: run CORRUPT, at state l overwrite positions S with the CLEAN run's states
  B (complement):                run CLEAN,   at state l overwrite all ctx positions with the CORRUPT states,
                                 so at layer l only B (and the final [SEP]) carries clean information
recovery = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt); 1 = segment at layer l carries the whole effect.
Pairs whose |LD_clean - LD_corrupt| is below --min_gap readout-sd are skipped (denominator too small).

Usage: python patching.py --encoder nli [--readout_layer 24] [--limit 10]
Writes results/patching_<encoder>_L<r>.csv (one row per pair x segment x layer) and a pairs file with the gaps.
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from common import load_feats, POSITIVE, RESULTS, SEGMENTS, device, encode_with_segments, load_all, load_encoder, load_tokenizer
from probe import C_REG


class Readout:
    def __init__(self, scaler, clf, layer, dev):
        self.layer = layer
        self.mu = torch.tensor(scaler.mean_, dtype=torch.float32, device=dev)
        self.sd = torch.tensor(scaler.scale_, dtype=torch.float32, device=dev)
        self.w = torch.tensor(clf.coef_[0], dtype=torch.float32, device=dev)
        self.b = float(clf.intercept_[0])

    def __call__(self, hidden_states, attn_mask):
        h = hidden_states[self.layer].float()
        m = attn_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1)
        return ((pooled - self.mu) / self.sd) @ self.w + self.b


def fold_readouts(df, X_layer, shape, layer, dev, seed=0):
    """Return {conversation_id: Readout fit on the folds that exclude it}."""
    pos = POSITIVE[shape]
    main_idx = np.where((df.type_candidate.values == shape) & (df.set.values == "main"))[0]
    y = (df.label.values[main_idx] == pos).astype(int)
    conv = df.conversation_id.values[main_idx]
    out, sds = {}, []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(main_idx, y, conv):
        sc = StandardScaler().fit(X_layer[main_idx[tr]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf = LogisticRegression(C=C_REG, max_iter=2000).fit(sc.transform(X_layer[main_idx[tr]]), y[tr])
        sds.append(clf.decision_function(sc.transform(X_layer[main_idx[te]])).std())
        ro = Readout(sc, clf, layer, dev)
        for c in set(conv[te]):
            out[c] = ro
    return out, float(np.mean(sds))


class Patcher:
    """Forward hooks that overwrite, for batch item i, the residual state `layer_of_item[i]` at `positions`
    with `source[layer][positions]`. State 0 = embeddings output, state l = output of encoder layer l-1."""

    def __init__(self, model):
        self.model, self.active, self.handles = model, None, []
        self.handles.append(model.embeddings.register_forward_hook(self._hook(0)))
        for i, layer in enumerate(model.encoder.layer):
            self.handles.append(layer.register_forward_hook(self._hook(i + 1)))

    def _hook(self, state):
        def fn(module, inputs, output):
            if self.active is None:
                return None
            items, positions, source = self.active
            h = output[0] if isinstance(output, tuple) else output
            sel = [i for i, l in enumerate(items) if l == state]
            if sel:
                h = h.clone()
                for i in sel:
                    h[i, positions] = source[state][positions].to(h.dtype)
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
            return None
        return fn

    def run(self, enc, items=None, positions=None, source=None):
        self.active = None if items is None else (items, positions, source)
        try:
            out = self.model(**enc, output_hidden_states=True)
        finally:
            self.active = None
        return out.hidden_states

    def remove(self):
        for h in self.handles:
            h.remove()


def batch_of(enc, n):
    return {k: v.expand(n, *v.shape[1:]).contiguous() for k, v in enc.items()}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="nli")
    ap.add_argument("--readout_layer", type=int, default=24)
    ap.add_argument("--min_gap", type=float, default=0.5, help="min |LD_clean - LD_corrupt| in readout sd units")
    ap.add_argument("--edit_types", default="cf,ts")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dev = device()
    df = load_all()
    X_r = load_feats(f"{args.encoder}_pair", df)[0][args.readout_layer, SEGMENTS.index("mean")].astype(np.float32)
    tok = load_tokenizer()
    model = load_encoder(args.encoder, dev)
    patcher = Patcher(model)
    n_states = model.config.num_hidden_layers + 1
    idx_of = {r: i for i, r in enumerate(df.row_id.values)}
    rows, pair_info = [], []
    t0 = time.time()
    for shape in POSITIVE:
        readouts, sd = fold_readouts(df, X_r, shape, args.readout_layer, dev)
        pairs = df[(df.type_candidate == shape) & (df.set == "pairs") & df.edit_type.isin(args.edit_types.split(","))]
        if args.limit:
            pairs = pairs.groupby("edit_type").head(args.limit)
        segs = ["cls", "A", "C", "ctx", "B"] if shape == "motte_bailey" else ["cls", "A", "ctx", "B"]
        for n_done, p in enumerate(pairs.itertuples()):
            if n_done % 25 == 0:
                print(f"  {shape} {n_done}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
            o = df.iloc[idx_of[p.source_pair_id]]
            two = pd.DataFrame([o, df.iloc[idx_of[p.row_id]]])
            enc_c, seg_c = encode_with_segments(tok, two.iloc[[0]], shape)
            enc_x, seg_x = encode_with_segments(tok, two.iloc[[1]], shape)
            k = int(seg_c["B"][0].nonzero()[0, 0])
            assert k == int(seg_x["B"][0].nonzero()[0, 0]) and torch.equal(enc_c["input_ids"][0, :k], enc_x["input_ids"][0, :k])
            enc_c, enc_x = enc_c.to(dev), enc_x.to(dev)
            ro = readouts[p.conversation_id]
            hs_c = patcher.run(enc_c)
            hs_x = patcher.run(enc_x)
            ld_c = ro(hs_c, enc_c["attention_mask"]).item()
            ld_x = ro(hs_x, enc_x["attention_mask"]).item()
            gap = (ld_c - ld_x) / sd
            info = dict(encoder=args.encoder, readout_layer=args.readout_layer, shape=shape, mp_id=p.row_id,
                        edit_type=p.edit_type, y_clean=int(o.label == POSITIVE[shape]), y_corrupt=int(p.label == POSITIVE[shape]),
                        ld_clean=ld_c, ld_corrupt=ld_x, gap_sd=gap)
            pair_info.append(info)
            if abs(gap) < args.min_gap:
                continue
            src_c = [h[0] for h in hs_c]
            src_x = [h[0] for h in hs_x]
            pos = {"cls": torch.tensor([0]), "A": seg_c["A"][0].nonzero()[:, 0], "C": seg_c["C"][0].nonzero()[:, 0],
                   "ctx": torch.arange(k)}
            items = list(range(n_states))
            for s in segs:
                if s == "B":     # clean run, corrupt context -> only clean B remains
                    hs = patcher.run(batch_of(enc_c, n_states), items, pos["ctx"].to(dev), src_x)
                    ld = ro(hs, batch_of(enc_c, n_states)["attention_mask"])
                else:
                    hs = patcher.run(batch_of(enc_x, n_states), items, pos[s].to(dev), src_c)
                    ld = ro(hs, batch_of(enc_x, n_states)["attention_mask"])
                rec = ((ld - ld_x) / (ld_c - ld_x)).cpu().numpy()
                for layer in range(n_states):
                    rows.append(dict(info, segment=s, layer=layer, ld_patched=float(ld[layer]), recovery=float(rec[layer])))
        print(f"{shape}: {len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    tag = f"{args.encoder}_L{args.readout_layer}" + ("_smoke" if args.limit else "")
    pd.DataFrame(rows).to_csv(RESULTS / f"patching_{tag}.csv", index=False)
    pd.DataFrame(pair_info).to_csv(RESULTS / f"patching_pairs_{tag}.csv", index=False)
    print(f"wrote results/patching_{tag}.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
