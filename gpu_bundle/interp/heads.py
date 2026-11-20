"""Experiments 3b / 4b: attention heads and MLPs.

Step 1  attribution patching (AtP, Nanda 2023; Zhang & Nanda 2024) on the counterfactual pairs: for every head
        (the head's slice of the input to attention.output.dense) and every MLP (output of layer.output.dense)
        estimate the effect of patching clean -> corrupt at the context positions ([CLS] A [SEP] C [SEP]; the
        positions that align between the two inputs) as  sum_pos (a_clean - a_corrupt) . dLD/da  on the
        corrupt run, normalised by the pair's gap LD_clean - LD_corrupt. A head scores high when it writes
        B-dependent information into the context positions, i.e. moves the B-vs-A relation into A / [CLS].
Step 2  verify the top-k components by real activation patching (same positions) and compare.
Step 3  mean-ablate the top-k heads (all positions, mean over main-set tokens) and score the held-out
        real examples with the fold readouts; control = k random heads from the same layers.

Usage: python heads.py --encoder nli [--top 10] [--edit_type cf]
Writes results/heads_atp_<enc>.csv, results/heads_verify_<enc>.csv, results/heads_ablation_<enc>.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from common import POSITIVE, RESULTS, SEGMENTS, device, encode_with_segments, load_all, load_encoder, load_feats, load_tokenizer
from patching import fold_readouts

N_HEADS, D_HEAD = 16, 64


class Components:
    """Capture (and optionally replace) per-layer head inputs to attention.output.dense and MLP outputs."""

    def __init__(self, model):
        self.model, self.cache, self.patch = model, {}, None
        self.handles = []
        for i, layer in enumerate(model.encoder.layer):
            self.handles.append(layer.attention.output.dense.register_forward_pre_hook(self._pre(i)))
            self.handles.append(layer.output.dense.register_forward_hook(self._post(i)))

    def _pre(self, i):
        def fn(module, args):
            x = args[0]
            if self.patch is not None:
                x = self.patch("attn", i, x)
            self.cache[("attn", i)] = x
            return (x,)
        return fn

    def _post(self, i):
        def fn(module, args, out):
            if self.patch is not None:
                out = self.patch("mlp", i, out)
            self.cache[("mlp", i)] = out
            return out
        return fn


def readout_logit(ro, hidden_states, mask):
    return ro(hidden_states, mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="nli")
    ap.add_argument("--readout_layer", type=int, default=24)
    ap.add_argument("--edit_type", default="cf")
    ap.add_argument("--min_gap", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dev = device()
    df = load_all()
    X, _ = load_feats(f"{args.encoder}_pair", df)
    X_r = X[args.readout_layer, SEGMENTS.index("mean")].astype(np.float32)
    tok = load_tokenizer()
    model = load_encoder(args.encoder, dev)
    comp = Components(model)
    idx_of = {r: i for i, r in enumerate(df.row_id.values)}
    n_layers = model.config.num_hidden_layers
    atp_rows, verify_rows, abl_rows = [], [], []
    t0 = time.time()
    tag = args.encoder + ("_smoke" if args.limit else "")
    for shape in POSITIVE:
        readouts, sd = fold_readouts(df, X_r, shape, args.readout_layer, dev)
        pairs = df[(df.type_candidate == shape) & (df.set == "pairs") & (df.edit_type == args.edit_type)]
        if args.limit:
            pairs = pairs.head(args.limit)
        kept = []
        # ---------------- step 1: attribution patching
        for p in pairs.itertuples():
            o = df.iloc[idx_of[p.source_pair_id]]
            enc_c, seg_c = encode_with_segments(tok, df.iloc[[idx_of[p.source_pair_id]]], shape)
            enc_x, _ = encode_with_segments(tok, df.iloc[[idx_of[p.row_id]]], shape)
            k = int(seg_c["B"][0].nonzero()[0, 0])
            enc_c, enc_x = enc_c.to(dev), enc_x.to(dev)
            ro = readouts[p.conversation_id]
            with torch.no_grad():
                comp.cache = {}
                ld_c = ro(model(**enc_c, output_hidden_states=True).hidden_states, enc_c["attention_mask"]).item()
                clean = {key: v.detach() for key, v in comp.cache.items()}
            comp.cache = {}
            out = model(**enc_x, output_hidden_states=True)
            ld_x = ro(out.hidden_states, enc_x["attention_mask"])
            gap = ld_c - ld_x.item()
            if abs(gap) / sd < args.min_gap:
                continue
            keys = list(comp.cache)
            grads = torch.autograd.grad(ld_x, [comp.cache[key] for key in keys])
            kept.append((p, ld_c, ld_x.item(), clean, k))
            for key, g in zip(keys, grads):
                diff = (clean[key][0, :k] - comp.cache[key][0, :k].detach()) * g[0, :k]   # [k, 1024]
                kind, layer = key
                if kind == "attn":
                    per_head = diff.view(k, N_HEADS, D_HEAD).sum((0, 2)).cpu().numpy()
                    for h in range(N_HEADS):
                        atp_rows.append(dict(shape=shape, mp_id=p.row_id, comp=f"L{layer}H{h}", kind="head", layer=layer,
                                             head=h, atp=float(per_head[h] / gap)))
                else:
                    atp_rows.append(dict(shape=shape, mp_id=p.row_id, comp=f"L{layer}MLP", kind="mlp", layer=layer,
                                         head=-1, atp=float(diff.sum().item() / gap)))
            model.zero_grad(set_to_none=True)
        A = pd.DataFrame([r for r in atp_rows if r["shape"] == shape])
        if A.empty:
            continue
        rank = A.groupby(["comp", "kind", "layer", "head"]).atp.mean().reset_index().sort_values("atp", ascending=False)
        top = rank[rank.kind == "head"].head(args.top)
        print(f"{shape}: AtP on {len(kept)} pairs, {time.time() - t0:.0f}s; top heads {list(top.comp)}", flush=True)

        # ---------------- step 2: verify top heads (and top MLPs) by real patching at context positions
        to_verify = pd.concat([top, rank[rank.kind == "mlp"].head(3)])
        with torch.no_grad():
            for p, ld_c, ld_x, clean, k in kept:
                enc_x, _ = encode_with_segments(tok, df.iloc[[idx_of[p.row_id]]], shape)
                enc_x = enc_x.to(dev)
                ro = readouts[p.conversation_id]
                for c in to_verify.itertuples():
                    def patch(kind, layer, x, c=c, clean=clean, k=k):
                        if kind != c.kind.replace("head", "attn") or layer != c.layer:
                            return x
                        x = x.clone()
                        if kind == "attn":
                            sl = slice(c.head * D_HEAD, (c.head + 1) * D_HEAD)
                            x[0, :k, sl] = clean[("attn", layer)][0, :k, sl]
                        else:
                            x[0, :k] = clean[("mlp", layer)][0, :k]
                        return x
                    comp.patch = patch
                    ld = ro(model(**enc_x, output_hidden_states=True).hidden_states, enc_x["attention_mask"]).item()
                    comp.patch = None
                    verify_rows.append(dict(shape=shape, mp_id=p.row_id, comp=c.comp, atp_mean=c.atp,
                                            recovery=(ld - ld_x) / (ld_c - ld_x)))

        # ---------------- step 3: mean-ablate top heads on the held-out real examples
        main = df[(df.type_candidate == shape) & (df.set == "main")]
        if args.limit:
            main = main.head(4 * args.limit)
        y = (main.label == POSITIVE[shape]).astype(int).values
        means = {}
        with torch.no_grad():                         # per-layer mean head input over main-set tokens
            for i in range(0, len(main), 16):
                enc, _ = encode_with_segments(tok, main.iloc[i:i + 16], shape)
                enc = enc.to(dev)
                comp.cache = {}
                model(**enc)
                m = enc["attention_mask"].unsqueeze(-1).float()
                for layer in range(n_layers):
                    s = (comp.cache[("attn", layer)] * m).sum((0, 1))
                    means[layer] = means.get(layer, 0) + s
            n_tok = sum(int(encode_with_segments(tok, main.iloc[i:i + 16], shape)[0]["attention_mask"].sum())
                        for i in range(0, len(main), 16))
            means = {l: v / n_tok for l, v in means.items()}
        rng = np.random.default_rng(0)
        layers_top = top.layer.values
        rand_heads = [(int(l), int(rng.integers(N_HEADS))) for l in layers_top]
        conditions = {"none": [], "top": list(zip(top.layer.astype(int), top["head"].astype(int))), "random": rand_heads}
        conditions.update({f"top1_{c}": [(int(l), int(h))] for c, l, h in zip(top.comp[:3], top.layer[:3], top["head"][:3])})
        with torch.no_grad():
            for cond, heads in conditions.items():
                def patch(kind, layer, x, heads=heads):
                    hs = [h for l, h in heads if l == layer and kind == "attn"]
                    if not hs:
                        return x
                    x = x.clone()
                    for h in hs:
                        sl = slice(h * D_HEAD, (h + 1) * D_HEAD)
                        x[..., sl] = means[layer][sl]
                    return x
                comp.patch = patch if heads else None
                logits = []
                for r in main.itertuples():
                    enc, _ = encode_with_segments(tok, main.loc[[r.Index]], shape)
                    enc = enc.to(dev)
                    ro = readouts[r.conversation_id]
                    logits.append(ro(model(**enc, output_hidden_states=True).hidden_states, enc["attention_mask"]).item())
                comp.patch = None
                abl_rows += [dict(shape=shape, condition=cond, row_id=rid, y=int(yy), logit=lg)
                             for rid, yy, lg in zip(main.row_id, y, logits)]
                print(f"  {shape} ablation {cond}: AUC {roc_auc_score(y, logits) if y.min() != y.max() else float('nan'):.3f}", flush=True)
    pd.DataFrame(atp_rows).to_csv(RESULTS / f"heads_atp_{tag}.csv", index=False)
    pd.DataFrame(verify_rows).to_csv(RESULTS / f"heads_verify_{tag}.csv", index=False)
    pd.DataFrame(abl_rows).to_csv(RESULTS / f"heads_ablation_{tag}.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
