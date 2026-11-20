"""Cache segment-pooled hidden states of every layer for all rows (683 main + 416 minimal-pair edits).

Output: cache/feats_<encoder>_<input>.npz with
  X        float16 [layers+1, n_segments, n_rows, hidden]   (layer 0 = embeddings)
  row_id   [n_rows] (order of common.load_all())
  seg_n    int [n_segments, n_rows]  number of tokens in each segment (0 = segment absent)
Usage: python extract.py --encoders nli,plain,random [--inputs pair,b_only] [--limit 40]
"""
import argparse
import time

import numpy as np
import torch

from common import CACHE, POSITIVE, SEGMENTS, device, encode_with_segments, load_all, load_encoder, load_tokenizer, pool


@torch.no_grad()
def extract(model, tok, df, input_mode, dev, bs=8):
    n = len(df)
    out, seg_n = None, np.zeros((len(SEGMENTS), n), dtype=np.int32)
    for shape in POSITIVE:
        idx = np.where(df.type_candidate.values == shape)[0]
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            enc, seg = encode_with_segments(tok, df.iloc[j], shape, input_mode)
            hs = model(**enc.to(dev), output_hidden_states=True).hidden_states   # (L+1) x [b, T, H]
            if out is None:
                out = np.zeros((len(hs), len(SEGMENTS), n, hs[0].shape[-1]), dtype=np.float16)
            for si, s in enumerate(SEGMENTS):
                m = seg[s].to(dev)
                seg_n[si, j] = m.sum(1).cpu().numpy()
                for li, h in enumerate(hs):
                    out[li, si, j] = pool(h.float(), m).cpu().numpy().astype(np.float16)
    return out, seg_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="nli,plain,random")
    ap.add_argument("--inputs", default="pair")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows per shape")
    args = ap.parse_args()
    dev = device()
    df = load_all()
    if args.limit:
        df = df.groupby("type_candidate", group_keys=False).head(args.limit).reset_index(drop=True)
    tok = load_tokenizer()
    CACHE.mkdir(exist_ok=True)
    for name in args.encoders.split(","):
        model = load_encoder(name, dev)
        for mode in args.inputs.split(","):
            t0 = time.time()
            X, seg_n = extract(model, tok, df, mode, dev)
            tag = f"feats_{name}_{mode}" + ("_smoke" if args.limit else "")
            np.savez(CACHE / f"{tag}.npz", X=X, row_id=np.array([str(r) for r in df.row_id], dtype="U"), seg_n=seg_n)
            print(f"{tag}: {X.shape} in {time.time() - t0:.0f}s on {dev}", flush=True)
        del model
        if dev.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
