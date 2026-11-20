"""Shared code for the interpretability experiments on frozen DeBERTa-v3-large encoders.

Inputs are built exactly as in the fine-tuning code of relational-fallacy-detection (finetune/train_cv.py):
  motte-and-bailey: tok(A, f"{C} [SEP] {B}")      strawman: tok(A, B)
so the probed representations are the ones the earlier classifiers saw.

Segments (token masks per example): cls, A, C, B, trig (trigger-phrase tokens inside B, extended lexicon),
brest (B minus trigger tokens), mean (all non-padding tokens).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from triggers_ext import find_triggers

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "relational-fallacy-detection"
CACHE = HERE / "cache"
RESULTS = HERE / "results"
NLI_CKPT = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
PLAIN_CKPT = "microsoft/deberta-v3-large"
MAX_LEN = 256
POSITIVE = {"motte_bailey": "MOTTE_BAILEY", "strawman": "STRAWMAN"}
SEGMENTS = ["cls", "A", "C", "B", "trig", "brest", "mean"]
ENCODERS = {"nli": NLI_CKPT, "plain": PLAIN_CKPT, "random": None}   # random = NLI architecture, random weights


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer():
    return AutoTokenizer.from_pretrained(NLI_CKPT)   # all three encoders share the DeBERTa-v3 vocabulary


def load_encoder(name, dev=None):
    if ENCODERS[name] is None:
        torch.manual_seed(0)
        model = AutoModel.from_config(AutoConfig.from_pretrained(NLI_CKPT))
    else:
        model = AutoModel.from_pretrained(ENCODERS[name], dtype=torch.float32)
    return model.to(dev or device()).eval()


# ----------------------------------------------------------------------------------------------- data
def load_main():
    """The 683-pair dataset of the earlier project, with NLI features and the extended trigger flag."""
    d = pd.read_csv(REPO / "finetune/data/dataset_augmented_all_v2.csv", keep_default_na=False)
    nli = pd.read_csv(REPO / "finetune/session_results/features/dataset_augmented_all_v2_nli_features.csv")
    d = d.merge(nli, on="pair_id", how="left")
    d["row_id"] = d["pair_id"]
    return d


def load_pairs():
    """Minimal-pair edits (one row per edited B), see build_minimal_pairs.py."""
    mp = pd.read_csv(HERE / "data/minimal_pairs.csv", keep_default_na=False)
    mp["row_id"] = mp["mp_id"]
    mp["kind"] = mp["edit_type"]
    return mp


def load_all():
    """Main dataset + minimal-pair edits in one frame (row_id unique), as cached by extract.py."""
    m, p = load_main(), load_pairs()
    m["set"], p["set"] = "main", "pairs"
    cols = ["row_id", "set", "type_candidate", "conversation_id", "claim_A", "challenge", "claim_B", "label", "kind", "source"]
    extra = [c for c in m.columns if c.startswith(("entail_", "neutral_", "contra_"))]
    out = pd.concat([m[cols + extra], p[cols + ["source_pair_id", "edit_type"]]], ignore_index=True)
    for c in ("source_pair_id", "edit_type"):
        out[c] = out[c].fillna("")
    return out


# ----------------------------------------------------------------------------------------------- encoding
def build_pair(row, shape, sep, input_mode="pair"):
    a = str(row.claim_A)
    c = str(row.challenge) if isinstance(row.challenge, str) else ""
    b = str(row.claim_B)
    if input_mode == "b_only":
        return b, None, None, 0
    if shape == "motte_bailey" and c:
        second = f"{c} {sep} {b}"
        return a, second, (0, len(c)), len(c) + len(sep) + 2
    return a, b, None, 0


def encode_with_segments(tok, rows, shape, input_mode="pair", max_len=MAX_LEN):
    """Tokenise `rows` and return (BatchEncoding without offsets, dict segment -> bool tensor [n, T])."""
    built = [build_pair(r, shape, tok.sep_token, input_mode) for r in rows.itertuples()]
    first = [x[0] for x in built]
    second = [x[1] for x in built]
    kw = dict(truncation=True, max_length=max_len, padding=True, return_tensors="pt", return_offsets_mapping=True)
    enc = tok(first, **kw) if input_mode == "b_only" else tok(first, second, **kw)
    offsets = enc.pop("offset_mapping")
    n, T = enc["input_ids"].shape
    seg = {s: torch.zeros(n, T, dtype=torch.bool) for s in SEGMENTS}
    for i, (r, (_, _, c_span, b_start)) in enumerate(zip(rows.itertuples(), built)):
        sid = enc.sequence_ids(i)
        trig_spans = [(s + b_start, e + b_start) for s, e in find_triggers(str(r.claim_B), shape)]
        b_seq = 0 if input_mode == "b_only" else 1
        for t in range(T):
            if enc["attention_mask"][i, t] == 0:
                continue
            seg["mean"][i, t] = True
            if t == 0:
                seg["cls"][i, t] = True
            s = sid[t]
            if s is None:
                continue
            st, en = offsets[i, t].tolist()
            if s == 0 and input_mode != "b_only":
                seg["A"][i, t] = True
            elif s == b_seq:
                if c_span is not None and en <= c_span[1]:
                    seg["C"][i, t] = True
                elif en > b_start or c_span is None:
                    seg["B"][i, t] = True
                    if any(st < te and en > ts for ts, te in trig_spans):
                        seg["trig"][i, t] = True
                    else:
                        seg["brest"][i, t] = True
    return enc, seg


def pool(hidden, mask):
    """hidden [n, T, H], mask [n, T] -> mean over masked tokens [n, H] (zeros when the mask is empty)."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1)


def labels_for(df, shape):
    return (df.label == POSITIVE[shape]).astype(int).values


def load_feats(tag, df=None):
    """Load cache/feats_<tag>.npz -> (X, seg_n); checks the row order against `df` (default load_all())."""
    z = np.load(CACHE / f"feats_{tag}.npz", allow_pickle=True)
    df = load_all() if df is None else df
    assert [str(r) for r in z["row_id"]] == [str(r) for r in df.row_id], "cache row order differs from load_all()"
    return z["X"], z["seg_n"]
