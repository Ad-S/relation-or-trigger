"""Merge the hand-written minimal-pair edits (data/minimal_pairs/batch_*.jsonl) with the original examples.

Edit types (A and C are never changed, so their token positions align between the two members of a pair):
  cf  counterfactual: minimal edit of B that FLIPS the label; the trigger phrase is kept where the original had one
  ts  trigger swap:   the trigger phrase is rewritten out of B (or added when the original had none); label KEPT

Writes data/minimal_pairs.csv (one row per edit) and data/minimal_pairs_review.csv (a random sample of 30
per type, with empty `accept` / `note` columns for hand checking). Prints automatic sanity checks.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "relational-fallacy-detection"
sys.path.insert(0, str(HERE))
from triggers_ext import has_trigger  # noqa: E402


FLIP = {"motte_bailey": {"MOTTE_BAILEY": "NONE", "NONE": "MOTTE_BAILEY"},
        "strawman": {"STRAWMAN": "NONE", "NONE": "STRAWMAN"}}



def main():
    data = pd.read_csv(REPO / "finetune/data/dataset_augmented_all_v2.csv", keep_default_na=False)
    orig = data[data.kind == "original"].set_index("pair_id")
    edits = []
    for f in sorted((HERE / "data/minimal_pairs").glob("batch_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                edits.append(json.loads(line))
    ids = [e["pair_id"] for e in edits]
    dup = {i for i in ids if ids.count(i) > 1}
    assert not dup, f"duplicate pair_ids: {dup}"
    missing = set(orig.index) - set(ids)
    print(f"{len(edits)} edited originals; {len(missing)} originals without edits: {sorted(missing)}")

    rows = []
    for e in edits:
        o = orig.loc[e["pair_id"]]
        shape = o.type_candidate
        for kind in ("cf", "ts"):
            b_new = e.get(f"{kind}_B", "").strip()
            if not b_new:
                continue
            rows.append(dict(
                mp_id=f"{e['pair_id']}_{kind}", source_pair_id=e["pair_id"], edit_type=kind, type_candidate=shape,
                conversation_id=o.conversation_id, source=o.source, claim_A=o.claim_A, challenge=o.challenge,
                claim_B_orig=o.claim_B, claim_B=b_new, label_orig=o.label,
                label=FLIP[shape][o.label] if kind == "cf" else o.label,
                trigger_orig=has_trigger(o.claim_B, shape), trigger_new=has_trigger(b_new, shape)))
    mp = pd.DataFrame(rows)
    out = HERE / "data/minimal_pairs.csv"
    mp.to_csv(out, index=False)

    print(f"wrote {out}: {len(mp)} edits")
    print(mp.groupby(["type_candidate", "edit_type", "label_orig", "label"]).size().to_string())
    cf, ts = mp[mp.edit_type == "cf"], mp[mp.edit_type == "ts"]
    cf_lost = cf[cf.trigger_orig & ~cf.trigger_new]
    ts_same = ts[ts.trigger_orig == ts.trigger_new]
    print(f"\ncf edits that lost the original trigger: {len(cf_lost)} / {int(cf.trigger_orig.sum())}  {list(cf_lost.mp_id)}")
    print(f"ts edits that did NOT toggle the trigger: {len(ts_same)} / {len(ts)}  {list(ts_same.mp_id)}")

    review = pd.concat([g.sample(min(30, len(g)), random_state=0) for _, g in mp.groupby("edit_type")])
    review = review[["mp_id", "edit_type", "type_candidate", "label_orig", "label", "claim_A", "challenge",
                     "claim_B_orig", "claim_B"]].assign(accept="", note="")
    review.to_csv(HERE / "data/minimal_pairs_review.csv", index=False)
    print(f"wrote data/minimal_pairs_review.csv ({len(review)} rows to hand-check)")


if __name__ == "__main__":
    main()
