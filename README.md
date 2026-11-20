# Relation or Trigger?

Probing and causally interpreting a cross-turn fallacy classifier (motte-and-bailey and strawman) built on DeBERTa-v3-large.

Aaditya Sharma (2019113009). Mechanistic Interpretability of Language Models, IIIT Hyderabad.

- `report/report.pdf`: final report (ACL format); `report/report.tex` is its source.
- `mech_interp_final_presentation.pptx`: final presentation.
- `proposal/`: project proposal.
- `interp/`: all experiments (probing, minimal-pair geometry, activation patching, attribution patching and head ablation, LEACE erasure), figures and result tables. Start with `interp/FINDINGS.md`; `interp/README.md` lists every script and how to run it.
- `gpu_bundle/`: the self-contained bundle used for the GPU part (patching, heads, forward erasure) on a single A100.
- `gpu_results/`: outputs and logs of that run.

## Setup

The dataset, NLI features and trigger list come from our earlier project, which must be cloned next to `interp/`:

```bash
git clone https://github.com/Ad-S/relational-fallacy-detection
cd interp
python -m venv .venv && source .venv/bin/activate
pip install torch transformers sentencepiece scikit-learn pandas matplotlib concept-erasure
```

For the GPU bundle, copy or clone the same repository into `gpu_bundle/` as well. Then follow the run order in `interp/README.md`. Cached activations (`interp/cache/`, about 1.5 GB) are not versioned and are rebuilt by `extract.py`.
