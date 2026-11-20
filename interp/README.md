# interp/: interpretability experiments for "Relation or Trigger?"

This builds on `../relational-fallacy-detection`, using its data, its NLI features and its trigger list, and leaves that folder untouched.
Start with `FINDINGS.md` (answers per research question), then `RESULTS.md` (all tables) and `figures/`.

| File | Purpose |
|---|---|
| `data/minimal_pairs/batch_*.jsonl` | minimal-pair edits of claim B (cf = counterfactual, ts = trigger swap) |
| `build_minimal_pairs.py` | -> `data/minimal_pairs.csv` (+ automatic checks) and `data/minimal_pairs_review.csv` (60 to hand-check) |
| `triggers_ext.py` | selection trigger list + close variants (defines the trigger target/span) |
| `common.py` | data loading, inputs identical to the old fine-tuning code, segment masks (cls/A/C/B/trig/brest/mean) |
| `extract.py` | cache segment-pooled hidden states, all 25 layers -> `cache/` (not versioned, ~1.5 GB) |
| `probe.py` | Exp 1 probes + permuted-label controls |
| `geometry.py` | Exp 2 minimal pairs along fallacy / trigger directions |
| `patching.py` | Exp 3 residual-stream patching (frozen encoder + linear readout) |
| `heads.py` | Exp 3b/4b attribution patching of heads/MLPs, verification, mean ablation |
| `erasure.py` | Exp 4 LEACE: amnesic probing (cached) and in-forward erasure |
| `cosine_entail.py` | RQ4 fallacy vs entailment direction |
| `analyze.py` | all figures + `RESULTS.md` |

Run (local, uv venv `.venv`, Python 3.12):
```bash
python build_minimal_pairs.py
python extract.py --encoders nli,plain,random --inputs pair && python extract.py --encoders nli --inputs b_only
for e in nli_pair plain_pair random_pair nli_b_only; do python probe.py --encoders $e --out probe_$e.csv; done
python geometry.py && python erasure.py --mode probe --segments mean,B && python cosine_entail.py
# GPU part (../gpu_bundle, run with jl on an A100, ~26 min): patching, heads, in-forward erasure for nli and plain
cd ../gpu_bundle && jl run . --script run_gpu.py --gpu A100 --keep --json --yes   # results -> ../gpu_results/
cd ../interp && python analyze.py
```
