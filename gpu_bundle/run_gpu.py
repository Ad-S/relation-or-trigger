"""Unattended GPU session: feature caches, residual patching, heads, forward erasure for nli and plain.
Each step is a subprocess with its own log; a failed step does not stop the session.
Results: interp/results/, logs: logs/, summary: logs/summary.json"""
import json, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
PY = sys.executable
LAYERS = ",".join(str(l) for l in range(0, 25, 2))
STEPS = [
    ("smoke_patch", [PY, "extract.py", "--encoders", "nli", "--limit", "8"]),
    ("extract", [PY, "extract.py", "--encoders", "nli,plain", "--inputs", "pair"]),
    ("patch_nli", [PY, "patching.py", "--encoder", "nli", "--readout_layer", "24"]),
    ("heads_nli", [PY, "heads.py", "--encoder", "nli", "--top", "10"]),
    ("erase_nli", [PY, "erasure.py", "--mode", "forward", "--encoder", "nli", "--layers", LAYERS, "--concepts", "trigger,entail"]),
    ("patch_plain", [PY, "patching.py", "--encoder", "plain", "--readout_layer", "24"]),
    ("heads_plain", [PY, "heads.py", "--encoder", "plain", "--top", "10"]),
    ("erase_plain", [PY, "erasure.py", "--mode", "forward", "--encoder", "plain", "--layers", LAYERS, "--concepts", "trigger,entail"]),
]
logs = HERE / "logs"; logs.mkdir(exist_ok=True)
(HERE / "interp" / "results").mkdir(exist_ok=True)
summary = []
for name, cmd in STEPS:
    t0 = time.time()
    with open(logs / f"{name}.log", "w") as f:
        rc = subprocess.run(cmd, cwd=HERE / "interp", stdout=f, stderr=subprocess.STDOUT).returncode
    summary.append(dict(step=name, rc=rc, minutes=round((time.time() - t0) / 60, 1)))
    print(json.dumps(summary[-1]), flush=True)
    (logs / "summary.json").write_text(json.dumps(summary, indent=1))
print("SESSION_DONE", flush=True)
