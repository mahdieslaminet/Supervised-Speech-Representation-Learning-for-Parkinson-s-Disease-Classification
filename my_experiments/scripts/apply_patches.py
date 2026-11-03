# -*- coding: utf-8 -*-
"""Idempotent compatibility patches for pddetection-reps-learning (modern torch/numpy/torchaudio).
Run from the repo root:  python my_experiments/scripts/apply_patches.py [repo_root]
Originals are saved next to each file as <name>.orig (only the first time).
"""
import re, shutil, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

try:
    import torch
    major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    can_weights_only = (major, minor) >= (1, 13)
    print(f"torch {torch.__version__}: weights_only patch {'ON' if can_weights_only else 'OFF (torch < 1.13)'}")
except Exception as e:  # torch not importable here -> assume modern
    can_weights_only = True
    print("torch not importable, assuming >= 1.13:", e)

def patch(rel, subs):
    p = root / rel
    s = p.read_text(encoding="utf-8")
    new = s; n_total = 0
    for pat, repl in subs:
        new, n = re.subn(pat, repl, new, flags=re.M)
        n_total += n
    if new != s:
        bak = p.with_suffix(p.suffix + ".orig")
        if not bak.exists():
            shutil.copy2(p, bak)
        p.write_text(new, encoding="utf-8")
    print(f"{rel}: {n_total} change(s)")

# 1) removed in torchaudio >= 2.1/2.9; wav reading uses soundfile anyway
patch("audio/audio_dataset.py", [(r'^torchaudio\.set_audio_backend\("sox_io"\)\s*$', "# (patched) torchaudio.set_audio_backend removed")])
# 2) np.Inf removed in NumPy 2.0
for f in ("downstream/ds_runner.py", "upstream_auxiliary/ups_runner_merge.py"):
    patch(f, [(r"\bnp\.Inf\b", "np.inf")])
# 3) np.array([...], dtype=object) raises ValueError when upstream/aux1/aux2 batches have the SAME size
#    (batch 128 as in the paper). Same semantics, built element by element.
patch("upstream_auxiliary/ups_runner_merge.py", [(
    r"(?m)^([ \t]*)returned_batch = np\.array\(\s*\[batch_longloader, batch_medloader, batch_shortloader\],\s*dtype=object,?\s*\)",
    r"\1returned_batch = np.empty(3, dtype=object)\n\1returned_batch[0], returned_batch[1], returned_batch[2] = batch_longloader, batch_medloader, batch_shortloader")])
# 4) torch >= 2.6: torch.load defaults to weights_only=True (checkpoints hold optimizer/config objects)
if can_weights_only:
    for f in ("downstream/ds_runner.py", "upstream_auxiliary/ups_runner_merge.py"):
        patch(f, [(r'torch\.load\(([^()\n]*?map_location="cpu")\)', r"torch.load(\1, weights_only=False)")])
print("done")
