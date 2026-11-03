# -*- coding: utf-8 -*-
"""Make the two things to upload to Google Drive (one big file is MUCH faster to upload and to
copy to Colab's local disk than thousands of small .npy files):
  colab_pack/data/<name>.tar   features (.npy) + fold CSVs only (no wavs)
  colab_pack/repo_code/        the repo code (+ my_experiments), without data/results/.venv
Run from the repo root:
    python my_experiments/scripts/pack_data.py --data_dir preprocess/pcgita_full_100spk
Then upload the folder colab_pack to Drive as  MyDrive/idiap_pd/  (-> idiap_pd/data, idiap_pd/repo_code).
"""
import argparse, shutil, tarfile, time
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=".")
ap.add_argument("--data_dir", default="preprocess/pcgita_full_100spk")
ap.add_argument("--out", default="colab_pack")
ap.add_argument("--feat", default="fbank")
a = ap.parse_args()
repo, out = Path(a.repo).resolve(), Path(a.out).resolve()
dd = repo / a.data_dir

# --- code (small) ---
skip_dirs = {".venv", "venv", ".git", "__pycache__", "results", "colab_pack", "segmented_audio_data", "features_data", "folds", "folds_spkID_task", "configs"}
def ignore(d, names):
    return [n for n in names if n in skip_dirs or n.endswith((".pyc", ".orig", ".tar", ".zip"))]
code = out / "repo_code"
if code.exists(): shutil.rmtree(code)
shutil.copytree(repo, code, ignore=ignore)
print("code ->", code, f"({sum(f.stat().st_size for f in code.rglob('*') if f.is_file())/1e6:.1f} MB)")

# --- data tar ---
tar_path = out / "data" / f"{dd.name}_light.tar"; tar_path.parent.mkdir(parents=True, exist_ok=True)
need = set()
for sub in ("folds", "folds_spkID_task"):
    for f in (dd / sub).glob("*.csv"):
        need.update(pd.read_csv(f).file_path.tolist()) if "offline" in f.name else None
feat_files = sorted((dd / "features_data" / a.feat).glob("*.npy"))
missing = [p for p in need if not (repo / p).exists()]
assert not missing, f"CSV points to missing files, e.g. {missing[:3]}"
t0 = time.time()
with tarfile.open(tar_path, "w") as tf:                       # no compression: .npy float32 barely compresses
    for f in (list(feat_files) + sorted((dd / "folds").glob("*.csv")) + sorted((dd / "folds_spkID_task").glob("*.csv"))
              + [dd / "folds_info.json", dd / "speakers.csv"]):
        if f.exists(): tf.add(f, arcname=f.relative_to(repo).as_posix())
print(f"data -> {tar_path}  ({tar_path.stat().st_size/1e9:.2f} GB, {len(feat_files)} feature files, {time.time()-t0:.0f}s)")
print("referenced by fold CSVs:", len(need), "files, all present in the tar:", need <= {p.relative_to(repo).as_posix() for p in feat_files})
