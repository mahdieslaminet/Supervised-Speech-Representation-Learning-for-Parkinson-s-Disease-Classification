# -*- coding: utf-8 -*-
"""Prepare a (fresh) Colab runtime in ~1-3 minutes. Safe to run again after every reconnect.
  1) code    : Drive/repo_code      -> /content/repo            (only if missing)
  2) data    : Drive/data/*.tar     -> /content/repo/preprocess (local disk = fast; only if not extracted yet)
  3) results : /content/repo/results -> symlink to Drive/results (checkpoints, logs, outputs survive disconnects)
  4) deps    : pip-installs only what is missing, runs apply_patches.py (idempotent)
Usage (after mounting Drive in a notebook cell):
    python ./my_experiments/scripts/colab_bootstrap.py
"""
import argparse, importlib, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--drive_root", default="/content/drive/MyDrive/idiap_pd")
ap.add_argument("--workdir", default="/content/repo")
ap.add_argument("--tar", default=None, help="tar name inside Drive/data (default: the only .tar there)")
a = ap.parse_args()
drive, work = Path(a.drive_root), Path(a.workdir)
assert drive.exists(), f"{drive} not found - did you mount Drive and upload colab_pack as idiap_pd?"

# 1) code
if not (work / "train_upstream.py").exists():
    shutil.copytree(drive / "repo_code", work, dirs_exist_ok=True); print("code copied ->", work)
else:
    print("code already in", work)

# 2) data (local disk)
tars = [drive / "data" / a.tar] if a.tar else sorted((drive / "data").glob("*.tar"))
assert len(tars) == 1 and tars[0].exists(), f"expected exactly one tar in {drive/'data'}, got {tars}"
tar = tars[0]; flag = work / f".extracted_{tar.stem}"
if not flag.exists():
    t0 = time.time(); print(f"extracting {tar.name} ({tar.stat().st_size/1e9:.2f} GB) to local disk ...")
    with tarfile.open(tar) as tf:
        try: tf.extractall(work, filter="data")
        except TypeError: tf.extractall(work)
    flag.write_text(time.strftime("%Y-%m-%d %H:%M:%S")); print(f"extracted in {time.time()-t0:.0f}s")
else:
    print("data already extracted locally")

# 3) results -> Drive
dres = drive / "results"; dres.mkdir(exist_ok=True)
link = work / "results"
if link.is_symlink():
    if link.resolve() != dres.resolve(): link.unlink(); link.symlink_to(dres, target_is_directory=True)
elif link.exists():
    if any(link.iterdir()): sys.exit(f"{link} is a non-empty real folder; move its content to {dres} first")
    link.rmdir(); link.symlink_to(dres, target_is_directory=True)
else:
    link.symlink_to(dres, target_is_directory=True)
print("results ->", link.resolve())

# 4) deps + patches
need = {"yaml": "pyyaml", "sklearn": "scikit-learn", "soundfile": "soundfile", "pandas": "pandas"}
miss = []
for mod, pipname in need.items():
    try: importlib.import_module(mod)
    except ImportError: miss.append(pipname)
if miss:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *miss])
subprocess.check_call([sys.executable, str(work / "my_experiments" / "scripts" / "apply_patches.py"), str(work)])
try:
    import torch; print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
    if not torch.cuda.is_available(): print("!! no GPU: Runtime > Change runtime type > GPU")
except ImportError:
    print("!! torch missing")
du = shutil.disk_usage(work); print(f"local disk free: {du.free/1e9:.0f} GB")
print("READY. cd", work)
