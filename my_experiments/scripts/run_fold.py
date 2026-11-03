# -*- coding: utf-8 -*-
"""Run upstream -> downstream for ONE fold, resumable, without ever using the repo's --newinit.

WHY no --newinit: in train_upstream.py / train_downstream.py, --newinit deletes the WHOLE
results/<exp>/ups-<exp>/ (or ds-<exp>/) folder - i.e. every other fold's checkpoints/outputs - whenever
this fold's checkpoint already exists. Here the config copies the scripts need are created by this
script instead, and the repo's own "load checkpoint if present" logic gives the resume.

Resume behaviour
  * stage finished (marker results/<exp>/status/fold<k>_<stage>.done)  -> skipped
  * stage interrupted, checkpoint exists  -> continues from the last BEST checkpoint
    (weights + optimizer incl. halved lr are restored; the repo's code resets its epoch counter,
     patience counter and best-val-loss, and RNG state is not restored).
    UPSTREAM never stops early (its val loss keeps falling), so a plain resume would run another
    full 100 epochs. Therefore the epochs already completed in earlier attempts are counted from the
    stage log and only the REMAINING epochs are run (total stays = --epochs). Disable with --no_smart_resume.
  * --restart  -> deletes only this fold's checkpoints/outputs/markers and starts clean
Logs are appended to results/<exp>/logs/.

Smoke test:  python my_experiments/scripts/run_fold.py --fold 1 --epochs 2 --batch 10 --method fusion --expname smoke_f1
Paper-like:  python my_experiments/scripts/run_fold.py --fold 1 --epochs 100 --batch 128 --method fusion --expname fusion_s0
"""
import argparse, glob, json, os, shutil, subprocess, sys, time
from pathlib import Path
import yaml

METHODS = {  # (auxiltr, w1, w2)   w1 = -lambda (speaker adversarial), w2 = alpha (PD classifier)
    "baseline": (False, 0.0, 0.0),
    "adv": (True, -0.01, 0.0),
    "pd": (True, 0.0, 0.01),
    "fusion": (True, -0.01, 0.01),
}
ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=".")
ap.add_argument("--data_dir", default="preprocess/pcgita_full_100spk")
ap.add_argument("--fold", type=int, default=1)
ap.add_argument("--method", choices=METHODS, default="fusion")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch", type=int, default=10, help="upstream & downstream batch size")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--expname", default="smoke")
ap.add_argument("--device", default=None, help="cpu or cuda:0 (default: repo decides)")
ap.add_argument("--dry_run", action="store_true")
ap.add_argument("--restart", action="store_true", help="discard this fold's previous checkpoints/outputs")
ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (speed only; 0 is the safe default)")
ap.add_argument("--no_smart_resume", action="store_true", help="resume upstream with the full epoch budget again")
a = ap.parse_args()

repo = Path(a.repo).resolve()
res = repo / "results" / a.expname
ups_main, ds_main = res / f"ups-{a.expname}", res / f"ds-{a.expname}"
status, logs = res / "status", res / "logs"
cfg_dir = repo / "my_experiments" / "configs" / a.expname
k = a.fold

# ---------------- pre-flight checks ----------------
info_all = json.load(open(repo / a.data_dir / "folds_info.json"))
info = info_all[str(k)]
n_cls, n_folds = info["n_spk_id_classes"], len(info_all)
folds, spk_folds = f"{a.data_dir}/folds/", f"{a.data_dir}/folds_spkID_task/"
n_online = len(glob.glob(str(repo / folds / "test_fold*_online.csv")))
if n_online != n_folds:
    sys.exit(f"STOP: found {n_online} test_fold*_online.csv in {folds} but {n_folds} folds exist. "
             "ds_runner.py counts these files to pick the upstream checkpoint for each fold; a wrong count "
             "makes later folds reuse the fold-1 upstream model (test-speaker leakage). Regenerate the missing CSVs.")
settings = dict(method=a.method, epochs=a.epochs, batch=a.batch, seed=a.seed, data_dir=a.data_dir, n_folds=n_folds)
sfile = res / "settings.json"
if sfile.exists() and json.load(open(sfile)) != settings:
    sys.exit(f"STOP: {sfile} has different settings:\n  {json.load(open(sfile))}\n  vs now {settings}\n"
             "Use another --expname so results of different settings never mix.")

# ---------------- configs (never touch repo/config) ----------------
def load(n): return yaml.safe_load(open(repo / "config" / n, encoding="utf-8"))
cfg = {}
cfg["upstream_config.yaml"] = load("upstream_config.yaml")
cfg["upstream_config.yaml"]["dataloader"].update(data_path=folds, batch_size=a.batch, num_workers=a.num_workers)
cfg["upstream_config.yaml"]["runner"]["Max_epoch"] = a.epochs
cfg["downstream_config.yaml"] = load("downstream_config.yaml")
cfg["downstream_config.yaml"]["dataloader"].update(data_path=folds, batch_size=a.batch, num_workers=a.num_workers)
cfg["downstream_config.yaml"]["runner"]["Max_epoch"] = a.epochs
cfg["upstream_auxiliary1_config.yaml"] = load("upstream_auxiliary1_config.yaml")
cfg["upstream_auxiliary1_config.yaml"]["dataloader"].update(data_path=spk_folds, num_workers=a.num_workers)
cfg["upstream_auxiliary1_config.yaml"]["MLP"]["outputdim"] = int(n_cls)
cfg["upstream_auxiliary2_config.yaml"] = load("upstream_auxiliary2_config.yaml")
cfg["upstream_auxiliary2_config.yaml"]["dataloader"].update(data_path=folds, num_workers=a.num_workers)
cfg["upstream_auxiliary2_config.yaml"]["MLP"]["outputdim"] = 2
cfg["audio_config.yaml"] = load("audio_config.yaml")
cfg_dir.mkdir(parents=True, exist_ok=True)
for n, d in cfg.items():
    yaml.safe_dump(d, open(cfg_dir / n, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)

auxtr, w1, w2 = METHODS[a.method]
dev = ["-d", a.device] if a.device else []
c = lambda n: str(cfg_dir / n)
ups = [sys.executable, "train_upstream.py", "-c", c("upstream_config.yaml"), "-ac1", c("upstream_auxiliary1_config.yaml"),
       "-ac2", c("upstream_auxiliary2_config.yaml"), "-a", c("audio_config.yaml"), "-n", a.expname, "-s", str(a.seed),
       "--fold", str(k), "--valmonitor"] + dev
if auxtr:
    ups += ["--auxiltr", "--auxlossw1", str(w1), "--auxlossw2", str(w2)]
dws = [sys.executable, "train_downstream.py", "-c", c("upstream_config.yaml"), "-dc", c("downstream_config.yaml"),
       "-ac", c("audio_config.yaml"), "-n", a.expname, "-s", str(a.seed), "--fold", str(k), "--mode", "train",
       "--valmonitor"] + dev
print(f"[{a.expname}] fold {k}: aux1 outputdim={n_cls}, method={a.method}, epochs={a.epochs}, batch={a.batch}")
if a.dry_run:
    for cmd in (ups, dws): print("$", " ".join(cmd))
    sys.exit(0)

# ---------------- state / config copies the repo scripts expect ----------------
for d in (ups_main, ds_main, status, logs):
    d.mkdir(parents=True, exist_ok=True)
if not sfile.exists():
    json.dump(settings, open(sfile, "w"), indent=1)
for src, dst in (("upstream_config.yaml", ups_main / "ups_config.yaml"), ("upstream_auxiliary1_config.yaml", ups_main / "aux1_config.yaml"),
                 ("upstream_auxiliary2_config.yaml", ups_main / "aux2_config.yaml"), ("audio_config.yaml", ups_main / "feat_config.yaml"),
                 ("downstream_config.yaml", ds_main / "ds_config.yaml"), ("upstream_config.yaml", ds_main / "ups_config.yaml"),
                 ("audio_config.yaml", ds_main / "feat_config.yaml")):
    shutil.copy2(cfg_dir / src, dst)

ckpt = {"ups": ups_main / "outputs" / "saved_model" / f"UPs_NN_fold{k}.ckpt",
        "ds": ds_main / "outputs" / "saved_model" / f"Ds_NN_fold{k}.ckpt"}
out_npy = ds_main / "outputs" / f"output_{k}.npy"
marker = lambda stage: status / f"fold{k}_{stage}.done"

if a.restart:
    for p in [ckpt["ups"], ckpt["ds"], out_npy, marker("ups"), marker("ds")]:
        if p.exists(): p.unlink(); print("removed", p)

def completed_epochs(logfile):
    """Epochs finished in earlier attempts, from lines like ' Epoch [ 85/100]-TRAIN: ...' (upstream log)."""
    import re
    total, last = 0, -1
    if not Path(logfile).exists():
        return 0
    for line in open(logfile, encoding="utf-8", errors="replace"):
        if line.startswith("===== 2") and " exit " not in line:      # new attempt header
            total += last + 1; last = -1
        m = re.search(r"Epoch \[\s*(\d+)/\s*\d+\]-TRAIN", line)
        if m: last = int(m.group(1))
    return total + last + 1

def run(stage, cmd):
    if marker(stage).exists():
        print(f"[skip] {stage} fold {k}: already finished"); return
    logf = logs / f"fold{k}_{stage}.log"
    fresh = not ckpt[stage].exists()
    if fresh and logf.exists():                       # a clean start: old attempts must not be counted
        logf.rename(logf.with_name(logf.name + time.strftime(".old-%Y%m%d-%H%M%S")))
    reduced = False
    if not fresh:
        print(f"[resume] {stage} fold {k}: continuing from the last best checkpoint")
        if stage == "ups" and not a.no_smart_resume:
            done_ep = completed_epochs(logf)
            remaining = max(1, a.epochs - done_ep)
            c = yaml.safe_load(open(cfg_dir / "upstream_config.yaml", encoding="utf-8"))
            c["runner"]["Max_epoch"] = remaining
            yaml.safe_dump(c, open(ups_main / "ups_config.yaml", "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
            reduced = True
            print(f"[resume] upstream: {done_ep} of {a.epochs} epochs were completed earlier -> running the remaining {remaining}")
    t0 = time.time()
    with open(logf, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in p.stdout:
            sys.stdout.write(line); lf.write(line)
        rc = p.wait()
        lf.write(f"===== exit {rc} after {time.time()-t0:.0f}s\n")
    if reduced:                                       # put the full-budget config back
        shutil.copy2(cfg_dir / "upstream_config.yaml", ups_main / "ups_config.yaml")
    if rc:
        sys.exit(f"FAILED: {stage} fold {k} (exit {rc}). Re-run the same command to resume. Log: {logf}")
    marker(stage).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    if hasattr(os, "sync"): os.sync()
    print(f"[done] {stage} fold {k} in {(time.time()-t0)/60:.1f} min")

run("ups", ups)
if not ckpt["ups"].exists():
    sys.exit(f"upstream checkpoint missing: {ckpt['ups']}")
run("ds", dws)
print(f"results: {out_npy}")
