# -*- coding: utf-8 -*-
"""Run many folds / seeds / methods with run_fold.py. Resumable at two levels: finished folds are skipped
(output_<fold>.npy + status marker); an interrupted fold continues from its last best checkpoint (see run_fold.py).
exp name = <method>_s<seed>  (all folds of one method+seed share one results folder, like the author's).

Rehearsal (baseline, seed 0, all 10 folds, paper settings):
    python my_experiments/scripts/run_all.py --methods baseline --seeds 0 --folds 1-10
Everything:
    python my_experiments/scripts/run_all.py --methods baseline adv pd fusion --seeds 0 1 2 3 4 --folds 1-10
Add --dry_run to print the plan only.
"""
import argparse, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=".")
ap.add_argument("--data_dir", default="preprocess/pcgita_full_100spk")
ap.add_argument("--methods", nargs="+", default=["baseline"], choices=["baseline", "adv", "pd", "fusion"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--folds", default="1-10", help="e.g. 1-10 or 1,3,5")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--device", default=None)
ap.add_argument("--num_workers", type=int, default=0)
ap.add_argument("--dry_run", action="store_true")
a = ap.parse_args()

if "-" in a.folds:
    lo, hi = a.folds.split("-"); folds = list(range(int(lo), int(hi) + 1))
else:
    folds = [int(x) for x in a.folds.split(",")]
repo = Path(a.repo).resolve()
here = Path(__file__).resolve().parent

plan = [(m, s, k) for m in a.methods for s in a.seeds for k in folds]
def done(m, s, k):
    r = repo / "results" / f"{m}_s{s}"
    return (r / f"ds-{m}_s{s}" / "outputs" / f"output_{k}.npy").exists() and (r / "status" / f"fold{k}_ds.done").exists()
todo = [(m, s, k) for m, s, k in plan if not done(m, s, k)]
print(f"planned {len(plan)} runs, already done {len(plan) - len(todo)}, to run {len(todo)}")
for i, (m, s, k) in enumerate(todo, 1):
    exp = f"{m}_s{s}"
    cmd = [sys.executable, str(here / "run_fold.py"), "--repo", str(repo), "--data_dir", a.data_dir,
           "--fold", str(k), "--method", m, "--seed", str(s), "--epochs", str(a.epochs),
           "--batch", str(a.batch), "--expname", exp] + (["--device", a.device] if a.device else []) + ["--num_workers", str(a.num_workers)]
    print(f"\n[{i}/{len(todo)}] {exp} fold {k}")
    if a.dry_run:
        print("  $", " ".join(cmd)); continue
    r = subprocess.run(cmd, cwd=repo)
    if r.returncode:
        print(f"!! {exp} fold {k} failed (exit {r.returncode}); continuing with the next run")
print("\nfinished. Aggregate with: python my_experiments/scripts/collect_results.py")
