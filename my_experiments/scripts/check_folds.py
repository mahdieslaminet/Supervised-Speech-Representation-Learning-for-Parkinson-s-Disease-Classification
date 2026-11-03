# -*- coding: utf-8 -*-
"""Sanity checks for the fold CSVs (run from repo root):
    python my_experiments/scripts/check_folds.py preprocess/pcgita_full_100spk
"""
import sys, glob, os
import numpy as np, pandas as pd

d = sys.argv[1] if len(sys.argv) > 1 else "preprocess/pcgita_full_100spk"
feat = "fbank"; MINF = 118  # 500 ms = 118 frames (32 ms window, 4 ms shift)
K = len(glob.glob(f"{d}/folds/test_fold*_{feat}_offline.csv"))
rd = lambda sub, s, k: pd.read_csv(f"{d}/{sub}/{s}_fold{k}_{feat}_offline.csv")
ok = True
def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg); ok &= bool(cond)

for sub in ("folds", "folds_spkID_task"):
    miss = [f"{s}_fold{k}_{t}.csv" for k in range(1, K + 1) for s in ("train", "val", "test") for t in (f"{feat}_offline", "online")
            if not os.path.exists(f"{d}/{sub}/{s}_fold{k}_{t}.csv")]
    check(not miss, f"{sub}: all offline+online CSVs exist for {K} folds" + (f" (missing: {miss[:4]})" if miss else ""))
n_on = len(glob.glob(f"{d}/folds/test_fold*_online.csv"))
check(n_on == K, f"test_fold*_online.csv count = {n_on} (ds_runner.py uses it to choose the upstream ckpt per fold; must be {K})")
tested = []
for k in range(1, K + 1):
    S = {s: set(rd("folds", s, k).ID) for s in ("train", "val", "test")}
    check(not (S["train"] & S["test"] or S["train"] & S["val"] or S["val"] & S["test"]), f"fold{k}: train/val/test speaker-disjoint")
    tested += sorted(S["test"])
    A = {s: rd("folds_spkID_task", s, k) for s in ("train", "val", "test")}
    ids = set().union(*[set(v.ID) for v in A.values()])
    allrows = pd.concat([rd("folds", s, k) for s in S])
    lab_of = allrows.groupby("ID").label.first()
    check(not (ids & S["test"]) and set(lab_of[list(ids)]) == {0}, f"fold{k}: spkID task = healthy speakers outside test fold only ({len(ids)} spk)")
    cl = sorted(set(A["train"].label)); check(cl == list(range(len(cl))), f"fold{k}: spkID labels contiguous 0..{len(cl)-1}")
    check(all(set(v.label) <= set(cl) for v in A.values()), f"fold{k}: val/test spkID labels all seen in train")
tested = np.array(tested)
check(len(tested) == len(set(tested)), f"every speaker tested at most once over {K} folds ({len(set(tested))} distinct)")

allf = pd.concat([rd("folds", s, 1) for s in ("train", "val", "test")])
missing = [p for p in allf.file_path if not os.path.exists(p)]
check(not missing, f"all {len(allf)} feature files exist" + (f" (missing e.g. {missing[0]})" if missing else ""))
keep = allf[allf.length >= MINF]
print(f"\nrows < 500 ms dropped by the dataset loader: {len(allf)-len(keep)} of {len(allf)} "
      f"({100*(len(allf)-len(keep))/len(allf):.1f}%), by class: " +
      str((allf[allf.length < MINF].groupby('label').size()).to_dict()))
seg = keep.assign(n=1 + (keep.length - MINF) // 59).groupby(["ID", "label"]).n.sum()
print(f"500 ms segments per speaker: min={seg.min()} median={int(seg.median())} max={seg.max()}; "
      f"mean HC={seg.xs(0, level='label').mean():.0f} PD={seg.xs(1, level='label').mean():.0f}")
check(seg.min() >= 10, "no speaker with <10 usable segments")
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
