# -*- coding: utf-8 -*-
"""Build an EVIDENCE PACK from what the pipeline already recorded (nothing is typed by hand).

    python my_experiments/scripts/make_report.py \
        --results /content/drive/MyDrive/idiap_pd/results \
        --data_dir /content/repo/preprocess/pcgita_full_100spk \
        --out /content/drive/MyDrive/idiap_pd/evidence

Sources used (all written by the training scripts themselves):
  results/<method>_s<seed>/ds-*/outputs/output_<fold>.npy   raw per-speaker scores (test/val/train)
  results/<method>_s<seed>/logs/fold<k>_{ups,ds}.log         full training logs
  results/<method>_s<seed>/status/*.done|.progress           finish markers + time stamps
  results/<method>_s<seed>/settings.json + config copies     hyper-parameters actually used
  <data_dir>/folds*/ , speakers.csv, folds_info.json         the inputs (CSV lists) of every stage
Every metric is RE-COMPUTED from the raw scores and compared with (a) the value stored by the repo in the .npy
and (b) the number printed in the training log. Mismatches are listed in INTEGRITY.md.
"""
import argparse, glob, hashlib, json, math, os, re, sys, time, zipfile, platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PAPER = {  # Table 1 of Janbakhshi & Kodrasi (ITG 2021): accuracy [%] mean, std, AUC mean
    "baseline": (66.20, 1.17, 0.77), "adv": (72.00, 5.62, 0.84), "pd": (71.00, 1.90, 0.78), "fusion": (75.4, 1.02, 0.80)}

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--data_dir", default="preprocess/pcgita_full_100spk")
ap.add_argument("--out", default="evidence")
ap.add_argument("--include_ckpt", action="store_true", help="also put the (large) checkpoints in the zip")
a = ap.parse_args()
res, dd, out = Path(a.results), Path(a.data_dir), Path(a.out)
out.mkdir(parents=True, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - w), 100 * (c + w)

def metrics(scores, target):
    pred = scores.argmax(1); t = target.astype(int)
    acc = 100 * (pred == t).mean()
    sen = 100 * ((pred == 1) & (t == 1)).sum() / max(1, (t == 1).sum())
    spe = 100 * ((pred == 0) & (t == 0)).sum() / max(1, (t == 0).sum())
    auc = roc_auc_score(t, 1 - scores[:, 0]) if len(set(t)) == 2 else float("nan")   # same formula as train_downstream.py
    return acc, sen, spe, auc

def log_test_block(logfile):
    """Speaker-level Acc/Sen/Spe/AUC printed in the LAST 'test data' block of a downstream log."""
    if not logfile.exists(): return None
    txt = logfile.read_text(encoding="utf-8", errors="replace")
    i = txt.rfind("---------test data----------")
    if i < 0: return None
    blk = txt[i:i + 1500]
    g = lambda k: (re.search(rf"\n\s*{k}:\s*([-\d.eE]+|None)", blk) or [None, None])[1]
    try: return tuple(float(g(k)) if g(k) not in (None, "None") else float("nan") for k in ("Acc", "Sen", "Spe", "AUC"))
    except Exception: return None

def attempts(logfile):
    if not logfile.exists(): return 0
    return sum(1 for l in open(logfile, encoding="utf-8", errors="replace") if l.startswith("===== 2") and " exit " not in l)

rows, spk_rows, manifest = [], [], []
fold_labels = {}
for exp in sorted(res.glob("*_s[0-9]*")):
    m = re.match(r"(.+)_s(\d+)$", exp.name)
    if not m or not (exp / f"ds-{exp.name}").exists(): continue
    method, seed = m.group(1), int(m.group(2))
    for f in sorted((exp / f"ds-{exp.name}" / "outputs").glob("output_*.npy"), key=lambda p: int(p.stem.split("_")[1])):
        k = int(f.stem.split("_")[1])
        d = np.load(f, allow_pickle=True).item()["test"]
        s_acc_chunk, scores, tgt, s_acc, s_sen, s_spe, s_auc = d
        scores, tgt = np.asarray(scores, float), np.asarray(tgt, float)
        acc, sen, spe, auc = metrics(scores, tgt)
        lg = log_test_block(exp / "logs" / f"fold{k}_ds.log")
        mk = exp / "status" / f"fold{k}_ds.done"
        # label order vs the fold's test CSV (speakers are expected in ascending-ID order)
        csvp = dd / "folds" / f"test_fold{k}_fbank_offline.csv"; order_ok = None; ids = [None] * len(tgt)
        if csvp.exists():
            c = pd.read_csv(csvp)
            cands = {"first-appearance": c.drop_duplicates("ID").set_index("ID").label,
                     "ascending-ID": c.groupby("ID").label.first().sort_index()}
            order_ok = False
            for nm, t in cands.items():          # test loader has shuffle=False -> first-appearance is the expected order
                if len(t) == len(tgt) and (t.values == tgt.astype(int)).all():
                    order_ok = True; ids = list(t.index); break
        rows.append(dict(method=method, seed=seed, fold=k, n_test_spk=len(tgt), n_PD=int(tgt.sum()),
                         acc=acc, sen=sen, spe=spe, auc=auc, acc_stored=s_acc, auc_stored=s_auc,
                         acc_log=lg[0] if lg else np.nan, auc_log=lg[3] if lg else np.nan,
                         ok_vs_stored=bool(abs(acc - s_acc) < 0.01 and (np.isnan(auc) or abs(auc - float(s_auc)) < 6e-4)),
                         ok_vs_log=None if not lg else bool(abs(acc - lg[0]) < 0.06 and (np.isnan(auc) or abs(auc - lg[3]) < 0.0015)),
                         label_order_matches_csv=order_ok,
                         attempts_ups=attempts(exp / "logs" / f"fold{k}_ups.log"), attempts_ds=attempts(exp / "logs" / f"fold{k}_ds.log"),
                         finished_at=time.strftime("%Y-%m-%d %H:%M", time.localtime(mk.stat().st_mtime)) if mk.exists() else "",
                         npy_sha256=sha(f)[:16]))
        for j, (sc, tg) in enumerate(zip(scores, tgt)):
            spk_rows.append(dict(method=method, seed=seed, fold=k, speaker_ID_assumed=ids[j], label=int(tg), P_PD=round(float(1 - sc[0]), 6), pred=int(sc.argmax())))
        manifest.append((sha(f), str(f)))
df = pd.DataFrame(rows)
if df.empty: sys.exit("no output_*.npy found under " + str(res))
df.to_csv(out / "all_folds.csv", index=False)
pd.DataFrame(spk_rows).to_csv(out / "per_speaker_scores.csv", index=False)

# ---------------- summary ----------------
lines = ["# Results summary (all numbers re-computed from raw per-speaker scores)\n"]
for (method, seed), g in df.groupby(["method", "seed"]):
    n = len(g); sp = pd.DataFrame(spk_rows); sp = sp[(sp.method == method) & (sp.seed == seed)]
    corr = int((sp.label == sp.pred).sum()); lo, hi = wilson(corr, len(sp))
    tp = int(((sp.label == 1) & (sp.pred == 1)).sum()); fn = int(((sp.label == 1) & (sp.pred == 0)).sum())
    tn = int(((sp.label == 0) & (sp.pred == 0)).sum()); fp = int(((sp.label == 0) & (sp.pred == 1)).sum())
    pa = PAPER.get(method)
    lines += [f"## {method}, seed {seed}  ({n} folds, {len(sp)} speakers tested)",
              f"- mean over folds: accuracy {g.acc.mean():.1f} +- {g.acc.std():.1f} (std over folds), sensitivity {g.sen.mean():.1f}, specificity {g.spe.mean():.1f}, AUC {g.auc.mean():.3f} +- {g.auc.std():.3f}",
              f"- pooled over all tested speakers: {corr}/{len(sp)} correct = {100*corr/len(sp):.1f}% (95% Wilson interval {lo:.1f} to {hi:.1f})",
              f"- confusion matrix (rows=true, cols=predicted): HC [{tn}, {fp}], PD [{fn}, {tp}]"]
    if pa: lines.append(f"- paper (Table 1, mean over 5 seeds): accuracy {pa[0]} +- {pa[1]}, AUC {pa[2]}")
    if n < 10: lines.append(f"- NOTE: only {n} of 10 folds finished")
    lines.append("")
lines.append("Limits: ONE seed per method, so the paper's '+-' (spread over seeds) is not reproduced; with 100 test speakers one speaker = 1 percentage point.")
(out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

# ---------------- integrity ----------------
L = ["# Integrity report\n", "## 1. Recomputation checks (per fold)",
     f"- folds checked: {len(df)}",
     f"- recomputed == value stored in output_k.npy: {int(df.ok_vs_stored.sum())}/{len(df)}",
     f"- recomputed == number printed in the training log: {int((df.ok_vs_log == True).sum())}/{int(df.ok_vs_log.notna().sum())} (logs found for {int(df.ok_vs_log.notna().sum())})",
     f"- label order of the .npy equals label order of the test CSV: {int((df.label_order_matches_csv == True).sum())}/{int(df.label_order_matches_csv.notna().sum())}"]
bad = df[(~df.ok_vs_stored) | (df.ok_vs_log == False)]
L.append("- MISMATCHES: " + ("none" if bad.empty else "\n" + bad[["method", "seed", "fold", "acc", "acc_stored", "acc_log", "auc", "auc_stored", "auc_log"]].to_string(index=False)))
L += ["\n## 2. Run history (honest list of interruptions)",
      "attempts>1 means the stage was interrupted and resumed/restarted; rotated logs are named *.old-* in results/<exp>/logs/."]
hist = df[(df.attempts_ups > 1) | (df.attempts_ds > 1)][["method", "seed", "fold", "attempts_ups", "attempts_ds"]]
L.append(hist.to_string(index=False) if not hist.empty else "no interrupted stage found in the current logs")
old = sorted(glob.glob(str(res / "*" / "logs" / "*.old-*")))
L.append(f"\nrotated (restarted) logs: {len(old)}" + ("".join(f"\n- {Path(p).relative_to(res)}" for p in old)))
# data partition checks
L.append("\n## 3. Input checks")
try:
    info = json.load(open(dd / "folds_info.json")); sp = pd.read_csv(dd / "speakers.csv")
    tested = []
    for k in sorted(int(x) for x in info):
        tested += sorted(set(pd.read_csv(dd / "folds" / f"test_fold{k}_fbank_offline.csv").ID))
    L += [f"- speakers.csv: {len(sp)} speakers (HC={int((sp.label==0).sum())}, PD={int((sp.label==1).sum())})",
          f"- every speaker is in exactly one test fold: {sorted(tested) == sorted(sp.ID.tolist())}",
          f"- folds: {len(info)}; speakers per fold (train/val/test) of fold 1: {info['1']['train_speakers']}/{info['1']['val_speakers']}/{info['1']['test_speakers']}"]
except Exception as e:
    L.append(f"- could not check inputs: {e}")
# settings
L.append("\n## 4. Settings actually used (settings.json of each experiment)")
for sj in sorted(res.glob("*_s[0-9]*/settings.json")): L.append(f"- {sj.parent.name}: {sj.read_text().strip()}")
env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
try:
    import sklearn; env["sklearn"] = sklearn.__version__
    import torch; env["torch"] = torch.__version__; env["cuda"] = torch.version.cuda
    if torch.cuda.is_available(): env["gpu"] = torch.cuda.get_device_name(0)
except Exception: pass
L += ["\n## 5. Environment (this report)", json.dumps(env), f"\nreport made: {time.strftime('%Y-%m-%d %H:%M')}",
      "\n## 6. What this does NOT prove",
      "- it cannot prove that the logs were not edited. The strongest protection is to give the supervisor VIEW access to the original Drive folder (Drive keeps version history) and let them re-run one fold.",
      "- a re-run is not bit-identical (GPU non-determinism); small differences are expected."]
(out / "INTEGRITY.md").write_text("\n".join(L), encoding="utf-8")

# ---------------- pipeline input/output documentation (real shapes/counts) ----------------
P = ["# Pipeline inputs and outputs (values read from the actual files)\n"]
def head(p, n=3):
    try: return pd.read_csv(p).head(n).to_string(index=False)
    except Exception as e: return f"(not found: {e})"
try:
    sp = pd.read_csv(dd / "speakers.csv"); ad = pd.read_csv(dd / "audio_data.csv"); fe = pd.read_csv(next(dd.glob("*_features_data.csv")))
    ex = np.load(Path(a.data_dir).parents[1] / fe.file_path.iloc[0]) if not Path(fe.file_path.iloc[0]).exists() else np.load(fe.file_path.iloc[0])
    P += ["## Stage 0 -> 1: raw PC-GITA recordings -> 16 kHz pieces (<= 8 s) -> fbank features",
          f"- speakers: {len(sp)}; pieces (wav): {len(ad)}; total speech after VAD: {ad.length.sum()/60000:.1f} min",
          f"- feature files: {len(fe)}; example shape (mel bins x frames): {ex.shape} (126 bins; 32 ms window, 4 ms shift)",
          "- CSV columns: " + ", ".join(fe.columns), "```\n" + head(dd / next(dd.glob('*_features_data.csv')).name) + "\n```"]
except Exception as e:
    P.append(f"(stage 1 details unavailable: {e})")
try:
    P += ["\n## Stage 1 -> 2/3: fold CSVs (the input list of every training script)"]
    for sub in ("folds", "folds_spkID_task"):
        c = {s: pd.read_csv(dd / sub / f"{s}_fold1_fbank_offline.csv") for s in ("train", "val", "test")}
        P.append(f"- {sub}/ fold 1: " + "; ".join(f"{s}: {len(v)} rows, {v.ID.nunique()} speakers" for s, v in c.items()))
    P.append("- 500 ms segment = 1 + (500-32)/4 = 118 frames -> model input 126 x 118 (min-max scaled per segment); rows shorter than 500 ms are dropped by the loader")
except Exception as e:
    P.append(f"(stage 2 inputs unavailable: {e})")
P += ["\n## Stage 2: upstream (auto-encoder [+ auxiliary tasks])",
      "- input: train/val/test fold CSVs (+ folds_spkID_task for the speaker-ID adversary)",
      "- output: results/<exp>/ups-<exp>/outputs/saved_model/UPs_NN_fold<k>.ckpt, logs/fold<k>_ups.log (losses per epoch)",
      "\n## Stage 3: downstream (MLP classifier on the 128-dim code)",
      "- input: UPs_NN_fold<k>.ckpt of the SAME fold + the fold CSVs",
      "- output: Ds_NN_fold<k>.ckpt, output_<k>.npy = {test, val, train} each [chunk_acc, speaker_scores(n x 2), target, speaker_acc, sen, spe, AUC], logs/fold<k>_ds.log",
      "- speaker decision = average of the segment softmax scores of that speaker (soft voting)",
      "\n## Stage 4: this report",
      "- input: all output_<k>.npy + logs; output: all_folds.csv, per_speaker_scores.csv, SUMMARY.md, INTEGRITY.md, MANIFEST.sha256"]
(out / "PIPELINE_IO.md").write_text("\n".join(P), encoding="utf-8")

# ---------------- manifest + zip ----------------
extra = [p for pat in ("*/settings.json", "*/status/*", "*/logs/*", "*/*/*_config.yaml", "*/*/feat_config.yaml") for p in res.glob(pat) if p.is_file()]
if a.include_ckpt: extra += list(res.glob("*/*/outputs/saved_model/*.ckpt"))
for p in extra: manifest.append((sha(p), str(p)))
for p in [dd / "speakers.csv", dd / "folds_info.json"] + list((dd).glob("folds*/*_offline.csv")):
    if p.exists(): manifest.append((sha(p), str(p)))
(out / "MANIFEST.sha256").write_text("\n".join(f"{h}  {p}" for h, p in sorted(manifest, key=lambda x: x[1])), encoding="utf-8")
zp = out / "evidence_pack.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in list(out.glob("*.md")) + list(out.glob("*.csv")) + [out / "MANIFEST.sha256"]: z.write(p, p.name)
    for p in extra:
        if p.suffix != ".ckpt" or a.include_ckpt: z.write(p, "raw/" + str(p.relative_to(res)))
    for f in res.glob("*/ds-*/outputs/output_*.npy"): z.write(f, "raw/" + str(f.relative_to(res)))
    for p in [dd / "speakers.csv", dd / "folds_info.json"] + list(dd.glob("folds*/*_offline.csv")):
        if p.exists(): z.write(p, "inputs/" + str(p.relative_to(dd)))
print(f"folds: {len(df)} | stored-match {int(df.ok_vs_stored.sum())}/{len(df)} | log-match {int((df.ok_vs_log==True).sum())}/{int(df.ok_vs_log.notna().sum())}")
print("written to", out, "->", [p.name for p in sorted(out.iterdir())])
