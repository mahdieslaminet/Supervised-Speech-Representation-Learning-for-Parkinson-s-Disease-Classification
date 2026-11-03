# -*- coding: utf-8 -*-
"""
PC-GITA preprocessing for pddetection-reps-learning (Idiap).

Put this file in:  <repo>/preprocess/pcgita_database/preprocess_pcgita.py
Run from anywhere:
    python preprocess/pcgita_database/preprocess_pcgita.py --pcgita_root "F:/PC-GITA" --n_folds 10
Pilot (small test):
    python preprocess/pcgita_database/preprocess_pcgita.py --pcgita_root "F:/PC-GITA" \
        --n_per_class 10 --n_folds 3 --out_name pcgita_pilot

What it does
  1. Scans the 4 tasks used in the paper: Words (24, 'petaka' excluded), sentences (6),
     sentences2 (4), read text (1)  -> 35 recordings per speaker.
  2. Speaker key = (AC | A) + 4 digits parsed from the file name (AC = healthy = label 0,
     A = PD = label 1). Folders named '*sobra*' (leftover speakers) are skipped.
  3. 44.1 kHz -> 16 kHz, optional energy VAD, cut into <= 8 s pieces (like the repo),
     fbank features via the repo's own audio.audio_utils.get_feat (D x T .npy).
  4. Writes CSVs in the repo format (uttr_num,ID,file_path,length,label,spk_ID) for
     folds/ and folds_spkID_task/, both *_online.csv and *_{feat}_offline.csv.
All CSV paths are relative to the repo root (run train_*.py from the repo root).
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
from audio.audio_utils import get_feat, get_config_args  # noqa: E402

TASK_DIRS = {"words": "Words", "sentences": "sentences",
             "sentences2": "sentences2", "readtext": "read text"}
ID_RX = re.compile(r"AVPEPUDE(AC|A)(\d{4})", re.IGNORECASE)
TARGET_FS = 16000


# ------------------------------- scanning --------------------------------- #
def scan(root, tasks, exclude_words):
    """Return {(task, item, speaker_key): path}; first file wins on duplicates."""
    found, dup, noid = {}, [], []
    for task in tasks:
        base = Path(root) / TASK_DIRS[task]
        for p in sorted(base.rglob("*.wav")):
            parts_low = [x.lower() for x in p.relative_to(root).parts]
            if any("sobra" in x for x in parts_low):
                continue  # leftover speakers (not in the 100 main speakers)
            if task == "words" and any(w in parts_low for w in exclude_words):
                continue
            m = ID_RX.search(p.name)
            if not m:
                noid.append(str(p)); continue
            spk = (m.group(1).upper(), m.group(2))
            item = re.sub(r"[_\s]+", "", p.stem[m.end():]).lower()
            key = (task, item, spk)
            if key in found:
                dup.append(str(p)); continue
            found[key] = str(p)
    return found, dup, noid


# ------------------------------- audio ------------------------------------ #
def load_16k(path):
    x, fs = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != TARGET_FS:
        g = np.gcd(fs, TARGET_FS)
        x = resample_poly(x, TARGET_FS // g, fs // g).astype("float32")
    return x


def energy_vad(x, fs=TARGET_FS, frame_ms=25, hop_ms=10, thr_db=35.0, hang_ms=100):
    """Keep frames within thr_db of the loudest frame (rough substitute for the
    Praat-based VAD used in the paper)."""
    fl, hp = int(fs * frame_ms / 1e3), int(fs * hop_ms / 1e3)
    if len(x) < fl:
        return x
    n = 1 + (len(x) - fl) // hp
    e = np.array([10 * np.log10(np.mean(x[i * hp:i * hp + fl] ** 2) + 1e-12) for i in range(n)])
    keep = e > e.max() - thr_db
    hang = hang_ms // hop_ms
    k2 = keep.copy()
    for i in np.where(keep)[0]:
        k2[max(0, i - hang):i + hang + 1] = True
    mask = np.zeros(len(x), bool)
    for i in np.where(k2)[0]:
        mask[i * hp:i * hp + fl] = True
    return x[mask]


# -------------------------------- folds ----------------------------------- #
def make_folds(spk_ids, labels, n_folds, seed):
    """ONE stratified split -> proper partition of speakers (test), plus a
    dev fold of the same size as test taken from the rest."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out = []
    for k, (tr_i, te_i) in enumerate(skf.split(spk_ids, labels), 1):
        tr_ids, tr_lab = spk_ids[tr_i], labels[tr_i]
        inner = StratifiedKFold(n_splits=max(n_folds - 1, 2), shuffle=True, random_state=seed + k)
        a, b = next(inner.split(tr_ids, tr_lab))
        out.append(dict(train=tr_ids[a], val=tr_ids[b], test=spk_ids[te_i]))
    return out


def save_pair(wav_df, feat_df, folder, split, k, feat_type):
    os.makedirs(folder, exist_ok=True)
    wav_df.to_csv(os.path.join(folder, f"{split}_fold{k}_online.csv"), index=False)
    feat_df.to_csv(os.path.join(folder, f"{split}_fold{k}_{feat_type}_offline.csv"), index=False)


# -------------------------------- main ------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcgita_root", required=True, help="folder containing Words, sentences, ...")
    ap.add_argument("--out_name", default="pcgita_database", help="folder under preprocess/")
    ap.add_argument("--tasks", nargs="+", default=list(TASK_DIRS), choices=list(TASK_DIRS))
    ap.add_argument("--exclude_words", nargs="*", default=["petaka"],
                    help="word folders to drop (paper: 24 words; 'petaka' is the 25th)")
    ap.add_argument("--n_per_class", type=int, default=0, help="0 = all 50 per class; else pilot subset")
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vad", choices=["energy", "none"], default="energy")
    ap.add_argument("--segmentlen", type=float, default=8.0, help="max piece length [s]")
    ap.add_argument("--min_files_spkid", type=int, default=3)
    ap.add_argument("--config_path", default="config/audio_config.yaml")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    out_dir = Path("preprocess") / a.out_name
    feat_type = get_config_args(a.config_path)["feat_type"]
    found, dup, noid = scan(a.pcgita_root, a.tasks, [w.lower() for w in a.exclude_words])
    print(f"files kept: {len(found)} | duplicates skipped: {len(dup)} | no-ID skipped: {len(noid)}")
    for d in dup:
        print("  dup:", d)

    # speakers
    per_spk = defaultdict(list)
    for (task, item, spk), p in found.items():
        per_spk[spk].append((task, item, p))
    hc = sorted(s for s in per_spk if s[0] == "AC")
    pd_ = sorted(s for s in per_spk if s[0] == "A")
    print(f"speakers found: HC={len(hc)} PD={len(pd_)}")
    if a.n_per_class:
        rng = np.random.RandomState(a.seed)
        hc = [hc[i] for i in sorted(rng.permutation(len(hc))[:a.n_per_class])]
        pd_ = [pd_[i] for i in sorted(rng.permutation(len(pd_))[:a.n_per_class])]
    spk_list = hc + pd_
    spk_int = {s: i for i, s in enumerate(spk_list)}          # global speaker ID
    spk_label = {s: (0 if s[0] == "AC" else 1) for s in spk_list}
    spk_name = {s: f"{s[0]}{s[1]}" for s in spk_list}
    counts = {spk_name[s]: len(per_spk[s]) for s in spk_list}
    odd = {k: v for k, v in counts.items() if v != 35 and len(a.tasks) == 4}
    print(f"selected speakers: {len(spk_list)} (HC={len(hc)}, PD={len(pd_)}); files/speaker != 35: {odd}")
    if a.dry_run:
        return

    # audio -> pieces -> features
    seg_dir = out_dir / "segmented_audio_data"
    feat_dir = out_dir / "features_data" / feat_type
    os.makedirs(seg_dir, exist_ok=True); os.makedirs(feat_dir, exist_ok=True)
    rows_wav, rows_feat = [], []
    from tqdm import tqdm
    for s in tqdm(spk_list):
        n = 0
        for task, item, p in sorted(per_spk[s]):
            x = load_16k(p)
            if a.vad == "energy":
                x = energy_vad(x)
            L = int(a.segmentlen * TARGET_FS)
            for j, st in enumerate(range(0, len(x), L)):
                piece = x[st:st + L]
                if len(piece) < 0.032 * TARGET_FS + 1:
                    continue
                stem = f"{spk_name[s]}_{task}_{item}_p{j}"
                wp = seg_dir / f"{stem}.wav"
                sf.write(str(wp), piece, TARGET_FS, subtype="PCM_16")
                feat = get_feat(str(wp), a.config_path).T            # D x T
                fp = feat_dir / f"{stem}.npy"
                np.save(str(fp), feat)
                base = dict(uttr_num=n, ID=spk_int[s], label=spk_label[s], spk_ID=spk_name[s])
                rows_wav.append({**base, "file_path": wp.as_posix(), "length": len(piece) * 1e3 / TARGET_FS})
                rows_feat.append({**base, "file_path": fp.as_posix(), "length": feat.shape[1]})
                n += 1
    cols = ["uttr_num", "ID", "file_path", "length", "label", "spk_ID"]
    wav_df = pd.DataFrame(rows_wav)[cols]; feat_df = pd.DataFrame(rows_feat)[cols]
    wav_df.to_csv(out_dir / "audio_data.csv", index=False)
    feat_df.to_csv(out_dir / f"{feat_type}_features_data.csv", index=False)
    pd.DataFrame([dict(ID=spk_int[s], spk_ID=spk_name[s], label=spk_label[s], n_files=len(per_spk[s]))
                  for s in spk_list]).to_csv(out_dir / "speakers.csv", index=False)

    # folds (speaker-independent) + speaker-ID folds
    ids = np.array([spk_int[s] for s in spk_list]); labs = np.array([spk_label[s] for s in spk_list])
    folds = make_folds(ids, labs, a.n_folds, a.seed)
    rng = np.random.RandomState(a.seed)
    info = {}
    for k, f in enumerate(folds, 1):
        for split in ("train", "val", "test"):
            sel = lambda df: df[df.ID.isin(f[split])]
            save_pair(sel(wav_df), sel(feat_df), out_dir / "folds", split, k, feat_type)
        # speaker-ID task: healthy speakers NOT in the test fold, utterances split 60/20/20
        hc_ids = sorted(i for i in np.concatenate([f["train"], f["val"]]) if labs[ids == i][0] == 0)
        cls = {sid: c for c, sid in enumerate(hc_ids)}
        parts = {"train": [], "val": [], "test": []}
        for sid in hc_ids:
            idx = np.where(wav_df.ID.values == sid)[0]
            if len(idx) < a.min_files_spkid:
                continue
            idx = rng.permutation(idx)
            n_tr, n_va = int(round(.6 * len(idx))), int(round(.2 * len(idx)))
            parts["train"] += list(idx[:n_tr]); parts["val"] += list(idx[n_tr:n_tr + n_va]); parts["test"] += list(idx[n_tr + n_va:])
        for split, ix in parts.items():
            w = wav_df.iloc[sorted(ix)].copy(); ft = feat_df.iloc[sorted(ix)].copy()
            w["label"] = w.ID.map(cls); ft["label"] = ft.ID.map(cls)
            save_pair(w, ft, out_dir / "folds_spkID_task", split, k, feat_type)
        info[k] = dict(n_spk_id_classes=len(hc_ids),
                       **{f"{s}_speakers": int(len(f[s])) for s in ("train", "val", "test")},
                       test_PD=int(labs[np.isin(ids, f["test"])].sum()))
    json.dump(info, open(out_dir / "folds_info.json", "w"), indent=2)
    print(json.dumps(info, indent=2))
    n_cls = sorted({v["n_spk_id_classes"] for v in info.values()})
    print(f"\nSet 'outputdim' in config/upstream_auxiliary1_config.yaml to {n_cls} "
          f"(per fold, see folds_info.json) and data_path to preprocess/{a.out_name}/folds[_spkID_task]/")


if __name__ == "__main__":
    main()
