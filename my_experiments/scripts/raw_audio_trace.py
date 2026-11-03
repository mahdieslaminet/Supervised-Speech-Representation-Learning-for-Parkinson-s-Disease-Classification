import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


# ----------------------------------------------------------------------------- raw-audio trace
# These two audio functions reproduce the submitted preprocess_pcgita.py;
# imports are local to avoid importing or executing that script's main program.
import re
import shutil
import tempfile
from datetime import datetime, timezone

TARGET_FS = 16000
TASK_DIRS = {"words": "Words", "sentences": "sentences",
             "sentences2": "sentences2", "readtext": "read text"}
ID_RX = re.compile(r"AVPEPUDE(AC|A)(\d{4})", re.IGNORECASE)


def load_16k(path):
    import soundfile as sf
    from scipy.signal import resample_poly
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


def vad_details(x):
    """Record the exact sample mask used by energy_vad, including its tail behavior."""
    fl, hp = 400, 160
    if len(x) < fl:
        return np.ones(len(x), bool), np.empty(0), np.empty(0, bool)
    n = 1 + (len(x) - fl) // hp
    energy = np.array([10 * np.log10(np.mean(x[i * hp:i * hp + fl] ** 2) + 1e-12) for i in range(n)])
    active = energy > energy.max() - 35.0
    expanded = active.copy()
    for i in np.where(active)[0]:
        expanded[max(0, i - 10):i + 11] = True
    mask = np.zeros(len(x), bool)
    for i in np.where(expanded)[0]:
        mask[i * hp:i * hp + fl] = True
    return mask, energy, active


def feature_key(path):
    m = re.fullmatch(r"((?:AC|A)\d{4})_(words|sentences|sentences2|readtext)_(.*)_p(\d+)", Path(path).stem)
    if not m:
        raise ValueError(f"Cannot map feature filename to original audio: {path}")
    return (m[1], m[2], m[3]), int(m[4])


def index_audio_zips(root, wanted):
    """Read ZIP directories only. Never extract an archive tree or touch source files."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw dataset ZIP folder not found: {root}")
    matches = {}
    for archive in sorted(root.rglob("*")):
        if not archive.is_file() or archive.suffix.lower() != ".zip":
            continue
        if not zipfile.is_zipfile(archive):
            raise ValueError(f"Unreadable ZIP (possibly split/encrypted): {archive}")
        with zipfile.ZipFile(archive) as z:
            for entry in sorted(z.infolist(), key=lambda v: v.filename):
                name = entry.filename.replace("\\", "/")
                parts = name.split("/")
                if entry.is_dir() or not name.endswith(".wav") or "__MACOSX" in parts:
                    continue
                for task, dirname in TASK_DIRS.items():
                    if dirname not in parts:
                        continue
                    # Same task-relative folder rules as the submitted scan().
                    rel = parts[parts.index(dirname):]
                    low = [v.lower() for v in rel]
                    if any("sobra" in v for v in low) or (task == "words" and "petaka" in low):
                        continue
                    m = ID_RX.search(parts[-1])
                    if not m:
                        continue
                    speaker = m[1].upper() + m[2]
                    item = re.sub(r"[_\s]+", "", Path(parts[-1]).stem[m.end():]).lower()
                    key = (speaker, task, item)
                    if key in wanted:
                        matches.setdefault(key, []).append((archive, entry.filename))
    for key in sorted(wanted):
        entries = matches.get(key, [])
        if not entries:
            raise FileNotFoundError(f"No original WAV in the ZIPs for feature key {key}")
        if len(entries) != 1:
            raise ValueError(f"Ambiguous original WAV for {key}: {entries}. Select the correct source; do not guess.")
    return {key: value[0] for key, value in matches.items()}


def new_output_dir(a, repo):
    parent = Path(a.out).expanduser().resolve()
    protected = [repo / "results", repo / a.data_dir, repo / "config", repo / "audio",
                 repo / "preprocess", repo / "ups_results", Path(a.raw_zip_dir).expanduser().resolve()]
    if getattr(a, "results_dir", None):
        protected.append(repo / a.results_dir)
    for path in protected:
        path = path.resolve()
        if parent == path or path in parent.parents:
            raise ValueError(f"Output must be outside inputs/results: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    prefix = f"mini_trace_{a.expname}_fold{a.fold}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
    if Path(prefix).name != prefix:
        raise ValueError("Experiment name must not contain directory separators")
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def plot_audio_stages(folder, before, after, feat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 7))
    for ax, audio, title in [(axes[0], before, "16 kHz mono, before VAD"),
                              (axes[1], after, "After energy VAD (concatenated speech)")]:
        step = max(1, len(audio) // 20000)
        ix = np.arange(0, len(audio), step)
        ax.plot(ix / TARGET_FS, audio[::step], lw=.5)
        ax.set_title(title); ax.set_xlabel("seconds")
    axes[2].imshow(feat, origin="lower", aspect="auto")
    axes[2].set_title("Regenerated Mel feature of one PCM16 piece")
    axes[2].set_xlabel("frame (4 ms)"); axes[2].set_ylabel("Mel bin")
    fig.tight_layout(); fig.savefig(folder / "audio_stages.png", dpi=110); plt.close(fig)


def regenerate_audio(rows, folder, sources, config_path, a):
    """Regenerate every model-input feature for this speaker; return original->new paths.

    Shapes must agree, since the original authors' dataset defines the segment table.
    Values are compared, not silently forced to match. Model inference uses NEW features.
    """
    import soundfile as sf
    from audio.audio_utils import get_feat, get_waveform
    folder.mkdir(exist_ok=False)
    grouped = {}
    for fp in rows.file_path.tolist():
        key, piece = feature_key(fp)
        grouped.setdefault(key, []).append((fp, piece))
    mapping, records = {}, []
    for number, (key, features) in enumerate(sorted(grouped.items())):
        speaker, task, item = key
        sd = folder / f"{number:03d}_{speaker}_{task}"
        sd.mkdir()
        archive, member = sources[key]
        raw_path = sd / "00_original.wav"
        with zipfile.ZipFile(archive) as z, z.open(member) as src, raw_path.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        wav_info = sf.info(str(raw_path))
        x = load_16k(str(raw_path))
        if not np.isfinite(x).all() or len(x) == 0:
            raise ValueError(f"Empty or nonfinite audio: {member}")
        # FLOAT WAVs retain the exact float32 arrays used by preprocessing.
        sf.write(str(sd / "01_mono_16k.wav"), x, TARGET_FS, subtype="FLOAT")
        mask, energy, active = vad_details(x)
        y = energy_vad(x) if a.vad == "energy" else x.copy()
        if a.vad == "none":
            mask = np.ones(len(x), bool)
        if not np.array_equal(y, x[mask]):
            raise AssertionError("Recorded VAD mask differs from actual VAD output")
        changes = np.diff(np.r_[False, mask, False].astype(int))
        intervals = np.column_stack((np.where(changes == 1)[0], np.where(changes == -1)[0]))
        np.savez_compressed(sd / "02_vad_details.npz", kept_intervals_samples=intervals,
                            frame_energy_db=energy, active_frames=active,
                            frame_ms=25, hop_ms=10, threshold_db=35., hang_ms=100,
                            vad_applied=a.vad == "energy")
        sf.write(str(sd / "02_after_vad.wav"), y, TARGET_FS, subtype="FLOAT")
        L = int(a.segmentlen * TARGET_FS)
        for fp, j in features:
            piece = y[j * L:(j + 1) * L]
            if len(piece) < .032 * TARGET_FS + 1:
                raise ValueError(f"Regenerated audio piece is too short: {fp}; check VAD/settings/source")
            pd_ = sd / f"piece_{j:03d}"; pd_.mkdir()
            wp = pd_ / "03_piece_pcm16.wav"
            # PCM16 quantization BEFORE get_feat is essential: original preprocessing does this.
            sf.write(str(wp), piece, TARGET_FS, subtype="PCM_16")
            fnew = get_feat(str(wp), str(config_path)).T
            if not np.isfinite(fnew).all():
                raise ValueError(f"Nonfinite regenerated Mel feature: {fp}")
            normal, fs = get_waveform(str(wp), normalization=False)
            np.save(pd_ / "04_dc_removed_peak_normalized_kaldi_input.npy", normal)
            if not np.isfinite(normal).all():
                raise ValueError(f"Nonfinite normalized audio: {fp}")
            sf.write(str(pd_ / "04_normalized_listen.wav"), normal / (2 ** 15), fs, subtype="FLOAT")
            new_path = pd_ / "05_regenerated_fbank.npy"
            np.save(new_path, fnew)
            original = np.load(fp, allow_pickle=False)
            np.save(pd_ / "06_original_saved_fbank.npy", original)
            same_shape = original.shape == fnew.shape
            diff = float(np.max(np.abs(original - fnew))) if same_shape and original.size else None
            close = bool(same_shape and np.allclose(original, fnew, rtol=1e-5, atol=1e-4))
            rec = dict(feature_file=str(fp), source_zip=str(archive), source_member=member,
                       original_wav_sha256=sha(raw_path, 64), original_sample_rate=wav_info.samplerate,
                       original_channels=wav_info.channels, original_frames=wav_info.frames,
                       samples_16k=len(x), samples_after_vad=len(y), piece_index=j,
                       piece_start_in_vad_samples=j * L, piece_samples=len(piece),
                       regenerated_feature=str(new_path.relative_to(folder.parent)),
                       original_shape=list(original.shape), regenerated_shape=list(fnew.shape),
                       same_shape=same_shape, max_abs_feature_difference=diff,
                       features_allclose=close, model_input_source="regenerated_from_raw_audio")
            records.append(rec)
            (pd_ / "comparison.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            if not same_shape:
                raise ValueError(f"Feature shape mismatch for {fp}: {original.shape} vs {fnew.shape}. "
                                 f"Cannot reuse the original segment table. See {pd_ / 'comparison.json'}")
            mapping[str(Path(fp).resolve())] = new_path
        if number < a.audio_plots:
            plot_audio_stages(sd, x, y, fnew)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 2.5))
            ax.plot(np.arange(normal.size) / fs, normal, lw=.4)
            ax.set_title("DC removed + peak normalized x32768 (last selected piece)")
            ax.set_xlabel("seconds"); fig.tight_layout()
            fig.savefig(sd / "normalized_waveform.png", dpi=110); plt.close(fig)
        (sd / "source.json").write_text(json.dumps(dict(source_zip=str(archive), source_member=member,
                                       source_wav_sha256=sha(raw_path, 64)), indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(folder / "audio_feature_checks.csv", index=False)
    return mapping, records


def write_manifest(out):
    entries = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "file_manifest.csv":
            entries.append(dict(path=path.relative_to(out).as_posix(), bytes=path.stat().st_size,
                                sha256=sha(path, 64)))
    pd.DataFrame(entries).to_csv(out / "file_manifest.csv", index=False)

