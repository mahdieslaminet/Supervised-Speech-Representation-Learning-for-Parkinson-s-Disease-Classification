# -*- coding: utf-8 -*-
"""Raw-audio mini-trace, based on the supplied mini_trace.py.

Reads selected WAV members without modifying the original ZIPs. Recreates mono
16 kHz -> VAD -> PCM16 <=8 s pieces -> authors' fbank -> dataset segmentation ->
model -> speaker vote. Original and regenerated features/scores are compared.
Outputs always use a new timestamped folder. See README_FA.md for Colab steps.
"""
import argparse, hashlib, json, os, platform, sys, time, zipfile
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd

from raw_audio_trace import (feature_key, index_audio_zips, new_output_dir,
                             regenerate_audio, write_manifest)

STAGES = [  # (stage, what happens, where in the authors' code)
    ("0a raw WAV", "copy only selected originals, with archive member path and SHA256", "raw_audio_trace.py: index_audio_zips"),
    ("0b mono + resample", "average channels; resample_poly to 16 kHz", "submitted preprocess_pcgita.py: load_16k"),
    ("0c VAD", "energy: 25 ms / 10 ms, 35 dB below max, +/-100 ms expansion; concatenate kept samples", "submitted preprocess_pcgita.py: energy_vad"),
    ("0d audio pieces", "cut <=8 s; write PCM16 before feature extraction", "submitted preprocess_pcgita.py: main"),
    ("0e waveform normalization", "remove DC; divide by peak; multiply by 32768 for Kaldi", "audio/audio_utils.py: get_waveform(normalization=False)"),
    ("0f Mel", "authors' get_feat; use saved experiment config; compare against original .npy", "audio/audio_utils.py: get_feat"),
    ("1 fold CSV", "list of feature files of the test speakers (rows shorter than 118 frames = 500 ms are dropped)", "audio/audio_dataset.py: AcousticDataset.__init__"),
    ("2 load features", "one .npy per piece of speech, array [126 mel bins x frames]", "audio/audio_dataset.py: OfflineAcousticDataset._load_feat"),
    ("3 segmentation", "sliding window of 118 frames (500 ms) with shift 59 (50% overlap)", "audio/audio_dataset.py: AcousticDataset._getindex / _sample"),
    ("4 min-max scaling", "each segment is scaled to [0,1] with ITS OWN min and max (the paper says z-score)", "audio/audio_dataset.py: AcousticDataset.standardize"),
    ("5 encoder CNN", "4 blocks conv(3x3) -> max-pool -> batch-norm -> leaky-ReLU", "upstream_auxiliary/ups_model.py: AECNNNetEnc.EncoderCNN"),
    ("6 flatten + encoder MLP", "3840 -> 256 -> 128: the 128-d code (bottleneck representation)", "upstream_auxiliary/ups_model.py: AECNNNetEnc.EncoderMLP"),
    ("7 classifier MLP", "128 -> 64 -> 64 -> 2 (HC, PD)", "downstream/ds_model.py: MLP.classifier"),
    ("8 softmax per segment", "probability of HC / PD for every segment", "downstream/ds_runner.py: DownsRunner._test_epoch"),
    ("9 soft voting", "speaker score = mean of the segment softmax scores of that speaker", "downstream/ds_runner.py: DownsRunner._test_epoch (np.mean)"),
    ("10 decision", "speaker is called PD if the PD score is larger than the HC score", "train_downstream.py (accuracy / AUC from speaker scores)"),
]


def sha(p, n=16):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


# ----------------------------------------------------------------------------- pure numpy helpers
def segment_table(dataset):
    """For every segment index of the authors' dataset: (utterance row, start frame, speaker ID)."""
    N = len(dataset)
    utt = np.zeros(N, int); start = np.zeros(N, int)
    for i in range(N):
        u, s = dataset._getindex(i)
        utt[i], start[i] = u, s
    spk = np.array(dataset.spkID)[utt]
    return utt, start, spk


def kept_order(spk, n_kept):
    """speakers in order of first appearance among the segments that are really used (drop_last removes the tail)."""
    seen, order = set(), []
    for s in spk[:n_kept]:
        if s not in seen:
            seen.add(s); order.append(int(s))
    return order


def choose_speakers(order, labels, n_spk, wanted=None):
    if wanted:
        return [int(s) for s in wanted]
    n_hc = n_spk // 2; n_pd = n_spk - n_hc
    hc = [s for s in order if labels[s] == 0][:n_hc]
    pd_ = [s for s in order if labels[s] == 1][:n_pd]
    return hc + pd_


# ----------------------------------------------------------------------------- torch backend (authors' classes)
class TorchBackend:
    def __init__(self):
        import torch
        self.torch = torch

    def load_ckpt(self, p):
        try:
            return self.torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            return self.torch.load(p, map_location="cpu")

    def make_models(self, ds_ck, ups_ck, freq, seq):
        return Models(self.torch, ds_ck, ups_ck, freq, seq)


class Models:
    def __init__(self, torch, ds_ck, ups_ck, freq, seq):
        from upstream_auxiliary.pretrain_ups_expert import FeatureExtractionPretrained, RepresentationPretrain
        from downstream.train_downs_expert import DownStreamTrain
        self.torch = torch
        self.ups_cfg, self.ds_cfg = ds_ck["UpsConfig"], ds_ck["DsConfig"]
        self.sel = self.ds_cfg["SelectedEncodedLayers"]
        self.up = FeatureExtractionPretrained(freq, seq, self.ups_cfg); self.up.load_model(ds_ck)
        dim = self.up.get_feature_dimension(selected_encoded_layers=self.sel)
        self.ds = DownStreamTrain(dim, self.ds_cfg); self.ds.load_model(ds_ck)
        self.up.eval(); self.ds.eval()
        self.ae = None
        if ups_ck is not None:
            cfg_u = ups_ck.get("UpsConfig", self.ups_cfg)
            self.ae = RepresentationPretrain(freq, seq, cfg_u); self.ae.load_model(ups_ck); self.ae.eval()

    def trace(self, xnp):
        """xnp [B,126,118] (already min-max scaled) -> dict with all layer outputs."""
        torch = self.torch
        # MINI_TRACE_DEVICE_ALIGNMENT_V4
        device = next(self.up.model.parameters()).device
        self.ds.model.to(device)
        if self.ae is not None:
            self.ae.model.to(device)
        x = torch.from_numpy(np.ascontiguousarray(xnp)).to(device=device, dtype=torch.float32)
        out = {"layers": OrderedDict()}
        with torch.no_grad():
            ref = self.ds(self.up(x, selected_encoded_layers=self.sel))      # the repo's own forward
            if self.sel == -1:
                enc = self.up.model.EncoderNet
                h = x.unsqueeze(1)
                for i, l in enumerate(enc.EncoderCNN):
                    h = l(h); out["layers"][f"enc_cnn_{i:02d}_{l.__class__.__name__}"] = h.cpu().numpy()
                h = h.reshape(-1, enc.lsize); out["layers"]["enc_flatten"] = h.cpu().numpy()
                for i, l in enumerate(enc.EncoderMLP):
                    h = l(h); out["layers"][f"enc_mlp_{i:02d}_{l.__class__.__name__}"] = h.cpu().numpy()
                out["code"] = h.cpu().numpy()
                for i, l in enumerate(self.ds.model.classifier):
                    h = l(h); out["layers"][f"cls_mlp_{i:02d}_{l.__class__.__name__}"] = h.cpu().numpy()
                out["manual_matches_repo_forward"] = bool(torch.allclose(h, ref, atol=1e-5))
                out["manual_vs_repo_maxdiff"] = float((h - ref).abs().max())
                logits = ref  # authoritative authors' forward, even if the diagnostic differs
            else:
                logits = ref; out["manual_matches_repo_forward"] = None; out["manual_vs_repo_maxdiff"] = None
            out["logits"] = logits.cpu().numpy()
            out["probs"] = torch.softmax(logits, dim=1).cpu().numpy()
            if self.ae is not None:
                try:
                    res = self.ae(x)
                    recon, code_u = (res[0], res[1]) if isinstance(res, (tuple, list)) else (res, None)
                    recon = recon.squeeze(1) if recon.dim() == 4 else recon
                    d, t = min(recon.shape[1], x.shape[1]), min(recon.shape[2], x.shape[2])
                    out["recon"] = recon.cpu().numpy()
                    out["recon_mse"] = ((recon[:, :d, :t] - x[:, :d, :t]) ** 2).mean(dim=(1, 2)).cpu().numpy()
                    if code_u is not None:
                        out["code_ups"] = code_u.reshape(code_u.shape[0], -1).cpu().numpy()
                except Exception as e:
                    out["ae_error"] = repr(e)
        return out


# ----------------------------------------------------------------------------- plots
def make_plots(figdir, sid, lab, seg0_raw, seg0_mm, conv0, code, p_pd, kept, mean_p):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        return [f"(matplotlib not available: {e})"]
    figdir.mkdir(exist_ok=True); made = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for a_, im, t in zip(ax, (seg0_raw, seg0_mm), ("segment 0 raw fbank", "segment 0 after min-max (model input)")):
        h = a_.imshow(im, origin="lower", aspect="auto"); a_.set_title(t); a_.set_xlabel("frame (4 ms)"); a_.set_ylabel("mel bin"); fig.colorbar(h, ax=a_)
    fig.tight_layout(); fig.savefig(figdir / "1_input_segment.png", dpi=110); plt.close(fig); made.append("1_input_segment.png")
    maps = [(k, v) for k, v in conv0.items() if k.endswith("LeakyReLU") and k.startswith("enc_cnn")]
    if maps:
        fig, ax = plt.subplots(1, len(maps), figsize=(3.2 * len(maps), 3))
        ax = np.atleast_1d(ax)
        for a_, (k, v) in zip(ax, maps):
            a_.imshow(v[0].mean(0), origin="lower", aspect="auto"); a_.set_title(f"{k[8:10]}: {v.shape[1]} ch x {v.shape[2]}x{v.shape[3]}\n(channel mean)", fontsize=8)
        fig.tight_layout(); fig.savefig(figdir / "2_conv_blocks_segment0.png", dpi=110); plt.close(fig); made.append("2_conv_blocks_segment0.png")
    if code is not None:
        fig, a_ = plt.subplots(figsize=(8, 3.5)); h = a_.imshow(code.T, aspect="auto", origin="lower")
        a_.set_xlabel("segment number"); a_.set_ylabel("code dimension (128)"); a_.set_title(f"128-d code of every segment, speaker {sid} ({lab})"); fig.colorbar(h, ax=a_)
        fig.tight_layout(); fig.savefig(figdir / "3_code_all_segments.png", dpi=110); plt.close(fig); made.append("3_code_all_segments.png")
    fig, a_ = plt.subplots(figsize=(8, 3.2)); idx = np.arange(len(p_pd))
    a_.plot(idx[kept], p_pd[kept], ".", ms=3, label="segments used"); a_.plot(idx[~kept], p_pd[~kept], "x", ms=4, c="r", label="dropped by drop_last")
    a_.axhline(0.5, c="gray", ls=":"); a_.axhline(mean_p, c="k", label=f"soft voting mean = {mean_p:.3f}")
    a_.set_ylim(0, 1); a_.set_xlabel("segment number"); a_.set_ylabel("P(PD)"); a_.set_title(f"speaker {sid} ({lab}): per-segment score"); a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figdir / "4_segment_scores_and_vote.png", dpi=110); plt.close(fig); made.append("4_segment_scores_and_vote.png")
    return made


# ----------------------------------------------------------------------------- main
def main(a, backend=None):
    repo = Path(a.repo).resolve(); os.chdir(repo); sys.path.insert(0, str(repo))
    import yaml
    from audio.audio_dataset import OfflineAcousticDataset
    backend = backend or TorchBackend()
    exp, k = a.expname, a.fold
    results_root = (Path(a.results_dir).expanduser().resolve()
                    if getattr(a, "results_dir", None) else repo / "results")
    res = results_root / exp
    p_ds = res / f"ds-{exp}/outputs/saved_model/Ds_NN_fold{k}.ckpt"
    p_up = res / f"ups-{exp}/outputs/saved_model/UPs_NN_fold{k}.ckpt"
    p_out = res / f"ds-{exp}/outputs/output_{k}.npy"
    p_feat = res / f"ds-{exp}/feat_config.yaml"
    p_csv = repo / a.data_dir / "folds" / f"test_fold{k}_fbank_offline.csv"
    for p in (p_ds, p_out, p_feat, p_csv):
        if not p.exists():
            sys.exit(f"missing: {p}")
    out = new_output_dir(a, repo)
    print("New output directory:", out, flush=True)
    provenance = out / "provenance"; provenance.mkdir()
    source_files = [p_feat, p_csv, Path(__file__).resolve(),
                    Path(__file__).resolve().with_name("raw_audio_trace.py")]
    for rel in ["audio/audio_utils.py", "audio/audio_dataset.py", "config/audio_config.yaml",
                "upstream_auxiliary/ups_model.py", "upstream_auxiliary/pretrain_ups_expert.py",
                "downstream/ds_model.py", "downstream/train_downs_expert.py", "downstream/ds_runner.py"]:
        path = repo / rel
        if path.is_file():
            target = provenance / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    for path in source_files:
        (provenance / path.name).write_bytes(path.read_bytes())

    ds_ck = backend.load_ckpt(p_ds); ups_ck = backend.load_ckpt(p_up) if p_up.exists() else None
    feat_cfg = yaml.safe_load(p_feat.read_text(encoding="utf-8"))
    # MINI_TRACE_RESOLVE_FEATURE_PATHS_V3
    trace_csv = provenance / "resolved_test_fold.csv"
    table = pd.read_csv(p_csv)
    feature_base = (repo / a.data_dir).resolve().parent.parent
    table["file_path"] = [str((feature_base / str(fp)).resolve())
                          for fp in table["file_path"]]
    missing = [fp for fp in table["file_path"] if not Path(fp).is_file()]
    if missing:
        raise FileNotFoundError("Missing feature files: " + repr(missing[:5]))
    table.to_csv(trace_csv, index=False)
    dataset = OfflineAcousticDataset(None, ds_ck["DsConfig"], str(trace_csv), feat_cfg)
    freq, seq = dataset.getDimension()
    bs = ds_ck["DsConfig"]["dataloader"]["batch_size"]
    ucfg = ds_ck["UpsConfig"]; drop_last = ucfg[ucfg["SelectedAENetwork"]].get("batchnorm", False)
    N = len(dataset); n_kept = (N // bs) * bs if drop_last else N
    utt, start, spk = segment_table(dataset)
    order = kept_order(spk, n_kept)
    labels = dataset.table.drop_duplicates("ID").set_index("ID").label.to_dict()
    chosen = choose_speakers(order, labels, a.n_spk, a.spk_ids)
    if not chosen or len(set(chosen)) != len(chosen) or any(s not in order for s in chosen):
        raise ValueError(f"Choose distinct test speakers with kept segments; available IDs: {order}")
    selected_rows = dataset.table[dataset.table.ID.isin(chosen)]
    wanted_keys = {feature_key(fp)[0] for fp in selected_rows.file_path.tolist()}
    print(f"Indexing ZIPs for {len(chosen)} speakers / {len(wanted_keys)} recordings...", flush=True)
    sources = index_audio_zips(a.raw_zip_dir, wanted_keys)
    stored = np.load(p_out, allow_pickle=True).item()["test"]
    st_scores, st_target = np.asarray(stored[1], float), np.asarray(stored[2], float)
    if st_scores.shape != (len(order), 2) or st_target.shape != (len(order),):
        raise ValueError("Stored speaker outputs do not align with inferred dataset order; refusing an unreliable comparison")
    if not np.array_equal(st_target, [labels[s] for s in order]):
        raise ValueError("Stored labels disagree with inferred speaker order; refusing an unreliable comparison")
    models = backend.make_models(ds_ck, ups_ck, freq, seq)

    info = dict(expname=exp, fold=k, repo=str(repo), results_dir=str(results_root),
                data_dir=str((repo / a.data_dir).resolve()), test_csv=str(p_csv), n_segments_total=N, batch_size=bs, drop_last=bool(drop_last),
                n_segments_used=int(n_kept), n_speakers_with_scores=len(order), n_speakers_in_stored_output=int(len(st_scores)),
                chosen_speakers=chosen, input_shape=[int(freq), int(seq)], selected_encoded_layers=ds_ck["DsConfig"]["SelectedEncodedLayers"],
                ckpt_downstream=f"{p_ds.name} sha256:{sha(p_ds, 64)}", ckpt_upstream=(f"{p_up.name} sha256:{sha(p_up, 64)}" if ups_ck else "missing"),
                stored_output=f"{p_out.name} sha256:{sha(p_out, 64)}", torch=getattr(getattr(backend, 'torch', None), '__version__', None),
                python=platform.python_version(), made=time.strftime("%Y-%m-%d %H:%M"),
                feature_source="regenerated_from_selected_raw_WAVs", raw_zip_dir=str(a.raw_zip_dir),
                vad=a.vad, segmentlen_seconds=a.segmentlen, complete_speaker_trace=(a.max_segments == 0),
                feature_config=feat_cfg, argv=sys.argv,
                preprocessing_note="Energy VAD from submitted project script, NOT Praat/paper VAD.")
    pd.DataFrame(STAGES, columns=["stage", "what_happens", "authors_code"]).to_csv(out / "stage_map.csv", index=False)
    rows_summary, speaker_md = [], []
    cache = {}
    feature_paths = {}
    all_audio_checks = []

    def feat(u):
        if u not in cache:
            cache.clear()
            original_path = str(Path(dataset.X[u]).resolve())
            cache[u] = dataset._load_feat(str(feature_paths[original_path]))
        return cache[u]

    for sid in chosen:
        lab = "PD" if labels[sid] == 1 else "HC"
        sd = out / f"spk_{sid}_{lab}"; sd.mkdir(exist_ok=True)
        idx = np.where(spk == sid)[0]
        if a.max_segments and len(idx) > a.max_segments:
            idx = idx[:a.max_segments]
        rows = dataset.table[dataset.table.ID == sid]
        rows.to_csv(sd / "01_fold_csv_rows.csv", index=False)
        print(f"Speaker {sid} ({lab}): regenerating {len(rows)} feature pieces from raw audio...", flush=True)
        cache.clear()
        feature_paths, audio_checks = regenerate_audio(rows, sd / "00_audio_pipeline", sources, p_feat, a)
        all_audio_checks.extend(dict(speaker_ID=sid, **r) for r in audio_checks)
        feature_match = all(r["features_allclose"] for r in audio_checks)
        if not feature_match:
            print("WARNING: regenerated features differ from saved features; see audio_feature_checks.csv", flush=True)
        feats_dir = sd / "02_feature_files_examples"; feats_dir.mkdir(exist_ok=True)
        for fp in rows.file_path.tolist()[:2]:
            try: np.save(feats_dir / Path(fp).name, np.load(fp, allow_pickle=True))
            except Exception: pass
        raw_list, mm_list = [], []
        for i in idx:
            r = dataset._sample(feat(int(utt[i])), int(start[i]))
            r = np.asarray(r).copy(); raw_list.append(r)
            mm_list.append(np.asarray(dataset.standardize(dataset._sample(feat(int(utt[i])), int(start[i])))))
        raw, mm = np.stack(raw_list), np.stack(mm_list)
        # Original dataset[i] loads saved features; the trace uses regenerated ones.
        # This diagnostic must NOT be forced to True when raw inputs/settings differ.
        same = bool(np.allclose(np.asarray(dataset[int(idx[0])][0]), mm[0], rtol=1e-5, atol=1e-4))
        K = a.save_segments
        np.savez_compressed(sd / "03_segments_raw_first.npz", segments=raw[:K], global_index=idx[:K])
        np.savez_compressed(sd / "04_segments_minmax_first.npz", segments=mm[:K], global_index=idx[:K])
        small, conv0, probs, code, recon_mse, code_ups = {}, None, [], [], [], []
        for b0 in range(0, len(idx), 64):
            r = models.trace(mm[b0:b0 + 64])
            if b0 == 0:
                conv0 = {kk: v[:1] for kk, v in r["layers"].items() if v[0].size > 4096}
            for kk, v in r["layers"].items():
                if v[0].size <= 4096:
                    small.setdefault(kk, []).append(v)
            small.setdefault("output_logits", []).append(r["logits"])
            probs.append(r["probs"]); code.append(r.get("code"))
            if "recon_mse" in r: recon_mse.append(r["recon_mse"])
            if "code_ups" in r: code_ups.append(r["code_ups"])
            if b0 == 0:
                first = r
        probs = np.concatenate(probs); kept = idx < n_kept
        p_pd = probs[:, 1]
        small = {kk: np.concatenate(v) for kk, v in small.items()}
        np.savez_compressed(sd / "05_activations_all_segments_small_layers.npz", **small)
        np.savez_compressed(sd / "05_activations_segment0_conv_layers.npz", **(conv0 or {}))
        if "recon" in first:
            np.savez_compressed(sd / "06_upstream_autoencoder_segment0.npz", input=mm[0], reconstruction=first["recon"][0])
        seg_df = pd.DataFrame(dict(global_index=idx, utterance_row=utt[idx], feature_file=[dataset.X[u] for u in utt[idx]],
                                   regenerated_feature_file=[str(feature_paths[str(Path(dataset.X[u]).resolve())].relative_to(out)) for u in utt[idx]],
                                   start_frame=start[idx], end_frame=start[idx] + seq, used_in_vote=kept,
                                   raw_min=[float(x.min()) for x in raw], raw_max=[float(x.max()) for x in raw],
                                   P_HC=probs[:, 0], P_PD=p_pd, predicted=probs.argmax(1)))
        if recon_mse: seg_df["ae_recon_mse"] = np.concatenate(recon_mse)
        seg_df.to_csv(sd / "07_segment_scores.csv", index=False)
        mean_sc = probs[kept].mean(0) if kept.any() else np.full(2, np.nan)
        pos = order.index(sid) if sid in order else None
        stored_sc = st_scores[pos] if pos is not None and pos < len(st_scores) else None
        diff = float(np.abs(stored_sc - mean_sc).max()) if (stored_sc is not None and not a.max_segments) else None
        dec = dict(speaker_ID=sid, true_label=lab, n_feature_files=int(len(rows)), n_segments=int(len(idx)), n_segments_dropped_by_drop_last=int((~kept).sum()),
                   mean_score_HC_PD=[float(x) for x in mean_sc], decision=("n/a (all segments dropped by drop_last)" if not kept.any() else ("PD" if mean_sc[1] > mean_sc[0] else "HC")),
                   stored_score_in_output_npy=None if stored_sc is None else [float(x) for x in stored_sc],
                   max_abs_diff_vs_stored=diff, stored_label_matches=None if pos is None or pos >= len(st_target) else bool(int(st_target[pos]) == int(labels[sid])),
                   feature_source="regenerated_from_raw_audio", all_regenerated_features_match_saved=feature_match,
                   n_raw_audio_sources=len({r["source_member"] for r in audio_checks}),
                   complete_speaker_trace=(a.max_segments == 0),
                   dataset_getitem_equals_trace=same, manual_forward_matches_repo_forward=first.get("manual_matches_repo_forward"),
                   manual_forward_maxdiff=first.get("manual_vs_repo_maxdiff"), ae_error=first.get("ae_error"))
        (sd / "08_speaker_decision.json").write_text(json.dumps(dec, indent=1), encoding="utf-8")
        figs = make_plots(sd / "figures", sid, lab, raw[0], mm[0], conv0 or {}, np.concatenate(code) if code and code[0] is not None else None, p_pd, kept, float(mean_sc[1]))
        rows_summary.append(dec)
        ok = "-" if diff is None else ("MATCH" if diff < 1e-4 else f"DIFF {diff:.2e}")
        speaker_md.append(f"### Speaker {sid} ({lab})\n- feature files: {len(rows)}; segments: {len(idx)} (dropped by drop_last: {int((~kept).sum())})\n"
                          f"- mean score (HC, PD): {mean_sc[0]:.4f}, {mean_sc[1]:.4f} -> decision **{dec['decision']}** (true: {lab})\n"
                          f"- stored score in output_{k}.npy: {None if stored_sc is None else np.round(stored_sc, 4).tolist()} -> check: **{ok}**\n"
                          f"- repo forward == step-by-step forward: {dec['manual_forward_matches_repo_forward']}; dataset[i] == traced segment: {same}\n"
                          f"- regenerated fbank matches saved fbank: **{feature_match}**; model inputs are regenerated from raw WAVs.\n"
                          f"- files: `spk_{sid}_{lab}/` (00_audio_pipeline: originals + mono/resample + VAD + PCM16 + normalization + Mel comparisons; 01 csv rows, 03/04 segments, 05 activations, 07 scores, 08 decision, figures/)\n"
                          + "".join(f"\n![{f}](spk_{sid}_{lab}/figures/{f})" for f in figs) + "\n")
    pd.DataFrame(rows_summary).to_csv(out / "00_selection_and_checks.csv", index=False)
    pd.DataFrame(all_audio_checks).to_csv(out / "00_audio_feature_checks.csv", index=False)
    info["n_raw_audio_sources"] = len(sources)
    info["all_regenerated_features_match_saved"] = all(r["features_allclose"] for r in all_audio_checks)
    info["saved_score_mapping_note"] = "Stored output has no IDs: order inferred from first appearance in the authors' test dataset; labels cross-checked."
    import importlib.metadata
    info["package_versions"] = {}
    for name in ["torch", "torchaudio", "numpy", "scipy", "soundfile", "pandas"]:
        try: info["package_versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    (out / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    md = ["# Mini-trace report\n",
          f"Experiment `{exp}`, fold {k}. Speakers followed: {chosen}. All numbers below were produced by this run from the saved checkpoints; nothing was typed by hand.\n",
          "## Pipeline stages (submitted preprocessing + authors' feature/model code)\n", "| stage | what happens | code |\n|---|---|---|",
          *[f"| {s} | {w} | `{c}` |" for s, w, c in STAGES],
          "\n## How the checks work\n- All traced model inputs are regenerated from the raw WAVs, not substituted from saved .npy files. Features are compared using rtol=1e-5, atol=1e-4; shape disagreement stops the trace rather than reusing incompatible segmentation indices. A value mismatch is recorded honestly and inference still uses the regenerated inputs.\n"
          "- The recomputed mean softmax is compared against output_<fold>.npy only for a complete speaker trace. Difference <1e-4 is numerical consistency evidence for these selected speakers, NOT proof of the entire training/evaluation protocol. Stored speaker positions are inferred from CSV order, not an explicit ID in output.npy.\n"
          "- `drop_last`: the authors' test loader drops the last incomplete batch (batch norm is on), so the last segments of the CSV order are not used; they are marked as dropped.\n"
          "- Segments come from the authors' `_getindex/_sample/standardize`; models are the authors' `FeatureExtractionPretrained` and `DownStreamTrain` loaded from `Ds_NN_fold<k>.ckpt` (encoder fine-tuned in the downstream stage). The upstream auto-encoder view uses `UPs_NN_fold<k>.ckpt`.\n",
          "## Speakers\n", *speaker_md,
          "## Scope and limitations\n- VAD reproduces the supplied project script's ENERGY VAD; it is not the paper's Praat VAD. VAD/8-second settings default to that script and are recorded; historical command-line settings are not independently established.\n- Small-layer activations are saved for all traced segments; large feature maps and AE reconstruction are saved for segment 0. The full normalized input arrays are saved for the first --save_segments windows.\n- The decoder is the upstream-checkpoint reconstruction, NOT part of the classification path. The classifier uses the downstream-checkpoint encoder.\n- This is inference only: no training is run, no fold split is changed, and no source dataset/results files are overwritten. Checkpoint hashes and source/config copies are included, but checkpoint weights are not bundled.\n- Raw speech is personal data; keep this archive private unless you have permission to share it.\n",
          "## Run info\n```\n" + json.dumps(info, indent=1) + "\n```"]
    (out / "TRACE_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    write_manifest(out)
    zp = Path(str(out) + ".zip")
    with zipfile.ZipFile(zp, "x", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file(): z.write(p, str(p.relative_to(out.parent)))
    print("speakers:", chosen, "| checks:", [(r["speaker_ID"], r["max_abs_diff_vs_stored"]) for r in rows_summary])
    print("zip:", zp)
    if a.download:
        try:
            from google.colab import files
            files.download(str(zp))
        except ImportError:
            print("Automatic download is only available inside Colab. ZIP is saved at the path above.")
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--results_dir", default=None, help="results root, absolute or relative to the repo (default: <repo>/results)")
    ap.add_argument("--expname", default="baseline_s0")
    ap.add_argument("--fold", type=int, default=6)
    ap.add_argument("--data_dir", default="preprocess/pcgita_full_100spk")
    ap.add_argument("--n_spk", type=int, default=4, help="half healthy, half PD (first ones in test order)")
    ap.add_argument("--spk_ids", nargs="*", type=int, default=None, help="choose speakers by ID instead")
    ap.add_argument("--save_segments", type=int, default=8, help="how many segments are saved in full (arrays are big); all get scores")
    ap.add_argument("--max_segments", type=int, default=0, help="0 = all segments of each speaker (needed for the stored-score check)")
    ap.add_argument("--out", default="/content/drive/MyDrive/idiap_pd/mini_trace_exports", help="parent folder; a new unique subfolder is always created")
    ap.add_argument("--raw_zip_dir", default="/content/drive/MyDrive/Session 1405.07.11/DataSet")
    ap.add_argument("--vad", choices=["energy", "none"], default="energy")
    ap.add_argument("--segmentlen", type=float, default=8.)
    ap.add_argument("--audio_plots", type=int, default=2, help="plot first N raw sources per speaker; all sources have arrays/audio")
    ap.add_argument("--download", action="store_true", help="download ZIP inside a Colab notebook kernel")
    args = ap.parse_args(argv)
    if not np.isfinite(args.segmentlen) or args.n_spk < 1 or args.save_segments < 1 or args.max_segments < 0 or args.audio_plots < 0 or args.segmentlen < .033:
        ap.error("Counts must be valid and segmentlen must be >=0.033 seconds")
    return args


if __name__ == "__main__":
    main(parse_args())
