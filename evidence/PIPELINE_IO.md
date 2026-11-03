# Pipeline inputs and outputs (values read from the actual files)

## Stage 0 -> 1: raw PC-GITA recordings -> 16 kHz pieces (<= 8 s) -> fbank features
- speakers: 100; pieces (wav): 3638; total speech after VAD: 105.8 min
- feature files: 3638; example shape (mel bins x frames): (126, 1993) (126 bins; 32 ms window, 4 ms shift)
- CSV columns: uttr_num, ID, file_path, length, label, spk_ID
```
 uttr_num  ID                                                                         file_path  length  label spk_ID
        0   0 preprocess/pcgita_full_100spk/features_data/fbank/AC0001_readtext_readtext_p0.npy    1993      0 AC0001
        1   0 preprocess/pcgita_full_100spk/features_data/fbank/AC0001_readtext_readtext_p1.npy    1386      0 AC0001
        2   0   preprocess/pcgita_full_100spk/features_data/fbank/AC0001_sentences_laura_p0.npy     551      0 AC0001
```

## Stage 1 -> 2/3: fold CSVs (the input list of every training script)
- folds/ fold 1: train: 2912 rows, 80 speakers; val: 364 rows, 10 speakers; test: 362 rows, 10 speakers
- folds_spkID_task/ fold 1: train: 993 rows, 45 speakers; val: 319 rows, 45 speakers; test: 323 rows, 45 speakers
- 500 ms segment = 1 + (500-32)/4 = 118 frames -> model input 126 x 118 (min-max scaled per segment); rows shorter than 500 ms are dropped by the loader

## Stage 2: upstream (auto-encoder [+ auxiliary tasks])
- input: train/val/test fold CSVs (+ folds_spkID_task for the speaker-ID adversary)
- output: results/<exp>/ups-<exp>/outputs/saved_model/UPs_NN_fold<k>.ckpt, logs/fold<k>_ups.log (losses per epoch)

## Stage 3: downstream (MLP classifier on the 128-dim code)
- input: UPs_NN_fold<k>.ckpt of the SAME fold + the fold CSVs
- output: Ds_NN_fold<k>.ckpt, output_<k>.npy = {test, val, train} each [chunk_acc, speaker_scores(n x 2), target, speaker_acc, sen, spe, AUC], logs/fold<k>_ds.log
- speaker decision = average of the segment softmax scores of that speaker (soft voting)

## Stage 4: this report
- input: all output_<k>.npy + logs; output: all_folds.csv, per_speaker_scores.csv, SUMMARY.md, INTEGRITY.md, MANIFEST.sha256