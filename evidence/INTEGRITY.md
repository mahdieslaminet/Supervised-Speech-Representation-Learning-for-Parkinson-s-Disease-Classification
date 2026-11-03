# Integrity report

## 1. Recomputation checks (per fold)
- folds checked: 12
- recomputed == value stored in output_k.npy: 12/12
- recomputed == number printed in the training log: 12/12 (logs found for 12)
- label order of the .npy equals label order of the test CSV: 12/12
- MISMATCHES: none

## 2. Run history (honest list of interruptions)
attempts>1 means the stage was interrupted and resumed/restarted; rotated logs are named *.old-* in results/<exp>/logs/.
  method  seed  fold  attempts_ups  attempts_ds
baseline     0     4             1            2

rotated (restarted) logs: 1
- fusion_s0/logs/fold1_ups.log.old-20261015-123448

## 3. Input checks
- speakers.csv: 100 speakers (HC=50, PD=50)
- every speaker is in exactly one test fold: True
- folds: 10; speakers per fold (train/val/test) of fold 1: 80/10/10

## 4. Settings actually used (settings.json of each experiment)
- adv_s0: {
 "method": "adv",
 "epochs": 100,
 "batch": 128,
 "seed": 0,
 "data_dir": "preprocess/pcgita_full_100spk",
 "n_folds": 10
}
- baseline_s0: {
 "method": "baseline",
 "epochs": 100,
 "batch": 128,
 "seed": 0,
 "data_dir": "preprocess/pcgita_full_100spk",
 "n_folds": 10
}
- fusion_s0: {
 "method": "fusion",
 "epochs": 100,
 "batch": 128,
 "seed": 0,
 "data_dir": "preprocess/pcgita_full_100spk",
 "n_folds": 10
}

## 5. Environment (this report)
{"python": "3.13.16", "numpy": "2.1.3", "pandas": "2.2.3", "sklearn": "1.6.1", "torch": "2.11.0+cu130", "cuda": "13.0", "gpu": "Tesla T4"}

report made: 2026-10-24 18:52

## 6. What this does NOT prove
- it cannot prove that the logs were not edited. The strongest protection is to give the supervisor VIEW access to the original Drive folder (Drive keeps version history) and let them re-run one fold.
- a re-run is not bit-identical (GPU non-determinism); small differences are expected.