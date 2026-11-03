# Results summary (all numbers re-computed from raw per-speaker scores)

## adv, seed 0  (1 folds, 10 speakers tested)
- mean over folds: accuracy 70.0 +- nan (std over folds), sensitivity 100.0, specificity 40.0, AUC 0.880 +- nan
- pooled over all tested speakers: 7/10 correct = 70.0% (95% Wilson interval 39.7 to 89.2)
- confusion matrix (rows=true, cols=predicted): HC [2, 3], PD [0, 5]
- paper (Table 1, mean over 5 seeds): accuracy 72.0 +- 5.62, AUC 0.84
- NOTE: only 1 of 10 folds finished

## baseline, seed 0  (10 folds, 100 speakers tested)
- mean over folds: accuracy 69.0 +- 11.0 (std over folds), sensitivity 70.0, specificity 68.0, AUC 0.772 +- 0.116
- pooled over all tested speakers: 69/100 correct = 69.0% (95% Wilson interval 59.4 to 77.2)
- confusion matrix (rows=true, cols=predicted): HC [34, 16], PD [15, 35]
- paper (Table 1, mean over 5 seeds): accuracy 66.2 +- 1.17, AUC 0.77

## fusion, seed 0  (1 folds, 10 speakers tested)
- mean over folds: accuracy 80.0 +- nan (std over folds), sensitivity 100.0, specificity 60.0, AUC 0.880 +- nan
- pooled over all tested speakers: 8/10 correct = 80.0% (95% Wilson interval 49.0 to 94.3)
- confusion matrix (rows=true, cols=predicted): HC [3, 2], PD [0, 5]
- paper (Table 1, mean over 5 seeds): accuracy 75.4 +- 1.02, AUC 0.8
- NOTE: only 1 of 10 folds finished

Limits: ONE seed per method, so the paper's '+-' (spread over seeds) is not reproduced; with 100 test speakers one speaker = 1 percentage point.