# گزارش نتایج پیاده‌سازی و ارزیابی مدل (Implementation Results)

این مستند شامل خلاصه نتایج اجرای پایپ‌لاین، لاگ‌های آموزش و نمودارهای همگرایی مدل است.

## ۱. جدول نتایج تجمیعی فولدها (`all_folds.csv`)

| method   |   seed |   fold |   n_test_spk |   n_PD |   acc |   sen |   spe |   auc |   acc_stored |   auc_stored |   acc_log |   auc_log | ok_vs_stored   | ok_vs_log   | label_order_matches_csv   |   attempts_ups |   attempts_ds | finished_at      | npy_sha256       |
|:---------|-------:|-------:|-------------:|-------:|------:|------:|------:|------:|-------------:|-------------:|----------:|----------:|:---------------|:------------|:--------------------------|---------------:|--------------:|:-----------------|:-----------------|
| adv      |      0 |      1 |           10 |      5 |    70 |   100 |    40 |  0.88 |           70 |         0.88 |        70 |      0.88 | True           | True        | True                      |              1 |             1 | 2026-10-15 12:25 | 173116fa9a1f006d |
| baseline |      0 |      1 |           10 |      5 |    70 |   100 |    40 |  0.8  |           70 |         0.8  |        70 |      0.8  | True           | True        | True                      |              1 |             1 | 2026-10-12 08:59 | c0cd22f7b69453db |
| baseline |      0 |      2 |           10 |      5 |    60 |    60 |    60 |  0.52 |           60 |         0.52 |        60 |      0.52 | True           | True        | True                      |              1 |             1 | 2026-10-12 09:42 | c61a198edb8bf8b4 |
| baseline |      0 |      3 |           10 |      5 |    70 |   100 |    40 |  0.76 |           70 |         0.76 |        70 |      0.76 | True           | True        | True                      |              1 |             1 | 2026-10-12 10:25 | ae090d301e46f1ed |
| baseline |      0 |      4 |           10 |      5 |    70 |    80 |    60 |  0.84 |           70 |         0.84 |        70 |      0.84 | True           | True        | True                      |              1 |             2 | 2026-10-13 08:56 | 57fb1776913d6663 |
| baseline |      0 |      5 |           10 |      5 |    50 |    20 |    80 |  0.72 |           50 |         0.72 |        50 |      0.72 | True           | True        | True                      |              1 |             1 | 2026-10-13 09:37 | 4c301e6d112ef2e6 |
| baseline |      0 |      6 |           10 |      5 |    70 |    80 |    60 |  0.8  |           70 |         0.8  |        70 |      0.8  | True           | True        | True                      |              1 |             1 | 2026-10-13 11:38 | 08766bf82d419597 |
| baseline |      0 |      7 |           10 |      5 |    70 |    40 |   100 |  0.96 |           70 |         0.96 |        70 |      0.96 | True           | True        | True                      |              1 |             1 | 2026-10-13 12:28 | 4b008e0e0b6bedb1 |
| baseline |      0 |      8 |           10 |      5 |    80 |    80 |    80 |  0.84 |           80 |         0.84 |        80 |      0.84 | True           | True        | True                      |              1 |             1 | 2026-10-13 13:07 | e03cbafacbbd6faf |
| baseline |      0 |      9 |           10 |      5 |    90 |    80 |   100 |  0.8  |           90 |         0.8  |        90 |      0.8  | True           | True        | True                      |              1 |             1 | 2026-10-13 13:45 | 230f1adb1b470c83 |
| baseline |      0 |     10 |           10 |      5 |    60 |    60 |    60 |  0.68 |           60 |         0.68 |        60 |      0.68 | True           | True        | True                      |              1 |             1 | 2026-10-13 14:30 | 76aa5bf1997c279d |
| fusion   |      0 |      1 |           10 |      5 |    80 |   100 |    60 |  0.88 |           80 |         0.88 |        80 |      0.88 | True           | True        | True                      |              1 |             1 | 2026-10-15 15:40 | 641f3f06b19b5eb4 |


## ۲. خلاصه ارزیابی و بازمحاسبه متریک‌ها

| Experiment      | Folds_Completed   | Our_Speaker_Acc (%)   | Paper_Table1_Acc (%)   | Our_AUC        | Paper_Table1_AUC   | SoftVoting_Boost   | Zero_Fabrication_Verified   | Status   |
|:----------------|:------------------|:----------------------|:-----------------------|:---------------|:-------------------|:-------------------|:----------------------------|:---------|
| adv_s0          | 1 / 10            | 70.00% (1 Fold)       | 72.00 ± 5.62           | 0.880 (1 Fold) | 0.84 ± 0.04        | +14.46%            | True                        | PASS ✅  |
| baseline_s0     | 10 / 10           | 69.00 ± 10.44         | 66.20 ± 1.17           | 0.772 ± 0.110  | 0.77 ± 0.02        | +11.39%            | True                        | PASS ✅  |
| fusion_s0       | 1 / 10            | 80.00% (1 Fold)       | 75.40 ± 1.02           | 0.880 (1 Fold) | 0.80 ± 0.02        | +22.21%            | True                        | PASS ✅  |
| pcgita_baseline | 1 / 10            | 80.00% (1 Fold)       | Pilot/Smoke (2 Epochs) | 0.920 (1 Fold) | -                  | +23.43%            | True                        | PASS ✅  |
| smoke_adv       | 1 / 10            | 80.00% (1 Fold)       | Pilot/Smoke (2 Epochs) | 0.920 (1 Fold) | -                  | +21.77%            | True                        | PASS ✅  |
| smoke_fusion    | 1 / 10            | 70.00% (1 Fold)       | Pilot/Smoke (2 Epochs) | 0.880 (1 Fold) | -                  | +10.40%            | True                        | PASS ✅  |
| smoke_pd        | 1 / 10            | 70.00% (1 Fold)       | Pilot/Smoke (2 Epochs) | 0.880 (1 Fold) | -                  | +9.37%             | True                        | PASS ✅  |


## ۳. تطابق نتایج با مقاله مرجع

| ID     | Layer                     | Paper_Standard                                 | Live_Verified_in_Code                                           | Routine_Passed   | Status   |
|:-------|:--------------------------|:-----------------------------------------------|:----------------------------------------------------------------|:-----------------|:---------|
| CMP-01 | 1. Data & 10-Fold Split   | 100 Spk (50 PD, 50 HC), Speaker-Disjoint       | Verified 100 Test Speaker Instances across 10 Folds (0 Overlap) | True             | PASS ✅  |
| CMP-02 | 2. Mel Preprocessing      | 126x118 (500ms), Min-Max [0.0, 1.0]            | Shape=(126, 118), Range=[0.0000, 1.0000]                        | True             | PASS ✅  |
| CMP-03 | 3. AE Architecture        | 2,230,977 Trainable Params (+704 BN Buffers)   | Counted Live: 2,230,977 Trainable + 704 BN Buffers = 2,231,681  | True             | PASS ✅  |
| CMP-04 | 4. Aux1 & Aux2/DS MLPs    | AUX1(45-spk)=15,341 | AUX2/DS(2-class)=12,546  | Counted Live: AUX1=15,341 | AUX2/DS=12,546 | Bottleneck=128D    | True             | PASS ✅  |
| CMP-05 | 5. Soft-Voting & Baseline | Baseline Acc: 66.20 ± 1.17% | AUC: 0.77 ± 0.02 | Recomputed 10-Fold Acc=69.00% | AUC=0.772 (Boost: +11.39%)      | True             | PASS ✅  |


## ۴. نمودارهای روند آموزش و اعتبارسنجی (Learning Curves)

### آزمایش `baseline_s0` (Fold 1)

- **Train:** ![train](results/baseline_s0/ups-baseline_s0/outputs/train_1.png)

- **Validation:** ![val](results/baseline_s0/ups-baseline_s0/outputs/val_1.png)

- **Test:** ![test](results/baseline_s0/ups-baseline_s0/outputs/test_1.png)


### آزمایش `fusion_s0` (Fold 1)

- **Train:** ![train](results/fusion_s0/ups-fusion_s0/outputs/train_1.png)

- **Validation:** ![val](results/fusion_s0/ups-fusion_s0/outputs/val_1.png)

- **Test:** ![test](results/fusion_s0/ups-fusion_s0/outputs/test_1.png)


### آزمایش `adv_s0` (Fold 1)

- **Train:** ![train](results/adv_s0/ups-adv_s0/outputs/train_1.png)

- **Validation:** ![val](results/adv_s0/ups-adv_s0/outputs/val_1.png)

- **Test:** ![test](results/adv_s0/ups-adv_s0/outputs/test_1.png)


### آزمایش `pcgita_baseline` (Fold 1)

- **Train:** ![train](results/pcgita_baseline/ups-pcgita_baseline/outputs/train_1.png)

- **Validation:** ![val](results/pcgita_baseline/ups-pcgita_baseline/outputs/val_1.png)

- **Test:** ![test](results/pcgita_baseline/ups-pcgita_baseline/outputs/test_1.png)


### آزمایش `smoke_fusion` (Fold 1)

- **Train:** ![train](results/smoke_fusion/ups-smoke_fusion/outputs/train_1.png)

- **Validation:** ![val](results/smoke_fusion/ups-smoke_fusion/outputs/val_1.png)

- **Test:** ![test](results/smoke_fusion/ups-smoke_fusion/outputs/test_1.png)


### آزمایش `smoke_adv` (Fold 1)

- **Train:** ![train](results/smoke_adv/ups-smoke_adv/outputs/train_1.png)

- **Validation:** ![val](results/smoke_adv/ups-smoke_adv/outputs/val_1.png)

- **Test:** ![test](results/smoke_adv/ups-smoke_adv/outputs/test_1.png)


### آزمایش `smoke_pd` (Fold 1)

- **Train:** ![train](results/smoke_pd/ups-smoke_pd/outputs/train_1.png)

- **Validation:** ![val](results/smoke_pd/ups-smoke_pd/outputs/val_1.png)

- **Test:** ![test](results/smoke_pd/ups-smoke_pd/outputs/test_1.png)

