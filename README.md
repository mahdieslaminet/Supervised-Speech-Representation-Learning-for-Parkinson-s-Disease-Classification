# Parkinson's Disease Speech Classification: Reproduction of Supervised Representation Learning (PC-GITA)

Colab-based reproduction and validation of the paper **"Supervised Speech Representation Learning for Parkinson's Disease Classification"** (Janbakhshi & Kodrasi, Idiap). The code builds on the authors' official repo, [idiap/pddetection-reps-learning](https://github.com/idiap/pddetection-reps-learning), and runs it on the full PC-GITA dataset.

## What this project does

A CNN auto-encoder learns a compact representation of speech Mel-spectrograms. Four training modes are compared:

| Method | Description |
|---|---|
| `baseline` | Unsupervised auto-encoder (no auxiliary task) |
| `adv` | Adversarial speaker-ID branch (speaker-invariant representation) |
| `pd` | Joint PD classifier (PD-discriminative representation) |
| `fusion` | Both auxiliary tasks together |

The bottleneck representation is then used to classify PD vs. neurotypical speakers, with soft voting over each speaker's segments.

## Setup (as in the paper)

- Data: PC-GITA, 100 speakers (50 PD, 50 neurotypical), Spanish
- Input: 500 ms Mel-spectrogram segments, 50% overlap, 126 Mel bands
- Evaluation: stratified speaker-independent 10-fold cross-validation
- Metrics: accuracy and AUC (PD task), accuracy and AUC (speaker-ID task)
- Hyper-parameters: lambda = alpha = 0.01
- Seeds: seed 0 only (the paper averages 5 seeds)

## Repository structure

```
repo_code/
  my_experiments/
    scripts/            # preprocessing, run_fold.py, run_all.py, check_folds.py,
                        # colab_bootstrap.py, apply_patches.py, make_report.py
  upstream_auxiliary/   # code from the original Idiap repo (patched)
notebooks/
  Session_*.ipynb       # running the experiments on Colab
  Validation_Notebook.ipynb   # validation checks (folds, soft voting, ROC, reconstruction)
preprocess/             # generated locally (not included)
results/                # generated per method/seed/fold (not included)
```

## How to run

1. Get the PC-GITA dataset from its authors (it is **not** included here, see the license).
2. Preprocess it into `preprocess/pcgita_full_100spk` (10 folds).
3. Open a notebook in Google Colab and mount Google Drive.
4. Prepare the environment and apply the patches:
   ```bash
   python my_experiments/scripts/colab_bootstrap.py
   python my_experiments/scripts/apply_patches.py /content/repo
   ```
5. Check the folds, then train:
   ```bash
   python my_experiments/scripts/check_folds.py preprocess/pcgita_full_100spk
   python my_experiments/scripts/run_all.py --methods baseline adv pd fusion --seeds 0 --folds 1-10
   ```
6. Run `Validation_Notebook.ipynb` to recompute metrics from `output_K.npy` and plot the results.

## Results

| Method | Accuracy (%) | AUC |
|---|---|---|
| Baseline | _fill in_ | _fill in_ |
| Adversarial | _fill in_ | _fill in_ |
| PD discriminative | _fill in_ | _fill in_ |
| Fusion | _fill in_ | _fill in_ |

Paper's reported values (mean of 5 seeds): baseline 66.2% / 0.77, adversarial 72.0% / 0.84, PD 71.0% / 0.78, fusion 75.4% / 0.80.

## Notes

- Results may differ slightly from the paper because only one seed was run.
- The dataset is not redistributed. Please request access from the PC-GITA authors.

## Credits and citation

- Paper: P. Janbakhshi and I. Kodrasi, *Supervised Speech Representation Learning for Parkinson's Disease Classification*, Idiap Research Institute (arXiv:2106.00531).
- Original code: https://github.com/idiap/pddetection-reps-learning
- Dataset: J. R. Orozco-Arroyave et al., PC-GITA, LREC 2014.

## License

Add your license here. Respect the licenses of the original repo and of PC-GITA.
