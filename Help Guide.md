# Run & Troubleshooting Guide

This document provides quick instructions on how to execute and verify the Parkinson project pipeline.

## 1. Environment Setup
Install the required dependencies in your terminal or Colab cell:
```bash
pip install -r req.yml
```

## 2. Quick Smoke Test (Using 6 Public Samples)
To test the pipeline on the sample dataset (`dummy_database`) with 2 epochs:
```bash
python3 my_experiments/scripts/run_fold.py --fold 1 --epochs 2 --batch 10 --method baseline --data_dir preprocess/dummy_database --expname smoke_test
```

## 3. Full Run (For Local/Private Full Dataset)
If you are running across all folds with the full private dataset:
```bash
python3 my_experiments/scripts/run_all.py --methods baseline --seeds 0 --folds 1-10 --data_dir preprocess/pcgita_full_100spk
```