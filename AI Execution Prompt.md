# AI Assistant Execution Prompt

If you are an AI model or an assistant tasked with running or auditing this Parkinson project repository, use the context and steps below.

## Repository Layout Context
- `repo_code/`: Core source files for upstream (autoencoder/features) and downstream (classification) models.
- `preprocess/dummy_database/`: Contains 6 public sample audio files (`spk1.wav` to `spk6.wav`) and configuration manifests.
- `my_experiments/scripts/`: Automation scripts including `run_fold.py` and `run_all.py`.
- `config/`: YAML configuration files for audio, upstream, and downstream models.

## Step-by-Step Execution Instructions for AI
1. **Verify Environment:** Ensure PyTorch, torchaudio, and standard ML libraries are installed.
2. **Execute Smoke Test:** Run a single fold test to verify code health without throwing errors:
   ```bash
   python3 my_experiments/scripts/run_fold.py --fold 1 --epochs 2 --batch 10 --method baseline --data_dir preprocess/dummy_database --expname ai_audit_test
   ```
3. **Inspect Output:** Confirm that output tensors (`.npy`) and status logs are correctly generated under the `results/` directory.