# Koopman AE Baseline Experiment - Command Reference

This document records all commands used to train and evaluate the Koopman AE Baseline model on the Duffing oscillator dataset.

## Experiment Overview

**Model**: Simplified Koopman Autoencoder Baseline  
**Dataset**: Duffing Oscillator with Noise (`duffing_with_noise.csv`)  
**State Variables**: `x`, `xdot` (2D system)  
**Latent Dimension**: 20 (10× state dimension)  
**Loss Functions**: 
- Reconstruction Loss: `||x - Decoder(Encoder(x))||²` (weight: 1.0)
- Koopman Linearity Loss: `||z(t+1) - A_f @ z(t)||²` (weight: 10.0)

---

## Step 1: Training the Model

### Basic Training Command

```bash
python train_kae.py \
    --csv_path duffing_with_noise.csv \
    --traj_id_column traj_id \
    --state_columns x xdot \
    --n_epochs 40 \
    --lam_rec 1.0 \
    --lam_lin 10.0 \
    --save_dir ./results_kae_duffing
```

### Full Training Command with All Options

```bash
python train_kae.py \
    --csv_path duffing_with_noise.csv \
    --traj_id_column traj_id \
    --state_columns x xdot \
    --n_z 20 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --n_epochs 40 \
    --batch_size 256 \
    --lam_rec 1.0 \
    --lam_lin 10.0 \
    --save_dir ./results_kae_duffing
```

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--csv_path` | `duffing_with_noise.csv` | Path to dataset CSV file |
| `--traj_id_column` | `traj_id` | Column name for trajectory IDs |
| `--state_columns` | `x xdot` | State variable column names |
| `--n_z` | `20` (auto) | Latent dimension (default: 10× state_dim) |
| `--train_ratio` | `0.7` | Training data ratio (70%) |
| `--val_ratio` | `0.15` | Validation data ratio (15%) |
| `--test_ratio` | `0.15` | Test data ratio (15%) |
| `--n_epochs` | `40` | Number of training epochs |
| `--batch_size` | `256` | Batch size for training |
| `--lam_rec` | `1.0` | Weight for reconstruction loss |
| `--lam_lin` | `10.0` | Weight for linearity loss |
| `--save_dir` | `./results_kae_duffing` | Directory to save results |

### Training Output Files

After training, the following files are saved in `--save_dir`:

- `best_model.pth` - Best model based on validation loss
- `final_model.pth` - Final model after all epochs
- `test_data.npz` - Test data for evaluation (saved automatically)
- `training_curves.png` - Training/validation loss curves

---

## Step 2: Evaluating the Model

### Basic Evaluation Command

```bash
python eval_kae.py \
    --model_path ./results_kae_duffing/best_model.pth \
    --test_data_path ./results_kae_duffing/test_data.npz \
    --save_dir ./results_kae_duffing
```

### Full Evaluation Command with All Options

```bash
python eval_kae.py \
    --model_path ./results_kae_duffing/best_model.pth \
    --test_data_path ./results_kae_duffing/test_data.npz \
    --save_dir ./results_kae_duffing \
    --horizons 1 5 10 20 50 100 \
    --long_horizon_steps 200
```

### Evaluation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--model_path` | `./results_kae_duffing/best_model.pth` | Path to trained model |
| `--test_data_path` | `./results_kae_duffing/test_data.npz` | Path to test data (from training) |
| `--save_dir` | `./results_kae_duffing` | Directory to save evaluation results |
| `--horizons` | `1 5 10 20 50 100` | Prediction horizons for multi-step NRMSE |
| `--long_horizon_steps` | `200` | Number of steps for long-horizon evaluation |

### Evaluation Output Files

After evaluation, the following files are saved in `--save_dir`:

- `evaluation_results.txt` - Text file with all metrics
- `eval_1step_mse.png` - 1-step MSE per dimension (bar chart)
- `eval_multistep_nrmse.png` - Multi-step NRMSE vs horizon (line plot)
- `eval_phase_portrait.png` - Phase portrait comparison (true vs predicted)
- `eval_reconstruction.png` - Reconstruction quality (time series plot)
- `eval_long_horizon.png` - Long-horizon error growth (log error vs time)

---

## Complete Experiment Workflow

### One-Line Commands (Copy-Paste Ready)

```bash
# 1. Train the model
python train_kae.py --csv_path duffing_with_noise.csv --traj_id_column traj_id --state_columns x xdot --n_epochs 40 --lam_rec 1.0 --lam_lin 10.0 --save_dir ./results_kae_duffing

# 2. Evaluate the model
python eval_kae.py --model_path ./results_kae_duffing/best_model.pth --test_data_path ./results_kae_duffing/test_data.npz --save_dir ./results_kae_duffing
```

### With Virtual Environment

If using a virtual environment:

```bash
# Activate virtual environment
source venv/bin/activate

# 1. Train the model
python train_kae.py \
    --csv_path duffing_with_noise.csv \
    --traj_id_column traj_id \
    --state_columns x xdot \
    --n_epochs 40 \
    --lam_rec 1.0 \
    --lam_lin 10.0 \
    --save_dir ./results_kae_duffing

# 2. Evaluate the model
python eval_kae.py \
    --model_path ./results_kae_duffing/best_model.pth \
    --test_data_path ./results_kae_duffing/test_data.npz \
    --save_dir ./results_kae_duffing
```

---

## Expected Results

### Training Metrics (from our experiment)

- **Final Train Loss**: 0.006867
  - Reconstruction: 0.006597
  - Linearity: 0.000027
- **Final Val Loss**: 0.005152
  - Reconstruction: 0.004874
  - Linearity: 0.000028

### Evaluation Metrics (from our experiment)

- **1-Step MSE**: 0.005882
- **Multi-Step NRMSE**:
  - Horizon 1: 0.063604
  - Horizon 5: 0.042619
  - Horizon 10: 0.058082
  - Horizon 20: 0.028169
  - Horizon 50: 0.987767
  - Horizon 100: 2.612319
- **Chamfer Distance (Phase Portrait)**: 0.000108
- **Spectral Radius of A_f**: 1.006336
- **Reconstruction Error**: 0.005202
- **Long-Horizon Divergence Rate**: 0.021798

---

## Running on Other Datasets

### Van der Pol Oscillator

```bash
# Training
python train_kae.py \
    --csv_path vanderpol_with_noise.csv \
    --traj_id_column traj_id \
    --state_columns x xdot \
    --save_dir ./results_kae_vanderpol

# Evaluation
python eval_kae.py \
    --model_path ./results_kae_vanderpol/best_model.pth \
    --test_data_path ./results_kae_vanderpol/test_data.npz \
    --save_dir ./results_kae_vanderpol
```

### Lorenz Attractor

```bash
# Training
python train_kae.py \
    --csv_path lorenz_data.csv \
    --traj_id_column traj_id \
    --state_columns x y z \
    --save_dir ./results_kae_lorenz

# Evaluation
python eval_kae.py \
    --model_path ./results_kae_lorenz/best_model.pth \
    --test_data_path ./results_kae_lorenz/test_data.npz \
    --save_dir ./results_kae_lorenz
```

### Double Pendulum

```bash
# Training
python train_kae.py \
    --csv_path double_pendulum_data.csv \
    --traj_id_column traj_id \
    --state_columns theta1 theta2 omega1 omega2 \
    --save_dir ./results_kae_double_pendulum

# Evaluation
python eval_kae.py \
    --model_path ./results_kae_double_pendulum/best_model.pth \
    --test_data_path ./results_kae_double_pendulum/test_data.npz \
    --save_dir ./results_kae_double_pendulum
```

---

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're in the project root directory
   ```bash
   cd "/Users/pujiacheng/Desktop/ESE 5380"
   ```

2. **CUDA Out of Memory**: Reduce batch size
   ```bash
   --batch_size 128  # or smaller
   ```

3. **File Not Found**: Check that CSV file exists and paths are correct

4. **Model Dimension Mismatch**: Ensure `--n_z` matches the model you're loading, or let it auto-compute

---

## File Structure

```
results_kae_duffing/
├── best_model.pth              # Best model checkpoint
├── final_model.pth             # Final model checkpoint
├── test_data.npz               # Test data for evaluation
├── training_curves.png          # Training/validation loss plots
├── evaluation_results.txt       # Evaluation metrics (text)
├── eval_1step_mse.png          # 1-step MSE plot
├── eval_multistep_nrmse.png    # Multi-step NRMSE plot
├── eval_phase_portrait.png      # Phase portrait comparison
├── eval_reconstruction.png      # Reconstruction quality plot
├── eval_long_horizon.png        # Long-horizon error growth
└── EXPERIMENT_COMMANDS.md       # This file
```

---

## Notes

- The model uses **temporal splitting** (no shuffling) to prevent data leakage
- Training data: 70%, Validation: 15%, Test: 15%
- All plots are automatically saved (no interactive display)
- The model learns a linear operator `A_f` in latent space: `z(t+1) = A_f @ z(t)`
- Spectral radius near 1.0 indicates stable dynamics

---

## Date

Experiment conducted on: December 6, 2024

