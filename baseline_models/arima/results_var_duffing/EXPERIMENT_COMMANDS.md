# VAR (Vector Autoregression) Baseline Experiment - Command Reference

This document records all commands used to train and evaluate the VAR model on the Duffing oscillator dataset.

## Experiment Overview

**Model**: VAR (Vector Autoregression)  
**Dataset**: Duffing Oscillator with Noise (`duffing_with_noise.csv`)  
**State Variables**: `x`, `xdot` (2D system)  
**Method**: VAR with automatic lag order selection  
**Lag Order**: Auto-selected (typically 10 for this dataset)

---

## Step 1: Training the VAR Model

### Basic Training Command

```bash
python arima.py \
    --csv_path duffing_with_noise.csv \
    --traj_id_column traj_id \
    --state_columns x xdot \
    --method var \
    --auto_order \
    --save_dir ./results_var_duffing
```

### Full Training Command with All Options

```bash
python arima.py \
    --csv_path duffing_with_noise.csv \
    --traj_id_column traj_id \
    --state_columns x xdot \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --method var \
    --auto_order \
    --save_dir ./results_var_duffing
```

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--csv_path` | `duffing_with_noise.csv` | Path to dataset CSV file |
| `--traj_id_column` | `traj_id` | Column name for trajectory IDs |
| `--state_columns` | `x xdot` | State variable column names |
| `--train_ratio` | `0.7` | Training data ratio (70%) |
| `--val_ratio` | `0.15` | Validation data ratio (15%) |
| `--test_ratio` | `0.15` | Test data ratio (15%) |
| `--method` | `var` | Use VAR (Vector Autoregression) |
| `--auto_order` | (flag) | Auto-select lag order |
| `--save_dir` | `./results_var_duffing` | Directory to save results |

### Training Output Files

After training, the following files are saved in `--save_dir`:

- `var_model.pkl` - Saved VAR model (pickle file)
- `test_data.npz` - Test data for evaluation
- `test_predictions.csv` - Test predictions with true/predicted values
- `train_predictions.csv` - Training predictions
- `arima_metrics.txt` - Basic metrics from training
- `arima_results.png` - Visualization plots

---

## Step 2: Evaluating the VAR Model

### Basic Evaluation Command

```bash
python eval_var.py \
    --model_path ./results_var_duffing/var_model.pkl \
    --test_data_path ./results_var_duffing/test_data.npz \
    --save_dir ./results_var_duffing
```

### Full Evaluation Command with All Options

```bash
python eval_var.py \
    --model_path ./results_var_duffing/var_model.pkl \
    --test_data_path ./results_var_duffing/test_data.npz \
    --save_dir ./results_var_duffing \
    --horizons 1 5 10 20 50 100 \
    --long_horizon_steps 200
```

### Evaluation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--model_path` | `./results_var_duffing/var_model.pkl` | Path to saved VAR model |
| `--test_data_path` | `./results_var_duffing/test_data.npz` | Path to test data |
| `--save_dir` | `./results_var_duffing` | Directory to save evaluation results |
| `--horizons` | `1 5 10 20 50 100` | Prediction horizons for multi-step NRMSE |
| `--long_horizon_steps` | `200` | Number of steps for long-horizon evaluation |

### Evaluation Output Files

After evaluation, the following files are saved in `--save_dir`:

- `evaluation_results.txt` - Text file with all metrics
- `eval_1step_mse.png` - 1-step MSE per dimension (bar chart)
- `eval_multistep_nrmse.png` - Multi-step NRMSE vs horizon (line plot)
- `eval_phase_portrait.png` - Phase portrait comparison (true vs predicted)
- `eval_long_horizon.png` - Long-horizon error growth (log error vs time)

**Note**: Reconstruction error is NOT computed as VAR is a prediction model, not an autoencoder.

---

## Complete Experiment Workflow

### One-Line Commands (Copy-Paste Ready)

```bash
# 1. Train the VAR model
python arima.py --csv_path duffing_with_noise.csv --traj_id_column traj_id --state_columns x xdot --method var --auto_order --save_dir ./results_var_duffing

# 2. Evaluate the VAR model
python eval_var.py --model_path ./results_var_duffing/var_model.pkl --test_data_path ./results_var_duffing/test_data.npz --save_dir ./results_var_duffing
```

### With Virtual Environment

If using a virtual environment:

```bash
# Activate virtual environment
source venv/bin/activate

# 1. Train the VAR model
python arima.py \
    --csv_path duffing_with_noise.csv \
    --traj_id_column traj_id \
    --state_columns x xdot \
    --method var \
    --auto_order \
    --save_dir ./results_var_duffing

# 2. Evaluate the VAR model
python eval_var.py \
    --model_path ./results_var_duffing/var_model.pkl \
    --test_data_path ./results_var_duffing/test_data.npz \
    --save_dir ./results_var_duffing
```

---

## Expected Results

### Training Metrics (from our experiment)

- **VAR Lag Order**: 10 (auto-selected)
- **Validation RMSE**: 1.223222
- **Validation MAE**: 1.010724
- **Test RMSE**: 1.275839
- **Test MAE**: 1.032965

### Evaluation Metrics (from our experiment)

- **1-Step MSE**: 0.450775
- **Multi-Step NRMSE**:
  - Horizon 1: 0.178019
  - Horizon 5: 0.164062
  - Horizon 10: 0.196903
  - Horizon 20: 0.225152
  - Horizon 50: 0.389369
  - Horizon 100: 1.993326
- **Chamfer Distance (Phase Portrait)**: 0.000108
- **Spectral Radius of VAR Companion Matrix**: ~1.0 (varies)
- **Long-Horizon Divergence Rate**: 0.015131

---

## Evaluation Functions Used

The evaluation script uses the following functions from `evaluation.py`:

✅ **Used (Suitable for VAR)**:
- `one_step_mse()` - 1-step prediction error
- `multi_step_nrmse()` - Multi-step normalized RMSE
- `chamfer_distance_phase()` - Phase portrait fidelity
- `long_horizon_divergence_rate()` - Long-term error growth
- `spectral_radius()` - Spectral properties of VAR companion matrix

❌ **Not Used (Not Suitable for VAR)**:
- `reconstruction_error()` - VAR is a prediction model, not an autoencoder that reconstructs input states

---

## VAR Model Details

### What is VAR?

Vector Autoregression (VAR) is a multivariate time series model that:
- Models each variable as a linear function of past values of all variables
- Captures cross-variable dependencies
- Uses a fixed lag order (number of past time steps)

### VAR Model Structure

For a 2D system with lag order `p`:
```
x(t) = A₁ @ x(t-1) + A₂ @ x(t-2) + ... + Aₚ @ x(t-p) + ε(t)
```

Where:
- `x(t)` is the state vector at time `t`
- `A₁, A₂, ..., Aₚ` are coefficient matrices
- `ε(t)` is the error term

### Companion Matrix

The VAR model can be represented as a companion matrix for spectral analysis:
- Size: `(n_x * lag_order) × (n_x * lag_order)`
- Eigenvalues indicate stability properties
- Spectral radius near 1.0 indicates stable dynamics

---

## Comparison with Other Baselines

### VAR vs Koopman AE Baseline

| Feature | VAR | Koopman AE Baseline |
|---------|-----|---------------------|
| **Approach** | Linear autoregression | Neural network + linear operator |
| **Observables** | Raw state | Learned latent space (20D) |
| **Interpretability** | High (linear coefficients) | Lower (learned features) |
| **1-Step MSE** | 0.450775 | 0.005882 |
| **Spectral Radius** | ~1.0 | 1.006336 |

### VAR vs eDMD Baseline

| Feature | VAR | eDMD Baseline |
|---------|-----|---------------|
| **Observables** | Raw state | Dictionary functions (10D) |
| **Approach** | Linear autoregression | Dictionary + linear operator |
| **1-Step MSE** | 0.450775 | 0.000823 |
| **Spectral Radius** | ~1.0 | 1.006770 |

### VAR vs TAE

| Feature | VAR | TAE |
|---------|-----|-----|
| **Approach** | Linear autoregression | Time-lagged autoencoder |
| **1-Step MSE** | 0.450775 | ~0.066 |

---

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're in the project root directory
   ```bash
   cd "/Users/pujiacheng/Desktop/ESE 5380"
   ```

2. **Model Not Found**: Ensure VAR model was saved during training
   - Check that `--method var` was used
   - Verify `var_model.pkl` exists in save directory

3. **File Not Found**: Check that CSV file exists and paths are correct

4. **Spectral Properties Warning**: If spectral properties can't be computed, it's not critical - other metrics will still be computed

---

## File Structure

```
results_var_duffing/
├── var_model.pkl              # Saved VAR model
├── test_data.npz               # Test data for evaluation
├── test_predictions.csv        # Test predictions
├── train_predictions.csv       # Training predictions
├── arima_metrics.txt           # Basic metrics from training
├── arima_results.png           # Visualization from training
├── evaluation_results.txt      # Detailed evaluation metrics
├── eval_1step_mse.png         # 1-step MSE plot
├── eval_multistep_nrmse.png   # Multi-step NRMSE plot
├── eval_phase_portrait.png     # Phase portrait comparison
├── eval_long_horizon.png       # Long-horizon error growth
└── EXPERIMENT_COMMANDS.md      # This file
```

---

## Key Insights

1. **Linear Model**: VAR is a purely linear model, which may struggle with highly nonlinear dynamics like the Duffing oscillator.

2. **Lag Order**: The auto-selected lag order (10) captures temporal dependencies but may not be optimal for all systems.

3. **Spectral Properties**: The VAR companion matrix provides insights into stability, similar to Koopman operators.

4. **Performance**: VAR achieves reasonable short-term prediction but may struggle with long-term forecasting due to error accumulation.

5. **Baseline Comparison**: VAR serves as a classical time series baseline, useful for comparing against more sophisticated methods.

---

## Notes

- The model uses **temporal splitting** (no shuffling) to prevent data leakage
- Training data: 70%, Validation: 15%, Test: 15%
- All plots are automatically saved (no interactive display)
- VAR is a classical statistical method, providing interpretable linear relationships
- The model is saved as a pickle file for easy loading and evaluation

---

## Date

Experiment conducted on: December 6, 2024

