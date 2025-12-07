# eDMD Baseline Experiment - Command Reference

This document records all commands used to train and evaluate the eDMD (extended Dynamic Mode Decomposition) Baseline model on the Duffing oscillator dataset.

## Experiment Overview

**Model**: eDMD Baseline with Dictionary Functions  
**Dataset**: Duffing Oscillator with Noise (`duffing_with_noise.csv`)  
**State Variables**: `x`, `xdot` (2D system)  
**Observables Dimension**: 10 (dictionary functions)  
**Dictionary Functions**: 
```
φ(x) = [x, xdot, x², x*xdot, xdot², x³, sin(x), cos(x), sin(xdot), cos(xdot)]
```

**Loss Function**: 
- Observables Prediction Loss: `||φ(x(t+1)) - K @ φ(x(t))||²`

Where `K` is the learned linear Koopman operator.

---

## Step 1: Training the Model

### Basic Training Command

```bash
python train_edmd.py \
    --csv_path duffing_with_noise.csv \
    --traj_id_column traj_id \
    --state_columns x xdot \
    --n_epochs 40 \
    --save_dir ./results_edmd_duffing
```

### Full Training Command with All Options

```bash
python train_edmd.py \
    --csv_path duffing_with_noise.csv \
    --traj_id_column traj_id \
    --state_columns x xdot \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --n_epochs 40 \
    --batch_size 256 \
    --save_dir ./results_edmd_duffing
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
| `--n_epochs` | `40` | Number of training epochs |
| `--batch_size` | `256` | Batch size for training |
| `--save_dir` | `./results_edmd_duffing` | Directory to save results |

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
python eval_edmd.py \
    --model_path ./results_edmd_duffing/best_model.pth \
    --test_data_path ./results_edmd_duffing/test_data.npz \
    --save_dir ./results_edmd_duffing
```

### Full Evaluation Command with All Options

```bash
python eval_edmd.py \
    --model_path ./results_edmd_duffing/best_model.pth \
    --test_data_path ./results_edmd_duffing/test_data.npz \
    --save_dir ./results_edmd_duffing \
    --horizons 1 5 10 20 50 100 \
    --long_horizon_steps 200
```

### Evaluation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--model_path` | `./results_edmd_duffing/best_model.pth` | Path to trained model |
| `--test_data_path` | `./results_edmd_duffing/test_data.npz` | Path to test data (from training) |
| `--save_dir` | `./results_edmd_duffing` | Directory to save evaluation results |
| `--horizons` | `1 5 10 20 50 100` | Prediction horizons for multi-step NRMSE |
| `--long_horizon_steps` | `200` | Number of steps for long-horizon evaluation |

### Evaluation Output Files

After evaluation, the following files are saved in `--save_dir`:

- `evaluation_results.txt` - Text file with all metrics
- `eval_1step_mse.png` - 1-step MSE per dimension (bar chart)
- `eval_multistep_nrmse.png` - Multi-step NRMSE vs horizon (line plot)
- `eval_phase_portrait.png` - Phase portrait comparison (true vs predicted)
- `eval_observables_mse.png` - Observables prediction MSE per dimension
- `eval_long_horizon.png` - Long-horizon error growth (log error vs time)

---

## Complete Experiment Workflow

### One-Line Commands (Copy-Paste Ready)

```bash
# 1. Train the model
python train_edmd.py --csv_path duffing_with_noise.csv --traj_id_column traj_id --state_columns x xdot --n_epochs 40 --save_dir ./results_edmd_duffing

# 2. Evaluate the model
python eval_edmd.py --model_path ./results_edmd_duffing/best_model.pth --test_data_path ./results_edmd_duffing/test_data.npz --save_dir ./results_edmd_duffing
```

### With Virtual Environment

If using a virtual environment:

```bash
# Activate virtual environment
source venv/bin/activate

# 1. Train the model
python train_edmd.py \
    --csv_path duffing_with_noise.csv \
    --traj_id_column traj_id \
    --state_columns x xdot \
    --n_epochs 40 \
    --save_dir ./results_edmd_duffing

# 2. Evaluate the model
python eval_edmd.py \
    --model_path ./results_edmd_duffing/best_model.pth \
    --test_data_path ./results_edmd_duffing/test_data.npz \
    --save_dir ./results_edmd_duffing
```

---

## Expected Results

### Training Metrics (from our experiment)

- **Final Train Loss**: 0.001882
- **Final Val Loss**: 0.002206
- **Observables Prediction Loss**: Both train and val losses represent the observables prediction error

### Evaluation Metrics (from our experiment)

- **1-Step MSE**: 0.000823
- **Multi-Step NRMSE**:
  - Horizon 1: 0.021387
  - Horizon 5: 0.025760
  - Horizon 10: 0.100102
  - Horizon 20: 0.187823
  - Horizon 50: 0.179617
  - Horizon 100: 0.588696
- **Chamfer Distance (Phase Portrait)**: 0.000108
- **Spectral Radius of K**: 1.006770
- **Observables Prediction MSE**: 0.002512
- **Observables Prediction RMSE**: 0.050119
- **Long-Horizon Divergence Rate**: 0.012079

---

## Dictionary Functions Details

The eDMD model uses the following 10 dictionary functions:

1. **x** - Position (linear term)
2. **xdot** - Velocity (linear term)
3. **x²** - Position squared (quadratic term)
4. **x*xdot** - Position × velocity (quadratic cross term)
5. **xdot²** - Velocity squared (quadratic term)
6. **x³** - Position cubed (cubic term)
7. **sin(x)** - Sine of position (trigonometric)
8. **cos(x)** - Cosine of position (trigonometric)
9. **sin(xdot)** - Sine of velocity (trigonometric)
10. **cos(xdot)** - Cosine of velocity (trigonometric)

These functions are hand-crafted (not learned) and provide interpretable observables for the Koopman operator.

---

## Running on Other Datasets

### Van der Pol Oscillator

```bash
# Training
python train_edmd.py \
    --csv_path vanderpol_with_noise.csv \
    --traj_id_column traj_id \
    --state_columns x xdot \
    --save_dir ./results_edmd_vanderpol

# Evaluation
python eval_edmd.py \
    --model_path ./results_edmd_vanderpol/best_model.pth \
    --test_data_path ./results_edmd_vanderpol/test_data.npz \
    --save_dir ./results_edmd_vanderpol
```

### Lorenz Attractor

**Note**: The current dictionary functions are designed for 2D systems (x, xdot). For 3D systems like Lorenz, you would need to modify the dictionary functions in `baseline_models/edmd_baseline.py`.

```bash
# Training (requires dictionary modification for 3D)
python train_edmd.py \
    --csv_path lorenz_data.csv \
    --traj_id_column traj_id \
    --state_columns x y z \
    --save_dir ./results_edmd_lorenz

# Evaluation
python eval_edmd.py \
    --model_path ./results_edmd_lorenz/best_model.pth \
    --test_data_path ./results_edmd_lorenz/test_data.npz \
    --save_dir ./results_edmd_lorenz
```

---

## Comparison with Other Baselines

### eDMD vs Koopman AE Baseline

| Feature | eDMD Baseline | Koopman AE Baseline |
|---------|---------------|---------------------|
| **Observables** | Hand-crafted dictionary (10D) | Learned neural network (20D) |
| **Interpretability** | High (explicit functions) | Lower (learned features) |
| **Flexibility** | Lower (fixed dictionary) | High (adapts to data) |
| **Training Speed** | Fast (simple linear operator) | Moderate (neural network) |
| **1-Step MSE** | 0.000823 | 0.005882 |
| **Spectral Radius** | 1.006770 | 1.006336 |

### eDMD vs TAE

| Feature | eDMD Baseline | TAE |
|---------|---------------|-----|
| **Approach** | Dictionary functions + linear operator | Time-lagged autoencoder |
| **Theoretical Basis** | Koopman operator theory | Standard autoencoder |
| **Linearity** | Explicit in observables space | Implicit |
| **1-Step MSE** | 0.000823 | ~0.066 (from TAE results) |

---

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're in the project root directory
   ```bash
   cd "/Users/pujiacheng/Desktop/ESE 5380"
   ```

2. **State Dimension Mismatch**: The dictionary functions are designed for 2D systems (x, xdot). For other dimensions, modify `DictionaryFunctions` in `baseline_models/edmd_baseline.py`.

3. **CUDA Out of Memory**: Reduce batch size
   ```bash
   --batch_size 128  # or smaller
   ```

4. **File Not Found**: Check that CSV file exists and paths are correct

5. **Model Dimension Mismatch**: Ensure the model dimensions match the data (n_x=2 for Duffing)

---

## File Structure

```
results_edmd_duffing/
├── best_model.pth              # Best model checkpoint
├── final_model.pth             # Final model checkpoint
├── test_data.npz               # Test data for evaluation
├── training_curves.png          # Training/validation loss plots
├── evaluation_results.txt       # Evaluation metrics (text)
├── eval_1step_mse.png          # 1-step MSE plot
├── eval_multistep_nrmse.png    # Multi-step NRMSE plot
├── eval_phase_portrait.png     # Phase portrait comparison
├── eval_observables_mse.png    # Observables prediction MSE
├── eval_long_horizon.png       # Long-horizon error growth
└── EXPERIMENT_COMMANDS.md       # This file
```

---

## Key Insights

1. **Dictionary Functions**: The hand-crafted dictionary provides interpretable observables, making it easier to understand what the model is learning.

2. **Linear Operator K**: The learned operator K maps observables from time t to t+1, providing a linear representation of the dynamics in the observables space.

3. **Spectral Properties**: The spectral radius near 1.0 (1.006770) indicates stable dynamics, which is consistent with the Duffing oscillator behavior.

4. **Performance**: eDMD achieves very low 1-step MSE (0.000823), indicating excellent short-term prediction capability.

5. **Observables Space**: The model learns to predict all 10 observables accurately, not just the state variables.

---

## Notes

- The model uses **temporal splitting** (no shuffling) to prevent data leakage
- Training data: 70%, Validation: 15%, Test: 15%
- All plots are automatically saved (no interactive display)
- The model learns a linear operator `K` in observables space: `φ(x(t+1)) = K @ φ(x(t))`
- Spectral radius near 1.0 indicates stable dynamics
- Dictionary functions are fixed (not learned), providing interpretability

---

## Date

Experiment conducted on: December 6, 2024

