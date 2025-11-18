# Training with CSV Data

This guide explains how to use `train_test_with_given_data.py` to train the Koopman Autoencoder with your own CSV data.

## CSV Format

The script supports two CSV formats:

### Format 1: Single Trajectory
If you have a single time series, your CSV should have columns for each state variable:

```csv
x,xdot
0.5,0.0
0.498,0.01
0.494,0.02
...
```

### Format 2: Multiple Trajectories
If you have multiple trajectories, include a trajectory ID column:

```csv
traj_id,x,xdot
0,0.5,0.0
0,0.498,0.01
0,0.494,0.02
1,0.3,0.0
1,0.298,0.01
...
```

### Optional Columns
- **Time column**: If your CSV has a time column, specify it with `--time_column`. It will be ignored.
- **State columns**: If not specified, the script will auto-detect state columns (all columns except time and traj_id).

## Usage

### Basic Usage (Auto-detect columns)

```bash
python train_test_with_given_data.py --csv_path data.csv
```

### Specify State Columns

```bash
python train_test_with_given_data.py \
    --csv_path data.csv \
    --state_columns x xdot
```

### Multiple Trajectories

```bash
python train_test_with_given_data.py \
    --csv_path data.csv \
    --traj_id_column traj_id \
    --state_columns x xdot
```

### Custom Split Ratios

```bash
python train_test_with_given_data.py \
    --csv_path data.csv \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

### Full Example with All Options

```bash
python train_test_with_given_data.py \
    --csv_path my_data.csv \
    --state_columns x xdot \
    --traj_id_column traj_id \
    --time_column time \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --n_z 8 \
    --p 30 \
    --n_epochs 50 \
    --batch_size 128 \
    --save_dir ./results/
```

## Arguments

- `--csv_path`: (Required) Path to CSV file
- `--state_columns`: Column names for state variables (space-separated)
- `--traj_id_column`: Column name for trajectory ID (if multiple trajectories)
- `--time_column`: Column name for time (optional, will be ignored)
- `--train_ratio`: Ratio for training set (default: 0.7)
- `--val_ratio`: Ratio for validation set (default: 0.15)
- `--test_ratio`: Ratio for test set (default: 0.15)
- `--n_z`: Latent dimension (default: 6)
- `--p`: Observables dimension (default: 20)
- `--n_epochs`: Number of training epochs (default: 40)
- `--batch_size`: Batch size (default: 256)
- `--save_dir`: Directory to save results (default: ./)

## Output Files

After training, the following files will be saved in `--save_dir`:

1. **best_model.pth**: Model checkpoint with best validation loss
2. **final_model.pth**: Final model after all epochs
3. **training_results.png**: Visualization of training curves and predictions
4. **test_metrics.txt**: Quantitative metrics on test set

## Example: Creating Test Data

You can create a test CSV from the Duffing oscillator simulation:

```python
import numpy as np
import pandas as pd
from data_simulation import generate_duffing_dataset

# Generate data
t, trajs = generate_duffing_dataset(n_traj=10, T=10.0, dt=0.01)

# Convert to DataFrame
data_list = []
for traj_id, traj in enumerate(trajs):
    for i, state in enumerate(traj):
        data_list.append({
            'traj_id': traj_id,
            'time': i * 0.01,
            'x': state[0],
            'xdot': state[1]
        })

df = pd.DataFrame(data_list)
df.to_csv('duffing_data.csv', index=False)
```

Then train with:
```bash
python train_test_with_given_data.py \
    --csv_path duffing_data.csv \
    --traj_id_column traj_id \
    --time_column time \
    --state_columns x xdot
```

