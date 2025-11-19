# Data Simulation Guide

The `data_simulation.py` script can simulate four types of dynamical systems and export data to CSV format.

## Supported Systems

1. **Duffing Oscillator** (bistable potential)
   - Equation: `x'' = -delta*x' - alpha*x - beta*x^3 + gamma*cos(omega*t)`
   - Default: `x'' = x - x^3` (conservative, bistable)
   - State: `[x, xdot]`
   - Derivatives: `[xdot, xddot]`

2. **Van der Pol Oscillator**
   - Equation: `x'' = mu*(1 - x^2)*x' - x`
   - State: `[x, xdot]`
   - Derivatives: `[xdot, xddot]`

3. **Lorenz Attractor**
   - Equations:
     - `dx/dt = sigma*(y - x)`
     - `dy/dt = x*(rho - z) - y`
     - `dz/dt = x*y - beta*z`
   - State: `[x, y, z]`
   - Derivatives: `[xdot, ydot, zdot]`

4. **Double Pendulum**
   - State: `[theta1, theta1_dot, theta2, theta2_dot]`
   - Derivatives: `[theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]`

## CSV Output Format

The CSV file contains the following columns:
- `traj_id`: Trajectory number (0 to n_traj-1)
- `time`: Absolute time
- `time_step`: Time step index
- State variables (e.g., `x`, `xdot`, `y`, `z`, `theta1`, etc.)
- Derivatives (e.g., `xdot`, `xddot`, `ydot`, `zdot`, etc.)

## Usage Examples

### Basic Usage

```bash
# Duffing oscillator (default parameters)
python data_simulation.py --system duffing --n_traj 100 --output duffing_data.csv

# Van der Pol oscillator
python data_simulation.py --system vanderpol --n_traj 100 --output vanderpol_data.csv

# Lorenz attractor
python data_simulation.py --system lorenz --n_traj 50 --output lorenz_data.csv

# Double pendulum
python data_simulation.py --system double_pendulum --n_traj 100 --output pendulum_data.csv
```

### With Custom Parameters

```bash
# Duffing with damping and forcing
python data_simulation.py --system duffing \
    --n_traj 100 \
    --T 20.0 \
    --dt 0.01 \
    --alpha 1.0 \
    --beta 1.0 \
    --delta 0.1 \
    --gamma 0.3 \
    --omega 1.2 \
    --output duffing_damped.csv

# Van der Pol with custom mu
python data_simulation.py --system vanderpol \
    --n_traj 100 \
    --mu 2.0 \
    --output vanderpol_mu2.csv

# Lorenz with custom parameters
python data_simulation.py --system lorenz \
    --n_traj 50 \
    --sigma 10.0 \
    --rho 28.0 \
    --beta_lorenz 2.67 \
    --output lorenz_custom.csv

# Double pendulum with custom masses and lengths
python data_simulation.py --system double_pendulum \
    --n_traj 100 \
    --L1 1.0 \
    --L2 0.8 \
    --m1 1.0 \
    --m2 0.5 \
    --g 9.81 \
    --output pendulum_custom.csv
```

### With Noise

```bash
# Add noise to simulation
python data_simulation.py --system duffing \
    --n_traj 100 \
    --noise_std 0.01 \
    --output duffing_noisy.csv
```

## Command Line Arguments

### Common Arguments
- `--system`: System type (`duffing`, `vanderpol`, `lorenz`, `double_pendulum`)
- `--n_traj`: Number of trajectories (default: 100)
- `--T`: Simulation time per trajectory (default: 10.0)
- `--dt`: Time step (default: 0.01)
- `--noise_std`: Noise standard deviation (default: 0.0)
- `--output`: Output CSV file path (default: `simulated_data.csv`)

### Duffing Oscillator Parameters
- `--alpha`: Alpha parameter (default: 1.0)
- `--beta`: Beta parameter (default: 1.0)
- `--delta`: Damping parameter (default: 0.0)
- `--gamma`: Forcing amplitude (default: 0.0)
- `--omega`: Forcing frequency (default: 0.0)

### Van der Pol Parameters
- `--mu`: Mu parameter (default: 1.0)

### Lorenz Parameters
- `--sigma`: Sigma parameter (default: 10.0)
- `--rho`: Rho parameter (default: 28.0)
- `--beta_lorenz`: Beta parameter (default: 8/3)

### Double Pendulum Parameters
- `--L1`: Length of first pendulum (default: 1.0)
- `--L2`: Length of second pendulum (default: 1.0)
- `--m1`: Mass of first pendulum (default: 1.0)
- `--m2`: Mass of second pendulum (default: 1.0)
- `--g`: Gravitational constant (default: 9.81)

## Python API

You can also use the functions programmatically:

```python
from data_simulation import generate_dataset, export_to_csv

# Generate Duffing oscillator data
t, trajs, derivatives = generate_dataset(
    system_type='duffing',
    n_traj=100,
    T=10.0,
    dt=0.01,
    noise_std=0.0,
    alpha=1.0,
    beta=1.0
)

# Export to CSV
df = export_to_csv(t, trajs, derivatives, 'duffing', 'output.csv')

# Or use backward compatibility functions
from data_simulation import generate_duffing_dataset
t, trajs = generate_duffing_dataset(n_traj=100, T=10.0, dt=0.01)
```

## Using Generated Data for Training

After generating CSV files, you can use them with the training script:

```bash
# Generate data
python data_simulation.py --system duffing --n_traj 100 --output duffing_data.csv

# Train model
python train_test_with_given_data.py \
    --csv_path duffing_data.csv \
    --traj_id_column traj_id \
    --time_column time \
    --state_columns x xdot
```

## Notes

- The simulation uses 4th-order Runge-Kutta integration
- Initial conditions are randomly sampled from reasonable ranges
- All trajectories have the same length (determined by T and dt)
- The CSV includes both state variables and their time derivatives
- Backward compatibility is maintained for existing code using `generate_duffing_dataset()`

