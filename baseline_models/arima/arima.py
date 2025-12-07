"""
ARIMA Baseline Model for Nonlinear Dynamical Systems

This script implements ARIMA (AutoRegressive Integrated Moving Average) models
as a baseline for comparison with Koopman Autoencoder.

For multivariate systems, we fit separate ARIMA models for each state dimension
or use Vector Autoregression (VAR) for joint modeling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_data_from_csv(csv_path, state_columns=None, traj_id_column=None, time_column=None):
    """
    Load trajectory data from CSV file
    
    Args:
        csv_path: path to CSV file
        state_columns: list of column names for state variables
        traj_id_column: column name for trajectory ID
        time_column: column name for time (optional)
    
    Returns:
        trajs: array of shape (n_traj, n_steps, n_x)
        n_x: state dimension
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Identify columns to exclude
    exclude_cols = []
    if time_column and time_column in df.columns:
        exclude_cols.append(time_column)
    if traj_id_column and traj_id_column in df.columns:
        exclude_cols.append(traj_id_column)
    
    # Auto-detect state columns if not provided
    if state_columns is None:
        state_columns = [col for col in df.columns if col not in exclude_cols]
        print(f"Auto-detected state columns: {state_columns}")
    
    # Extract state data
    state_data = df[state_columns].values
    n_x = len(state_columns)
    
    # Handle multiple trajectories
    if traj_id_column and traj_id_column in df.columns:
        traj_ids = df[traj_id_column].unique()
        trajs = []
        for traj_id in traj_ids:
            traj_data = df[df[traj_id_column] == traj_id][state_columns].values
            if len(traj_data) >= 3:
                trajs.append(traj_data.astype(np.float32))
        
        try:
            trajs = np.stack(trajs)
        except ValueError:
            min_len = min(len(t) for t in trajs)
            trajs = np.stack([t[:min_len] for t in trajs])
        print(f"Found {len(trajs)} trajectories")
    else:
        if len(state_data) < 3:
            raise ValueError("Need at least 3 data points")
        trajs = state_data.reshape(1, -1, n_x)
        print(f"Single trajectory with {len(state_data)} time steps")
    
    print(f"Data shape: {trajs.shape} (n_traj, n_steps, n_x)")
    return trajs, n_x, state_columns


def check_stationarity(series, alpha=0.05):
    """
    Check if a time series is stationary using Augmented Dickey-Fuller test
    
    Returns:
        is_stationary: boolean
        p_value: p-value from ADF test
    """
    result = adfuller(series)
    p_value = result[1]
    is_stationary = p_value < alpha
    return is_stationary, p_value


def fit_arima_univariate(train_data, order=(1, 1, 1), max_p=5, max_d=2, max_q=5):
    """
    Fit ARIMA model to univariate time series with automatic order selection
    
    Args:
        train_data: 1D array of training data
        order: (p, d, q) order tuple, or None for auto-selection
        max_p, max_d, max_q: maximum orders for auto-selection
    
    Returns:
        fitted model
        selected order
    """
    if order is None:
        # Auto-select order using AIC
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        # Check if stationary
        is_stationary, _ = check_stationarity(train_data)
        d_range = [0] if is_stationary else [0, 1, 2]
        
        for p in range(max_p + 1):
            for d in d_range:
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(train_data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        order = best_order
        print(f"  Auto-selected ARIMA order: {order} (AIC: {best_aic:.2f})")
    
    model = ARIMA(train_data, order=order)
    fitted = model.fit()
    return fitted, order


def fit_arima_multivariate(train_data, method='separate', order=None):
    """
    Fit ARIMA models for multivariate time series
    
    Args:
        train_data: array of shape (n_steps, n_x)
        method: 'separate' (fit ARIMA for each dimension) or 'var' (Vector Autoregression)
        order: ARIMA order for separate method, or VAR lag order
    
    Returns:
        fitted models (list for separate, single VAR model for var method)
        method used
    """
    n_steps, n_x = train_data.shape
    
    if method == 'separate':
        # Fit separate ARIMA models for each dimension
        models = []
        orders = []
        for i in range(n_x):
            print(f"Fitting ARIMA for dimension {i+1}/{n_x}...")
            model, order_used = fit_arima_univariate(train_data[:, i], order=order)
            models.append(model)
            orders.append(order_used)
        return models, orders, 'separate'
    
    elif method == 'var':
        # Use Vector Autoregression (VAR) for joint modeling
        if order is None:
            # Auto-select VAR lag order
            model = VAR(train_data)
            lag_order = model.select_order(maxlags=10)
            order = lag_order.selected_orders['aic']
            print(f"Auto-selected VAR lag order: {order}")
        else:
            model = VAR(train_data)
        
        fitted = model.fit(maxlags=order)
        return fitted, order, 'var'
    
    else:
        raise ValueError(f"Unknown method: {method}")


def predict_arima_separate(models, n_steps, last_values=None):
    """
    Predict using separate ARIMA models
    
    Args:
        models: list of fitted ARIMA models
        n_steps: number of steps to predict
        last_values: last observed values (shape: (n_x,)) for starting prediction
    
    Returns:
        predictions: array of shape (n_steps, n_x)
    """
    predictions = []
    for i, model in enumerate(models):
        # Use forecast method which uses the fitted model's last values
        pred = model.forecast(steps=n_steps)
        predictions.append(pred)
    return np.column_stack(predictions)


def predict_arima_var(model, n_steps, last_values):
    """
    Predict using VAR model
    
    Args:
        model: fitted VAR model
        n_steps: number of steps to predict
        last_values: last observed values (shape: (lag_order, n_x))
    
    Returns:
        predictions: array of shape (n_steps, n_x)
    """
    predictions = model.forecast(last_values, steps=n_steps)
    return predictions


def evaluate_arima_predictions(true, preds):
    """Compute evaluation metrics"""
    mse = np.mean((true - preds)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true - preds))
    
    # Phase space error
    phase_error = np.linalg.norm(true - preds, axis=1)
    mean_phase_error = np.mean(phase_error)
    max_phase_error = np.max(phase_error)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mean_phase_error': mean_phase_error,
        'max_phase_error': max_phase_error
    }


def main():
    parser = argparse.ArgumentParser(description='ARIMA Baseline for Dynamical Systems')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--state_columns', type=str, nargs='+', default=None,
                       help='Column names for state variables')
    parser.add_argument('--traj_id_column', type=str, default=None,
                       help='Column name for trajectory ID')
    parser.add_argument('--time_column', type=str, default=None,
                       help='Column name for time (optional)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio for training (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Ratio for validation (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Ratio for testing (default: 0.15)')
    parser.add_argument('--method', type=str, default='separate',
                       choices=['separate', 'var'],
                       help='Method: separate ARIMA per dimension or VAR (default: separate)')
    parser.add_argument('--order', type=int, nargs=3, default=None,
                       help='ARIMA order (p,d,q) for separate method, or VAR lag for var method')
    parser.add_argument('--auto_order', action='store_true',
                       help='Auto-select model order (overrides --order)')
    parser.add_argument('--save_dir', type=str, default='./results_arima',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Validate split ratios
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "Train, validation, and test ratios must sum to 1.0"
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    trajs, n_x, state_columns = load_data_from_csv(
        args.csv_path,
        state_columns=args.state_columns,
        traj_id_column=args.traj_id_column,
        time_column=args.time_column
    )
    
    # Split trajectories temporally
    n_traj, n_steps, _ = trajs.shape
    train_end = int(n_steps * args.train_ratio)
    val_end = int(n_steps * (args.train_ratio + args.val_ratio))
    
    print(f"\nTemporal split per trajectory:")
    print(f"  Train: steps 0 to {train_end-1} ({train_end/n_steps*100:.1f}%)")
    print(f"  Val:   steps {train_end} to {val_end-1} ({(val_end-train_end)/n_steps*100:.1f}%)")
    print(f"  Test:  steps {val_end} to {n_steps-1} ({(n_steps-val_end)/n_steps*100:.1f}%)")
    
    # Prepare training data (concatenate all trajectories)
    train_trajs = trajs[:, :train_end, :]  # (n_traj, train_steps, n_x)
    val_trajs = trajs[:, train_end:val_end, :]
    test_trajs = trajs[:, val_end:, :]
    
    # Flatten across trajectories for training
    train_data = train_trajs.reshape(-1, n_x)  # (n_traj * train_steps, n_x)
    val_data = val_trajs.reshape(-1, n_x)
    test_data = test_trajs.reshape(-1, n_x)
    
    print(f"\nTraining data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Fit ARIMA models
    print(f"\nFitting ARIMA models (method: {args.method})...")
    order = tuple(args.order) if args.order else None
    if args.auto_order:
        order = None
    
    if args.method == 'separate':
        models, orders, method_used = fit_arima_multivariate(train_data, method='separate', order=order)
        print(f"\nFitted {len(models)} separate ARIMA models")
        print(f"Orders used: {orders}")
    else:
        model, lag_order, method_used = fit_arima_multivariate(train_data, method='var', order=order)
        print(f"\nFitted VAR model with lag order: {lag_order}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    n_val_steps = val_data.shape[0]
    
    if method_used == 'separate':
        # For separate ARIMA, models use their own last values from training
        val_preds = predict_arima_separate(models, n_steps=n_val_steps)
    else:
        # For VAR, need last lag_order values from training
        lag_order = model.k_ar
        last_values = train_data[-lag_order:].copy()
        val_preds = predict_arima_var(model, n_steps=n_val_steps, last_values=last_values)
    
    val_metrics = evaluate_arima_predictions(val_data, val_preds)
    print(f"Validation Metrics:")
    print(f"  RMSE: {val_metrics['rmse']:.6f}")
    print(f"  MAE: {val_metrics['mae']:.6f}")
    print(f"  Mean Phase Error: {val_metrics['mean_phase_error']:.6f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    n_test_steps = test_data.shape[0]
    
    if method_used == 'separate':
        # For separate ARIMA, models use their own last values from training
        # Note: This is a limitation - ARIMA models predict from their training end
        # For proper evaluation, we'd need to refit with validation data included
        test_preds = predict_arima_separate(models, n_steps=n_test_steps)
    else:
        # Use last values from validation set (or training if validation is empty)
        if len(val_data) >= lag_order:
            last_values = val_data[-lag_order:].copy()
        else:
            last_values = train_data[-lag_order:].copy()
        test_preds = predict_arima_var(model, n_steps=n_test_steps, last_values=last_values)
    
    test_metrics = evaluate_arima_predictions(test_data, test_preds)
    print(f"\nTest Metrics:")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  MAE: {test_metrics['mae']:.6f}")
    print(f"  Mean Phase Error: {test_metrics['mean_phase_error']:.6f}")
    print(f"  Max Phase Error: {test_metrics['max_phase_error']:.6f}")
    
    # Save test predictions to CSV
    print("\nSaving test predictions to CSV...")
    test_df = pd.DataFrame()
    for i, col_name in enumerate(state_columns):
        test_df[f'{col_name}_true'] = test_data[:, i]
        test_df[f'{col_name}_pred'] = test_preds[:, i]
        test_df[f'{col_name}_error'] = test_data[:, i] - test_preds[:, i]
    test_df['phase_error'] = np.linalg.norm(test_data - test_preds, axis=1)
    test_csv_path = os.path.join(args.save_dir, 'test_predictions.csv')
    test_df.to_csv(test_csv_path, index=False)
    print(f"Test predictions saved to: {test_csv_path}")
    
    # Evaluate on training set (for comparison)
    print("\nEvaluating on training set...")
    n_train_steps = train_data.shape[0]
    
    if method_used == 'separate':
        train_preds = predict_arima_separate(models, n_steps=n_train_steps)
    else:
        # For VAR, use last values from training (circular, but for comparison)
        last_values = train_data[-lag_order:].copy()
        train_preds = predict_arima_var(model, n_steps=n_train_steps, last_values=last_values)
    
    train_metrics = evaluate_arima_predictions(train_data, train_preds)
    print(f"Training Metrics:")
    print(f"  RMSE: {train_metrics['rmse']:.6f}")
    print(f"  MAE: {train_metrics['mae']:.6f}")
    print(f"  Mean Phase Error: {train_metrics['mean_phase_error']:.6f}")
    
    # Save training predictions to CSV
    print("\nSaving training predictions to CSV...")
    train_df = pd.DataFrame()
    for i, col_name in enumerate(state_columns):
        train_df[f'{col_name}_true'] = train_data[:, i]
        train_df[f'{col_name}_pred'] = train_preds[:, i]
        train_df[f'{col_name}_error'] = train_data[:, i] - train_preds[:, i]
    train_df['phase_error'] = np.linalg.norm(train_data - train_preds, axis=1)
    train_csv_path = os.path.join(args.save_dir, 'train_predictions.csv')
    train_df.to_csv(train_csv_path, index=False)
    print(f"Training predictions saved to: {train_csv_path}")
    
    # Evaluate on a continuous test trajectory for visualization
    print("\nEvaluating on continuous test trajectory...")
    test_traj = test_trajs[0]  # Use first trajectory
    n_eval_steps = min(100, len(test_traj) - 1)
    
    # For fair comparison, refit models including validation data
    # This allows prediction starting from test trajectory beginning
    print("Refitting models with train+val data for trajectory prediction...")
    train_val_data = np.vstack([train_data, val_data])
    
    if method_used == 'separate':
        # Refit models with train+val data
        traj_models = []
        for i in range(n_x):
            model, _ = fit_arima_univariate(train_val_data[:, i], order=orders[i])
            traj_models.append(model)
        # Predict from end of train+val (which is start of test)
        traj_preds = predict_arima_separate(traj_models, n_steps=n_eval_steps)
    else:
        # Refit VAR with train+val data
        lag_order = model.k_ar
        traj_model = VAR(train_val_data).fit(maxlags=lag_order)
        last_values = train_val_data[-lag_order:].copy()
        traj_preds = predict_arima_var(traj_model, n_steps=n_eval_steps, last_values=last_values)
    
    traj_true = test_traj[:n_eval_steps+1]  # n_eval_steps+1 points (includes initial condition)
    # traj_preds has n_eval_steps points (predictions from step 1 to n_eval_steps)
    # Align: compare predictions with true values from step 1 onwards
    traj_true_aligned = traj_true[1:]  # Skip initial condition, compare from step 1
    traj_preds_aligned = traj_preds[:n_eval_steps]  # First n_eval_steps predictions
    
    # Ensure same length
    min_len = min(len(traj_true_aligned), len(traj_preds_aligned))
    traj_true_aligned = traj_true_aligned[:min_len]
    traj_preds_aligned = traj_preds_aligned[:min_len]
    
    traj_metrics = evaluate_arima_predictions(traj_true_aligned, traj_preds_aligned)
    
    print(f"Trajectory Metrics ({min_len} steps):")
    print(f"  RMSE: {traj_metrics['rmse']:.6f}")
    print(f"  MAE: {traj_metrics['mae']:.6f}")
    print(f"  Mean Phase Error: {traj_metrics['mean_phase_error']:.6f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Phase space plot (for 2D systems)
    if n_x == 2:
        plt.subplot(1, 3, 1)
        plt.plot(traj_true[:, 0], traj_true[:, 1], '-o', label='True', markersize=3, alpha=0.7)
        # Plot predictions aligned with true (skip initial point)
        pred_with_init = np.vstack([traj_true[0:1], traj_preds_aligned])  # Add initial point
        plt.plot(pred_with_init[:, 0], pred_with_init[:, 1], 
                '-x', label='ARIMA Pred', markersize=3, alpha=0.7)
        plt.xlabel(state_columns[0])
        plt.ylabel(state_columns[1])
        plt.title('Phase Space: ARIMA Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Time series plot (first dimension)
    plt.subplot(1, 3, 2)
    time_axis = np.arange(len(traj_true))
    plt.plot(time_axis, traj_true[:, 0], '-o', label='True', markersize=2, alpha=0.7)
    # Plot predictions aligned
    pred_with_init = np.vstack([traj_true[0:1], traj_preds_aligned])
    pred_time_axis = np.arange(len(pred_with_init))
    plt.plot(pred_time_axis, pred_with_init[:, 0], '-x', label='ARIMA Pred', markersize=2, alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel(state_columns[0])
    plt.title(f'Time Series: {state_columns[0]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error over time
    plt.subplot(1, 3, 3)
    errors = np.linalg.norm(traj_true_aligned - traj_preds_aligned, axis=1)
    error_time_axis = np.arange(len(errors))
    plt.plot(error_time_axis, errors, '-', linewidth=2, label='Prediction Error')
    plt.xlabel('Time Step')
    plt.ylabel('Phase Space Error')
    plt.title('Prediction Error Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    results_path = os.path.join(args.save_dir, 'arima_results.png')
    plt.savefig(results_path, dpi=150)
    print(f"\nResults saved to '{results_path}'")
    plt.show()
    
    # Save VAR model if using VAR method
    if method_used == 'var':
        import pickle
        var_model_path = os.path.join(args.save_dir, 'var_model.pkl')
        with open(var_model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"VAR model saved to '{var_model_path}'")
        
        # Also save test data for evaluation
        test_data_path = os.path.join(args.save_dir, 'test_data.npz')
        np.savez(test_data_path,
                 x_test=test_data,
                 x_test_next=test_data,  # For VAR, we predict from previous values
                 state_columns=state_columns,
                 train_data=train_data)
        print(f"Test data saved to '{test_data_path}'")
    
    # Save metrics
    metrics_path = os.path.join(args.save_dir, 'arima_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("ARIMA Baseline Metrics\n")
        f.write("="*50 + "\n")
        f.write(f"Method: {method_used}\n")
        if method_used == 'separate':
            f.write(f"Orders: {orders}\n")
        else:
            f.write(f"VAR Lag Order: {lag_order}\n")
        f.write("\nTraining Metrics:\n")
        for key, value in train_metrics.items():
            f.write(f"  {key}: {value:.6f}\n")
        f.write("\nValidation Metrics:\n")
        for key, value in val_metrics.items():
            f.write(f"  {key}: {value:.6f}\n")
        f.write("\nTest Metrics:\n")
        for key, value in test_metrics.items():
            f.write(f"  {key}: {value:.6f}\n")
        f.write("\nTrajectory Metrics:\n")
        for key, value in traj_metrics.items():
            f.write(f"  {key}: {value:.6f}\n")
    print(f"Metrics saved to '{metrics_path}'")


if __name__ == "__main__":
    main()

