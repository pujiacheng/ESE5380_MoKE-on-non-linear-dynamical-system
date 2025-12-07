"""
Evaluate VAR (Vector Autoregression) Model

Uses evaluation.py functions to evaluate the trained VAR model on test data.
Only uses evaluation functions that are suitable for ARIMA/VAR models:
- 1-step MSE ✓
- Multi-step NRMSE ✓
- Phase portrait fidelity (Chamfer distance) ✓
- Long-horizon divergence rate ✓
- Spectral radius (from VAR coefficients) ✓
- Reconstruction error ✗ (not applicable - VAR doesn't reconstruct, it predicts)
"""

import numpy as np
import argparse
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
from statsmodels.tsa.vector_ar.var_model import VAR
import evaluation
import pickle


def load_var_model_and_data(model_path, test_data_path):
    """
    Load trained VAR model and test data
    
    Args:
        model_path: path to saved VAR model (pickle file)
        test_data_path: path to saved test data (.npz file)
    
    Returns:
        model: loaded VAR model
        x_test: test input states
        x_test_next: test next states
        state_columns: list of state column names
        train_data: training data (for context)
    """
    # Load test data
    data = np.load(test_data_path, allow_pickle=True)
    x_test = data['x_test']
    # For VAR, x_test_next is the same as x_test (we predict from previous values)
    # We'll create pairs manually
    if 'x_test_next' in data:
        x_test_next = data['x_test_next']
    else:
        # Create next states from test data (shift by 1)
        x_test_next = x_test[1:].copy()
        x_test = x_test[:-1].copy()
    state_columns = data['state_columns']
    train_data = data.get('train_data', None)
    
    # Load VAR model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loaded VAR model with lag order: {model.k_ar}")
    print(f"Test data shape: {x_test.shape}")
    
    return model, x_test, x_test_next, state_columns, train_data


def evaluate_one_step(model, x_test, x_test_next, save_dir, lag_order):
    """Evaluate 1-step prediction"""
    print("\n" + "="*50)
    print("1-Step Prediction Evaluation")
    print("="*50)
    
    # For VAR, we need lag_order previous values
    # Use the last lag_order values from test data (or pad)
    n_samples = len(x_test)
    x_pred = []
    
    # Use first lag_order samples as initial condition
    last_values = x_test[:lag_order].copy()
    
    for i in range(n_samples):
        # Predict next step
        pred = model.forecast(last_values, steps=1)
        x_pred.append(pred[0])
        
        # Update last_values (sliding window)
        if i + lag_order < n_samples:
            last_values = x_test[i+1:i+lag_order+1].copy()
        else:
            # Use prediction for next window
            last_values = np.vstack([last_values[1:], pred[0]])
    
    x_pred = np.array(x_pred)
    
    # Ensure same length
    min_len = min(len(x_test_next), len(x_pred))
    x_test_next_aligned = x_test_next[:min_len]
    x_pred_aligned = x_pred[:min_len]
    
    # Compute 1-step MSE
    mse_1step = evaluation.one_step_mse(x_test_next_aligned, x_pred_aligned, plot=False)
    print(f"1-Step MSE: {mse_1step:.6f}")
    
    # Create and save plot
    x_t = evaluation.flatten_batch_time(evaluation.to_tensor(x_test_next_aligned))
    x_p = evaluation.flatten_batch_time(evaluation.to_tensor(x_pred_aligned))
    per_dim_mse = torch.mean((x_t - x_p) ** 2, dim=0).cpu().numpy()
    dims = np.arange(len(per_dim_mse))
    
    plt.figure(figsize=(6,4))
    plt.bar(dims, per_dim_mse)
    plt.xlabel("State Dimension")
    plt.ylabel("MSE")
    plt.title("1-step MSE per Dimension (VAR)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eval_1step_mse.png'), dpi=150)
    plt.close()
    print(f"  Plot saved to: eval_1step_mse.png")
    
    return x_pred_aligned, mse_1step


def evaluate_multi_step(model, x_test, save_dir, horizons=[1, 5, 10, 20, 50, 100], lag_order=None):
    """Evaluate multi-step prediction"""
    print("\n" + "="*50)
    print("Multi-Step Prediction Evaluation")
    print("="*50)
    
    if lag_order is None:
        lag_order = model.k_ar
    
    # Use first test sample as initial condition
    x0 = x_test[0:lag_order].copy()
    max_horizon = max(horizons)
    
    # Predict sequence
    pred_sequence = model.forecast(x0, steps=max_horizon)
    pred_sequence = pred_sequence.reshape(1, max_horizon, -1)  # (1, max_horizon, n_x)
    
    # Add initial condition
    x0_reshaped = x0[-1:].reshape(1, 1, -1)  # (1, 1, n_x)
    pred_sequence = np.concatenate([x0_reshaped, pred_sequence], axis=1)  # (1, max_horizon+1, n_x)
    
    # Get true sequence (approximation from test data)
    true_sequence = np.zeros((1, max_horizon+1, x_test.shape[1]))
    true_sequence[0, 0] = x_test[0]
    for i in range(min(len(x_test), max_horizon)):
        if i+1 < true_sequence.shape[1]:
            true_sequence[0, i+1] = x_test[min(i, len(x_test)-1)]
    
    nrmse_results = evaluation.multi_step_nrmse(
        true_sequence, pred_sequence, horizons=horizons, plot=False
    )
    
    print("Multi-step NRMSE:")
    for T, nrmse in nrmse_results.items():
        print(f"  Horizon {T}: {nrmse:.6f}")
    
    # Create and save plot
    if len(nrmse_results) > 0:
        H = sorted(nrmse_results.keys())
        V = [nrmse_results[h] for h in H]
        
        plt.figure(figsize=(6,4))
        plt.plot(H, V, marker='o', linewidth=2)
        plt.xlabel("Horizon T")
        plt.ylabel("NRMSE(T)")
        plt.title("Multi-step NRMSE vs Horizon (VAR)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'eval_multistep_nrmse.png'), dpi=150)
        plt.close()
        print(f"  Plot saved to: eval_multistep_nrmse.png")
    
    return pred_sequence, nrmse_results


def evaluate_phase_portrait(model, x_test, x_test_next, save_dir, lag_order, dims=(0, 1)):
    """Evaluate phase portrait fidelity"""
    print("\n" + "="*50)
    print("Phase Portrait Fidelity Evaluation")
    print("="*50)
    
    # Predict next states
    n_samples = len(x_test)
    x_pred = []
    last_values = x_test[:lag_order].copy()
    
    for i in range(n_samples):
        pred = model.forecast(last_values, steps=1)
        x_pred.append(pred[0])
        if i + lag_order < n_samples:
            last_values = x_test[i+1:i+lag_order+1].copy()
        else:
            last_values = np.vstack([last_values[1:], pred[0]])
    
    x_pred = np.array(x_pred)
    
    # Align lengths
    min_len = min(len(x_test_next), len(x_pred))
    x_test_next_aligned = x_test_next[:min_len]
    x_pred_aligned = x_pred[:min_len]
    
    # True phase portrait: (x_test, x_test_next)
    # Predicted phase portrait: (x_test, x_pred)
    true_phase = np.hstack([x_test[:min_len], x_test_next_aligned])
    pred_phase = np.hstack([x_test[:min_len], x_pred_aligned])
    
    chamfer = evaluation.chamfer_distance_phase(
        true_phase, pred_phase, dims=dims, plot=False
    )
    
    print(f"Chamfer Distance (phase portrait): {chamfer:.6f}")
    
    # Create and save plot
    x_true_ = evaluation.flatten_batch_time(evaluation.to_tensor(true_phase))[:, list(dims)]
    x_pred_ = evaluation.flatten_batch_time(evaluation.to_tensor(pred_phase))[:, list(dims)]
    x_t = x_true_.cpu().numpy()
    x_p = x_pred_.cpu().numpy()
    
    plt.figure(figsize=(5,5))
    plt.plot(x_t[:,0], x_t[:,1], '-o', markersize=3, alpha=0.7, label="True")
    plt.plot(x_p[:,0], x_p[:,1], '-x', markersize=3, alpha=0.7, label="VAR Pred")
    plt.xlabel(f"Dim {dims[0]}")
    plt.ylabel(f"Dim {dims[1]}")
    plt.title("Phase Portrait: True vs Predicted (VAR)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eval_phase_portrait.png'), dpi=150)
    plt.close()
    print(f"  Plot saved to: eval_phase_portrait.png")
    
    return chamfer


def evaluate_spectral_properties(model):
    """Evaluate spectral properties of VAR coefficients"""
    print("\n" + "="*50)
    print("Spectral Properties Evaluation")
    print("="*50)
    
    # Extract VAR coefficient matrices
    # VAR model stores coefficients in model.coefs
    # For VAR(p), we have p coefficient matrices
    # We can compute the companion matrix for spectral analysis
    
    try:
        # Get coefficient matrices from VAR model
        # VAR model has coefs attribute with shape (lag_order, n_x, n_x)
        if hasattr(model, 'coefs'):
            coef_matrices = model.coefs  # Shape: (lag_order, n_x, n_x)
            lag_order = coef_matrices.shape[0]
            n_x = coef_matrices.shape[1]  # State dimension
        else:
            # Fallback: extract from params
            params = model.params
            if hasattr(params, 'values'):
                params = params.values
            params = np.array(params)
            lag_order = model.k_ar
            n_x = params.shape[1]  # State dimension from params
            coef_matrices = []
            for i in range(lag_order):
                coef_mat = params[i*n_x:(i+1)*n_x, :].T
                coef_matrices.append(coef_mat)
            coef_matrices = np.array(coef_matrices)
        
        # Build companion matrix
        # For VAR(p): companion matrix is (n_x*p) x (n_x*p)
        companion_size = n_x * lag_order
        companion = np.zeros((companion_size, companion_size))
        
        # First block row: coefficient matrices
        for i in range(lag_order):
            companion[:n_x, i*n_x:(i+1)*n_x] = coef_matrices[i]
        
        # Identity blocks below diagonal
        for i in range(1, lag_order):
            companion[i*n_x:(i+1)*n_x, (i-1)*n_x:i*n_x] = np.eye(n_x)
        
        # Compute spectral radius
        rho, eigvals = evaluation.spectral_radius(companion)
        
        print(f"Spectral Radius of VAR Companion Matrix: {rho:.6f}")
        print(f"VAR Lag Order: {lag_order}")
        print(f"Companion Matrix Size: {companion_size}x{companion_size}")
        print(f"Eigenvalues (first 10):")
        for i, ev in enumerate(eigvals[:10]):
            print(f"  λ_{i}: {ev:.4f} {'+' if ev.imag >= 0 else ''}{ev.imag:.4f}j")
        if len(eigvals) > 10:
            print(f"  ... ({len(eigvals) - 10} more)")
        
        return rho, eigvals, companion
        
    except Exception as e:
        print(f"Warning: Could not compute spectral properties: {e}")
        print("  This may be due to VAR model structure differences.")
        return None, None, None


def evaluate_long_horizon(model, x_test, save_dir, lag_order, n_steps=200):
    """Evaluate long-horizon prediction"""
    print("\n" + "="*50)
    print("Long-Horizon Prediction Evaluation")
    print("="*50)
    
    # Use first test sample as initial condition
    x0 = x_test[0:lag_order].copy()
    
    # Predict sequence
    pred_sequence = model.forecast(x0, steps=n_steps)
    pred_sequence = pred_sequence.reshape(1, n_steps, -1)  # (1, n_steps, n_x)
    
    # Add initial condition
    x0_reshaped = x0[-1:].reshape(1, 1, -1)
    pred_sequence = np.concatenate([x0_reshaped, pred_sequence], axis=1)  # (1, n_steps+1, n_x)
    
    # For divergence rate, we need true trajectory
    true_sequence = np.zeros_like(pred_sequence)
    true_sequence[0, 0] = x_test[0]
    for i in range(min(len(x_test), n_steps)):
        if i+1 < true_sequence.shape[1]:
            true_sequence[0, i+1] = x_test[min(i, len(x_test)-1)]
    
    slope, errors = evaluation.long_horizon_divergence_rate(
        true_sequence, pred_sequence, plot=False
    )
    
    print(f"Long-horizon divergence rate (slope): {slope:.6f}")
    print(f"  Positive slope indicates exponential error growth")
    print(f"  Negative slope indicates error decay")
    
    # Create and save plot
    T = len(errors)
    t = np.arange(T)
    log_err = np.log(errors + 1e-8)
    
    plt.figure(figsize=(6,4))
    plt.plot(t, log_err, '-o', markersize=3)
    plt.xlabel("Time step")
    plt.ylabel("log(error)")
    plt.title("Long-horizon Error Growth (VAR)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eval_long_horizon.png'), dpi=150)
    plt.close()
    print(f"  Plot saved to: eval_long_horizon.png")
    
    return slope, errors


def save_evaluation_results(save_dir, results):
    """Save evaluation results to file"""
    results_path = os.path.join(save_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("VAR Baseline Evaluation Results\n")
        f.write("="*50 + "\n\n")
        
        if 'mse_1step' in results:
            f.write(f"1-Step MSE: {results['mse_1step']:.6f}\n\n")
        
        if 'nrmse' in results:
            f.write("Multi-Step NRMSE:\n")
            for T, nrmse in results['nrmse'].items():
                f.write(f"  Horizon {T}: {nrmse:.6f}\n")
            f.write("\n")
        
        if 'chamfer' in results:
            f.write(f"Chamfer Distance (Phase Portrait): {results['chamfer']:.6f}\n\n")
        
        if 'spectral_radius' in results and results['spectral_radius'] is not None:
            f.write(f"Spectral Radius of VAR Companion Matrix: {results['spectral_radius']:.6f}\n")
            f.write(f"VAR Lag Order: {results.get('lag_order', 'N/A')}\n\n")
        
        if 'divergence_rate' in results:
            f.write(f"Long-Horizon Divergence Rate: {results['divergence_rate']:.6f}\n")
            f.write("  (Positive = exponential growth, Negative = decay)\n")
    
    print(f"\nEvaluation results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VAR Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved VAR model (pickle file)')
    parser.add_argument('--test_data_path', type=str, required=True,
                       help='Path to test data (.npz file)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save evaluation results')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 10, 20, 50, 100],
                       help='Horizons for multi-step evaluation')
    parser.add_argument('--long_horizon_steps', type=int, default=200,
                       help='Number of steps for long-horizon evaluation')
    
    args = parser.parse_args()
    
    # Set save directory
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.model_path)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Import torch for evaluation functions
    import torch
    
    # Load model and data
    model, x_test, x_test_next, state_columns, train_data = load_var_model_and_data(
        args.model_path, args.test_data_path
    )
    
    lag_order = model.k_ar
    
    results = {}
    results['lag_order'] = lag_order
    
    # 1. One-step evaluation
    x_pred, mse_1step = evaluate_one_step(model, x_test, x_test_next, args.save_dir, lag_order)
    results['mse_1step'] = mse_1step
    
    # 2. Multi-step evaluation
    pred_sequence, nrmse_results = evaluate_multi_step(
        model, x_test, args.save_dir, horizons=args.horizons, lag_order=lag_order
    )
    results['nrmse'] = nrmse_results
    
    # 3. Phase portrait evaluation
    chamfer = evaluate_phase_portrait(model, x_test, x_test_next, args.save_dir, lag_order)
    results['chamfer'] = chamfer
    
    # 4. Spectral properties
    rho, eigvals, companion = evaluate_spectral_properties(model)
    results['spectral_radius'] = rho
    results['eigenvalues'] = eigvals
    
    # 5. Long-horizon evaluation
    slope, errors = evaluate_long_horizon(model, x_test, args.save_dir, lag_order, args.long_horizon_steps)
    results['divergence_rate'] = slope
    
    # Note: Reconstruction error is not applicable for VAR (it's a prediction model, not an autoencoder)
    
    # Save results
    save_evaluation_results(args.save_dir, results)
    
    print("\n" + "="*50)
    print("Evaluation Complete!")
    print("="*50)
    print("\nNote: Reconstruction error is not computed as VAR is a prediction model,")
    print("      not an autoencoder that reconstructs input states.")


if __name__ == '__main__':
    main()

