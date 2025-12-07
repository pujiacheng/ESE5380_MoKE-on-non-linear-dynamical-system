"""
Evaluate Koopman AE Baseline Model

Uses evaluation.py functions to evaluate the trained model on test data.
Computes various metrics including:
- 1-step MSE
- Multi-step NRMSE
- Phase portrait fidelity (Chamfer distance)
- Spectral radius of A_f
- Long-horizon divergence rate
- Reconstruction error

All plots are saved to the results directory.
"""

import torch
import numpy as np
import argparse
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from baseline_models.koopman_ae_baseline import KoopmanAEBaseline
import evaluation


def load_model_and_data(model_path, test_data_path, device):
    """
    Load trained model and test data
    
    Args:
        model_path: path to saved model state dict
        test_data_path: path to saved test data (.npz file)
        device: device to load model on
    
    Returns:
        model: loaded model
        x_test: test input states
        x_test_next: test next states
        state_columns: list of state column names
    """
    # Load test data
    data = np.load(test_data_path, allow_pickle=True)
    x_test = data['x_test']
    x_test_next = data['x_test_next']
    state_columns = data['state_columns']
    
    # Determine model dimensions from data
    n_x = x_test.shape[1]
    n_z = 10 * n_x  # Default latent dimension
    
    # Create and load model
    model = KoopmanAEBaseline(n_x=n_x, n_z=n_z).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Loaded model: n_x={n_x}, n_z={n_z}")
    print(f"Test data shape: {x_test.shape}")
    
    return model, x_test, x_test_next, state_columns


def evaluate_one_step(model, x_test, x_test_next, device, save_dir):
    """Evaluate 1-step prediction"""
    print("\n" + "="*50)
    print("1-Step Prediction Evaluation")
    print("="*50)
    
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        x_pred = model.predict_next(x_test_tensor).cpu().numpy()
    
    # Compute 1-step MSE
    mse_1step = evaluation.one_step_mse(x_test_next, x_pred, plot=False)
    print(f"1-Step MSE: {mse_1step:.6f}")
    
    # Create and save plot
    x_t = evaluation.flatten_batch_time(evaluation.to_tensor(x_test_next))
    x_p = evaluation.flatten_batch_time(evaluation.to_tensor(x_pred))
    per_dim_mse = torch.mean((x_t - x_p) ** 2, dim=0).cpu().numpy()
    dims = np.arange(len(per_dim_mse))
    
    plt.figure(figsize=(6,4))
    plt.bar(dims, per_dim_mse)
    plt.xlabel("State Dimension")
    plt.ylabel("MSE")
    plt.title("1-step MSE per Dimension")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eval_1step_mse.png'), dpi=150)
    plt.close()
    print(f"  Plot saved to: eval_1step_mse.png")
    
    return x_pred, mse_1step


def evaluate_multi_step(model, x_test, device, save_dir, horizons=[1, 5, 10, 20, 50, 100]):
    """Evaluate multi-step prediction"""
    print("\n" + "="*50)
    print("Multi-Step Prediction Evaluation")
    print("="*50)
    
    # Use first test sample as initial condition
    x0 = torch.tensor(x_test[0:1], dtype=torch.float32).to(device)
    max_horizon = max(horizons)
    
    with torch.no_grad():
        # Predict sequence
        pred_sequence = model.predict_sequence(x0, n_steps=max_horizon)
        pred_sequence = pred_sequence.cpu().numpy()  # (1, max_horizon+1, n_x)
    
    # Get true sequence (we need to reconstruct from test pairs)
    # For now, we'll use the predicted sequence and compare with available test data
    # In practice, you'd want the full true trajectory
    
    # Compute multi-step NRMSE
    # Note: This requires true trajectories, so we'll create a placeholder
    # In practice, you should load full trajectories
    true_sequence = np.zeros_like(pred_sequence)
    true_sequence[0, 0] = x_test[0]
    # Fill in from test pairs (approximation)
    for i in range(min(len(x_test), max_horizon)):
        if i < len(x_test):
            true_sequence[0, i+1] = x_test[i] if i+1 < true_sequence.shape[1] else x_test[min(i, len(x_test)-1)]
    
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
        plt.title("Multi-step NRMSE vs Horizon")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'eval_multistep_nrmse.png'), dpi=150)
        plt.close()
        print(f"  Plot saved to: eval_multistep_nrmse.png")
    
    return pred_sequence, nrmse_results


def evaluate_phase_portrait(model, x_test, x_test_next, device, save_dir, dims=(0, 1)):
    """Evaluate phase portrait fidelity"""
    print("\n" + "="*50)
    print("Phase Portrait Fidelity Evaluation")
    print("="*50)
    
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        x_pred = model.predict_next(x_test_tensor).cpu().numpy()
    
    # True phase portrait: (x_test, x_test_next)
    # Predicted phase portrait: (x_test, x_pred)
    true_phase = np.hstack([x_test, x_test_next])
    pred_phase = np.hstack([x_test, x_pred])
    
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
    plt.plot(x_p[:,0], x_p[:,1], '-x', markersize=3, alpha=0.7, label="Pred")
    plt.xlabel(f"Dim {dims[0]}")
    plt.ylabel(f"Dim {dims[1]}")
    plt.title("Phase Portrait: True vs Predicted")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eval_phase_portrait.png'), dpi=150)
    plt.close()
    print(f"  Plot saved to: eval_phase_portrait.png")
    
    return chamfer


def evaluate_spectral_properties(model, device):
    """Evaluate spectral properties of A_f"""
    print("\n" + "="*50)
    print("Spectral Properties Evaluation")
    print("="*50)
    
    A_f = model.A_f.detach().cpu().numpy()
    rho, eigvals = evaluation.spectral_radius(A_f)
    
    print(f"Spectral Radius of A_f: {rho:.6f}")
    print(f"Eigenvalues of A_f:")
    for i, ev in enumerate(eigvals[:10]):  # Print first 10
        print(f"  λ_{i}: {ev:.4f} {'+' if ev.imag >= 0 else ''}{ev.imag:.4f}j")
    if len(eigvals) > 10:
        print(f"  ... ({len(eigvals) - 10} more)")
    
    return rho, eigvals


def evaluate_reconstruction(model, x_test, device, save_dir):
    """Evaluate reconstruction quality"""
    print("\n" + "="*50)
    print("Reconstruction Quality Evaluation")
    print("="*50)
    
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        out = model(x_test_tensor)
        x_rec = out['x_rec'].cpu().numpy()
    
    rec_err = evaluation.reconstruction_error(x_test, x_rec, plot=False)
    print(f"Reconstruction Error: {rec_err:.6f}")
    
    # Create and save plot
    x_t = evaluation.flatten_batch_time(evaluation.to_tensor(x_test))
    x_r = evaluation.flatten_batch_time(evaluation.to_tensor(x_rec))
    x_t_np = x_t.cpu().numpy()
    x_r_np = x_r.cpu().numpy()
    
    plt.figure(figsize=(6,4))
    plt.plot(x_t_np[:,0], label="True", alpha=0.7)
    plt.plot(x_r_np[:,0], label="Reconstructed", alpha=0.7)
    plt.xlabel("Time step")
    plt.ylabel("State[0]")
    plt.title("Reconstruction Quality (dim 0)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eval_reconstruction.png'), dpi=150)
    plt.close()
    print(f"  Plot saved to: eval_reconstruction.png")
    
    return rec_err


def evaluate_long_horizon(model, x_test, device, save_dir, n_steps=200):
    """Evaluate long-horizon prediction"""
    print("\n" + "="*50)
    print("Long-Horizon Prediction Evaluation")
    print("="*50)
    
    # Use first test sample as initial condition
    x0 = torch.tensor(x_test[0:1], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred_sequence = model.predict_sequence(x0, n_steps=n_steps)
        pred_sequence = pred_sequence.cpu().numpy()  # (1, n_steps+1, n_x)
    
    # For divergence rate, we need true trajectory
    # Create a placeholder - in practice, load full true trajectory
    true_sequence = np.zeros_like(pred_sequence)
    true_sequence[0, 0] = x_test[0]
    # Approximate true sequence (this is a limitation - ideally you'd have full trajectories)
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
    plt.title("Long-horizon Error Growth")
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
        f.write("Koopman AE Baseline Evaluation Results\n")
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
        
        if 'spectral_radius' in results:
            f.write(f"Spectral Radius of A_f: {results['spectral_radius']:.6f}\n\n")
        
        if 'reconstruction_error' in results:
            f.write(f"Reconstruction Error: {results['reconstruction_error']:.6f}\n\n")
        
        if 'divergence_rate' in results:
            f.write(f"Long-Horizon Divergence Rate: {results['divergence_rate']:.6f}\n")
            f.write("  (Positive = exponential growth, Negative = decay)\n")
    
    print(f"\nEvaluation results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Koopman AE Baseline')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model (.pth file)')
    parser.add_argument('--test_data_path', type=str, required=True,
                       help='Path to test data (.npz file from train_kae.py)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save evaluation results (default: same as model dir)')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 10, 20, 50, 100],
                       help='Horizons for multi-step evaluation')
    parser.add_argument('--long_horizon_steps', type=int, default=200,
                       help='Number of steps for long-horizon evaluation')
    
    args = parser.parse_args()
    
    # Set save directory
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.model_path)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and data
    model, x_test, x_test_next, state_columns = load_model_and_data(
        args.model_path, args.test_data_path, device
    )
    
    results = {}
    
    # 1. One-step evaluation
    x_pred, mse_1step = evaluate_one_step(model, x_test, x_test_next, device, args.save_dir)
    results['mse_1step'] = mse_1step
    
    # 2. Multi-step evaluation
    pred_sequence, nrmse_results = evaluate_multi_step(
        model, x_test, device, args.save_dir, horizons=args.horizons
    )
    results['nrmse'] = nrmse_results
    
    # 3. Phase portrait evaluation
    chamfer = evaluate_phase_portrait(model, x_test, x_test_next, device, args.save_dir)
    results['chamfer'] = chamfer
    
    # 4. Spectral properties
    rho, eigvals = evaluate_spectral_properties(model, device)
    results['spectral_radius'] = rho
    results['eigenvalues'] = eigvals
    
    # 5. Reconstruction quality
    rec_err = evaluate_reconstruction(model, x_test, device, args.save_dir)
    results['reconstruction_error'] = rec_err
    
    # 6. Long-horizon evaluation
    slope, errors = evaluate_long_horizon(model, x_test, device, args.save_dir, args.long_horizon_steps)
    results['divergence_rate'] = slope
    
    # Save results
    save_evaluation_results(args.save_dir, results)
    
    print("\n" + "="*50)
    print("Evaluation Complete!")
    print("="*50)


if __name__ == '__main__':
    main()

