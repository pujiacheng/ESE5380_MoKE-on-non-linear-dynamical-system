"""
Smart Hyperparameter Search using Bayesian Optimization

Instead of exhaustive grid search, this uses intelligent sampling based on
observed trends to find optimal hyperparameters efficiently.

Uses:
- Bayesian optimization for continuous parameters
- Early stopping when trend is clear
- Adaptive sampling around promising regions

Installation:
    pip install scikit-optimize

Usage:
    python hyperparam_search_smart.py \
        --system duffing \
        --max_trials 50 \
        --output_dir ./smart_results_duffing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import time
import json
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.plots import plot_convergence, plot_objective
from skopt.utils import use_named_args

from koopman_mixture_neural_network import (
    KoopmanAE,
    spectral_radius_penalty,
    hankel_stack_batch,
    compute_hankel_svd
)
from train_test_with_given_data import (
    load_data_from_csv,
    prepare_data_from_trajectories,
    sample_sequence_batch,
    evaluate_model,
    compute_metrics
)


# System configurations
SYSTEM_CONFIGS = {
    'duffing': {
        'csv_path': 'duffing_with_noise.csv',
        'state_columns': ['x', 'xdot'],
        'n_z_range': (4, 50),
    },
    'vanderpol': {
        'csv_path': 'vanderpol_with_noise.csv',
        'state_columns': ['x', 'xdot'],
        'n_z_range': (4, 50),
    },
    'lorenz': {
        'csv_path': 'lorenz_data.csv',
        'state_columns': ['x', 'y', 'z'],
        'n_z_range': (6, 90),
    },
    'double_pendulum': {
        'csv_path': 'double_pendulum_data.csv',
        'state_columns': ['theta1', 'theta1_dot', 'theta2', 'theta2_dot'],
        'n_z_range': (8, 100),
    },
}


def compute_spectral_radius(A, iters=20):
    """Compute spectral radius using power iteration"""
    A_np = A.detach().cpu().numpy()
    v = np.random.randn(A_np.shape[0])
    v = v / np.linalg.norm(v)
    for _ in range(iters):
        v = A_np @ v
        v = v / (np.linalg.norm(v) + 1e-12)
    Av = A_np @ v
    rho = np.dot(v, Av)
    return abs(rho)


def compute_loss_batch_custom(model, x0, x1, x2, all_X, device, n_x, n_z,
                               hankel_batch_size=64, hankel_Tseq=8, L=4,
                               lambda_rec=1.0, lambda_lin=10.0, lambda_ms=2.0,
                               lambda_hankel=1.0, lambda_bi=1.0, lambda_spec=1.0,
                               lambda_sparse=1e-4, spectral_target=1.1, spectral_iters=8):
    """Compute loss with custom hyperparameters"""
    mse = nn.MSELoss()

    out0 = model(x0)
    out1 = model(x1)
    out2 = model(x2)

    z0 = out0['z']
    z1 = out1['z']
    z2 = out2['z']
    xrec0 = out0['x_rec']

    loss_rec = mse(xrec0, x0)
    z_pred = z0 @ model.A_f.T
    loss_lin = mse(z_pred, z1)
    z_pred2 = z_pred @ model.A_f.T
    loss_ms = mse(z_pred2, z2)

    Id = torch.eye(model.A_f.shape[0], device=device)
    loss_bi = (model.A_f @ model.A_b - Id).norm()**2 + (model.A_b @ model.A_f - Id).norm()**2

    seqs = sample_sequence_batch(all_X, batch_size=hankel_batch_size,
                                Tseq=hankel_Tseq, device=device)

    with torch.no_grad():
        z_seq = model.encoder(seqs.reshape(-1, n_x)).reshape(
            seqs.shape[0], seqs.shape[1], n_z
        )

    H = hankel_stack_batch(z_seq, L=L)
    U, S, Vt = compute_hankel_svd(H)

    r = min(8, Vt.shape[0])
    V_r = Vt[:r].T

    Hmat = H.reshape(-1, H.shape[-1]).cpu().numpy()
    Vcoords = (Hmat @ V_r).reshape(H.shape[0], H.shape[1], r)
    Vcoords = torch.tensor(Vcoords, dtype=torch.float32, device=device)

    v_t = Vcoords[:, :-1, :].reshape(-1, r)
    v_tp1 = Vcoords[:, 1:, :].reshape(-1, r)

    reg = 1e-6
    vt = v_t.detach().cpu().numpy()
    vtp1 = v_tp1.detach().cpu().numpy()
    G = vt.T @ vt + reg * np.eye(r)
    A_v = (vtp1.T @ vt) @ np.linalg.inv(G)
    A_v = torch.tensor(A_v, dtype=torch.float32, device=device)

    v_pred = v_t @ A_v.T
    loss_hankel = mse(v_pred, v_tp1)

    loss_spec = spectral_radius_penalty(model.A_f, iters=spectral_iters, target=spectral_target)
    sparsity_term = model.sparsity_loss(mode="l1")

    loss = (lambda_rec * loss_rec + lambda_lin * loss_lin + lambda_ms * loss_ms +
            lambda_hankel * loss_hankel + lambda_bi * loss_bi +
            lambda_spec * loss_spec + lambda_sparse * sparsity_term)

    return loss


def train_and_evaluate(params_dict, data_dict, device, n_epochs=20):
    """Train model with given parameters and return validation loss"""

    n_z = params_dict['n_z']
    n_x = data_dict['n_x']

    # Create model
    model = KoopmanAE(n_x=n_x, n_z=n_z, p=20, use_observables=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    best_val_loss = float('inf')

    for ep in range(n_epochs):
        model.train()
        for x0, x1, x2 in data_dict['train_loader']:
            x0, x1, x2 = x0.to(device), x1.to(device), x2.to(device)
            loss = compute_loss_batch_custom(
                model, x0, x1, x2, data_dict['all_X_train'], device, n_x, n_z,
                hankel_batch_size=params_dict.get('hankel_batch_size', 64),
                hankel_Tseq=params_dict.get('hankel_Tseq', 8),
                L=params_dict.get('L', 4),
                lambda_rec=1.0,
                lambda_lin=params_dict.get('lambda_lin', 10.0),
                lambda_ms=params_dict.get('lambda_ms', 2.0),
                lambda_hankel=params_dict.get('lambda_hankel', 1.0),
                lambda_bi=params_dict.get('lambda_bi', 1.0),
                lambda_spec=params_dict.get('lambda_spec', 1.0),
                lambda_sparse=params_dict.get('lambda_sparse', 1e-4),
                spectral_target=params_dict.get('spectral_target', 1.1),
                spectral_iters=params_dict.get('spectral_iters', 8)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x0, x1, x2 in data_dict['val_loader']:
                x0, x1, x2 = x0.to(device), x1.to(device), x2.to(device)
                loss = compute_loss_batch_custom(
                    model, x0, x1, x2, data_dict['all_X_train'], device, n_x, n_z,
                    hankel_batch_size=params_dict.get('hankel_batch_size', 64),
                    hankel_Tseq=params_dict.get('hankel_Tseq', 8),
                    L=params_dict.get('L', 4),
                    lambda_rec=1.0,
                    lambda_lin=params_dict.get('lambda_lin', 10.0),
                    lambda_ms=params_dict.get('lambda_ms', 2.0),
                    lambda_hankel=params_dict.get('lambda_hankel', 1.0),
                    lambda_bi=params_dict.get('lambda_bi', 1.0),
                    lambda_spec=params_dict.get('lambda_spec', 1.0),
                    lambda_sparse=params_dict.get('lambda_sparse', 1e-4),
                    spectral_target=params_dict.get('spectral_target', 1.1),
                    spectral_iters=params_dict.get('spectral_iters', 8)
                )
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    # Final evaluation
    test_traj_sample = data_dict['test_traj_sample']
    true, preds = evaluate_model(model, test_traj_sample, device, n_steps=50)
    metrics = compute_metrics(true, preds)
    rho = compute_spectral_radius(model.A_f)

    return {
        'val_loss': best_val_loss,
        'test_rmse': metrics['rmse'],
        'test_mae': metrics['mae'],
        'spectral_radius': rho,
    }


def smart_search_all_hyperparams(system_name, system_config, device, n_epochs, max_trials, output_dir):
    """
    Smart search over ALL hyperparameters using Bayesian optimization
    """
    print("\n" + "="*80)
    print(f"SMART HYPERPARAMETER SEARCH: {system_name.upper()}")
    print(f"Max trials: {max_trials} (vs 100+ in grid search!)")
    print("="*80)

    # Load data once
    trajs, n_x = load_data_from_csv(
        system_config['csv_path'], system_config['state_columns'],
        'traj_id', 'time'
    )
    n_traj, n_steps, _ = trajs.shape

    train_end = int(n_steps * 0.7)
    val_end = int(n_steps * 0.85)

    train_trajs = trajs[:, :train_end, :]
    val_trajs = trajs[:, train_end:val_end, :]
    test_trajs = trajs[:, val_end:, :]

    xt_train, xt1_train, xt2_train = prepare_data_from_trajectories(train_trajs)
    xt_val, xt1_val, xt2_val = prepare_data_from_trajectories(val_trajs)
    xt_test, xt1_test, xt2_test = prepare_data_from_trajectories(test_trajs)

    train_loader = DataLoader(TensorDataset(xt_train, xt1_train, xt2_train),
                              batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(xt_val, xt1_val, xt2_val),
                           batch_size=256, shuffle=False)

    all_X_train = torch.cat([xt_train, xt1_train, xt2_train], dim=0)
    test_traj_sample = torch.cat([xt_test[:50], xt1_test[49:50], xt2_test[49:50]], dim=0).numpy()

    data_dict = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'all_X_train': all_X_train,
        'test_traj_sample': test_traj_sample,
        'n_x': n_x,
    }

    # Define search space
    n_z_min, n_z_max = system_config['n_z_range']

    space = [
        Integer(n_z_min, n_z_max, name='n_z'),
        Real(0.1, 100.0, prior='log-uniform', name='lambda_lin'),
        Real(0.01, 10.0, prior='log-uniform', name='lambda_spec'),
        Real(0.1, 5.0, prior='log-uniform', name='lambda_hankel'),
        Integer(2, 8, name='L'),
        Integer(4, 16, name='hankel_Tseq'),
        Real(0.95, 1.15, name='spectral_target'),
        Integer(4, 20, name='spectral_iters'),
    ]

    # Store all results
    all_results = []
    iteration_count = [0]

    @use_named_args(space)
    def objective(**params):
        iteration_count[0] += 1
        print(f"\n[Trial {iteration_count[0]}/{max_trials}]")
        print(f"  n_z={params['n_z']}, λ_lin={params['lambda_lin']:.2f}, "
              f"λ_spec={params['lambda_spec']:.2f}, λ_hankel={params['lambda_hankel']:.2f}")
        print(f"  L={params['L']}, T_seq={params['hankel_Tseq']}, "
              f"target={params['spectral_target']:.2f}, iters={params['spectral_iters']}")

        start_time = time.time()

        try:
            result = train_and_evaluate(params, data_dict, device, n_epochs)
            val_loss = result['val_loss']

            # Clamp infinite/nan values to prevent GP fitting issues
            if np.isnan(val_loss) or np.isinf(val_loss):
                print(f"  → Val Loss: {val_loss} (invalid, clamping to 1e6)")
                val_loss = 1e6

            # Store result
            result_record = {**params, **result, 'trial': iteration_count[0],
                           'training_time': time.time() - start_time}
            all_results.append(result_record)

            if val_loss < 1e6:
                print(f"  → Val Loss: {val_loss:.6f}, RMSE: {result['test_rmse']:.4f}, "
                      f"ρ: {result['spectral_radius']:.4f}")

            return val_loss

        except Exception as e:
            print(f"  ERROR: {e}")
            return 1e6  # Return large value on error

    # Run Bayesian optimization
    print("\nStarting Bayesian optimization...")

    # Ensure n_initial_points doesn't exceed max_trials
    n_initial = min(10, max(3, max_trials // 2))

    result = gp_minimize(
        objective,
        space,
        n_calls=max_trials,
        n_initial_points=n_initial,  # Random exploration first (adaptive)
        acq_func='EI',  # Expected Improvement
        random_state=42,
        verbose=False
    )

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, f'{system_name}_smart_search_results.csv'), index=False)

    # Find best
    best_idx = df['val_loss'].idxmin()
    best_config = df.loc[best_idx].to_dict()

    # Save optimal config
    optimal_config = {
        'n_z': int(best_config['n_z']),
        'lambda_lin': float(best_config['lambda_lin']),
        'lambda_spec': float(best_config['lambda_spec']),
        'lambda_hankel': float(best_config['lambda_hankel']),
        'L': int(best_config['L']),
        'hankel_Tseq': int(best_config['hankel_Tseq']),
        'spectral_target': float(best_config['spectral_target']),
        'spectral_iters': int(best_config['spectral_iters']),
        'val_loss': float(best_config['val_loss']),
        'test_rmse': float(best_config['test_rmse']),
        'spectral_radius': float(best_config['spectral_radius']),
    }

    with open(os.path.join(output_dir, f'{system_name}_optimal_config.json'), 'w') as f:
        json.dump(optimal_config, f, indent=2)

    # Create visualizations
    create_smart_search_plots(df, result, system_name, output_dir)

    print("\n" + "="*80)
    print("SMART SEARCH COMPLETE!")
    print("="*80)
    print("\nOptimal Configuration:")
    print(json.dumps(optimal_config, indent=2))

    return optimal_config, df


def create_smart_search_plots(df, bayesopt_result, system_name, output_dir):
    """Create visualization plots for smart search results"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Convergence plot
    ax = axes[0, 0]
    trials = df['trial'].values
    val_losses = df['val_loss'].values
    best_so_far = np.minimum.accumulate(val_losses)

    ax.plot(trials, val_losses, 'o-', alpha=0.6, label='Trial loss')
    ax.plot(trials, best_so_far, 'r-', linewidth=2, label='Best so far')
    ax.set_xlabel('Trial', fontweight='bold')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('Convergence: Smart Search', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Val loss vs n_z
    ax = axes[0, 1]
    scatter = ax.scatter(df['n_z'], df['val_loss'], c=df['trial'], cmap='viridis', s=100, alpha=0.7)
    best_idx = df['val_loss'].idxmin()
    ax.scatter(df.loc[best_idx, 'n_z'], df.loc[best_idx, 'val_loss'],
              color='red', s=300, marker='*', edgecolors='black', linewidths=2,
              label=f'Best: n_z={int(df.loc[best_idx, "n_z"])}')
    ax.set_xlabel('$n_z$', fontweight='bold')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('Latent Dimension Exploration', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Trial #')

    # 3. Val loss vs lambda_lin
    ax = axes[0, 2]
    ax.scatter(df['lambda_lin'], df['val_loss'], c=df['trial'], cmap='plasma', s=100, alpha=0.7)
    ax.scatter(df.loc[best_idx, 'lambda_lin'], df.loc[best_idx, 'val_loss'],
              color='red', s=300, marker='*', edgecolors='black', linewidths=2)
    ax.set_xscale('log')
    ax.set_xlabel('$\\lambda_{lin}$', fontweight='bold')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('Linearity Weight Exploration', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Val loss vs lambda_spec
    ax = axes[1, 0]
    ax.scatter(df['lambda_spec'], df['val_loss'], c=df['trial'], cmap='coolwarm', s=100, alpha=0.7)
    ax.scatter(df.loc[best_idx, 'lambda_spec'], df.loc[best_idx, 'val_loss'],
              color='red', s=300, marker='*', edgecolors='black', linewidths=2)
    ax.set_xscale('log')
    ax.set_xlabel('$\\lambda_{spec}$', fontweight='bold')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('Spectral Weight Exploration', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 5. Test RMSE vs Val Loss
    ax = axes[1, 1]
    scatter = ax.scatter(df['val_loss'], df['test_rmse'], c=df['spectral_radius'],
                        cmap='RdYlGn_r', s=100, alpha=0.7, vmin=0.95, vmax=1.15)
    ax.scatter(df.loc[best_idx, 'val_loss'], df.loc[best_idx, 'test_rmse'],
              color='red', s=300, marker='*', edgecolors='black', linewidths=2, label='Best')
    ax.set_xlabel('Validation Loss', fontweight='bold')
    ax.set_ylabel('Test RMSE', fontweight='bold')
    ax.set_title('Generalization', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='$\\rho(A_f)$')

    # 6. Hyperparameter importance (top 5 trials)
    ax = axes[1, 2]
    top5 = df.nsmallest(5, 'val_loss')
    params = ['n_z', 'lambda_lin', 'lambda_spec', 'lambda_hankel', 'L']
    normalized_vals = []
    for param in params:
        vals = top5[param].values
        normalized = (vals - df[param].min()) / (df[param].max() - df[param].min() + 1e-8)
        normalized_vals.append(normalized)

    x = np.arange(len(params))
    for i in range(5):
        ax.plot(x, [normalized_vals[j][i] for j in range(len(params))],
               'o-', label=f'Rank {i+1}', markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(['$n_z$', '$\\lambda_{lin}$', '$\\lambda_{spec}$', '$\\lambda_{hankel}$', '$L$'])
    ax.set_ylabel('Normalized Value', fontweight='bold')
    ax.set_title('Top 5 Configurations', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Smart Hyperparameter Search: {system_name.capitalize()}',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{system_name}_smart_search_plots.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, f'{system_name}_smart_search_plots.pdf'))
    print(f"\n✓ Plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Smart Hyperparameter Search')
    parser.add_argument('--system', type=str, required=True,
                       choices=['duffing', 'vanderpol', 'lorenz', 'double_pendulum'])
    parser.add_argument('--max_trials', type=int, default=50,
                       help='Maximum number of trials (default: 50, vs 100+ in grid search)')
    parser.add_argument('--n_epochs', type=int, default=20,
                       help='Epochs per trial (default: 20, faster than full training)')
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    system_name = args.system
    system_config = SYSTEM_CONFIGS[system_name]

    if args.output_dir is None:
        output_dir = f'./smart_results_{system_name}'
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nDevice: {device}")
    print(f"System: {system_name}")
    print(f"Max trials: {args.max_trials}")
    print(f"Epochs per trial: {args.n_epochs}")
    print(f"Output: {output_dir}")

    # Run smart search
    optimal_config, results_df = smart_search_all_hyperparams(
        system_name, system_config, device, args.n_epochs, args.max_trials, output_dir
    )

    print(f"\n✓ All results saved to: {output_dir}/")
    print(f"\nFiles created:")
    print(f"  - {system_name}_smart_search_results.csv")
    print(f"  - {system_name}_optimal_config.json")
    print(f"  - {system_name}_smart_search_plots.png/pdf")


if __name__ == "__main__":
    main()
