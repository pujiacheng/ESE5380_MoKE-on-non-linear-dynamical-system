"""
Training script for Mixture of Experts Koopman Autoencoder

Features:
- Load balancing loss (ensure all experts are used)
- Diversity loss (encourage expert specialization)
- Per-expert reconstruction, linearity, and multi-step losses
- Bidirectional constraint per expert
- Spectral radius penalty per expert
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from koopman_moe_neural_network import (
    KoopmanMoE,
    spectral_radius_penalty,
    hankel_stack_batch,
    compute_hankel_svd
)


def prepare_data_from_trajectories(trajs, n_steps=8):
    """
    Convert trajectory data to training sequences (x_t, x_{t+1}, ..., x_{t+n_steps})
    
    Args:
        trajs: array of shape (n_traj, n_timesteps, n_x)
        n_steps: number of future steps to include
    
    Returns:
        list of tensors [xt, xt1, ..., xt_n] for consecutive states
    """
    n_traj, n_timesteps, n_x = trajs.shape
    
    all_states = [[] for _ in range(n_steps + 1)]  # xt, xt1, ..., xt8
    
    for traj in trajs:
        if n_timesteps >= n_steps + 1:
            for i in range(n_steps + 1):
                if i == 0:
                    # x_t: from start to -(n_steps)
                    all_states[i].append(traj[:-n_steps])
                elif i == n_steps:
                    # x_{t+n_steps}: from n_steps to end
                    all_states[i].append(traj[n_steps:])
                else:
                    # x_{t+i}: from i to -(n_steps-i)
                    all_states[i].append(traj[i:-(n_steps-i)])
    
    # Concatenate and convert to tensors
    state_tensors = []
    for states in all_states:
        concatenated = np.concatenate(states, axis=0)
        state_tensors.append(torch.tensor(concatenated, dtype=torch.float32))
    
    return state_tensors


def compute_loss_moe(model, state_sequence, device):
    """
    Compute loss for MoE Koopman model
    
    Args:
        state_sequence: list of tensors [x0, x1, ..., x8] for consecutive states
        device: device to run on
    
    Includes:
    1. Reconstruction loss (per expert, weighted by gating)
    2. Prediction loss (1-step)
    3. Multi-step loss (8-step)
    4. Latent linearity (per expert)
    5. Load balancing (ensure all experts used)
    6. Diversity loss (encourage specialization)
    7. Bidirectional constraint (per expert)
    8. Spectral radius penalty (per expert)
    """
    # Hyperparameters (rebalanced for Koopman + MoE)
    lam_rec = 1.0       # Reconstruction (baseline)
    lam_pred = 5.0      # 1-step prediction (reduced from 10.0)
    lam_ms = 10.0       # 8-step multi-step (INCREASED - most important!)
    lam_lin = 8.0       # Linearity (INCREASED - enforce Koopman structure)
    lam_balance = 1.0   # Load balancing
    lam_diversity_latent = 0.5     # Latent diversity (increased)
    lam_diversity_operator = 0.5   # Operator diversity (increased)
    lam_bi = 2.0        # Bidirectional (INCREASED - stability)
    lam_spec = 1.5      # Spectral radius (INCREASED - prevent explosion)
    
    # Unpack states
    x0 = state_sequence[0]
    x1 = state_sequence[1]
    x8 = state_sequence[8]  # 8-step ahead
    
    mse = nn.MSELoss()
    
    # Forward pass for current state
    out0 = model(x0)
    weights0 = out0['weights']  # (batch, n_experts)
    expert_recs = out0['expert_recs']  # List of (batch, n_x)
    expert_latents = out0['expert_latents']  # List of (batch, n_z)
    x_rec_blended = out0['x_rec']  # (batch, n_x)
    
    # === 1. Reconstruction Loss ===
    # Blended reconstruction should match input
    loss_rec = mse(x_rec_blended, x0)
    
    # === 2. Prediction Loss (1-step) ===
    # Each expert predicts next state
    loss_pred = 0
    expert_preds = []
    for expert in model.experts:
        x1_pred = expert.predict_next(x0)
        expert_preds.append(x1_pred)
    
    # Blend predictions
    x1_pred_blended = model.blending(expert_preds, weights0)
    loss_pred = mse(x1_pred_blended, x1)
    
    # === 3. Multi-Step Loss (8-step) ===
    # Predict 8 steps ahead using iterated predictions
    loss_ms = 0
    expert_preds_8step = []
    for expert in model.experts:
        x_pred = x0
        # Iterate 8 times
        for _ in range(8):
            x_pred = expert.predict_next(x_pred)
        expert_preds_8step.append(x_pred)
    
    # Blend 8-step predictions
    x8_pred_blended = model.blending(expert_preds_8step, weights0)
    loss_ms = mse(x8_pred_blended, x8)
    
    # === 4. Latent Linearity (per expert) ===
    # z_{t+1} should equal A_f @ z_t
    loss_lin = 0
    for i, expert in enumerate(model.experts):
        z0 = expert.encoder(x0)
        z1_true = expert.encoder(x1)
        z1_pred = z0 @ expert.A_f.T
        # Weight by how much this expert was active
        loss_lin += (weights0[:, i:i+1] * (z1_pred - z1_true)**2).mean()
    
    # === 5. Load Balancing Loss ===
    # Ensure all experts are used roughly equally
    avg_weights = weights0.mean(dim=0)  # Average over batch
    target_weight = 1.0 / model.n_experts
    loss_balance = ((avg_weights - target_weight)**2).sum()
    
    # === 6. Diversity Loss ===
    # 6a. Latent diversity (encourage different encodings)
    loss_diversity_latent = 0
    for i in range(model.n_experts):
        for j in range(i+1, model.n_experts):
            # Cosine similarity (penalize if similar)
            similarity = F.cosine_similarity(
                expert_latents[i], 
                expert_latents[j], 
                dim=-1
            ).mean()
            loss_diversity_latent -= similarity.abs()
    # Normalize by number of pairs
    n_pairs = model.n_experts * (model.n_experts - 1) / 2
    if n_pairs > 0:
        loss_diversity_latent /= n_pairs
    
    # 6b. Operator diversity (encourage different Koopman operators)
    loss_diversity_operator = 0
    for i in range(model.n_experts):
        for j in range(i+1, model.n_experts):
            # Frobenius inner product
            A_sim = (model.experts[i].A_f * model.experts[j].A_f).sum()
            loss_diversity_operator -= A_sim.abs() / (model.n_z ** 2)
    if n_pairs > 0:
        loss_diversity_operator /= n_pairs
    
    # === 7. Bidirectional Constraint (per expert) ===
    # A_f @ A_b ≈ I
    loss_bi = 0
    I = torch.eye(model.n_z, device=device)
    for expert in model.experts:
        loss_bi += (expert.A_f @ expert.A_b - I).norm()**2
        loss_bi += (expert.A_b @ expert.A_f - I).norm()**2
    loss_bi /= model.n_experts
    
    # === 8. Spectral Radius Penalty (per expert) ===
    # Prevent explosive dynamics
    loss_spec = 0
    for expert in model.experts:
        loss_spec += spectral_radius_penalty(expert.A_f, iters=8, target=1.1)
    loss_spec /= model.n_experts
    
    # === Total Loss ===
    loss_total = (
        lam_rec * loss_rec +
        lam_pred * loss_pred +
        lam_ms * loss_ms +
        lam_lin * loss_lin +
        lam_balance * loss_balance +
        lam_diversity_latent * loss_diversity_latent +
        lam_diversity_operator * loss_diversity_operator +
        lam_bi * loss_bi +
        lam_spec * loss_spec
    )
    
    return {
        'total': loss_total,
        'rec': loss_rec,
        'pred': loss_pred,
        'ms': loss_ms,
        'lin': loss_lin,
        'balance': loss_balance,
        'diversity_latent': loss_diversity_latent,
        'diversity_operator': loss_diversity_operator,
        'bi': loss_bi,
        'spec': loss_spec
    }


def train_model_moe(model, train_loader, device, n_epochs=40, val_loader=None, 
                    early_stopping=False, patience=20):
    """
    Train the MoE Koopman model
    
    Args:
        model: KoopmanMoE instance
        train_loader: DataLoader with state sequences [x0, x1, ..., x8]
        device: device to train on
        n_epochs: number of training epochs
        val_loader: optional validation DataLoader
        early_stopping: whether to use early stopping
        patience: number of epochs to wait for improvement
    
    Returns:
        log: list of dicts with loss history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    log = []
    # Multi-criteria early stopping
    best_val_ms = float('inf')        # Best 8-step prediction
    best_val_lin = float('inf')       # Best linearity
    best_combined = float('inf')      # Best combined score
    patience_counter = 0
    best_model_state = None
    
    # Print header
    print("\n" + "="*110)
    if val_loader is not None:
        if early_stopping:
            print(f"{'Epoch':<8} {'Train MS(8)':<12} {'Train Lin':<12} "
                  f"{'Val MS(8)':<12} {'Val Lin':<12} {'Combined':<12} {'Status':<22}")
        else:
            print(f"{'Epoch':<8} {'Train MS(8)':<12} {'Train Lin':<12} "
                  f"{'Val MS(8)':<12} {'Val Lin':<12}")
    else:
        print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Pred':<12} {'Train MS(8)':<12} {'Balance':<12}")
    print("="*110)
    
    for ep in range(n_epochs):
        # Training
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            # batch is a list of 9 tensors: [x0, x1, ..., x8]
            state_sequence = [state.to(device) for state in batch]
            
            # Compute losses
            losses = compute_loss_moe(model, state_sequence, device)
            
            # Backprop
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            
            epoch_losses.append({k: v.item() for k, v in losses.items()})
        
        # Average training losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in epoch_losses])
        
        # Validation (if provided)
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    state_sequence = [state.to(device) for state in batch]
                    losses = compute_loss_moe(model, state_sequence, device)
                    val_losses.append({k: v.item() for k, v in losses.items()})
            
            # Average validation losses
            avg_val_losses = {}
            for key in val_losses[0].keys():
                avg_val_losses[key] = np.mean([l[key] for l in val_losses])
            
            avg_losses['val_total'] = avg_val_losses['total']
            avg_losses['val_pred'] = avg_val_losses['pred']
            avg_losses['val_ms'] = avg_val_losses['ms']
            avg_losses['val_lin'] = avg_val_losses['lin']
            
            # Multi-criteria early stopping
            if early_stopping:
                # Combined score: weighted sum of prediction accuracy + linearity
                # Both should be minimized
                weight_ms = 0.7   # 70% weight on prediction accuracy
                weight_lin = 0.3  # 30% weight on maintaining linearity
                
                current_combined = weight_ms * avg_val_losses['ms'] + weight_lin * avg_val_losses['lin']
                
                # Check if this is the best model
                is_best = False
                if current_combined < best_combined:
                    best_combined = current_combined
                    best_val_ms = avg_val_losses['ms']
                    best_val_lin = avg_val_losses['lin']
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    is_best = True
                    status = "✓ Best (MS+Lin)"
                else:
                    patience_counter += 1
                    # Show what would have been best
                    if avg_val_losses['ms'] < best_val_ms:
                        status = f"MS↓ Lin↑ | {patience_counter}/{patience}"
                    elif avg_val_losses['lin'] < best_val_lin:
                        status = f"MS↑ Lin↓ | {patience_counter}/{patience}"
                    else:
                        status = f"Both↑ | {patience_counter}/{patience}"
                    
                    if patience_counter >= patience:
                        print(f"{'':8} {'':12} {'':12} {'':12} {'':12} {'':12} {'Early Stop!':<22}")
                        print("="*110)
                        print(f"Training stopped early at epoch {ep}")
                        print(f"Best validation:")
                        print(f"  Combined score: {best_combined:.6f}")
                        print(f"  8-step loss: {best_val_ms:.6f}")
                        print(f"  Linearity loss: {best_val_lin:.6f}")
                        print("="*110)
                        # Restore best model
                        model.load_state_dict(best_model_state)
                        log.append(avg_losses)
                        break
        
        log.append(avg_losses)
        
        # Print progress every epoch
        if val_loader is not None:
            if early_stopping:
                # Calculate current combined score for display
                weight_ms = 0.7
                weight_lin = 0.3
                current_combined = weight_ms * avg_losses['val_ms'] + weight_lin * avg_losses['val_lin']
                
                print(f"{ep:<8} {avg_losses['ms']:<12.6f} {avg_losses['lin']:<12.6f} "
                      f"{avg_losses['val_ms']:<12.6f} {avg_losses['val_lin']:<12.6f} "
                      f"{current_combined:<12.6f} {status:<22}")
            else:
                print(f"{ep:<8} {avg_losses['ms']:<12.6f} {avg_losses['lin']:<12.6f} "
                      f"{avg_losses['val_ms']:<12.6f} {avg_losses['val_lin']:<12.6f}")
        else:
            print(f"{ep:<8} {avg_losses['total']:<12.6f} {avg_losses['pred']:<12.6f} "
                  f"{avg_losses['ms']:<12.6f} {avg_losses['balance']:<12.6f}")
    
    print("="*110 + "\n")
    return log


def evaluate_model_moe(model, test_traj, device, n_steps=100):
    """
    Evaluate MoE model by predicting forward in time
    
    Args:
        model: trained KoopmanMoE model
        test_traj: test trajectory array of shape (n_steps, n_x)
        device: device to run on
        n_steps: number of steps to predict
    
    Returns:
        true: true trajectory
        preds: predicted trajectory
        weights: expert weights over time
        metrics: dict with errors at different horizons
    """
    model.eval()
    
    with torch.no_grad():
        x0 = torch.tensor(test_traj[0], dtype=torch.float32).unsqueeze(0).to(device)
        preds, weights = model.predict(x0, n_steps=n_steps)
        
        preds = preds.squeeze(1).cpu().numpy()  # Remove batch dim
        weights = weights.squeeze(1).cpu().numpy()  # Remove batch dim
        true = test_traj[:n_steps+1]
        
        # Compute errors at training horizons and beyond
        metrics = {}
        
        # 1-step error (what we trained on)
        if len(true) > 1:
            metrics['error_1step'] = np.mean((preds[1] - true[1])**2)
        
        # 8-step error (what we trained on)
        if len(true) > 8:
            metrics['error_8step'] = np.mean((preds[8] - true[8])**2)
        
        # Longer horizons (generalization)
        for horizon in [20, 50, 100]:
            if len(true) > horizon:
                metrics[f'error_{horizon}step'] = np.mean((preds[horizon] - true[horizon])**2)
        
        # Overall MSE across all steps
        metrics['error_overall'] = np.mean((preds - true)**2)
    
    return true, preds, weights, metrics


def visualize_expert_usage(weights, save_path='expert_usage.png'):
    """
    Visualize which experts are active over time
    
    Args:
        weights: array of shape (n_steps, n_experts)
    """
    n_steps, n_experts = weights.shape
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Stacked area chart
    axes[0].stackplot(
        range(n_steps),
        *[weights[:, i] for i in range(n_experts)],
        labels=[f'Expert {i+1}' for i in range(n_experts)],
        alpha=0.8
    )
    axes[0].set_xlabel('Time Step', fontsize=12)
    axes[0].set_ylabel('Gating Weight', fontsize=12)
    axes[0].set_title('Expert Activation Over Time (Stacked)', fontsize=14)
    axes[0].legend(loc='upper right', ncol=4)
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Individual expert trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, n_experts))
    for i in range(n_experts):
        axes[1].plot(weights[:, i], label=f'Expert {i+1}',
                    linewidth=2, alpha=0.7, color=colors[i])
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_ylabel('Gating Weight', fontsize=12)
    axes[1].set_title('Individual Expert Weights Over Time', fontsize=14)
    axes[1].legend(loc='upper right', ncol=4)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Expert usage visualization saved to {save_path}")
    plt.close()


def main():
    """Main training script for MoE Koopman"""
    # Command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train MoE Koopman on various dynamical systems')
    parser.add_argument('--system', type=str, default='duffing',
                       choices=['duffing', 'vanderpol', 'lorenz', 'double_pendulum'],
                       help='Dynamical system to model')
    parser.add_argument('--n_traj', type=int, default=100,
                       help='Number of trajectories')
    parser.add_argument('--T', type=float, default=10.0,
                       help='Time horizon')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step')
    parser.add_argument('--noise_std', type=float, default=0.0,
                       help='Noise standard deviation')
    parser.add_argument('--n_experts', type=int, default=4,
                       help='Number of experts')
    parser.add_argument('--n_epochs', type=int, default=40,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio (default: 0.1 = 10%%)')
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (epochs)')
    parser.add_argument('--save_prefix', type=str, default='',
                       help='Prefix for saved files')
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"System: {args.system}")
    
    # Generate or load data based on system
    print(f"\nGenerating {args.system} data...")
    from data_simulation import generate_dataset
    
    # System configurations
    system_configs = {
        'duffing': {
            'n_x': 2,
            'name': 'Duffing Oscillator',
            'labels': ['x', 'xdot']
        },
        'vanderpol': {
            'n_x': 2,
            'name': 'Van der Pol Oscillator',
            'labels': ['x', 'xdot']
        },
        'lorenz': {
            'n_x': 3,
            'name': 'Lorenz Attractor',
            'labels': ['x', 'y', 'z']
        },
        'double_pendulum': {
            'n_x': 4,
            'name': 'Double Pendulum',
            'labels': ['theta1', 'theta1_dot', 'theta2', 'theta2_dot']
        }
    }
    
    config = system_configs[args.system]
    n_x = config['n_x']
    n_z = n_x * 10  # Latent dimension = 10× input dimension
    system_name = config['name']
    state_labels = config['labels']
    
    # Generate training + validation data
    t, trajs_train, _ = generate_dataset(
        args.system, 
        n_traj=args.n_traj, 
        T=args.T, 
        dt=args.dt, 
        noise_std=args.noise_std
    )
    
    # Generate separate TEST data (completely unseen)
    t_test, trajs_test, _ = generate_dataset(
        args.system,
        n_traj=10,  # 10 test trajectories
        T=args.T,
        dt=args.dt,
        noise_std=args.noise_std
    )
    
    print(f"Generated {args.n_traj} training trajectories, each with {trajs_train.shape[1]} time steps")
    print(f"Generated 10 test trajectories (unseen data)")
    print(f"State dimension: {n_x}D")
    
    # Prepare training data
    print("\nPreparing training data...")
    state_sequence = prepare_data_from_trajectories(trajs_train, n_steps=8)
    # state_sequence is a list of 9 tensors: [x0, x1, ..., x8]
    
    # Split into train/val
    n_samples = len(state_sequence[0])
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val
    
    # Split each tensor
    train_states = [s[:n_train] for s in state_sequence]
    val_states = [s[n_train:] for s in state_sequence]
    
    train_loader = DataLoader(
        TensorDataset(*train_states),
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = None
    if n_val > 0:
        val_loader = DataLoader(
            TensorDataset(*val_states),
            batch_size=args.batch_size,
            shuffle=False
        )
        print(f"Training samples: {n_train}, Validation samples: {n_val}")
    else:
        print(f"Training samples: {n_train} (no validation split)")
    
    print(f"Training batches: {len(train_loader)}")
    
    # Create MoE model
    n_experts = args.n_experts
    
    model = KoopmanMoE(n_x=n_x, n_z=n_z, n_experts=n_experts).to(device)
    print(f"\nMoE Model created for {system_name}")
    print(f"  State dimension: n_x={n_x}")
    print(f"  Latent dimension: n_z={n_z}")
    print(f"  Number of experts: n_experts={n_experts}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {n_params:,}")
    
    # Train model
    print("\nStarting training...")
    if args.early_stopping:
        print(f"Early stopping enabled (patience={args.patience})")
    
    log = train_model_moe(
        model=model,
        train_loader=train_loader,
        device=device,
        n_epochs=args.n_epochs,
        val_loader=val_loader,
        early_stopping=args.early_stopping,
        patience=args.patience
    )
    
    # Evaluate on UNSEEN test trajectory
    print("\nEvaluating model on test data...")
    test_traj = trajs_test[0]  # Use first TEST trajectory (completely unseen)
    true, preds, weights, metrics = evaluate_model_moe(model, test_traj, device, n_steps=100)
    
    # Print evaluation metrics
    print("\n=== Test Set Performance ===")
    print(f"1-step MSE:   {metrics.get('error_1step', 0):.6f}  (trained on this)")
    print(f"8-step MSE:   {metrics.get('error_8step', 0):.6f}  (trained on this)")
    print(f"20-step MSE:  {metrics.get('error_20step', 0):.6f}  (generalization)")
    print(f"50-step MSE:  {metrics.get('error_50step', 0):.6f}  (generalization)")
    print(f"100-step MSE: {metrics.get('error_100step', 0):.6f} (generalization)")
    print(f"Overall MSE:  {metrics.get('error_overall', 0):.6f}")
    print("="*30)
    
    # Compute prediction error at each timestep
    prediction_errors = np.sqrt(np.sum((preds - true)**2, axis=1))  # Euclidean distance
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    # Plot 1: Phase space (system-dependent)
    if n_x == 2:
        # 2D systems: standard phase plot
        axes[0, 0].plot(true[:, 0], true[:, 1], '-o', label='True',
                       markersize=3, alpha=0.7, linewidth=2)
        axes[0, 0].plot(preds[:, 0], preds[:, 1], '-x', label='MoE Prediction',
                       markersize=3, alpha=0.7, linewidth=2)
        axes[0, 0].set_xlabel(state_labels[0], fontsize=12)
        axes[0, 0].set_ylabel(state_labels[1], fontsize=12)
        axes[0, 0].set_title(f'{system_name}: Phase Space', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    elif n_x == 3:
        # 3D systems: project to 2D (x-y plane)
        axes[0, 0].plot(true[:, 0], true[:, 1], '-o', label='True',
                       markersize=3, alpha=0.7, linewidth=2)
        axes[0, 0].plot(preds[:, 0], preds[:, 1], '-x', label='MoE Prediction',
                       markersize=3, alpha=0.7, linewidth=2)
        axes[0, 0].set_xlabel(f'{state_labels[0]}', fontsize=12)
        axes[0, 0].set_ylabel(f'{state_labels[1]}', fontsize=12)
        axes[0, 0].set_title(f'{system_name}: {state_labels[0]}-{state_labels[1]} Projection', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    elif n_x == 4:
        # 4D systems: plot first angle vs second angle
        axes[0, 0].plot(true[:, 0], true[:, 2], '-o', label='True',
                       markersize=3, alpha=0.7, linewidth=2)
        axes[0, 0].plot(preds[:, 0], preds[:, 2], '-x', label='MoE Prediction',
                       markersize=3, alpha=0.7, linewidth=2)
        axes[0, 0].set_xlabel(state_labels[0], fontsize=12)
        axes[0, 0].set_ylabel(state_labels[2], fontsize=12)
        axes[0, 0].set_title(f'{system_name}: Configuration Space', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Time series (first state variable)
    time_axis = np.arange(len(true)) * args.dt
    axes[0, 1].plot(time_axis, true[:, 0], '-o', label=f'True {state_labels[0]}',
                   markersize=2, alpha=0.7, linewidth=2)
    axes[0, 1].plot(time_axis, preds[:, 0], '-x', label=f'Pred {state_labels[0]}',
                   markersize=2, alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('Time', fontsize=12)
    axes[0, 1].set_ylabel(state_labels[0], fontsize=12)
    axes[0, 1].set_title(f'Time Series: {state_labels[0]}', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Prediction error over time
    time_axis_pred = np.arange(len(prediction_errors)) * args.dt
    axes[0, 2].plot(time_axis_pred, prediction_errors, linewidth=2, color='red')
    axes[0, 2].axvline(args.dt * 1, color='green', linestyle='--', alpha=0.5, label='1-step (trained)')
    axes[0, 2].axvline(args.dt * 8, color='blue', linestyle='--', alpha=0.5, label='8-step (trained)')
    axes[0, 2].set_xlabel('Time', fontsize=12)
    axes[0, 2].set_ylabel('Prediction Error (Euclidean)', fontsize=12)
    axes[0, 2].set_title('Error Growth Over Time', fontsize=14)
    axes[0, 2].legend()
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Loss curves
    epochs = range(len(log))
    axes[1, 0].plot(epochs, [l['total'] for l in log], label='Total', linewidth=2)
    axes[1, 0].plot(epochs, [l['pred'] for l in log], label='1-step Pred', linewidth=2)
    axes[1, 0].plot(epochs, [l['ms'] for l in log], label='8-step MS', linewidth=2, linestyle='--')
    axes[1, 0].plot(epochs, [l['balance'] for l in log], label='Load Balance', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Training Loss (log scale)', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Average expert usage
    avg_expert_usage = weights.mean(axis=0)
    axes[1, 1].bar(range(n_experts), avg_expert_usage)
    axes[1, 1].axhline(1.0/n_experts, color='r', linestyle='--',
                      label=f'Equal ({1.0/n_experts:.3f})')
    axes[1, 1].set_xlabel('Expert ID', fontsize=12)
    axes[1, 1].set_ylabel('Average Weight', fontsize=12)
    axes[1, 1].set_title('Average Expert Usage', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Error at specific horizons
    horizons = [1, 8, 20, 50, 100]
    horizon_errors = [metrics.get(f'error_{h}step', np.nan) for h in horizons]
    axes[1, 2].bar(range(len(horizons)), horizon_errors, color=['green', 'blue', 'orange', 'orange', 'red'])
    axes[1, 2].set_xticks(range(len(horizons)))
    axes[1, 2].set_xticklabels([f'{h}' for h in horizons])
    axes[1, 2].set_xlabel('Prediction Horizon (steps)', fontsize=12)
    axes[1, 2].set_ylabel('MSE', fontsize=12)
    axes[1, 2].set_title('Error at Different Horizons', fontsize=14)
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    results_file = f'{args.save_prefix}{args.system}_moe_results.png'
    plt.savefig(results_file, dpi=150, bbox_inches='tight')
    print(f"Results saved to '{results_file}'")
    plt.close()  # Close figure to free memory
    
    # Visualize expert usage over time
    expert_usage_file = f'{args.save_prefix}{args.system}_expert_usage.png'
    visualize_expert_usage(weights, save_path=expert_usage_file)
    
    # Save model
    model_file = f'{args.save_prefix}{args.system}_moe_model.pth'
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to '{model_file}'")
    
    # Print expert usage statistics
    print("\n=== Expert Usage Statistics ===")
    for i in range(n_experts):
        avg_weight = weights[:, i].mean()
        max_weight = weights[:, i].max()
        min_weight = weights[:, i].min()
        std_weight = weights[:, i].std()
        print(f"Expert {i+1}: avg={avg_weight:.3f}, max={max_weight:.3f}, "
              f"min={min_weight:.3f}, std={std_weight:.3f}")


if __name__ == "__main__":
    main()

