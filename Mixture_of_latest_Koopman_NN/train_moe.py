"""
Training script for Mixture of Experts Koopman Autoencoder

Features:
- Load balancing loss (ensure all experts are used)
- Per-expert reconstruction, linearity, and multi-step losses
- Bidirectional constraint per expert
- Spectral radius penalty per expert
- Hankel-based linearity constraint (HAVOK)
- Sparsity regularization on encoder/decoder weights
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from koopman_moe_neural_network import (
    KoopmanMoE,
    spectral_radius_penalty,
    compute_hankel_linearity_loss
)
from evaluation import (
    one_step_mse,
    multi_step_nrmse,
    chamfer_distance_phase,
    spectral_radius,
    long_horizon_divergence_rate,
    reconstruction_error
)


# ==============================================================================
# Training and Evaluation Horizons
# ==============================================================================
# TRAINING: Dense horizons (1-100) for proper Koopman linearity enforcement
TRAINING_HORIZONS = list(range(1, 101))  # [1, 2, 3, ..., 100]

# EVALUATION: Sparse horizons including extrapolation beyond training
EVAL_HORIZONS = [1, 10, 50, 100, 500, 1000]


def prepare_data_from_trajectories(trajs, hankel_seq_len=16):
    """
    Convert trajectory data to training tuples for multi-step linearity
    
    Args:
        trajs: array of shape (n_traj, n_timesteps, n_x)
        hankel_seq_len: sequence length for Hankel loss computation
    
    Returns:
        dict with 'x0', 'x1', 'x_k' for k in [10, 20, ..., 50], and 'sequences' for Hankel
    """
    n_traj, n_timesteps, n_x = trajs.shape
    
    # Dense horizons for proper Koopman linearity enforcement
    horizons = TRAINING_HORIZONS  # [1, 2, 3, ..., 100]
    max_horizon = max(horizons)
    
    # Initialize lists for each horizon
    data_lists = {h: [] for h in horizons}
    x0_list = []
    
    for traj in trajs:
        if n_timesteps > max_horizon:
            # x_t: from start to -(max_horizon)
            x0_list.append(traj[:-max_horizon])
            
            # x_{t+k} for each horizon
            for h in horizons:
                if h == max_horizon:
                    data_lists[h].append(traj[h:])
                else:
                    data_lists[h].append(traj[h:-(max_horizon-h)])
    
    # Concatenate and convert to tensors
    result = {
        'x0': torch.tensor(np.concatenate(x0_list, axis=0), dtype=torch.float32)
    }
    
    for h in horizons:
        result[f'x{h}'] = torch.tensor(np.concatenate(data_lists[h], axis=0), dtype=torch.float32)
    
    # Also prepare sequence data for Hankel loss
    # Extract contiguous sequences of length hankel_seq_len from each trajectory
    sequences = []
    for traj in trajs:
        if n_timesteps >= hankel_seq_len:
            n_seqs = n_timesteps - hankel_seq_len + 1
            for start in range(0, n_seqs, hankel_seq_len // 2):  # 50% overlap
                seq = traj[start:start + hankel_seq_len]
                if len(seq) == hankel_seq_len:
                    sequences.append(seq)
    
    if sequences:
        result['sequences'] = torch.tensor(np.stack(sequences, axis=0), dtype=torch.float32)
    
    return result


def sample_sequence_batch(sequences, batch_size):
    """
    Sample a batch of sequences for Hankel loss computation
    
    Args:
        sequences: tensor of shape (N, T, n_x)
        batch_size: number of sequences to sample
    
    Returns:
        batch of sequences (batch_size, T, n_x)
    """
    n_seqs = sequences.shape[0]
    indices = np.random.choice(n_seqs, size=min(batch_size, n_seqs), replace=False)
    return sequences[indices]


def compute_loss_moe(model, data_batch, device, sequences=None):
    """
    Compute loss for MoKE with TRUE Linear Dynamics (IC Gating).
    
    KEY: Initial-Condition (IC) Gating
    - K_eff = Σ π_k(z_0) · K_k  is computed ONCE from z_0
    - Then: z_t = K_eff^t · z_0  (true linear dynamics!)
    
    This preserves Koopman structure for multi-step prediction.
    
    Args:
        model: KoopmanMoE model
        data_batch: dict with 'x0', 'x1', 'x10', 'x20', ..., 'x100'
        device: device to run on
        sequences: optional tensor (batch, T, n_x) for Hankel loss
    """
    # Hyperparameters
    lam_rec = 2.0       # 1. Reconstruction
    lam_pred = 15.0     # 2. 1-step prediction (PRIMARY)
    lam_lin = 12.0      # 3. Multi-step linearity (KOOPMAN CORE)
    lam_balance = 1.0   # 4. Load balancing
    lam_bi = 1.0        # 5. Bidirectional
    lam_spec = 5.0      # 6. Spectral radius
    lam_hankel = 1.0    # 7. Hankel linearity
    lam_sparse = 1e-4   # 8. Sparsity
    
    # Extract data
    x0 = data_batch['x0']
    x1 = data_batch['x1']
    n_experts = model.n_experts
    
    mse = nn.MSELoss()
    
    # === Encode initial state ===
    z0 = model.encoder(x0)
    
    # === Compute K_eff ONCE from z_0 (IC Gating!) ===
    K_eff, pi = model.compute_effective_K(z0)
    # K_eff: (batch, n_z, n_z) - FIXED for this sample's trajectory
    # pi: (batch, n_experts) - gating weights
    
    # === 1. Reconstruction Loss ===
    x_rec = model.decoder(z0)
    loss_rec = mse(x_rec, x0)
    
    # === 2. Prediction Loss (1-step with K_eff) ===
    # z1 = K_eff @ z0 (batched matrix-vector multiply)
    z1_pred = torch.bmm(K_eff, z0.unsqueeze(-1)).squeeze(-1)
    x1_pred = model.decoder(z1_pred)
    loss_pred = mse(x1_pred, x1)
    
    # === 3. Multi-Step Latent Linearity with FIXED K_eff ===
    # z_h = K_eff^h @ z_0 (true Koopman!)
    horizons = TRAINING_HORIZONS  # Dense: [1, 2, 3, ..., 100]
    
    loss_lin = 0
    count = 0
    
    # Pre-compute K_eff^h using matrix powers
    K_eff_powers = {1: K_eff}
    K_power = K_eff.clone()
    prev_h = 1
    for h in horizons[1:]:
        for _ in range(h - prev_h):
            K_power = torch.bmm(K_power, K_eff)
        K_eff_powers[h] = K_power.clone()
        prev_h = h
    
    for h in horizons:
        if f'x{h}' not in data_batch:
            continue
        x_h = data_batch[f'x{h}']
        zh_true = model.encoder(x_h)
        
        # z_h = K_eff^h @ z_0 (using pre-computed powers)
        zh_pred = torch.bmm(K_eff_powers[h], z0.unsqueeze(-1)).squeeze(-1)
        
        loss_lin += mse(zh_pred, zh_true)
        count += 1
    
    loss_lin = loss_lin / max(count, 1)
    
    # === 4. Load Balancing Loss ===
    avg_pi = pi.mean(dim=0)
    target_weight = 1.0 / n_experts
    loss_balance = ((avg_pi - target_weight)**2).sum()
    
    # === 5. Bidirectional Constraint ===
    loss_bi = 0
    I = torch.eye(model.n_z, device=device)
    for k in range(n_experts):
        loss_bi += (model.K[k] @ model.K_b[k] - I).norm()**2
        loss_bi += (model.K_b[k] @ model.K[k] - I).norm()**2
    loss_bi /= n_experts
    
    # === 6. Spectral Radius Penalty ===
    loss_spec = 0
    for k in range(n_experts):
        loss_spec += spectral_radius_penalty(model.K[k], iters=8, target=1.005, lower=0.995)
    loss_spec /= n_experts
    
    # === 7. Hankel-Based Linearity Loss ===
    loss_hankel = torch.tensor(0.0, device=device)
    if sequences is not None and len(sequences) > 0:
        loss_hankel = model.hankel_linearity_loss(sequences, L=4, r=8, device=device)
    
    # === 8. Sparsity Regularization ===
    loss_sparse = model.sparsity_loss(mode="l1")
    
    # === Total Loss ===
    loss_total = (
        lam_rec * loss_rec +
        lam_pred * loss_pred +
        lam_lin * loss_lin +
        lam_balance * loss_balance +
        lam_bi * loss_bi +
        lam_spec * loss_spec +
        lam_hankel * loss_hankel +
        lam_sparse * loss_sparse
    )
    
    return {
        'total': loss_total,
        'rec': loss_rec,
        'pred': loss_pred,
        'lin': loss_lin,
        'balance': loss_balance,
        'bi': loss_bi,
        'spec': loss_spec,
        'hankel': loss_hankel,
        'sparse': loss_sparse
    }


def train_model_moe(model, train_loader, device, n_epochs=40, val_loader=None, 
                    early_stopping=False, patience=20, checkpoint_path=None,
                    train_sequences=None, val_sequences=None, hankel_batch_size=32):
    """
    Train the MoE Koopman model
    
    Args:
        model: KoopmanMoE instance
        train_loader: DataLoader with state sequences [x0, x1, ..., x50]
        device: device to train on
        n_epochs: number of training epochs
        val_loader: optional validation DataLoader
        early_stopping: whether to use early stopping
        patience: number of epochs to wait for improvement
        checkpoint_path: path to save best model checkpoint (optional)
        train_sequences: tensor of shape (N, T, n_x) for Hankel loss during training
        val_sequences: tensor of shape (N, T, n_x) for Hankel loss during validation
        hankel_batch_size: batch size for Hankel sequence sampling
    
    Returns:
        log: list of dicts with loss history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    log = []
    # Early stopping based on total validation loss
    best_val_total = float('inf')     # Best total validation loss
    patience_counter = 0
    best_model_state = None
    
    # Print header
    print("\n" + "="*110)
    print("🎯 TRAINING WITH MULTI-STEP LINEARITY (1, 10, 20, 30, 40, 50 steps)")
    print("   Weights: uniform (all 1.0) - equal importance for all horizons")
    print("="*110)
    if val_loader is not None:
        if early_stopping:
            print(f"{'Epoch':<8} {'Train Total':<12} {'Train Pred':<12} "
                  f"{'Val Total':<12} {'Val Pred':<12} {'Status':<22}")
        else:
            print(f"{'Epoch':<8} {'Train Total':<12} {'Train Pred':<12} "
                  f"{'Val Total':<12} {'Val Pred':<12}")
    else:
        print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Pred':<12} {'Train Lin':<12} {'Balance':<12}")
    print("="*110)
    
    for ep in range(n_epochs):
        # Debug prints at epoch 0 to understand initial state
        if ep == 0:
            model.eval()
            with torch.no_grad():
                # Get a sample batch for analysis
                sample_batch = next(iter(train_loader))
                x0_sample = sample_batch[0][:32].to(device)  # First 32 samples
                x50_sample = sample_batch[6][:32].to(device)  # x50
                
                print("\n" + "="*80)
                print("🔍 DEBUG: Epoch 0 - MoKE with IC Gating (True Linear Dynamics)")
                print("="*80)
                
                # Input data statistics
                print(f"\n📊 Input Data (x0):")
                print(f"   Shape: {x0_sample.shape}")
                print(f"   Mean: {x0_sample.mean().item():.6f}, Std: {x0_sample.std().item():.6f}")
                print(f"   Min: {x0_sample.min().item():.6f}, Max: {x0_sample.max().item():.6f}")
                
                print(f"\n📊 Input Data (x50):")
                print(f"   Mean: {x50_sample.mean().item():.6f}, Std: {x50_sample.std().item():.6f}")
                print(f"   |x50 - x0| mean: {(x50_sample - x0_sample).abs().mean().item():.6f}")
                
                # Encode and compute K_eff
                z0 = model.encoder(x0_sample)
                z50_true = model.encoder(x50_sample)
                K_eff, pi = model.compute_effective_K(z0)
                
                print(f"\n🔧 Shared Encoder/Decoder:")
                print(f"   Latent z0:  mean={z0.mean().item():.4f}, std={z0.std().item():.4f}")
                print(f"   Latent z50: mean={z50_true.mean().item():.4f}, std={z50_true.std().item():.4f}")
                
                print(f"\n🎯 IC Gating (computed from z_0, FIXED for trajectory):")
                print(f"   Mean weights: {pi.mean(dim=0).cpu().numpy()}")
                
                # K_eff analysis
                print(f"\n📐 Effective K_eff = Σ π_k K_k:")
                K_eff_mean = K_eff.mean(dim=0)
                eigvals_eff = torch.linalg.eigvals(K_eff_mean)
                spec_radius_eff = eigvals_eff.abs().max().item()
                print(f"   K_eff spectral radius (mean): {spec_radius_eff:.4f}")
                
                # K_eff^50 prediction (TRUE Koopman!)
                K_power = K_eff.clone()
                for _ in range(49):
                    K_power = torch.bmm(K_power, K_eff)
                z50_pred = torch.bmm(K_power, z0.unsqueeze(-1)).squeeze(-1)
                print(f"   z50 = K_eff^50 @ z0 error: {(z50_pred - z50_true).abs().mean().item():.6f}")
                
                # Per-Koopman operator analysis
                print(f"\n🧠 Individual Koopman Operators:")
                for i in range(model.n_experts):
                    K = model.K[i]
                    eigvals = torch.linalg.eigvals(K)
                    spec_radius = eigvals.abs().max().item()
                    print(f"   K_{i+1}: spectral radius = {spec_radius:.4f}")
                
                print("\n" + "="*80 + "\n")
            model.train()
        
        # Training
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            # batch is a tuple of tensors: (x0, x1, x2, ..., x100)
            horizons = TRAINING_HORIZONS  # Dense: [1, 2, 3, ..., 100]
            data_batch = {'x0': batch[0].to(device)}
            for idx, h in enumerate(horizons):
                data_batch[f'x{h}'] = batch[idx + 1].to(device)
            
            # Sample sequences for Hankel loss (if available)
            seq_batch = None
            if train_sequences is not None and len(train_sequences) > 0:
                seq_batch = sample_sequence_batch(train_sequences, hankel_batch_size).to(device)
            
            # Compute losses
            losses = compute_loss_moe(model, data_batch, device, sequences=seq_batch)
            
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
                    horizons = TRAINING_HORIZONS  # Dense: [1, 2, 3, ..., 100]
                    data_batch = {'x0': batch[0].to(device)}
                    for idx, h in enumerate(horizons):
                        data_batch[f'x{h}'] = batch[idx + 1].to(device)
                    
                    # Sample sequences for Hankel loss (if available)
                    seq_batch = None
                    if val_sequences is not None and len(val_sequences) > 0:
                        seq_batch = sample_sequence_batch(val_sequences, hankel_batch_size).to(device)
                    
                    losses = compute_loss_moe(model, data_batch, device, sequences=seq_batch)
                    val_losses.append({k: v.item() for k, v in losses.items()})
            
            # Average validation losses
            avg_val_losses = {}
            for key in val_losses[0].keys():
                avg_val_losses[key] = np.mean([l[key] for l in val_losses])
            
            avg_losses['val_total'] = avg_val_losses['total']
            avg_losses['val_pred'] = avg_val_losses['pred']
            avg_losses['val_lin'] = avg_val_losses['lin']
            
            # Early stopping based on total validation loss
            if early_stopping:
                current_val_total = avg_val_losses['total']
                
                # Check if this is the best model
                if current_val_total < best_val_total:
                    best_val_total = current_val_total
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    status = "✓ Best"
                    
                    # Save best checkpoint to disk immediately
                    if checkpoint_path:
                        torch.save(model.state_dict(), checkpoint_path)
                else:
                    patience_counter += 1
                    status = f"Wait {patience_counter}/{patience}"
                    
                    if patience_counter >= patience:
                        print(f"{'':8} {'':12} {'':12} {'':12} {'':12} {'Early Stop!':<22}")
                        print("="*110)
                        print(f"Training stopped early at epoch {ep}")
                        print(f"Best validation total loss: {best_val_total:.6f}")
                        print("="*110)
                        # Restore best model
                        model.load_state_dict(best_model_state)
                        log.append(avg_losses)
                        break
        
        log.append(avg_losses)
        
        # Print progress every epoch
        if val_loader is not None:
            if early_stopping:
                print(f"{ep:<8} {avg_losses['total']:<12.6f} {avg_losses['pred']:<12.6f} "
                      f"{avg_losses['val_total']:<12.6f} {avg_losses['val_pred']:<12.6f} "
                      f"{status:<22}")
            else:
                print(f"{ep:<8} {avg_losses['total']:<12.6f} {avg_losses['pred']:<12.6f} "
                      f"{avg_losses['val_total']:<12.6f} {avg_losses['val_pred']:<12.6f}")
        else:
            print(f"{ep:<8} {avg_losses['total']:<12.6f} {avg_losses['pred']:<12.6f} "
                  f"{avg_losses['lin']:<12.6f} {avg_losses['balance']:<12.6f}")
    
    print("="*110 + "\n")
    return log


def evaluate_model_moe(model, test_traj, device, n_steps=100):
    """
    Evaluate MoKE model by predicting forward in time
    
    Args:
        model: trained KoopmanMoE model
        test_traj: test trajectory array of shape (n_steps, n_x)
        device: device to run on
        n_steps: number of steps to predict
    
    Returns:
        true: true trajectory
        preds: predicted trajectory
        weights: expert weights over time
        x_rec: reconstructed initial condition
    """
    model.eval()
    
    with torch.no_grad():
        x0 = torch.tensor(test_traj[0], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Get reconstruction of initial condition
        out = model(x0)
        x_rec = out['x_rec'].cpu().numpy()
        
        # Get predictions using the new architecture
        preds, weights = model.predict(x0, n_steps=n_steps)
        
        preds = preds.squeeze(1).cpu().numpy()  # Remove batch dim
        weights = weights.squeeze(1).cpu().numpy()  # Remove batch dim
        true = test_traj[:n_steps+1]
    
    return true, preds, weights, x_rec


def compute_all_metrics_moe(model, true, preds, x_rec, device):
    """
    Compute all evaluation metrics from evaluation.py for MoKE model
    
    Args:
        model: trained KoopmanMoE model
        true: true trajectory (n_steps+1, n_x)
        preds: predicted trajectory (n_steps+1, n_x)
        x_rec: reconstructed initial condition (1, n_x)
        device: device
    
    Returns:
        dict of all metrics
    """
    metrics = {}
    
    # Reshape for evaluation functions: (1, n_steps+1, n_x)
    true_3d = true[np.newaxis, :, :]
    preds_3d = preds[np.newaxis, :, :]
    
    # 1-step MSE
    metrics['one_step_mse'] = one_step_mse(true_3d[:, :2, :], preds_3d[:, :2, :])
    
    # Multi-step NRMSE at various horizons
    n_steps = true.shape[0] - 1
    horizons = [h for h in [1, 10, 20, 50, 100] if h < n_steps]
    nrmse_dict = multi_step_nrmse(true_3d, preds_3d, horizons)
    for h, val in nrmse_dict.items():
        metrics[f'nrmse_{h}step'] = val
    
    # Chamfer distance in phase space (use first 2 dims)
    n_x = true.shape[1]
    dims = (0, 1) if n_x >= 2 else (0,)
    if len(dims) == 2:
        metrics['chamfer_distance'] = chamfer_distance_phase(true_3d, preds_3d, dims=dims)
    
    # Spectral radius of each Koopman operator (report max)
    max_rho = 0.0
    for i in range(model.n_experts):
        K = model.K[i].detach().cpu().numpy()
        rho, _ = spectral_radius(K)
        max_rho = max(max_rho, rho)
    metrics['spectral_radius_max'] = max_rho
    
    # Long-horizon divergence rate
    slope, _ = long_horizon_divergence_rate(true_3d, preds_3d)
    metrics['divergence_rate'] = slope
    
    # Reconstruction error (initial condition)
    metrics['reconstruction_error'] = reconstruction_error(true[0:1], x_rec)
    
    return metrics


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
    parser.add_argument('--inference_only', action='store_true',
                       help='Skip training, load model and generate plots only')
    parser.add_argument('--model_path', type=str, default='',
                       help='Path to pre-trained model weights (.pth file)')
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
    n_z = n_x * 5  # Latent dimension = 5× input dimension
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
    
    # Prepare training data with multi-step horizons and Hankel sequences
    print("\nPreparing training data with multi-step linearity horizons...")
    data_dict = prepare_data_from_trajectories(trajs_train, hankel_seq_len=16)
    
    # Dense horizons for proper Koopman linearity enforcement
    horizons = TRAINING_HORIZONS  # [1, 2, 3, ..., 100]
    print(f"  Linearity horizons: {len(horizons)} steps (dense 1-100)")
    print(f"  Horizon weights: uniform (all 1.0)")
    
    # Hankel sequences for HAVOK constraint
    train_sequences = data_dict.get('sequences', None)
    if train_sequences is not None:
        print(f"  Hankel sequences: {train_sequences.shape[0]} sequences of length {train_sequences.shape[1]}")
    
    # Split into train/val
    n_samples = len(data_dict['x0'])
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val
    
    # Create tensor lists for DataLoader
    train_tensors = [data_dict['x0'][:n_train]]
    val_tensors = [data_dict['x0'][n_train:]]
    
    for h in horizons:
        train_tensors.append(data_dict[f'x{h}'][:n_train])
        val_tensors.append(data_dict[f'x{h}'][n_train:])
    
    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = None
    if n_val > 0:
        val_loader = DataLoader(
            TensorDataset(*val_tensors),
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
    
    # Check for inference-only mode
    if args.inference_only:
        # Load pre-trained model
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = f'{args.save_prefix}{args.system}_moe_model.pth'
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            print("Please train the model first or provide a valid --model_path")
            return
        
        print(f"\n=== INFERENCE ONLY MODE ===")
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
        
        # Create dummy log (no training curves to show)
        log = None
    else:
        # Train model
        print("\nStarting training...")
        if args.early_stopping:
            print(f"Early stopping enabled (patience={args.patience})")
        
        # Define model save path for checkpointing
        model_file = f'{args.save_prefix}{args.system}_moe_model.pth'
        
        log = train_model_moe(
            model=model,
            train_loader=train_loader,
            device=device,
            n_epochs=args.n_epochs,
            val_loader=val_loader,
            early_stopping=args.early_stopping,
            patience=args.patience,
            checkpoint_path=model_file,
            train_sequences=train_sequences,
            val_sequences=train_sequences,  # Use same sequences for val (or could split)
            hankel_batch_size=32
        )
    
    # Evaluate on ALL test trajectories
    n_test = len(trajs_test)
    print(f"\nEvaluating model on {n_test} test trajectories...")
    
    all_results = []
    all_metrics = []
    all_weights = []
    all_x_rec = []
    
    for i, test_traj in enumerate(trajs_test):
        true, preds, weights, x_rec = evaluate_model_moe(model, test_traj, device, n_steps=100)
        all_results.append({'true': true, 'preds': preds})
        all_x_rec.append(x_rec)
        all_weights.append(weights)
        
        # Compute metrics for this trajectory
        metrics = compute_all_metrics_moe(model, true, preds, x_rec, device)
        all_metrics.append(metrics)
    
    # Compute average metrics across all test trajectories
    avg_metrics = {}
    std_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        std_metrics[key] = np.std([m[key] for m in all_metrics])
    
    # Print evaluation metrics (from evaluation.py)
    print("\n" + "="*70)
    print(f"EVALUATION METRICS (from evaluation.py, averaged over {n_test} trajectories)")
    print("="*70)
    print(f"  1-step MSE:           {avg_metrics['one_step_mse']:.6f} ± {std_metrics['one_step_mse']:.6f}")
    print(f"  Reconstruction Error: {avg_metrics['reconstruction_error']:.6f} ± {std_metrics['reconstruction_error']:.6f}")
    print(f"  Spectral Radius (max):{avg_metrics['spectral_radius_max']:.4f}")
    print(f"  Divergence Rate:      {avg_metrics['divergence_rate']:.6f} ± {std_metrics['divergence_rate']:.6f}")
    if 'chamfer_distance' in avg_metrics:
        print(f"  Chamfer Distance:     {avg_metrics['chamfer_distance']:.6f} ± {std_metrics['chamfer_distance']:.6f}")
    print("\n  Multi-step NRMSE:")
    for key in sorted(avg_metrics.keys()):
        if key.startswith('nrmse_'):
            horizon = key.replace('nrmse_', '').replace('step', '')
            print(f"    Horizon {horizon:>3}: {avg_metrics[key]:.6f} ± {std_metrics[key]:.6f}")
    print("="*70)
    
    # Colors for different trajectories
    traj_colors = plt.cm.tab10(np.linspace(0, 1, n_test))
    
    # Visualize results - dynamic layout based on state dimension
    # Row 1: All state variables vs time (all trajectories overlaid)
    # Row 2: Phase space, Loss curves, Expert usage, Error at horizons
    n_cols = max(n_x, 4)  # At least 4 columns for row 2
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
    
    time_axis = np.arange(len(all_results[0]['true'])) * args.dt
    
    # Row 1: Time series for ALL state variables (all trajectories)
    for i in range(n_x):
        for j, res in enumerate(all_results):
            true_j = res['true']
            preds_j = res['preds']
            alpha = 0.6 if j > 0 else 1.0
            label_true = 'True' if j == 0 else None
            label_pred = 'Predicted' if j == 0 else None
            axes[0, i].plot(time_axis, true_j[:, i], '-', color=traj_colors[j],
                           linewidth=1.5, alpha=alpha, label=label_true)
            axes[0, i].plot(time_axis, preds_j[:, i], '--', color=traj_colors[j],
                           linewidth=1.5, alpha=alpha, label=label_pred)
        axes[0, i].set_xlabel('Time', fontsize=12)
        axes[0, i].set_ylabel(state_labels[i], fontsize=12)
        axes[0, i].set_title(f'{state_labels[i]} vs Time ({n_test} trajs)', fontsize=14)
        if i == 0:
            axes[0, i].legend(['True', 'Pred'], loc='upper right')
        axes[0, i].grid(True, alpha=0.3)
    
    # Hide unused subplots in row 1
    for i in range(n_x, n_cols):
        axes[0, i].axis('off')
    
    # Row 2, Col 0: Phase space (all trajectories)
    for j, res in enumerate(all_results):
        true_j = res['true']
        preds_j = res['preds']
        alpha = 0.6 if j > 0 else 1.0
        if n_x == 2:
            axes[1, 0].plot(true_j[:, 0], true_j[:, 1], '-', color=traj_colors[j],
                           linewidth=1.5, alpha=alpha)
            axes[1, 0].plot(preds_j[:, 0], preds_j[:, 1], '--', color=traj_colors[j],
                           linewidth=1.5, alpha=alpha)
            axes[1, 0].plot(true_j[0, 0], true_j[0, 1], 'o', color=traj_colors[j], markersize=6)
        elif n_x == 3:
            axes[1, 0].plot(true_j[:, 0], true_j[:, 2], '-', color=traj_colors[j],
                           linewidth=1.5, alpha=alpha)
            axes[1, 0].plot(preds_j[:, 0], preds_j[:, 2], '--', color=traj_colors[j],
                           linewidth=1.5, alpha=alpha)
            axes[1, 0].plot(true_j[0, 0], true_j[0, 2], 'o', color=traj_colors[j], markersize=6)
        elif n_x == 4:
            axes[1, 0].plot(true_j[:, 0], true_j[:, 2], '-', color=traj_colors[j],
                           linewidth=1.5, alpha=alpha)
            axes[1, 0].plot(preds_j[:, 0], preds_j[:, 2], '--', color=traj_colors[j],
                           linewidth=1.5, alpha=alpha)
            axes[1, 0].plot(true_j[0, 0], true_j[0, 2], 'o', color=traj_colors[j], markersize=6)
    
    if n_x == 2:
        axes[1, 0].set_xlabel(state_labels[0], fontsize=12)
        axes[1, 0].set_ylabel(state_labels[1], fontsize=12)
        axes[1, 0].set_title(f'Phase Space ({n_test} trajs)', fontsize=14)
    elif n_x == 3:
        axes[1, 0].set_xlabel(state_labels[0], fontsize=12)
        axes[1, 0].set_ylabel(state_labels[2], fontsize=12)
        axes[1, 0].set_title(f'Phase ({state_labels[0]} vs {state_labels[2]}, {n_test} trajs)', fontsize=14)
    elif n_x == 4:
        axes[1, 0].set_xlabel(state_labels[0], fontsize=12)
        axes[1, 0].set_ylabel(state_labels[2], fontsize=12)
        axes[1, 0].set_title(f'Config Space ({n_test} trajs)', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Row 2, Col 1: Loss curves (or message if inference only)
    if log is not None:
        epochs = range(len(log))
        axes[1, 1].plot(epochs, [l['total'] for l in log], label='Total', linewidth=2)
        axes[1, 1].plot(epochs, [l['pred'] for l in log], label='1-step Pred', linewidth=2)
        axes[1, 1].plot(epochs, [l['lin'] for l in log], label='Linearity', linewidth=2, linestyle='--')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Loss', fontsize=12)
        axes[1, 1].set_title('Training Loss', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Inference Only\n(No Training Curves)', 
                       ha='center', va='center', fontsize=14, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Training Loss', fontsize=14)
        axes[1, 1].axis('off')
    
    # Row 2, Col 2: Average expert usage (averaged over all trajectories)
    avg_weights = np.mean([w.mean(axis=0) for w in all_weights], axis=0)
    axes[1, 2].bar(range(n_experts), avg_weights)
    axes[1, 2].axhline(1.0/n_experts, color='r', linestyle='--',
                      label=f'Equal ({1.0/n_experts:.3f})')
    axes[1, 2].set_xlabel('Expert ID', fontsize=12)
    axes[1, 2].set_ylabel('Average Weight', fontsize=12)
    axes[1, 2].set_title(f'Expert Usage ({n_test} trajs avg)', fontsize=14)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # Row 2, Col 3: NRMSE at specific horizons (averaged)
    eval_horizons = EVAL_HORIZONS  # [1, 10, 50, 100, 500, 1000]
    horizon_nrmse = [avg_metrics.get(f'nrmse_{h}step', np.nan) for h in eval_horizons]
    colors = ['green' if h <= 50 else 'red' for h in eval_horizons]
    axes[1, 3].bar(range(len(eval_horizons)), horizon_nrmse, color=colors)
    axes[1, 3].set_xticks(range(len(eval_horizons)))
    axes[1, 3].set_xticklabels([f'{h}' for h in eval_horizons])
    axes[1, 3].set_xlabel('Horizon (steps)', fontsize=12)
    axes[1, 3].set_ylabel('NRMSE (avg)', fontsize=12)
    axes[1, 3].set_title(f'NRMSE by Horizon ({n_test} trajs avg)', fontsize=14)
    axes[1, 3].grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots in row 2
    for i in range(4, n_cols):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    results_file = f'{args.save_prefix}{args.system}_moe_results.png'
    plt.savefig(results_file, dpi=150, bbox_inches='tight')
    print(f"Results saved to '{results_file}'")
    plt.close()
    
    # === NEW: Individual trajectory plots in a grid ===
    # Plot each of the 10 trajectories in separate subplots
    n_rows_grid = 2
    n_cols_grid = 5
    
    # For each state variable, create a separate grid figure
    for state_idx in range(n_x):
        fig_grid, axes_grid = plt.subplots(n_rows_grid, n_cols_grid, figsize=(20, 8))
        axes_grid = axes_grid.flatten()
        
        for traj_idx, res in enumerate(all_results):
            if traj_idx >= n_rows_grid * n_cols_grid:
                break
            
            ax = axes_grid[traj_idx]
            true_traj = res['true']
            pred_traj = res['preds']
            
            ax.plot(time_axis, true_traj[:, state_idx], '-', color='blue', 
                   linewidth=1.5, label='True')
            ax.plot(time_axis, pred_traj[:, state_idx], '--', color='red', 
                   linewidth=1.5, label='Pred')
            
            # Calculate MSE for this trajectory
            traj_mse = np.mean((pred_traj[:, state_idx] - true_traj[:, state_idx])**2)
            
            ax.set_title(f'IC {traj_idx+1} (MSE: {traj_mse:.4f})', fontsize=11)
            ax.set_xlabel('Time', fontsize=9)
            ax.set_ylabel(state_labels[state_idx], fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            if traj_idx == 0:
                ax.legend(fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(all_results), n_rows_grid * n_cols_grid):
            axes_grid[idx].axis('off')
        
        fig_grid.suptitle(f'{system_name}: {state_labels[state_idx]} - {n_test} Initial Conditions', 
                         fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        grid_file = f'{args.save_prefix}{args.system}_{state_labels[state_idx]}_grid.png'
        plt.savefig(grid_file, dpi=150, bbox_inches='tight')
        print(f"Grid plot saved to '{grid_file}'")
        plt.close()
    
    # Visualize expert usage over time (use first trajectory)
    expert_usage_file = f'{args.save_prefix}{args.system}_expert_usage.png'
    visualize_expert_usage(all_weights[0], save_path=expert_usage_file)
    
    # Save final model (only if we trained)
    if not args.inference_only:
        model_file = f'{args.save_prefix}{args.system}_moe_model.pth'
        torch.save(model.state_dict(), model_file)
        print(f"Final model saved to '{model_file}'")
        print(f"  (Best checkpoint was saved during training whenever validation improved)")
    
    # Print expert usage statistics (averaged over all trajectories)
    print(f"\n=== Expert Usage Statistics ({n_test} trajs) ===")
    # Concatenate all weights
    all_weights_concat = np.concatenate(all_weights, axis=0)
    for i in range(n_experts):
        avg_weight = all_weights_concat[:, i].mean()
        max_weight = all_weights_concat[:, i].max()
        min_weight = all_weights_concat[:, i].min()
        std_weight = all_weights_concat[:, i].std()
        print(f"Expert {i+1}: avg={avg_weight:.3f}, max={max_weight:.3f}, "
              f"min={min_weight:.3f}, std={std_weight:.3f}")


if __name__ == "__main__":
    main()

