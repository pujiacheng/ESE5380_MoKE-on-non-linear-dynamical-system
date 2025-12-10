"""
Main training script for Koopman Autoencoder on Duffing oscillator data

This script:
1. Generates Duffing oscillator trajectories
2. Prepares data for training with multi-step horizons
3. Trains the Koopman Autoencoder model
4. Evaluates and visualizes predictions

Loss components:
- Reconstruction loss
- Multi-step latent linearity (horizons: 1, 10, 20, 30, 40, 50)
- Hankel-based linearity (HAVOK)
- Bidirectional constraint (A_f @ A_b ≈ I)
- Spectral radius penalty
- Sparsity regularization
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from data_simulation import generate_duffing_dataset
from koopman_mixture_neural_network import (
    KoopmanAE, 
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
        trajs: array of shape (n_traj, n_steps, n_x)
        hankel_seq_len: sequence length for Hankel loss computation
    
    Returns:
        dict with 'x0', 'x1', 'x_k' for k in [10, 20, ..., 50], and 'sequences' for Hankel
    """
    n_traj, n_timesteps, n_x = trajs.shape
    
    # Horizons for multi-step linearity (up to 50 steps) - same as MoE
    horizons = TRAINING_HORIZONS  # Dense: [1, 2, 3, ..., 100]
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


def compute_loss(model, data_batch, device, sequences=None):
    """
    Compute loss for Koopman Autoencoder model
    
    Args:
        model: KoopmanAE model
        data_batch: dict with 'x0', 'x1', 'x10', 'x20', ..., 'x50'
        device: device to run on
        sequences: optional tensor (batch, T, n_x) for Hankel loss
    
    Loss components (aligned with MoE for fair comparison):
    1. Reconstruction loss
    2. Prediction loss (1-step) - PRIMARY OBJECTIVE
    3. Multi-step latent linearity (1, 10, 20, ..., 50 steps)
    4. Bidirectional constraint
    5. Spectral radius penalty
    6. Hankel-based linearity (HAVOK)
    7. Sparsity regularization
    """
    # Hyperparameters - aligned with MoE for fair comparison
    lam_rec = 2.0       # 1. Reconstruction
    lam_pred = 15.0     # 2. 1-step prediction (PRIMARY - same as MoE!)
    lam_lin = 12.0      # 3. Multi-step linearity (KOOPMAN CORE)
    lam_bi = 1.0        # 4. Bidirectional
    lam_spec = 5.0      # 5. Spectral radius (stability)
    lam_hankel = 1.0    # 6. Hankel linearity (HAVOK)
    lam_sparse = 1e-4   # 7. Sparsity (regularization)
    
    # Extract data
    x0 = data_batch['x0']
    x1 = data_batch['x1']  # For 1-step prediction
    
    mse = nn.MSELoss()
    
    # Forward pass for current state
    out0 = model(x0)
    x_rec = out0['x_rec']
    
    # === 1. Reconstruction Loss ===
    loss_rec = mse(x_rec, x0)
    
    # === 2. Prediction Loss (1-step) - PRIMARY OBJECTIVE ===
    # Predict next state using Koopman operator
    z0 = model.encoder(x0)
    z1_pred = z0 @ model.A_f.T
    x1_pred = model.decoder(z1_pred)
    loss_pred = mse(x1_pred, x1)
    
    # === 3. Multi-Step Latent Linearity ===
    # z_{t+k} should equal A_f^k @ z_t for k = 1, 10, 20, ..., 50
    # Uniform weights (same as MoE)
    
    # Multi-step linearity - compute A^k incrementally (efficient for dense horizons)
    loss_lin = 0
    A_k = model.A_f.clone()
    for k in TRAINING_HORIZONS:  # [1, 2, 3, ..., 100]
        x_k = data_batch[f'x{k}']
        zk_true = model.encoder(x_k)
        zk_pred = z0 @ A_k.T
        loss_lin += mse(zk_pred, zk_true)
        A_k = A_k @ model.A_f  # A^(k+1) = A^k @ A
    loss_lin /= len(TRAINING_HORIZONS)
    
    # === 4. Bidirectional Constraint ===
    # A_f @ A_b ≈ I (ensures reversibility)
    I = torch.eye(model.n_z, device=device)
    loss_bi = (model.A_f @ model.A_b - I).norm()**2 + (model.A_b @ model.A_f - I).norm()**2
    
    # === 5. Spectral Radius Penalty ===
    # Same target as MoE (0.99)
    loss_spec = spectral_radius_penalty(model.A_f, iters=8, target=1.005, lower=0.995)
    
    # === 6. Hankel-Based Linearity Loss ===
    loss_hankel = torch.tensor(0.0, device=device)
    if sequences is not None and len(sequences) > 0:
        batch, T, n_x = sequences.shape
        z_seq = model.encoder(sequences.reshape(-1, n_x)).reshape(batch, T, -1)
        loss_hankel = compute_hankel_linearity_loss(z_seq, L=4, r=8, device=device)
    
    # === 7. Sparsity Regularization ===
    loss_sparse = model.sparsity_loss(mode="l1")
    
    # === Total Loss (aligned with MoE) ===
    loss_total = (
        lam_rec * loss_rec +
        lam_pred * loss_pred +
        lam_lin * loss_lin +
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
        'bi': loss_bi,
        'spec': loss_spec,
        'hankel': loss_hankel,
        'sparse': loss_sparse
    }


def train_model(model, train_loader, device, n_epochs=40, val_loader=None,
                early_stopping=False, patience=20, checkpoint_path=None,
                train_sequences=None, val_sequences=None, hankel_batch_size=32):
    """
    Train the Koopman Autoencoder model with validation and early stopping
    
    Args:
        model: KoopmanAE model instance
        train_loader: DataLoader with training data
        device: device to train on
        n_epochs: number of training epochs
        val_loader: optional validation DataLoader
        early_stopping: whether to use early stopping
        patience: number of epochs to wait for improvement
        checkpoint_path: path to save best model checkpoint
        train_sequences: tensor for Hankel loss (N, T, n_x)
        val_sequences: tensor for Hankel loss validation
        hankel_batch_size: batch size for Hankel computation
    
    Returns:
        log: list of training losses per epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    log = []
    horizons = TRAINING_HORIZONS  # Dense: [1, 2, 3, ..., 100]
    
    # Early stopping tracking
    best_val_total = float('inf')
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
        print(f"{'Epoch':<8} {'Total':<12} {'Pred':<10} {'Rec':<10} {'Lin':<10} {'Spec':<10}")
    print("="*110)
    
    for ep in range(n_epochs):
        # === Training phase ===
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            # batch is a tuple of tensors: (x0, x1, x10, x20, ..., x50)
            data_batch = {'x0': batch[0].to(device)}
            for idx, h in enumerate(horizons):
                data_batch[f'x{h}'] = batch[idx + 1].to(device)
            
            # Sample sequences for Hankel loss
            seq_batch = None
            if train_sequences is not None and len(train_sequences) > 0:
                seq_batch = sample_sequence_batch(train_sequences, hankel_batch_size).to(device)
            
            # Compute losses
            losses = compute_loss(model, data_batch, device, sequences=seq_batch)
            
            # Backprop
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            
            epoch_losses.append({k: v.item() for k, v in losses.items()})
        
        # Average training losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in epoch_losses])
        
        # === Validation phase ===
        status = ""
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    data_batch = {'x0': batch[0].to(device)}
                    for idx, h in enumerate(horizons):
                        data_batch[f'x{h}'] = batch[idx + 1].to(device)
                    
                    seq_batch = None
                    if val_sequences is not None and len(val_sequences) > 0:
                        seq_batch = sample_sequence_batch(val_sequences, hankel_batch_size).to(device)
                    
                    losses = compute_loss(model, data_batch, device, sequences=seq_batch)
                    val_losses.append({k: v.item() for k, v in losses.items()})
            
            # Average validation losses
            avg_val_losses = {}
            for key in val_losses[0].keys():
                avg_val_losses[key] = np.mean([l[key] for l in val_losses])
            
            avg_losses['val_total'] = avg_val_losses['total']
            avg_losses['val_pred'] = avg_val_losses['pred']
            avg_losses['val_rec'] = avg_val_losses['rec']
            
            # Early stopping check
            if early_stopping:
                current_val_total = avg_val_losses['total']
                
                if current_val_total < best_val_total:
                    best_val_total = current_val_total
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    status = "✓ Best"
                    
                    # Save checkpoint immediately
                    if checkpoint_path:
                        torch.save(model.state_dict(), checkpoint_path)
                else:
                    patience_counter += 1
                    status = f"Wait {patience_counter}/{patience}"
                    
                    if patience_counter >= patience:
                        print(f"{'':8} {'':12} {'':10} {'':12} {'':10} {'Early Stop!':<22}")
                        print("="*110)
                        print(f"Training stopped early at epoch {ep}")
                        print(f"Best validation loss: {best_val_total:.6f}")
                        print("="*110)
                        # Restore best model
                        model.load_state_dict(best_model_state)
                        log.append(avg_losses)
                        break
        
        log.append(avg_losses)
        
        # Print progress
        if ep % 5 == 0 or ep == n_epochs - 1:
            if val_loader is not None:
                if early_stopping:
                    print(f"{ep:<8} {avg_losses['total']:<12.6f} {avg_losses['pred']:<12.6f} "
                          f"{avg_losses['val_total']:<12.6f} {avg_losses['val_pred']:<12.6f} "
                          f"{status:<22}")
                else:
                    print(f"{ep:<8} {avg_losses['total']:<12.6f} {avg_losses['pred']:<12.6f} "
                          f"{avg_losses['val_total']:<12.6f} {avg_losses['val_pred']:<12.6f}")
            else:
                print(f"{ep:<8} {avg_losses['total']:<12.6f} {avg_losses['pred']:<10.6f} "
                      f"{avg_losses['rec']:<10.6f} {avg_losses['lin']:<10.6f} "
                      f"{avg_losses['spec']:<10.6f}")
    
    print("="*110 + "\n")
    return log


def evaluate_model(model, test_traj, device, n_steps=100):
    """
    Evaluate model by predicting forward in time
    
    Args:
        model: trained KoopmanAE model
        test_traj: test trajectory array of shape (n_steps, n_x)
        device: device to run on
        n_steps: number of steps to predict
    
    Returns:
        true_traj: true trajectory (n_steps+1, n_x)
        pred_traj: predicted trajectory (n_steps+1, n_x)
        x_rec: reconstructed initial condition (1, n_x)
    """
    model.eval()
    with torch.no_grad():
        # Take initial condition from test trajectory
        x0 = torch.tensor(test_traj[0], dtype=torch.float32).unsqueeze(0).to(device)
        z0 = model.encoder(x0)
        
        # Reconstruction of initial condition
        x_rec = model.decoder(z0).cpu().numpy()
        
        zs = [z0]
        z = z0
        for i in range(n_steps):
            z = z @ model.A_f.T
            zs.append(z)
        
        zs = torch.cat(zs, dim=0)  # (n_steps+1, n_z)
        preds = model.decoder(zs).cpu().numpy()  # (n_steps+1, n_x)
        
        true = test_traj[:n_steps+1]
        
    return true, preds, x_rec


def compute_all_metrics(model, true, preds, x_rec, device):
    """
    Compute all evaluation metrics from evaluation.py
    
    Args:
        model: trained KoopmanAE model (for spectral radius)
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
    
    # Multi-step NRMSE at various horizons (including extrapolation)
    horizons = EVAL_HORIZONS  # [1, 10, 50, 100, 500, 1000]
    nrmse_dict = multi_step_nrmse(true_3d, preds_3d, horizons)
    for h, val in nrmse_dict.items():
        metrics[f'nrmse_{h}step'] = val
    
    # Chamfer distance in phase space
    metrics['chamfer_distance'] = chamfer_distance_phase(true_3d, preds_3d, dims=(0, 1))
    
    # Spectral radius of Koopman operator
    K = model.A_f.detach().cpu().numpy()
    rho, _ = spectral_radius(K)
    metrics['spectral_radius'] = rho
    
    # Long-horizon divergence rate
    slope, _ = long_horizon_divergence_rate(true_3d, preds_3d)
    metrics['divergence_rate'] = slope
    
    # Reconstruction error (initial condition)
    metrics['reconstruction_error'] = reconstruction_error(true[0:1], x_rec)
    
    return metrics


def main():
    """Main training script for Koopman Autoencoder (Model 4)"""
    # Command-line arguments (same style as train_moe.py)
    parser = argparse.ArgumentParser(description='Train Koopman Autoencoder on Duffing oscillator')
    parser.add_argument('--n_traj', type=int, default=100,
                       help='Number of trajectories')
    parser.add_argument('--T', type=float, default=10.0,
                       help='Time horizon')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step')
    parser.add_argument('--noise_std', type=float, default=0.0,
                       help='Noise standard deviation')
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
                       help='Run inference only (skip training)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model for inference (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Generate Duffing oscillator data
    print("\nGenerating Duffing oscillator trajectories...")
    t, trajs = generate_duffing_dataset(
        n_traj=args.n_traj, T=args.T, dt=args.dt, noise_std=args.noise_std
    )
    print(f"Generated {args.n_traj} trajectories, each with {trajs.shape[1]} time steps")
    
    # Split into train and test (keep 10 for testing like MoE)
    n_test = min(10, args.n_traj // 10)
    trajs_train = trajs[:-n_test] if n_test > 0 else trajs
    trajs_test = trajs[-n_test:] if n_test > 0 else trajs[:1]
    print(f"Train: {len(trajs_train)} trajectories, Test: {len(trajs_test)} trajectories")
    
    # Create model
    n_x = 2  # Duffing oscillator state dimension
    n_z = 5 * n_x  # Latent dimension: 5× state dimension (same as MoE)
    
    model = KoopmanAE(n_x=n_x, n_z=n_z).to(device)
    print(f"\nModel created: n_x={n_x}, n_z={n_z} (5×{n_x})")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_params:,}")
    
    # Model file path
    model_file = f'{args.save_prefix}koopman_model.pth'
    
    if args.inference_only:
        # Load existing model
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = model_file
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            return
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from '{model_path}'")
    else:
        # Prepare training data with multi-step horizons
        print("\nPreparing training data with multi-step linearity horizons...")
        data_dict = prepare_data_from_trajectories(trajs_train, hankel_seq_len=16)
        
        horizons = TRAINING_HORIZONS  # Dense: [1, 2, 3, ..., 100]
        print(f"  Linearity horizons: {horizons}")
        print(f"  Horizon weights: uniform (all 1.0)")
        
        # Get sequences for Hankel loss
        train_sequences = data_dict.get('sequences', None)
        if train_sequences is not None:
            print(f"  Hankel sequences: {train_sequences.shape[0]} sequences of length {train_sequences.shape[1]}")
        
        # Split into train/val
        n_samples = len(data_dict['x0'])
        n_val = int(n_samples * args.val_split)
        n_train = n_samples - n_val
        print(f"\nData split: {n_train} train, {n_val} validation samples")
        
        # Create tensors for train and val
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
        val_loader = DataLoader(
            TensorDataset(*val_tensors),
            batch_size=args.batch_size,
            shuffle=False
        )
        print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
        
        # Train model
        print("\nStarting training...")
        if args.early_stopping:
            print(f"Early stopping enabled (patience={args.patience})")
        
        log = train_model(
            model=model,
            train_loader=train_loader,
            device=device,
            n_epochs=args.n_epochs,
            val_loader=val_loader,
            early_stopping=args.early_stopping,
            patience=args.patience,
            checkpoint_path=model_file,
            train_sequences=train_sequences,
            val_sequences=train_sequences,  # Use same sequences for val
            hankel_batch_size=32
        )
        
        # Save final model (best was already saved during training if early stopping)
        torch.save(model.state_dict(), model_file)
        print(f"Final model saved to '{model_file}'")
        if args.early_stopping:
            print(f"  (Best checkpoint was saved during training whenever validation improved)")
    
    # Evaluate on ALL test trajectories
    n_test_eval = len(trajs_test)
    print(f"\nEvaluating model on {n_test_eval} test trajectories...")
    
    all_results = []
    all_metrics = []
    
    for i, test_traj in enumerate(trajs_test):
        true, preds, x_rec = evaluate_model(model, test_traj, device, n_steps=100)
        all_results.append({'true': true, 'preds': preds})
        
        # Compute metrics for this trajectory
        metrics = compute_all_metrics(model, true, preds, x_rec, device)
        all_metrics.append(metrics)
    
    # Compute average metrics across all test trajectories
    avg_metrics = {}
    std_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        std_metrics[key] = np.std([m[key] for m in all_metrics])
    
    # Print evaluation metrics (from evaluation.py)
    print("\n" + "="*70)
    print(f"EVALUATION METRICS (from evaluation.py, averaged over {n_test_eval} trajectories)")
    print("="*70)
    print(f"  1-step MSE:           {avg_metrics['one_step_mse']:.6f} ± {std_metrics['one_step_mse']:.6f}")
    print(f"  Reconstruction Error: {avg_metrics['reconstruction_error']:.6f} ± {std_metrics['reconstruction_error']:.6f}")
    print(f"  Spectral Radius:      {avg_metrics['spectral_radius']:.4f}")
    print(f"  Divergence Rate:      {avg_metrics['divergence_rate']:.6f} ± {std_metrics['divergence_rate']:.6f}")
    print(f"  Chamfer Distance:     {avg_metrics['chamfer_distance']:.6f} ± {std_metrics['chamfer_distance']:.6f}")
    print("\n  Multi-step NRMSE:")
    for key in sorted(avg_metrics.keys()):
        if key.startswith('nrmse_'):
            horizon = key.replace('nrmse_', '').replace('step', '')
            print(f"    Horizon {horizon:>3}: {avg_metrics[key]:.6f} ± {std_metrics[key]:.6f}")
    print("="*70)
    
    # Visualize results (use first test trajectory)
    true = all_results[0]['true']
    preds = all_results[0]['preds']
    
    plt.figure(figsize=(12, 5))
    
    # Phase space plot (all trajectories)
    plt.subplot(1, 2, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, n_test_eval))
    for j, res in enumerate(all_results):
        alpha = 0.7 if j == 0 else 0.4
        label_true = 'True' if j == 0 else None
        label_pred = 'Predicted' if j == 0 else None
        plt.plot(res['true'][:, 0], res['true'][:, 1], '-', color=colors[j],
                linewidth=1.5, alpha=alpha, label=label_true)
        plt.plot(res['preds'][:, 0], res['preds'][:, 1], '--', color=colors[j],
                linewidth=1.5, alpha=alpha, label=label_pred)
    plt.xlabel('x')
    plt.ylabel('xdot')
    plt.title(f'Phase Space ({n_test_eval} trajs)')
    plt.legend(['True', 'Predicted'])
    plt.grid(True, alpha=0.3)
    
    # Time series plot (all trajectories)
    plt.subplot(1, 2, 2)
    time_axis = np.arange(len(true)) * args.dt
    for j, res in enumerate(all_results):
        alpha = 0.7 if j == 0 else 0.4
        plt.plot(time_axis, res['true'][:, 0], '-', color=colors[j], linewidth=1.5, alpha=alpha)
        plt.plot(time_axis, res['preds'][:, 0], '--', color=colors[j], linewidth=1.5, alpha=alpha)
    plt.xlabel('Time')
    plt.ylabel('x')
    plt.title(f'Time Series: Position ({n_test_eval} trajs)')
    plt.legend(['True', 'Predicted'])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    results_file = f'{args.save_prefix}koopman_prediction_results.png'
    plt.savefig(results_file, dpi=150)
    print(f"Results saved to '{results_file}'")
    plt.close()


if __name__ == "__main__":
    main()
