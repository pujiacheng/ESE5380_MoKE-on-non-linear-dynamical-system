"""
Train and test Koopman Autoencoder with data from CSV file

This script:
1. Loads data from CSV file
2. Splits data into train/validation/test sets
3. Trains the Koopman Autoencoder model with multi-step linearity
4. Validates during training
5. Evaluates on test set
6. Saves model and results

Loss components (aligned with MoE):
- Reconstruction loss
- Multi-step latent linearity (horizons: 1, 10, 20, 30, 40, 50)
- Hankel-based linearity (HAVOK)
- Bidirectional constraint (A_f @ A_b ≈ I)
- Spectral radius penalty
- Sparsity regularization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
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


def load_data_from_csv(csv_path, state_columns=None, traj_id_column=None, time_column=None):
    """
    Load trajectory data from CSV file
    
    Args:
        csv_path: path to CSV file
        state_columns: list of column names for state variables (e.g., ['x', 'xdot'])
                      If None, will try to auto-detect (exclude time and traj_id columns)
        traj_id_column: column name for trajectory ID (if multiple trajectories in CSV)
                       If None, assumes single trajectory
        time_column: column name for time (optional, will be ignored if provided)
    
    Returns:
        trajs: array of shape (n_traj, n_steps, n_x) where n_x is state dimension
        n_x: state dimension
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"CSV shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
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
    
    # Handle multiple trajectories or single trajectory
    if traj_id_column and traj_id_column in df.columns:
        # Multiple trajectories
        traj_ids = df[traj_id_column].unique()
        trajs = []
        for traj_id in traj_ids:
            traj_data = df[df[traj_id_column] == traj_id][state_columns].values
            if len(traj_data) >= 3:  # Need at least 3 points for triplets
                trajs.append(traj_data.astype(np.float32))
        
        # Convert to regular array if all trajectories have same length
        try:
            trajs = np.stack(trajs)
        except ValueError:
            # Trajectories have different lengths - truncate to minimum
            lengths = [len(t) for t in trajs]
            min_len = min(lengths)
            max_len = max(lengths)
            print(f"WARNING: Trajectories have different lengths ({min_len} to {max_len})")
            print(f"         Truncating all to minimum length: {min_len}")
            trajs = np.stack([t[:min_len] for t in trajs])
        print(f"Found {len(trajs)} trajectories")
    else:
        # Single trajectory
        if len(state_data) < 3:
            raise ValueError("Need at least 3 data points for training")
        trajs = state_data.reshape(1, -1, n_x)
        print(f"Single trajectory with {len(state_data)} time steps")
    
    print(f"Data shape: {trajs.shape} (n_traj, n_steps, n_x)")
    return trajs, n_x


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
    horizons = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    max_horizon = max(horizons)
    
    # Initialize lists for each horizon
    data_lists = {h: [] for h in horizons}
    x0_list = []
    
    for traj in trajs:
        if n_timesteps > max_horizon:
            x0_list.append(traj[:-max_horizon])
            
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
    """Sample a batch of sequences for Hankel loss computation"""
    n_seqs = sequences.shape[0]
    indices = np.random.choice(n_seqs, size=min(batch_size, n_seqs), replace=False)
    return sequences[indices]


def compute_loss(model, data_batch, device, sequences=None):
    """
    Compute loss for Koopman Autoencoder model (aligned with MoE for fair comparison)
    
    Loss components:
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
    
    x0 = data_batch['x0']
    x1 = data_batch['x1']  # For 1-step prediction
    mse = nn.MSELoss()
    
    # Forward pass
    out0 = model(x0)
    x_rec = out0['x_rec']
    
    # === 1. Reconstruction Loss ===
    loss_rec = mse(x_rec, x0)
    
    # === 2. Prediction Loss (1-step) - PRIMARY OBJECTIVE ===
    z0 = model.encoder(x0)
    z1_pred = z0 @ model.A_f.T
    x1_pred = model.decoder(z1_pred)
    loss_pred = mse(x1_pred, x1)
    
    # === 3. Multi-Step Latent Linearity ===
    horizons = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    horizon_weights = {1: 1.0, 10: 1.0, 20: 1.0, 30: 1.0, 40: 1.0, 50: 1.0, 60: 1.0, 70: 1.0, 80: 1.0, 90: 1.0, 100: 1.0}
    
    loss_lin = 0
    
    # Pre-compute A^k
    A_powers = {1: model.A_f}
    A_k = model.A_f.clone()
    for k in [10, 20, 30, 40, 50]:
        prev_k = horizons[horizons.index(k) - 1]
        for _ in range(k - prev_k):
            A_k = A_k @ model.A_f
        A_powers[k] = A_k.clone()
    
    # z0 already computed above
    for k in horizons:
        x_k = data_batch[f'x{k}']
        w_k = horizon_weights[k]
        zk_true = model.encoder(x_k)
        zk_pred = z0 @ A_powers[k].T
        loss_lin += w_k * mse(zk_pred, zk_true)
    
    total_weight = sum(horizon_weights.values())
    loss_lin = loss_lin / total_weight
    
    # === 4. Bidirectional Constraint ===
    I = torch.eye(model.n_z, device=device)
    loss_bi = (model.A_f @ model.A_b - I).norm()**2 + (model.A_b @ model.A_f - I).norm()**2
    
    # === 5. Spectral Radius Penalty ===
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


def train_model(model, train_loader, val_loader, device, n_epochs=40,
                early_stopping=False, patience=20,
                train_sequences=None, val_sequences=None, hankel_batch_size=32, save_dir='./'):
    """
    Train the Koopman Autoencoder model with validation and early stopping
    
    Args:
        model: KoopmanAE model instance
        train_loader: DataLoader with training data
        val_loader: DataLoader with validation data
        device: device to train on
        n_epochs: number of training epochs
        early_stopping: whether to use early stopping
        patience: number of epochs to wait for improvement
        train_sequences: tensor for Hankel loss (N, T, n_x)
        val_sequences: tensor for Hankel loss validation
        hankel_batch_size: batch size for Hankel computation
        save_dir: directory to save model checkpoints
    """
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_log = []
    val_log = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    horizons = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    print("\n" + "="*110)
    print("🎯 TRAINING WITH MULTI-STEP LINEARITY (1, 10, 20, 30, 40, 50 steps)")
    print("="*110)
    if early_stopping:
        print(f"{'Epoch':<8} {'Train':<12} {'Val':<12} {'Status':<22}")
    else:
        print(f"{'Epoch':<8} {'Train':<12} {'Val':<12} {'Status':<15}")
    print("="*110)
    
    for ep in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch in train_loader:
            data_batch = {'x0': batch[0].to(device)}
            for idx, h in enumerate(horizons):
                data_batch[f'x{h}'] = batch[idx + 1].to(device)
            
            seq_batch = None
            if train_sequences is not None and len(train_sequences) > 0:
                seq_batch = sample_sequence_batch(train_sequences, hankel_batch_size).to(device)
            
            losses = compute_loss(model, data_batch, device, sequences=seq_batch)
            
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            
            train_losses.append(losses['total'].item())
        
        avg_train_loss = np.mean(train_losses)
        train_log.append(avg_train_loss)
        
        # Validation phase
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
                val_losses.append(losses['total'].item())
        
        avg_val_loss = np.mean(val_losses)
        val_log.append(avg_val_loss)
        
        # Save best model and check early stopping
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            status = "✓ Best"
        else:
            if early_stopping:
                patience_counter += 1
                status = f"Wait {patience_counter}/{patience}"
                
                if patience_counter >= patience:
                    print(f"{'':8} {'':12} {'':12} {'Early Stop!':<22}")
                    print("="*110)
                    print(f"Training stopped early at epoch {ep}")
                    print(f"Best validation loss: {best_val_loss:.6f}")
                    print("="*110)
                    # Restore best model
                    model.load_state_dict(best_model_state)
                    return train_log, val_log
        
        if ep % 5 == 0 or ep == n_epochs - 1:
            if early_stopping:
                print(f"{ep:<8} {avg_train_loss:<12.6f} {avg_val_loss:<12.6f} {status:<22}")
            else:
                print(f"{ep:<8} {avg_train_loss:<12.6f} {avg_val_loss:<12.6f} {status:<15}")
    
    print("="*110 + "\n")
    return train_log, val_log


def evaluate_model(model, test_traj, device, n_steps=None):
    """Evaluate model by predicting forward in time"""
    model.eval()
    if n_steps is None:
        n_steps = len(test_traj) - 1
    
    with torch.no_grad():
        x0 = torch.tensor(test_traj[0], dtype=torch.float32).unsqueeze(0).to(device)
        z0 = model.encoder(x0)
        
        # Reconstruction of initial condition
        x_rec = model.decoder(z0).cpu().numpy()
        
        zs = [z0]
        z = z0
        for i in range(n_steps):
            z = z @ model.A_f.T
            zs.append(z)
        
        zs = torch.cat(zs, dim=0)
        preds = model.decoder(zs).cpu().numpy()
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
    parser = argparse.ArgumentParser(description='Train Koopman Autoencoder from CSV data')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--state_columns', type=str, nargs='+', default=None,
                       help='Column names for state variables (e.g., x xdot)')
    parser.add_argument('--traj_id_column', type=str, default=None,
                       help='Column name for trajectory ID (if multiple trajectories)')
    parser.add_argument('--time_column', type=str, default=None,
                       help='Column name for time (optional)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio of data for training (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Ratio of data for validation (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Ratio of data for testing (default: 0.15)')
    parser.add_argument('--n_z', type=int, default=None,
                       help='Latent dimension (default: 5×state_dim)')
    parser.add_argument('--n_epochs', type=int, default=40,
                       help='Number of training epochs (default: 40)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (epochs)')
    parser.add_argument('--save_dir', type=str, default='./',
                       help='Directory to save model and results (default: ./)')
    parser.add_argument('--inference_only', action='store_true',
                       help='Run inference only (skip training)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model for inference (default: best_model.pth in save_dir)')
    
    args = parser.parse_args()
    
    # Validate split ratios
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "Train, validation, and test ratios must sum to 1.0"
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    trajs, n_x = load_data_from_csv(
        args.csv_path,
        state_columns=args.state_columns,
        traj_id_column=args.traj_id_column,
        time_column=args.time_column
    )
    
    # Compute latent dimension: 5× state dimension (same as MoE)
    if args.n_z is None:
        n_z = 5 * n_x
        print(f"Auto-computing latent dimension: n_z = 5 × {n_x} = {n_z}")
    else:
        n_z = args.n_z
        print(f"Using provided latent dimension: n_z = {n_z}")
    
    # Split trajectories temporally
    n_traj, n_steps, n_x = trajs.shape
    train_end = int(n_steps * args.train_ratio)
    val_end = int(n_steps * (args.train_ratio + args.val_ratio))
    
    train_trajs = trajs[:, :train_end, :]
    val_trajs = trajs[:, train_end:val_end, :]
    test_trajs = trajs[:, val_end:, :]
    
    print(f"\nTemporal split per trajectory:")
    print(f"  Train: steps 0 to {train_end-1} ({train_end/n_steps*100:.1f}%)")
    print(f"  Val:   steps {train_end} to {val_end-1} ({(val_end-train_end)/n_steps*100:.1f}%)")
    print(f"  Test:  steps {val_end} to {n_steps-1} ({(n_steps-val_end)/n_steps*100:.1f}%)")
    
    # Prepare data with multi-step horizons
    print("\nPreparing training data with multi-step linearity horizons...")
    train_data = prepare_data_from_trajectories(train_trajs, hankel_seq_len=16)
    val_data = prepare_data_from_trajectories(val_trajs, hankel_seq_len=16)
    
    horizons = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    print(f"  Linearity horizons: {horizons}")
    print(f"  Horizon weights: uniform (all 1.0)")
    
    train_sequences = train_data.get('sequences', None)
    val_sequences = val_data.get('sequences', None)
    
    if train_sequences is not None:
        print(f"  Train Hankel sequences: {train_sequences.shape[0]}")
    
    # Create DataLoaders
    train_tensors = [train_data['x0']]
    val_tensors = [val_data['x0']]
    for h in horizons:
        train_tensors.append(train_data[f'x{h}'])
        val_tensors.append(val_data[f'x{h}'])
    
    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=args.batch_size, shuffle=False)
    
    print(f"Training batches: {len(train_loader)}")
    
    # Create model
    model = KoopmanAE(n_x=n_x, n_z=n_z).to(device)
    print(f"\nModel created: n_x={n_x}, n_z={n_z} (5×{n_x})")
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_params:,}")
    
    if args.inference_only:
        # Load existing model for inference
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = os.path.join(args.save_dir, 'best_model.pth')
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            return
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from '{model_path}'")
        train_log, val_log = [], []  # Empty logs for inference mode
    else:
        # Train model
        print("\nStarting training...")
        if args.early_stopping:
            print(f"Early stopping enabled (patience={args.patience})")
        
        train_log, val_log = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            n_epochs=args.n_epochs,
            early_stopping=args.early_stopping,
            patience=args.patience,
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            hankel_batch_size=32,
            save_dir=args.save_dir
        )
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
        print("Loaded best model based on validation loss")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_traj = test_trajs[0]
    true, preds, x_rec = evaluate_model(model, test_traj, device, n_steps=min(100, len(test_traj)-1))
    
    # Compute all metrics from evaluation.py
    metrics = compute_all_metrics(model, true, preds, x_rec, device)
    
    print("\n" + "="*60)
    print("EVALUATION METRICS (from evaluation.py)")
    print("="*60)
    print(f"  1-step MSE:           {metrics['one_step_mse']:.6f}")
    print(f"  Reconstruction Error: {metrics['reconstruction_error']:.6f}")
    print(f"  Spectral Radius:      {metrics['spectral_radius']:.4f}")
    print(f"  Divergence Rate:      {metrics['divergence_rate']:.6f}")
    if 'chamfer_distance' in metrics:
        print(f"  Chamfer Distance:     {metrics['chamfer_distance']:.6f}")
    print("\n  Multi-step NRMSE:")
    for key, val in metrics.items():
        if key.startswith('nrmse_'):
            horizon = key.replace('nrmse_', '').replace('step', '')
            print(f"    Horizon {horizon:>3}: {val:.6f}")
    print("="*60)
    
    # Visualize results
    if len(train_log) > 0:
        # Full plot with training curves
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_log, label='Train Loss', linewidth=2)
        plt.plot(val_log, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(true[:, 0], true[:, 1], '-o', label='True', markersize=3, alpha=0.7)
        plt.plot(preds[:, 0], preds[:, 1], '-x', label='Predicted', markersize=3, alpha=0.7)
        plt.xlabel('x')
        plt.ylabel('xdot')
        plt.title('Phase Space: Test Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        time_axis = np.arange(len(true))
        plt.plot(time_axis, true[:, 0], '-o', label='True x', markersize=2, alpha=0.7)
        plt.plot(time_axis, preds[:, 0], '-x', label='Pred x', markersize=2, alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('x')
        plt.title('Time Series: Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        results_path = os.path.join(args.save_dir, 'training_results.png')
    else:
        # Inference-only mode: just show predictions
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(true[:, 0], true[:, 1], '-o', label='True', markersize=3, alpha=0.7)
        plt.plot(preds[:, 0], preds[:, 1], '-x', label='Predicted', markersize=3, alpha=0.7)
        plt.xlabel('x')
        plt.ylabel('xdot')
        plt.title('Phase Space: Test Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        time_axis = np.arange(len(true))
        plt.plot(time_axis, true[:, 0], '-o', label='True x', markersize=2, alpha=0.7)
        plt.plot(time_axis, preds[:, 0], '-x', label='Pred x', markersize=2, alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('x')
        plt.title('Time Series: Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        results_path = os.path.join(args.save_dir, 'inference_results.png')
    
    plt.tight_layout()
    plt.savefig(results_path, dpi=150)
    print(f"\nResults saved to '{results_path}'")
    plt.close()
    
    # Save final model (only if training was done)
    if not args.inference_only:
        final_model_path = os.path.join(args.save_dir, 'final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to '{final_model_path}'")
    
    # Save metrics
    metrics_path = os.path.join(args.save_dir, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Test Set Metrics (from evaluation.py)\n")
        f.write("="*50 + "\n")
        f.write(f"1-step MSE:           {metrics['one_step_mse']:.6f}\n")
        f.write(f"Reconstruction Error: {metrics['reconstruction_error']:.6f}\n")
        f.write(f"Spectral Radius:      {metrics['spectral_radius']:.4f}\n")
        f.write(f"Divergence Rate:      {metrics['divergence_rate']:.6f}\n")
        if 'chamfer_distance' in metrics:
            f.write(f"Chamfer Distance:     {metrics['chamfer_distance']:.6f}\n")
        f.write("\nMulti-step NRMSE:\n")
        for key, val in metrics.items():
            if key.startswith('nrmse_'):
                horizon = key.replace('nrmse_', '').replace('step', '')
                f.write(f"  Horizon {horizon:>3}: {val:.6f}\n")
    print(f"Metrics saved to '{metrics_path}'")


if __name__ == "__main__":
    main()
