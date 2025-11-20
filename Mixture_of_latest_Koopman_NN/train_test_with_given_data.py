"""
Train and test Koopman Autoencoder with data from CSV file

This script:
1. Loads data from CSV file
2. Splits data into train/validation/test sets
3. Trains the Koopman Autoencoder model
4. Validates during training
5. Evaluates on test set
6. Saves model and results
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
    hankel_stack_batch, 
    compute_hankel_svd
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
            # Trajectories have different lengths, pad or truncate
            min_len = min(len(t) for t in trajs)
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


def prepare_data_from_trajectories(trajs):
    """
    Convert trajectory data to training triplets (x_t, x_{t+1}, x_{t+2})
    
    Args:
        trajs: array of shape (n_traj, n_steps, n_x)
    
    Returns:
        xt, xt1, xt2: tensors of shape (N, n_x) where N is total number of triplets
    """
    n_traj, n_steps, n_x = trajs.shape
    
    # Flatten trajectories and create triplets
    all_xt = []
    all_xt1 = []
    all_xt2 = []
    
    for traj in trajs:
        # For each trajectory, create triplets
        if n_steps >= 3:
            xt = traj[:-2]
            xt1 = traj[1:-1]
            xt2 = traj[2:]
            all_xt.append(xt)
            all_xt1.append(xt1)
            all_xt2.append(xt2)
    
    # Concatenate all trajectories
    xt = np.concatenate(all_xt, axis=0)
    xt1 = np.concatenate(all_xt1, axis=0)
    xt2 = np.concatenate(all_xt2, axis=0)
    
    # Convert to tensors
    xt_t = torch.tensor(xt, dtype=torch.float32)
    xt1_t = torch.tensor(xt1, dtype=torch.float32)
    xt2_t = torch.tensor(xt2, dtype=torch.float32)
    
    return xt_t, xt1_t, xt2_t


def sample_sequence_batch(all_X, batch_size, Tseq=8, device='cpu'):
    """
    Sample random contiguous sequences from the dataset
    
    Args:
        all_X: tensor of all states, shape (N, n_x)
        batch_size: number of sequences to sample
        Tseq: length of each sequence
        device: device to place tensors on
    
    Returns:
        tensor of shape (batch_size, Tseq, n_x)
    """
    max_start = all_X.shape[0] - Tseq
    if max_start <= 0:
        # If dataset is too small, pad or repeat
        return all_X.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    starts = np.random.randint(0, max_start, size=batch_size)
    seqs = [all_X[s:s+Tseq] for s in starts]
    return torch.stack(seqs, dim=0).to(device)


def compute_loss_batch(model, x0, x1, x2, all_X, device, n_x, n_z, 
                       hankel_batch_size=64, hankel_Tseq=8, L=4):
    """
    Compute all loss terms for a batch
    
    Returns:
        dict with individual losses and total loss
    """
    # Hyperparameters for loss weights
    lam_rec, lam_lin, lam_ms = 1.0, 10.0, 2.0
    lam_edmd, lam_hankel = 1.0, 1.0
    lam_bi, lam_spec, lam_sparse = 1.0, 1.0, 1e-4
    
    mse = nn.MSELoss()
    
    out0 = model(x0)
    out1 = model(x1)
    out2 = model(x2)
    
    z0 = out0['z']
    z1 = out1['z']
    z2 = out2['z']
    xrec0 = out0['x_rec']
    
    # Reconstruction loss
    loss_rec = mse(xrec0, x0)
    
    # Latent linearity (1-step)
    z_pred = z0 @ model.A_f.T
    loss_lin = mse(z_pred, z1)
    
    # Multi-step (2-step)
    z_pred2 = z_pred @ model.A_f.T
    loss_ms = mse(z_pred2, z2)
    
    # eDMD observables regression (only if observables are enabled)
    if model.use_observables and out0['g'] is not None:
        g0 = out0['g']
        g1 = out1['g']
        g_pred = g0 @ model.A_g.T
        loss_edmd = mse(g_pred, g1)
    else:
        loss_edmd = torch.tensor(0.0, device=device)
    
    # Bidirectional constraint
    Id = torch.eye(model.A_f.shape[0], device=device)
    loss_bi = (model.A_f @ model.A_b - Id).norm()**2 + (model.A_b @ model.A_f - Id).norm()**2
    
    # Hankel term
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
    
    # Spectral penalty
    loss_spec = spectral_radius_penalty(model.A_f, iters=8, target=1.1)
    
    # Sparsity
    sparsity_term = model.sparsity_loss(mode="l1")
    
    # Total loss
    loss = (lam_rec * loss_rec + lam_lin * loss_lin + lam_ms * loss_ms +
            lam_edmd * loss_edmd + lam_hankel * loss_hankel + lam_bi * loss_bi +
            lam_spec * loss_spec + lam_sparse * sparsity_term)
    
    return {
        'total': loss,
        'rec': loss_rec,
        'lin': loss_lin,
        'ms': loss_ms,
        'edmd': loss_edmd,
        'hankel': loss_hankel,
        'bi': loss_bi,
        'spec': loss_spec
    }


def train_model(model, train_loader, val_loader, all_X_train, device, n_epochs=40, 
                batch_size=256, hankel_batch_size=64, hankel_Tseq=8, L=4, 
                n_x=2, n_z=20, save_dir='./'):
    """
    Train the Koopman Autoencoder model with validation
    
    Args:
        model: KoopmanAE model instance
        train_loader: DataLoader with training triplets
        val_loader: DataLoader with validation triplets
        all_X_train: all training state data for Hankel sampling
        device: device to train on
        n_epochs: number of training epochs
        batch_size: batch size for main training
        hankel_batch_size: batch size for Hankel computation
        hankel_Tseq: sequence length for Hankel computation
        L: Hankel window length
        n_x: state dimension
        n_z: latent dimension
        save_dir: directory to save model checkpoints (will be created if doesn't exist)
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_log = []
    val_log = []
    best_val_loss = float('inf')
    
    for ep in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for x0, x1, x2 in train_loader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            
            losses = compute_loss_batch(
                model, x0, x1, x2, all_X_train, device, n_x, n_z,
                hankel_batch_size, hankel_Tseq, L
            )
            
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
            for x0, x1, x2 in val_loader:
                x0 = x0.to(device)
                x1 = x1.to(device)
                x2 = x2.to(device)
                
                losses = compute_loss_batch(
                    model, x0, x1, x2, all_X_train, device, n_x, n_z,
                    hankel_batch_size, hankel_Tseq, L
                )
                val_losses.append(losses['total'].item())
        
        avg_val_loss = np.mean(val_losses)
        val_log.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        
        if ep % 5 == 0:
            print(f"Epoch {ep:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    return train_log, val_log


def evaluate_model(model, test_traj, device, n_steps=None):
    """
    Evaluate model by predicting forward in time
    
    Args:
        model: trained KoopmanAE model
        test_traj: test trajectory array of shape (n_steps, n_x)
        device: device to run on
        n_steps: number of steps to predict (if None, uses full trajectory)
    
    Returns:
        true_traj: true trajectory
        pred_traj: predicted trajectory
    """
    model.eval()
    if n_steps is None:
        n_steps = len(test_traj) - 1
    
    with torch.no_grad():
        x0 = torch.tensor(test_traj[0], dtype=torch.float32).unsqueeze(0).to(device)
        z0 = model.encoder(x0)
        
        zs = [z0]
        z = z0
        for i in range(n_steps):
            z = z @ model.A_f.T
            zs.append(z)
        
        zs = torch.cat(zs, dim=0)
        preds = model.decoder(zs).cpu().numpy()
        true = test_traj[:n_steps+1]
    
    return true, preds


def compute_metrics(true, preds):
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
                       help='Latent dimension (default: 10×state_dim, auto-computed)')
    parser.add_argument('--p', type=int, default=20,
                       help='Observables dimension (default: 20)')
    parser.add_argument('--n_epochs', type=int, default=40,
                       help='Number of training epochs (default: 40)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--save_dir', type=str, default='./',
                       help='Directory to save model and results (default: ./)')
    
    args = parser.parse_args()
    
    # Validate split ratios
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "Train, validation, and test ratios must sum to 1.0"
    
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    trajs, n_x = load_data_from_csv(
        args.csv_path,
        state_columns=args.state_columns,
        traj_id_column=args.traj_id_column,
        time_column=args.time_column
    )
    
    # Compute latent dimension: 10 × state dimension (or use provided value)
    if args.n_z is None:
        n_z = 10 * n_x
        print(f"Auto-computing latent dimension: n_z = 10 × {n_x} = {n_z}")
    else:
        n_z = args.n_z
        print(f"Using provided latent dimension: n_z = {n_z}")
    
    # Split trajectories temporally (chronologically) to avoid data leakage
    # For each trajectory: early time steps -> train, middle -> val, late -> test
    n_traj, n_steps, n_x = trajs.shape
    
    # Calculate split points for each trajectory
    train_end = int(n_steps * args.train_ratio)
    val_end = int(n_steps * (args.train_ratio + args.val_ratio))
    
    # Split each trajectory temporally
    train_trajs = trajs[:, :train_end, :]  # Early time steps
    val_trajs = trajs[:, train_end:val_end, :]  # Middle time steps
    test_trajs = trajs[:, val_end:, :]  # Late time steps
    
    print(f"\nTemporal split per trajectory:")
    print(f"  Train: steps 0 to {train_end-1} ({train_end/n_steps*100:.1f}%)")
    print(f"  Val:   steps {train_end} to {val_end-1} ({(val_end-train_end)/n_steps*100:.1f}%)")
    print(f"  Test:  steps {val_end} to {n_steps-1} ({(n_steps-val_end)/n_steps*100:.1f}%)")
    
    # Prepare triplets from each split
    xt_train, xt1_train, xt2_train = prepare_data_from_trajectories(train_trajs)
    xt_val, xt1_val, xt2_val = prepare_data_from_trajectories(val_trajs)
    xt_test, xt1_test, xt2_test = prepare_data_from_trajectories(test_trajs)
    
    total_samples = len(xt_train) + len(xt_val) + len(xt_test)
    print(f"\nData split (temporal, no shuffling):")
    print(f"  Training:   {len(xt_train)} samples ({len(xt_train)/total_samples*100:.1f}%)")
    print(f"  Validation: {len(xt_val)} samples ({len(xt_val)/total_samples*100:.1f}%)")
    print(f"  Test:       {len(xt_test)} samples ({len(xt_test)/total_samples*100:.1f}%)")
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(xt_train, xt1_train, xt2_train),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(xt_val, xt1_val, xt2_val),
        batch_size=args.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(xt_test, xt1_test, xt2_test),
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Flatten all training data for Hankel sampling
    all_X_train = torch.cat([xt_train, xt1_train, xt2_train], dim=0)
    
    # Create model
    model = KoopmanAE(n_x=n_x, n_z=n_z, p=args.p, use_observables=False).to(device)
    print(f"\nModel created: n_x={n_x}, n_z={n_z} (10×{n_x}), p={args.p}")
    
    # Train model
    print("\nStarting training...")
    train_log, val_log = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        all_X_train=all_X_train,
        device=device,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        hankel_batch_size=64,
        hankel_Tseq=8,
        L=4,
        n_x=n_x,
        n_z=n_z,
        save_dir=args.save_dir
    )
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    print("\nLoaded best model based on validation loss")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_losses = []
    
    with torch.no_grad():
        for x0, x1, x2 in test_loader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            
            losses = compute_loss_batch(
                model, x0, x1, x2, all_X_train, device, n_x, n_z,
                64, 8, 4
            )
            test_losses.append(losses['total'].item())
    
    avg_test_loss = np.mean(test_losses)
    print(f"Test Loss: {avg_test_loss:.6f}")
    
    # Predict on a sample test trajectory
    # Reconstruct test trajectories from test data
    test_traj_sample = torch.cat([xt_test[:50], xt1_test[49:50], xt2_test[49:50]], dim=0).numpy()
    true, preds = evaluate_model(model, test_traj_sample, device, n_steps=50)
    
    # Compute metrics
    metrics = compute_metrics(true, preds)
    print(f"\nTest Metrics:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  Mean Phase Error: {metrics['mean_phase_error']:.6f}")
    print(f"  Max Phase Error: {metrics['max_phase_error']:.6f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_log, label='Train Loss', linewidth=2)
    plt.plot(val_log, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Phase space plot
    plt.subplot(1, 3, 2)
    plt.plot(true[:, 0], true[:, 1], '-o', label='True', markersize=3, alpha=0.7)
    plt.plot(preds[:, 0], preds[:, 1], '-x', label='Predicted', markersize=3, alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('xdot')
    plt.title('Phase Space: Test Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Time series plot
    plt.subplot(1, 3, 3)
    time_axis = np.arange(len(true))
    plt.plot(time_axis, true[:, 0], '-o', label='True x', markersize=2, alpha=0.7)
    plt.plot(time_axis, preds[:, 0], '-x', label='Pred x', markersize=2, alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('x')
    plt.title('Time Series: Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    results_path = os.path.join(args.save_dir, 'training_results.png')
    plt.savefig(results_path, dpi=150)
    print(f"\nResults saved to '{results_path}'")
    plt.show()
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to '{final_model_path}'")
    
    # Save metrics
    metrics_path = os.path.join(args.save_dir, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Test Set Metrics\n")
        f.write("="*50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        f.write(f"\nTest Loss: {avg_test_loss:.6f}\n")
    print(f"Metrics saved to '{metrics_path}'")


if __name__ == "__main__":
    main()

