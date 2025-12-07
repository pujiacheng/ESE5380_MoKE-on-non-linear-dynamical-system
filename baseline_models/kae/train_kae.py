"""
Train Simplified Koopman Autoencoder Baseline

This script trains the simplified Koopman AE with only:
- Reconstruction loss: ||x - Decoder(Encoder(x))||²
- Koopman Linearity loss: ||z(t+1) - A_f @ z(t)||²

No other loss terms (no Hankel, no bidirectional, no eDMD, etc.)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from baseline_models.koopman_ae_baseline import KoopmanAEBaseline


def load_data_from_csv(csv_path, state_columns=None, traj_id_column=None, time_column=None):
    """
    Load trajectory data from CSV file
    
    Args:
        csv_path: path to CSV file
        state_columns: list of column names for state variables
        traj_id_column: column name for trajectory ID
        time_column: column name for time (optional, will be ignored)
    
    Returns:
        trajs: array of shape (n_traj, n_steps, n_x)
        n_x: state dimension
        state_columns: list of state column names
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
        traj_ids = df[traj_id_column].unique()
        trajs = []
        for traj_id in traj_ids:
            traj_data = df[df[traj_id_column] == traj_id][state_columns].values
            if len(traj_data) >= 2:  # Need at least 2 points for pairs
                trajs.append(traj_data.astype(np.float32))
        
        # Convert to regular array if all trajectories have same length
        try:
            trajs = np.stack(trajs)
        except ValueError:
            min_len = min(len(t) for t in trajs)
            trajs = np.stack([t[:min_len] for t in trajs])
        print(f"Found {len(trajs)} trajectories")
    else:
        if len(state_data) < 2:
            raise ValueError("Need at least 2 data points for training")
        trajs = state_data.reshape(1, -1, n_x).astype(np.float32)
        print(f"Single trajectory with {len(state_data)} time steps")
    
    print(f"Data shape: {trajs.shape} (n_traj, n_steps, n_x)")
    return trajs, n_x, state_columns


def prepare_time_pairs(trajs):
    """
    Prepare time pairs (x(t), x(t+1)) from trajectories
    
    Args:
        trajs: array of shape (n_traj, n_steps, n_x)
    
    Returns:
        x_t: array of shape (n_samples, n_x) - current states
        x_t1: array of shape (n_samples, n_x) - next states
    """
    n_traj, n_steps, n_x = trajs.shape
    x_t_list, x_t1_list = [], []
    
    for traj in trajs:
        for t in range(n_steps - 1):
            x_t_list.append(traj[t])
            x_t1_list.append(traj[t + 1])
    
    x_t = np.array(x_t_list, dtype=np.float32)
    x_t1 = np.array(x_t1_list, dtype=np.float32)
    
    return x_t, x_t1


def compute_loss_batch(model, x_t, x_t1, device, lam_rec=1.0, lam_lin=10.0):
    """
    Compute loss for a batch
    
    Args:
        model: KoopmanAEBaseline model
        x_t: current states (batch_size, n_x)
        x_t1: next states (batch_size, n_x)
        device: device to compute on
        lam_rec: weight for reconstruction loss
        lam_lin: weight for linearity loss
    
    Returns:
        dict with individual losses and total loss
    """
    mse = nn.MSELoss()
    
    # Forward pass for x(t)
    out_t = model(x_t)
    z_t = out_t['z']
    x_rec_t = out_t['x_rec']
    
    # Forward pass for x(t+1)
    out_t1 = model(x_t1)
    z_t1 = out_t1['z']
    
    # Reconstruction loss: ||x(t) - Decoder(Encoder(x(t)))||²
    loss_rec = mse(x_rec_t, x_t)
    
    # Koopman Linearity loss: ||z(t+1) - A_f @ z(t)||²
    z_pred_t1 = z_t @ model.A_f.T
    loss_lin = mse(z_pred_t1, z_t1)
    
    # Total loss
    loss = lam_rec * loss_rec + lam_lin * loss_lin
    
    return {
        'total': loss,
        'rec': loss_rec,
        'lin': loss_lin
    }


def train_model(model, train_loader, val_loader, device, n_epochs=40, 
                lam_rec=1.0, lam_lin=10.0, save_dir='./'):
    """
    Train the Koopman AE Baseline model
    
    Args:
        model: KoopmanAEBaseline model instance
        train_loader: DataLoader with training pairs
        val_loader: DataLoader with validation pairs
        device: device to train on
        n_epochs: number of training epochs
        lam_rec: weight for reconstruction loss
        lam_lin: weight for linearity loss
        save_dir: directory to save model checkpoints
    """
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_log = []
    val_log = []
    best_val_loss = float('inf')
    
    for ep in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_rec_losses = []
        train_lin_losses = []
        
        for x_t, x_t1 in train_loader:
            x_t = x_t.to(device)
            x_t1 = x_t1.to(device)
            
            optimizer.zero_grad()
            loss_dict = compute_loss_batch(model, x_t, x_t1, device, lam_rec, lam_lin)
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_rec_losses.append(loss_dict['rec'].item())
            train_lin_losses.append(loss_dict['lin'].item())
        
        avg_train_loss = np.mean(train_losses)
        avg_train_rec = np.mean(train_rec_losses)
        avg_train_lin = np.mean(train_lin_losses)
        train_log.append({
            'total': avg_train_loss,
            'rec': avg_train_rec,
            'lin': avg_train_lin
        })
        
        # Validation phase
        model.eval()
        val_losses = []
        val_rec_losses = []
        val_lin_losses = []
        
        with torch.no_grad():
            for x_t, x_t1 in val_loader:
                x_t = x_t.to(device)
                x_t1 = x_t1.to(device)
                
                loss_dict = compute_loss_batch(model, x_t, x_t1, device, lam_rec, lam_lin)
                val_losses.append(loss_dict['total'].item())
                val_rec_losses.append(loss_dict['rec'].item())
                val_lin_losses.append(loss_dict['lin'].item())
        
        avg_val_loss = np.mean(val_losses) if len(val_losses) > 0 else float('inf')
        avg_val_rec = np.mean(val_rec_losses) if len(val_rec_losses) > 0 else 0.0
        avg_val_lin = np.mean(val_lin_losses) if len(val_lin_losses) > 0 else 0.0
        val_log.append({
            'total': avg_val_loss,
            'rec': avg_val_rec,
            'lin': avg_val_lin
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{n_epochs}: "
                  f"Train Loss = {avg_train_loss:.6f} "
                  f"(Rec: {avg_train_rec:.6f}, Lin: {avg_train_lin:.6f}), "
                  f"Val Loss = {avg_val_loss:.6f} "
                  f"(Rec: {avg_val_rec:.6f}, Lin: {avg_val_lin:.6f})")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    
    # Plot training curves
    epochs = range(1, n_epochs + 1)
    plt.figure(figsize=(15, 5))
    
    # Total loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, [l['total'] for l in train_log], label='Train Total', linewidth=2)
    plt.plot(epochs, [l['total'] for l in val_log], label='Val Total', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Reconstruction loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, [l['rec'] for l in train_log], label='Train Rec', linewidth=2)
    plt.plot(epochs, [l['rec'] for l in val_log], label='Val Rec', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Linearity loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, [l['lin'] for l in train_log], label='Train Lin', linewidth=2)
    plt.plot(epochs, [l['lin'] for l in val_log], label='Val Lin', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Linearity Loss')
    plt.title('Koopman Linearity Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    return train_log, val_log


def main():
    parser = argparse.ArgumentParser(description='Train Koopman AE Baseline')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--state_columns', type=str, nargs='+', default=None,
                       help='Column names for state variables')
    parser.add_argument('--traj_id_column', type=str, default=None,
                       help='Column name for trajectory ID')
    parser.add_argument('--time_column', type=str, default=None,
                       help='Column name for time (optional)')
    parser.add_argument('--n_z', type=int, default=None,
                       help='Latent dimension (default: 10×state_dim)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training data ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation data ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test data ratio (default: 0.15)')
    parser.add_argument('--n_epochs', type=int, default=40,
                       help='Number of training epochs (default: 40)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--lam_rec', type=float, default=1.0,
                       help='Weight for reconstruction loss (default: 1.0)')
    parser.add_argument('--lam_lin', type=float, default=10.0,
                       help='Weight for linearity loss (default: 10.0)')
    parser.add_argument('--save_dir', type=str, default='./results_kae',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Validate split ratios
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "Train, validation, and test ratios must sum to 1.0"
    
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    trajs, n_x, state_columns = load_data_from_csv(
        args.csv_path, args.state_columns, args.traj_id_column, args.time_column
    )
    
    # Temporal split per trajectory
    n_traj, n_steps, _ = trajs.shape
    train_end = int(n_steps * args.train_ratio)
    val_end = int(n_steps * (args.train_ratio + args.val_ratio))
    
    train_trajs = trajs[:, :train_end, :]
    val_trajs = trajs[:, train_end:val_end, :]
    test_trajs = trajs[:, val_end:, :]
    
    print(f"\nTemporal split per trajectory:")
    print(f"  Train: steps 0 to {train_end-1} ({args.train_ratio*100:.1f}%)")
    print(f"  Val:   steps {train_end} to {val_end-1} ({args.val_ratio*100:.1f}%)")
    print(f"  Test:  steps {val_end} to {n_steps-1} ({args.test_ratio*100:.1f}%)")
    
    # Prepare time pairs
    x_train, x_train_next = prepare_time_pairs(train_trajs)
    x_val, x_val_next = prepare_time_pairs(val_trajs)
    x_test, x_test_next = prepare_time_pairs(test_trajs)
    
    print(f"\nTime pairs:")
    print(f"  Train: {len(x_train)} pairs")
    print(f"  Val:   {len(x_val)} pairs")
    print(f"  Test:  {len(x_test)} pairs")
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(x_train_next)),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val), torch.tensor(x_val_next)),
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Set latent dimension
    if args.n_z is None:
        n_z = 10 * n_x
    else:
        n_z = args.n_z
    
    # Create model
    model = KoopmanAEBaseline(n_x=n_x, n_z=n_z).to(device)
    print(f"\nCreated Koopman AE Baseline model: n_x={n_x}, n_z={n_z}")
    print(f"Loss weights: λ_rec={args.lam_rec}, λ_lin={args.lam_lin}")
    
    # Train model
    print(f"\nStarting training for {args.n_epochs} epochs...")
    train_log, val_log = train_model(
        model, train_loader, val_loader, device, args.n_epochs,
        args.lam_rec, args.lam_lin, args.save_dir
    )
    
    # Save test data for evaluation
    test_data_path = os.path.join(args.save_dir, 'test_data.npz')
    np.savez(test_data_path,
              x_test=x_test,
              x_test_next=x_test_next,
              state_columns=state_columns)
    print(f"\nTest data saved to {test_data_path} for evaluation")
    
    print(f"\nTraining complete! Model saved to {args.save_dir}")


if __name__ == '__main__':
    main()

