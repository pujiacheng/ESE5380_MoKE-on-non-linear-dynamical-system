"""
Train eDMD Baseline Model

This script trains the eDMD model with dictionary functions.
The model learns a linear operator K such that:
    φ(x(t+1)) ≈ K @ φ(x(t))

Loss functions:
- Observables prediction loss: ||φ(x(t+1)) - K @ φ(x(t))||²
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from baseline_models.edmd_baseline import EDMDModel, DictionaryFunctions


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


def compute_loss_batch(model, x_t, x_t1, device):
    """
    Compute loss for a batch
    
    Args:
        model: EDMDModel
        x_t: current states (batch_size, n_x)
        x_t1: next states (batch_size, n_x)
        device: device to compute on
    
    Returns:
        dict with loss
    """
    mse = nn.MSELoss()
    
    # Compute observables
    phi_t = model.dictionary(x_t)      # φ(x(t))
    phi_t1 = model.dictionary(x_t1)    # φ(x(t+1))
    
    # Predict next observables
    phi_t1_pred = phi_t @ model.K.T   # K @ φ(x(t))
    
    # Loss: ||φ(x(t+1)) - K @ φ(x(t))||²
    loss = mse(phi_t1_pred, phi_t1)
    
    return {
        'total': loss,
        'observables': loss
    }


def train_model(model, train_loader, val_loader, device, n_epochs=40, save_dir='./'):
    """
    Train the eDMD model
    
    Args:
        model: EDMDModel instance
        train_loader: DataLoader with training pairs
        val_loader: DataLoader with validation pairs
        device: device to train on
        n_epochs: number of training epochs
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
        
        for x_t, x_t1 in train_loader:
            x_t = x_t.to(device)
            x_t1 = x_t1.to(device)
            
            optimizer.zero_grad()
            loss_dict = compute_loss_batch(model, x_t, x_t1, device)
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        train_log.append({
            'total': avg_train_loss,
            'observables': avg_train_loss
        })
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for x_t, x_t1 in val_loader:
                x_t = x_t.to(device)
                x_t1 = x_t1.to(device)
                
                loss_dict = compute_loss_batch(model, x_t, x_t1, device)
                val_losses.append(loss_dict['total'].item())
        
        avg_val_loss = np.mean(val_losses) if len(val_losses) > 0 else float('inf')
        val_log.append({
            'total': avg_val_loss,
            'observables': avg_val_loss
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{n_epochs}: "
                  f"Train Loss = {avg_train_loss:.6f}, "
                  f"Val Loss = {avg_val_loss:.6f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    
    # Plot training curves
    epochs = range(1, n_epochs + 1)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [l['total'] for l in train_log], label='Train', linewidth=2)
    plt.plot(epochs, [l['total'] for l in val_log], label='Val', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [l['observables'] for l in train_log], label='Train Observables', linewidth=2)
    plt.plot(epochs, [l['observables'] for l in val_log], label='Val Observables', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Observables Loss')
    plt.title('Observables Prediction Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    return train_log, val_log


def main():
    parser = argparse.ArgumentParser(description='Train eDMD Baseline')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--state_columns', type=str, nargs='+', default=None,
                       help='Column names for state variables')
    parser.add_argument('--traj_id_column', type=str, default=None,
                       help='Column name for trajectory ID')
    parser.add_argument('--time_column', type=str, default=None,
                       help='Column name for time (optional)')
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
    parser.add_argument('--save_dir', type=str, default='./results_edmd',
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
    
    # Check state dimension
    if n_x != 2:
        print(f"Warning: Dictionary functions are designed for 2D systems (x, xdot).")
        print(f"Current system has {n_x} dimensions. Results may vary.")
    
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
    
    # Create model
    n_obs = 10  # Dictionary functions: [x, xdot, x^2, x*xdot, xdot^2, x^3, sin(x), cos(x), sin(xdot), cos(xdot)]
    model = EDMDModel(n_x=n_x, n_obs=n_obs).to(device)
    print(f"\nCreated eDMD model: n_x={n_x}, n_obs={n_obs}")
    print(f"Dictionary functions: [x, xdot, x^2, x*xdot, xdot^2, x^3, sin(x), cos(x), sin(xdot), cos(xdot)]")
    
    # Train model
    print(f"\nStarting training for {args.n_epochs} epochs...")
    train_log, val_log = train_model(
        model, train_loader, val_loader, device, args.n_epochs, args.save_dir
    )
    
    # Save test data for evaluation
    test_data_path = os.path.join(args.save_dir, 'test_data.npz')
    np.savez(test_data_path,
              x_test=x_test,
              x_test_next=x_test_next,
              state_columns=state_columns)
    print(f"\nTest data saved to {test_data_path} for evaluation")
    
    # Print final Koopman operator
    print(f"\nFinal Koopman operator K (first 5x5 block):")
    K_np = model.K.detach().cpu().numpy()
    print(K_np[:5, :5])
    
    print(f"\nTraining complete! Model saved to {args.save_dir}")


if __name__ == '__main__':
    main()

