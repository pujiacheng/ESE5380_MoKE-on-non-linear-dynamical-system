"""
Comprehensive Comparison Script: Train and Evaluate All Models

This script trains and evaluates:
1. Baselines: ARIMA (VAR), eDMD, Basic KAE
2. Advanced KAE (Model 4) - 1 Expert
3. MoE KAE (Model 5) - 2, 3, 4 Experts

All models are trained on the same data and evaluated with the same metrics.
Results are saved to CSV for easy comparison.

Usage:
    python train_all_models.py --system duffing --n_traj 10000 --n_epochs 100 --patience 10
"""

import os
import sys
import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add baseline_models to path
BASELINE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 
    '..', 'baseline_models'))
sys.path.insert(0, BASELINE_PATH)
sys.path.insert(0, os.path.join(BASELINE_PATH, 'arima'))
sys.path.insert(0, os.path.join(BASELINE_PATH, 'edmd'))
sys.path.insert(0, os.path.join(BASELINE_PATH, 'kae'))

# Import baseline models
from edmd_baseline import EDMDModel, compute_koopman_operator_analytical
from koopman_ae_baseline import KoopmanAEBaseline
from statsmodels.tsa.vector_ar.var_model import VAR

# Import our models
from data_simulation import (
    generate_duffing_dataset,
    generate_vanderpol_dataset,
    generate_lorenz_dataset,
    generate_double_pendulum_dataset
)
from koopman_mixture_neural_network import KoopmanAE, spectral_radius_penalty, compute_hankel_linearity_loss
from koopman_moe_neural_network import KoopmanMoE
from evaluation import (
    one_step_mse,
    multi_step_nrmse,
    chamfer_distance_phase,
    spectral_radius as compute_spectral_radius,
    long_horizon_divergence_rate,
    reconstruction_error
)


# ==============================================================================
# Data Generation
# ==============================================================================

def generate_dataset(system, n_traj, T, dt, noise_std=0.0):
    """Generate dataset for specified system"""
    if system == 'duffing':
        t, trajs = generate_duffing_dataset(n_traj=n_traj, T=T, dt=dt, noise_std=noise_std)
        config = {'name': 'Duffing Oscillator', 'n_x': 2, 'state_labels': ['x', 'xdot']}
    elif system == 'vanderpol':
        t, trajs = generate_vanderpol_dataset(n_traj=n_traj, T=T, dt=dt, noise_std=noise_std)
        config = {'name': 'Van der Pol Oscillator', 'n_x': 2, 'state_labels': ['x', 'xdot']}
    elif system == 'lorenz':
        t, trajs = generate_lorenz_dataset(n_traj=n_traj, T=T, dt=dt, noise_std=noise_std)
        config = {'name': 'Lorenz Attractor', 'n_x': 3, 'state_labels': ['x', 'y', 'z']}
    elif system == 'double_pendulum':
        t, trajs = generate_double_pendulum_dataset(n_traj=n_traj, T=T, dt=dt, noise_std=noise_std)
        config = {'name': 'Double Pendulum', 'n_x': 4, 'state_labels': ['θ1', 'θ2', 'ω1', 'ω2']}
    else:
        raise ValueError(f"Unknown system: {system}")
    
    return t, trajs, config


def prepare_training_data(trajs, horizons=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], hankel_seq_len=16):
    """Prepare training data with multi-step horizons"""
    n_traj, n_timesteps, n_x = trajs.shape
    max_horizon = max(horizons)
    
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
    
    result = {'x0': torch.tensor(np.concatenate(x0_list, axis=0), dtype=torch.float32)}
    for h in horizons:
        result[f'x{h}'] = torch.tensor(np.concatenate(data_lists[h], axis=0), dtype=torch.float32)
    
    # Hankel sequences
    sequences = []
    for traj in trajs:
        if n_timesteps >= hankel_seq_len:
            n_seqs = n_timesteps - hankel_seq_len + 1
            for start in range(0, n_seqs, hankel_seq_len // 2):
                seq = traj[start:start + hankel_seq_len]
                if len(seq) == hankel_seq_len:
                    sequences.append(seq)
    
    if sequences:
        result['sequences'] = torch.tensor(np.stack(sequences, axis=0), dtype=torch.float32)
    
    return result


# ==============================================================================
# Evaluation Functions
# ==============================================================================

def evaluate_predictions(model_name, true_trajs, pred_trajs, n_x, dt):
    """
    Evaluate predictions using metrics from evaluation.py
    
    Returns dict of metrics
    """
    metrics = {'model': model_name}
    
    # Stack trajectories for batch evaluation: (n_test, n_steps+1, n_x)
    n_test = len(true_trajs)
    
    all_one_step_mse = []
    all_nrmse = {h: [] for h in [1, 10, 20, 50, 100]}
    all_chamfer = []
    all_divergence = []
    all_recon = []
    
    for i in range(n_test):
        true = true_trajs[i]
        pred = pred_trajs[i]
        
        # Reshape for evaluation functions: (1, n_steps, n_x)
        true_3d = true[np.newaxis, :, :]
        pred_3d = pred[np.newaxis, :, :]
        
        # 1-step MSE
        if true.shape[0] >= 2:
            all_one_step_mse.append(one_step_mse(true_3d[:, :2, :], pred_3d[:, :2, :]))
        
        # Multi-step NRMSE
        n_steps = true.shape[0] - 1
        horizons = [h for h in [1, 10, 20, 50, 100] if h < n_steps]
        nrmse_dict = multi_step_nrmse(true_3d, pred_3d, horizons)
        for h, val in nrmse_dict.items():
            if h in all_nrmse:
                all_nrmse[h].append(val)
        
        # Chamfer distance (2D systems)
        if n_x >= 2:
            all_chamfer.append(chamfer_distance_phase(true_3d, pred_3d, dims=(0, 1)))
        
        # Divergence rate
        slope, _ = long_horizon_divergence_rate(true_3d, pred_3d)
        all_divergence.append(slope)
        
        # Reconstruction error (first point)
        all_recon.append(float(np.mean((true[0] - pred[0])**2)))
    
    # Average metrics
    metrics['one_step_mse'] = np.mean(all_one_step_mse) if all_one_step_mse else np.nan
    metrics['one_step_mse_std'] = np.std(all_one_step_mse) if all_one_step_mse else np.nan
    
    for h in [1, 10, 20, 50, 100]:
        if all_nrmse[h]:
            metrics[f'nrmse_{h}step'] = np.mean(all_nrmse[h])
            metrics[f'nrmse_{h}step_std'] = np.std(all_nrmse[h])
        else:
            metrics[f'nrmse_{h}step'] = np.nan
            metrics[f'nrmse_{h}step_std'] = np.nan
    
    if all_chamfer:
        metrics['chamfer_distance'] = np.mean(all_chamfer)
        metrics['chamfer_distance_std'] = np.std(all_chamfer)
    else:
        metrics['chamfer_distance'] = np.nan
        metrics['chamfer_distance_std'] = np.nan
    
    metrics['divergence_rate'] = np.mean(all_divergence)
    metrics['divergence_rate_std'] = np.std(all_divergence)
    
    metrics['reconstruction_error'] = np.mean(all_recon)
    metrics['reconstruction_error_std'] = np.std(all_recon)
    
    return metrics


# ==============================================================================
# Model Training Functions
# ==============================================================================

def train_var_model(train_data, save_dir):
    """Train VAR (ARIMA for multivariate) model"""
    print("\n" + "="*60)
    print("Training VAR Model (ARIMA Baseline)")
    print("="*60)
    
    model = VAR(train_data)
    lag_order_result = model.select_order(maxlags=10)
    optimal_lag = lag_order_result.selected_orders['aic']
    print(f"Selected VAR lag order: {optimal_lag}")
    
    fitted_model = model.fit(maxlags=optimal_lag)
    
    # Save model
    model_path = os.path.join(save_dir, 'var_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(fitted_model, f)
    print(f"VAR model saved to {model_path}")
    
    return fitted_model, optimal_lag


def predict_var(model, x0, n_steps, lag_order):
    """Predict using VAR model"""
    # VAR needs lag_order initial values
    # Use x0 repeated for simplicity (not ideal but consistent)
    x0_np = x0.cpu().numpy() if isinstance(x0, torch.Tensor) else x0
    
    # Create initial lag values
    last_values = np.tile(x0_np, (lag_order, 1))
    
    predictions = model.forecast(last_values, steps=n_steps)
    # Prepend initial condition
    full_pred = np.vstack([x0_np, predictions])
    return full_pred


def train_edmd_model(train_x0, train_x1, n_x, save_dir, device):
    """Train eDMD model"""
    print("\n" + "="*60)
    print("Training eDMD Model")
    print("="*60)
    
    model = EDMDModel(n_x=n_x).to(device)
    
    # Compute observables
    with torch.no_grad():
        phi_t = model(train_x0)
        phi_t1 = model(train_x1)
    
    # Compute Koopman operator analytically
    K = compute_koopman_operator_analytical(phi_t, phi_t1, reg=1e-6)
    model.K.data = K.to(device)
    
    print(f"eDMD Koopman operator computed (shape: {K.shape})")
    
    # Compute spectral radius
    eigvals = np.linalg.eigvals(K.numpy())
    rho = np.max(np.abs(eigvals))
    print(f"Spectral radius: {rho:.4f}")
    
    # Save model
    model_path = os.path.join(save_dir, 'edmd_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"eDMD model saved to {model_path}")
    
    return model


def save_checkpoint(model, optimizer, epoch, best_loss, patience_counter, save_path):
    """Save full training checkpoint for resumption"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'patience_counter': patience_counter,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load checkpoint and resume training state"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        patience_counter = checkpoint['patience_counter']
        print(f"Resuming from epoch {start_epoch}, best_loss={best_loss:.6f}")
        return start_epoch, best_loss, patience_counter
    return 0, float('inf'), 0


def train_kae_baseline(train_loader, n_x, n_z, device, n_epochs, patience, save_dir, resume=True):
    """Train simplified KAE baseline with checkpointing for resumption"""
    print("\n" + "="*60)
    print("Training KAE Baseline (Simplified)")
    print("="*60)
    
    model = KoopmanAEBaseline(n_x=n_x, n_z=n_z).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    
    # Checkpoint paths
    checkpoint_path = os.path.join(save_dir, 'kae_baseline_checkpoint.pth')
    best_model_path = os.path.join(save_dir, 'kae_baseline_best.pth')
    
    # Try to resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    patience_counter = 0
    
    if resume:
        start_epoch, best_loss, patience_counter = load_checkpoint(
            model, optimizer, checkpoint_path, device
        )
    
    best_state = None
    
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            x0, x1 = batch[0].to(device), batch[1].to(device)
            
            # Loss: reconstruction + prediction + linearity
            out = model(x0)
            loss_rec = mse(out['x_rec'], x0)
            
            x1_pred = model.predict_next(x0)
            loss_pred = mse(x1_pred, x1)
            
            z0 = out['z']
            z1_true = model.encoder(x1)
            z1_pred = z0 @ model.A_f.T
            loss_lin = mse(z1_pred, z1_true)
            
            loss = 2.0 * loss_rec + 15.0 * loss_pred + 12.0 * loss_lin
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # Save checkpoint every epoch for resumption
        save_checkpoint(model, optimizer, epoch, best_loss, patience_counter, checkpoint_path)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f} ✓ Best (saved)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f} (patience {patience_counter}/{patience})")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    elif os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    return model


def train_advanced_kae(train_loader, val_loader, n_x, n_z, device, n_epochs, patience, 
                       save_dir, train_sequences=None, resume=True):
    """Train Advanced KAE (Model 4) with all loss components and checkpointing"""
    print("\n" + "="*60)
    print("Training Advanced KAE (Model 4 - 1 Expert)")
    print("="*60)
    
    model = KoopmanAE(n_x=n_x, n_z=n_z).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    
    # Checkpoint paths
    checkpoint_path = os.path.join(save_dir, 'advanced_kae_checkpoint.pth')
    best_model_path = os.path.join(save_dir, 'advanced_kae_best.pth')
    
    horizons = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Try to resume
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if resume:
        start_epoch, best_val_loss, patience_counter = load_checkpoint(
            model, optimizer, checkpoint_path, device
        )
    
    best_state = None
    
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            data_batch = {'x0': batch[0].to(device)}
            for idx, h in enumerate(horizons):
                data_batch[f'x{h}'] = batch[idx + 1].to(device)
            
            x0 = data_batch['x0']
            x1 = data_batch['x1']
            
            # Forward pass
            out = model(x0)
            loss_rec = mse(out['x_rec'], x0)
            
            # Prediction loss
            z0 = model.encoder(x0)
            z1_pred = z0 @ model.A_f.T
            x1_pred = model.decoder(z1_pred)
            loss_pred = mse(x1_pred, x1)
            
            # Multi-step linearity
            loss_lin = 0
            A_powers = {1: model.A_f}
            A_k = model.A_f.clone()
            for k in [10, 20, 30, 40, 50]:
                prev_k = horizons[horizons.index(k) - 1]
                for _ in range(k - prev_k):
                    A_k = A_k @ model.A_f
                A_powers[k] = A_k.clone()
            
            for k in horizons:
                x_k = data_batch[f'x{k}']
                zk_true = model.encoder(x_k)
                zk_pred = z0 @ A_powers[k].T
                loss_lin += mse(zk_pred, zk_true)
            loss_lin /= len(horizons)
            
            # Bidirectional + Spectral
            I = torch.eye(model.n_z, device=device)
            loss_bi = (model.A_f @ model.A_b - I).norm()**2
            loss_spec = spectral_radius_penalty(model.A_f, iters=8, target=1.005, lower=0.995)
            
            # Sparsity
            loss_sparse = model.sparsity_loss(mode="l1")
            
            loss = (2.0 * loss_rec + 15.0 * loss_pred + 12.0 * loss_lin +
                   1.0 * loss_bi + 5.0 * loss_spec + 1e-4 * loss_sparse)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                data_batch = {'x0': batch[0].to(device)}
                for idx, h in enumerate(horizons):
                    data_batch[f'x{h}'] = batch[idx + 1].to(device)
                
                x0 = data_batch['x0']
                x1 = data_batch['x1']
                out = model(x0)
                z0 = model.encoder(x0)
                z1_pred = z0 @ model.A_f.T
                x1_pred = model.decoder(z1_pred)
                val_loss += mse(x1_pred, x1).item()
                n_val += 1
        
        avg_val_loss = val_loss / n_val if n_val > 0 else float('inf')
        
        # Save checkpoint every epoch for resumption
        save_checkpoint(model, optimizer, epoch, best_val_loss, patience_counter, checkpoint_path)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch}: Train = {avg_loss:.6f}, Val = {avg_val_loss:.6f} ✓ Best")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"Epoch {epoch}: Train = {avg_loss:.6f}, Val = {avg_val_loss:.6f} (patience {patience_counter}/{patience})")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    elif os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    return model


def train_moe(train_loader, val_loader, n_x, n_z, n_experts, device, n_epochs, patience,
              save_dir, train_sequences=None, resume=True):
    """Train MoE Koopman (Model 5) with checkpointing for resumption"""
    print("\n" + "="*60)
    print(f"Training MoE Koopman ({n_experts} Experts)")
    print("="*60)
    
    model = KoopmanMoE(n_x=n_x, n_z=n_z, n_experts=n_experts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    
    # Checkpoint paths
    checkpoint_path = os.path.join(save_dir, f'moe_{n_experts}expert_checkpoint.pth')
    best_model_path = os.path.join(save_dir, f'moe_{n_experts}expert_best.pth')
    
    horizons = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Try to resume
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if resume:
        start_epoch, best_val_loss, patience_counter = load_checkpoint(
            model, optimizer, checkpoint_path, device
        )
    
    best_state = None
    
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            data_batch = {'x0': batch[0].to(device)}
            for idx, h in enumerate(horizons):
                data_batch[f'x{h}'] = batch[idx + 1].to(device)
            
            x0 = data_batch['x0']
            x1 = data_batch['x1']
            
            # Forward pass
            out = model(x0)
            weights0 = out['weights']
            loss_rec = mse(out['x_rec'], x0)
            
            # Prediction loss (blended)
            expert_preds = []
            for expert in model.experts:
                x1_pred = expert.predict_next(x0)
                expert_preds.append(x1_pred)
            x1_pred_blended = model.blending(expert_preds, weights0)
            loss_pred = mse(x1_pred_blended, x1)
            
            # Multi-step linearity per expert
            loss_lin = 0
            A_powers = {}
            for i, expert in enumerate(model.experts):
                A_powers[i] = {1: expert.A_f}
                A_k = expert.A_f.clone()
                for k in [10, 20, 30, 40, 50]:
                    prev_k = horizons[horizons.index(k) - 1]
                    for _ in range(k - prev_k):
                        A_k = A_k @ expert.A_f
                    A_powers[i][k] = A_k.clone()
            
            for k in horizons:
                x_k = data_batch[f'x{k}']
                for i, expert in enumerate(model.experts):
                    z0 = expert.encoder(x0)
                    zk_true = expert.encoder(x_k)
                    zk_pred = z0 @ A_powers[i][k].T
                    loss_lin += (weights0[:, i:i+1] * (zk_pred - zk_true)**2).mean()
            loss_lin /= len(horizons)
            
            # Load balancing
            avg_weights = weights0.mean(dim=0)
            target_weight = 1.0 / n_experts
            loss_balance = ((avg_weights - target_weight)**2).sum()
            
            # Bidirectional + Spectral per expert
            loss_bi = 0
            loss_spec = 0
            I = torch.eye(model.n_z, device=device)
            for expert in model.experts:
                loss_bi += (expert.A_f @ expert.A_b - I).norm()**2
                loss_spec += spectral_radius_penalty(expert.A_f, iters=8, target=1.005, lower=0.995)
            loss_bi /= n_experts
            loss_spec /= n_experts
            
            # Sparsity
            loss_sparse = model.sparsity_loss(mode="l1")
            
            loss = (2.0 * loss_rec + 15.0 * loss_pred + 12.0 * loss_lin +
                   1.0 * loss_balance + 1.0 * loss_bi + 5.0 * loss_spec + 1e-4 * loss_sparse)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # Validation
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                data_batch = {'x0': batch[0].to(device)}
                for idx, h in enumerate(horizons):
                    data_batch[f'x{h}'] = batch[idx + 1].to(device)
                
                x0 = data_batch['x0']
                x1 = data_batch['x1']
                out = model(x0)
                weights0 = out['weights']
                
                expert_preds = []
                for expert in model.experts:
                    x1_pred = expert.predict_next(x0)
                    expert_preds.append(x1_pred)
                x1_pred_blended = model.blending(expert_preds, weights0)
                val_loss += mse(x1_pred_blended, x1).item()
                n_val += 1
        
        avg_val_loss = val_loss / n_val if n_val > 0 else float('inf')
        
        # Save checkpoint every epoch for resumption
        save_checkpoint(model, optimizer, epoch, best_val_loss, patience_counter, checkpoint_path)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch}: Train = {avg_loss:.6f}, Val = {avg_val_loss:.6f} ✓ Best")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"Epoch {epoch}: Train = {avg_loss:.6f}, Val = {avg_val_loss:.6f} (patience {patience_counter}/{patience})")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    elif os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    return model


# ==============================================================================
# Prediction Functions
# ==============================================================================

def predict_pytorch_model(model, x0, n_steps, device, is_moe=False):
    """Predict using PyTorch model"""
    model.eval()
    with torch.no_grad():
        x0_tensor = torch.tensor(x0, dtype=torch.float32).unsqueeze(0).to(device)
        
        if is_moe:
            preds, _ = model.predict(x0_tensor, n_steps=n_steps)
            preds = preds.squeeze(1).cpu().numpy()
        else:
            # For single models
            if hasattr(model, 'predict_sequence'):
                preds = model.predict_sequence(x0_tensor, n_steps)
                preds = preds.squeeze(0).cpu().numpy()
            else:
                # Manual prediction
                z = model.encoder(x0_tensor)
                preds = [x0_tensor.cpu().numpy().squeeze()]
                for _ in range(n_steps):
                    z = z @ model.A_f.T
                    x = model.decoder(z)
                    preds.append(x.cpu().numpy().squeeze())
                preds = np.array(preds)
    
    return preds


# ==============================================================================
# Main Script
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train and compare all models')
    parser.add_argument('--system', type=str, default='duffing',
                       choices=['duffing', 'vanderpol', 'lorenz', 'double_pendulum'])
    parser.add_argument('--n_traj', type=int, default=10000)
    parser.add_argument('--T', type=float, default=100.0)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of trajectories for training (default: 0.8)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Fraction of trajectories for validation (default: 0.1)')
    parser.add_argument('--test_split', type=float, default=0.1,
                       help='Fraction of trajectories for testing (default: 0.1)')
    parser.add_argument('--output_dir', type=str, default='comparison_results')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume training from checkpoints if available')
    parser.add_argument('--run_dir', type=str, default=None,
                       help='Specific run directory to resume (e.g., duffing_20231207_143022)')
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create or use existing output directory
    if args.run_dir:
        # Resume from specific run directory
        output_dir = os.path.join(args.output_dir, args.run_dir)
        if not os.path.exists(output_dir):
            raise ValueError(f"Run directory does not exist: {output_dir}")
        print(f"Resuming from: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"{args.system}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    resume = args.resume or args.run_dir is not None
    
    # Validate split ratios
    total_split = args.train_split + args.val_split + args.test_split
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"train_split + val_split + test_split must equal 1.0, got {total_split}")
    
    # Generate data
    print(f"\n{'='*60}")
    print(f"Generating {args.system} data: {args.n_traj} trajectories")
    print(f"{'='*60}")
    
    t, trajs, config = generate_dataset(args.system, args.n_traj, args.T, args.dt)
    n_x = config['n_x']
    n_z = 5 * n_x  # Latent dimension
    
    # =========================================================================
    # PROPER 3-WAY SPLIT OF TRAJECTORIES
    # =========================================================================
    n_total = args.n_traj
    n_train_traj = int(n_total * args.train_split)
    n_val_traj = int(n_total * args.val_split)
    n_test_traj = n_total - n_train_traj - n_val_traj  # Remaining goes to test
    
    # Shuffle trajectories before splitting (for randomness)
    np.random.seed(42)
    indices = np.random.permutation(n_total)
    
    train_indices = indices[:n_train_traj]
    val_indices = indices[n_train_traj:n_train_traj + n_val_traj]
    test_indices = indices[n_train_traj + n_val_traj:]
    
    trajs_train = trajs[train_indices]
    trajs_val = trajs[val_indices]
    trajs_test = trajs[test_indices]
    
    print(f"\n{'='*60}")
    print(f"DATA SPLIT (Trajectory-level, completely disjoint)")
    print(f"{'='*60}")
    print(f"  Train:      {len(trajs_train):6d} trajectories ({args.train_split*100:.0f}%)")
    print(f"  Validation: {len(trajs_val):6d} trajectories ({args.val_split*100:.0f}%)")
    print(f"  Test:       {len(trajs_test):6d} trajectories ({args.test_split*100:.0f}%)")
    print(f"{'='*60}")
    print(f"State dimension: {n_x}")
    print(f"Latent dimension: {n_z}")
    
    # Prepare TRAINING data
    print("\nPreparing training data...")
    train_data_dict = prepare_training_data(trajs_train)
    train_sequences = train_data_dict.get('sequences', None)
    
    # Prepare VALIDATION data (from separate trajectories!)
    print("Preparing validation data...")
    val_data_dict = prepare_training_data(trajs_val)
    
    horizons = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Training tensors
    train_tensors = [train_data_dict['x0']]
    for h in horizons:
        train_tensors.append(train_data_dict[f'x{h}'])
    
    # Validation tensors (from DIFFERENT trajectories than training)
    val_tensors = [val_data_dict['x0']]
    for h in horizons:
        val_tensors.append(val_data_dict[f'x{h}'])
    
    print(f"  Training samples: {len(train_data_dict['x0'])}")
    print(f"  Validation samples: {len(val_data_dict['x0'])}")
    print(f"  Test trajectories: {len(trajs_test)} (completely unseen)")
    
    # Save split indices for reproducibility
    split_path = os.path.join(output_dir, 'data_split.npz')
    np.savez(split_path,
             train_indices=train_indices,
             val_indices=val_indices,
             test_indices=test_indices)
    print(f"\nData split indices saved to {split_path}")
    
    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_tensors), batch_size=args.batch_size, shuffle=False)
    
    # Simple loader for baselines (x0, x1 only)
    simple_train = DataLoader(
        TensorDataset(train_data_dict['x0'], train_data_dict['x1']),
        batch_size=args.batch_size, shuffle=True
    )
    
    # Store results
    all_results = []
    models_info = {}
    
    # Number of test trajectories for evaluation
    n_eval_trajs = min(10, len(trajs_test))
    
    # ===========================================================================
    # 1. VAR (ARIMA) Baseline
    # ===========================================================================
    try:
        var_dir = os.path.join(output_dir, 'var')
        os.makedirs(var_dir, exist_ok=True)
        
        # Flatten TRAINING data only for VAR
        train_flat = trajs_train.reshape(-1, n_x)
        var_model, lag_order = train_var_model(train_flat, var_dir)
        
        # Evaluate on UNSEEN TEST trajectories
        true_trajs = []
        pred_trajs = []
        for test_traj in trajs_test[:n_eval_trajs]:
            x0 = test_traj[0]
            n_steps = min(100, len(test_traj) - 1)
            pred = predict_var(var_model, x0, n_steps, lag_order)
            true_trajs.append(test_traj[:n_steps+1])
            pred_trajs.append(pred[:n_steps+1])
        
        metrics = evaluate_predictions("VAR (ARIMA)", true_trajs, pred_trajs, n_x, args.dt)
        metrics['n_params'] = lag_order * n_x * n_x
        all_results.append(metrics)
        models_info['VAR'] = {'lag_order': lag_order}
        print(f"VAR: 1-step MSE = {metrics['one_step_mse']:.6f}")
    except Exception as e:
        print(f"VAR training failed: {e}")
    
    # ===========================================================================
    # 2. eDMD Baseline
    # ===========================================================================
    try:
        edmd_dir = os.path.join(output_dir, 'edmd')
        os.makedirs(edmd_dir, exist_ok=True)
        
        # Use TRAINING data only
        train_x0 = train_data_dict['x0'].to(device)
        train_x1 = train_data_dict['x1'].to(device)
        edmd_model = train_edmd_model(train_x0, train_x1, n_x, edmd_dir, device)
        
        # Evaluate on UNSEEN TEST trajectories
        true_trajs = []
        pred_trajs = []
        for test_traj in trajs_test[:n_eval_trajs]:
            x0 = test_traj[0]
            n_steps = min(100, len(test_traj) - 1)
            pred = predict_pytorch_model(edmd_model, x0, n_steps, device)
            true_trajs.append(test_traj[:n_steps+1])
            pred_trajs.append(pred[:n_steps+1])
        
        metrics = evaluate_predictions("eDMD", true_trajs, pred_trajs, n_x, args.dt)
        metrics['n_params'] = edmd_model.K.numel()
        all_results.append(metrics)
        print(f"eDMD: 1-step MSE = {metrics['one_step_mse']:.6f}")
    except Exception as e:
        print(f"eDMD training failed: {e}")
    
    # ===========================================================================
    # 3. KAE Baseline (Simplified)
    # ===========================================================================
    try:
        kae_base_dir = os.path.join(output_dir, 'kae_baseline')
        os.makedirs(kae_base_dir, exist_ok=True)
        
        kae_baseline = train_kae_baseline(simple_train, n_x, n_z, device, 
                                          args.n_epochs, args.patience, kae_base_dir, resume=resume)
        
        # Evaluate on UNSEEN TEST trajectories
        true_trajs = []
        pred_trajs = []
        for test_traj in trajs_test[:n_eval_trajs]:
            x0 = test_traj[0]
            n_steps = min(100, len(test_traj) - 1)
            pred = predict_pytorch_model(kae_baseline, x0, n_steps, device)
            true_trajs.append(test_traj[:n_steps+1])
            pred_trajs.append(pred[:n_steps+1])
        
        metrics = evaluate_predictions("KAE Baseline", true_trajs, pred_trajs, n_x, args.dt)
        metrics['n_params'] = sum(p.numel() for p in kae_baseline.parameters())
        all_results.append(metrics)
        print(f"KAE Baseline: 1-step MSE = {metrics['one_step_mse']:.6f}")
    except Exception as e:
        print(f"KAE Baseline training failed: {e}")
    
    # ===========================================================================
    # 4. Advanced KAE (Model 4 - 1 Expert)
    # ===========================================================================
    try:
        adv_kae_dir = os.path.join(output_dir, 'advanced_kae')
        os.makedirs(adv_kae_dir, exist_ok=True)
        
        adv_kae = train_advanced_kae(train_loader, val_loader, n_x, n_z, device,
                                     args.n_epochs, args.patience, adv_kae_dir, train_sequences, resume=resume)
        
        # Evaluate on UNSEEN TEST trajectories
        true_trajs = []
        pred_trajs = []
        for test_traj in trajs_test[:n_eval_trajs]:
            x0 = test_traj[0]
            n_steps = min(100, len(test_traj) - 1)
            pred = predict_pytorch_model(adv_kae, x0, n_steps, device)
            true_trajs.append(test_traj[:n_steps+1])
            pred_trajs.append(pred[:n_steps+1])
        
        metrics = evaluate_predictions("Advanced KAE (1 Expert)", true_trajs, pred_trajs, n_x, args.dt)
        metrics['n_params'] = sum(p.numel() for p in adv_kae.parameters())
        metrics['spectral_radius'] = float(compute_spectral_radius(adv_kae.A_f.detach().cpu().numpy())[0])
        all_results.append(metrics)
        print(f"Advanced KAE: 1-step MSE = {metrics['one_step_mse']:.6f}")
    except Exception as e:
        print(f"Advanced KAE training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ===========================================================================
    # 5. MoE with 2, 3, 4 Experts
    # ===========================================================================
    for n_experts in [2, 3, 4]:
        try:
            moe_dir = os.path.join(output_dir, f'moe_{n_experts}expert')
            os.makedirs(moe_dir, exist_ok=True)
            
            moe_model = train_moe(train_loader, val_loader, n_x, n_z, n_experts, device,
                                  args.n_epochs, args.patience, moe_dir, train_sequences, resume=resume)
            
            # Evaluate on UNSEEN TEST trajectories
            true_trajs = []
            pred_trajs = []
            for test_traj in trajs_test[:n_eval_trajs]:
                x0 = test_traj[0]
                n_steps = min(100, len(test_traj) - 1)
                pred = predict_pytorch_model(moe_model, x0, n_steps, device, is_moe=True)
                true_trajs.append(test_traj[:n_steps+1])
                pred_trajs.append(pred[:n_steps+1])
            
            metrics = evaluate_predictions(f"MoE ({n_experts} Experts)", true_trajs, pred_trajs, n_x, args.dt)
            metrics['n_params'] = sum(p.numel() for p in moe_model.parameters())
            
            # Max spectral radius across experts
            max_rho = 0
            for expert in moe_model.experts:
                rho, _ = compute_spectral_radius(expert.A_f.detach().cpu().numpy())
                max_rho = max(max_rho, rho)
            metrics['spectral_radius'] = max_rho
            
            all_results.append(metrics)
            print(f"MoE {n_experts} Expert: 1-step MSE = {metrics['one_step_mse']:.6f}")
        except Exception as e:
            print(f"MoE {n_experts} Expert training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ===========================================================================
    # Save Results
    # ===========================================================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns
    cols_order = ['model', 'n_params', 'one_step_mse', 'one_step_mse_std',
                  'nrmse_1step', 'nrmse_10step', 'nrmse_20step', 'nrmse_50step', 'nrmse_100step',
                  'chamfer_distance', 'divergence_rate', 'reconstruction_error', 'spectral_radius']
    cols_order = [c for c in cols_order if c in results_df.columns]
    results_df = results_df[cols_order + [c for c in results_df.columns if c not in cols_order]]
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'comparison_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(results_df[['model', 'n_params', 'one_step_mse', 'nrmse_50step', 'nrmse_100step']].to_string(index=False))
    print("="*80)
    
    # Save config
    config_path = os.path.join(output_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write(f"System: {args.system}\n")
        f.write(f"N Trajectories Total: {args.n_traj}\n")
        f.write(f"T: {args.T}, dt: {args.dt}\n")
        f.write(f"N Epochs: {args.n_epochs}\n")
        f.write(f"Patience: {args.patience}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"\n--- Data Split (Trajectory-level) ---\n")
        f.write(f"Train:      {n_train_traj} trajectories ({args.train_split*100:.0f}%)\n")
        f.write(f"Validation: {n_val_traj} trajectories ({args.val_split*100:.0f}%)\n")
        f.write(f"Test:       {n_test_traj} trajectories ({args.test_split*100:.0f}%)\n")
        f.write(f"Train Samples: {len(train_data_dict['x0'])}\n")
        f.write(f"Val Samples: {len(val_data_dict['x0'])}\n")
        f.write(f"\nState Dim: {n_x}, Latent Dim: {n_z}\n")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()

