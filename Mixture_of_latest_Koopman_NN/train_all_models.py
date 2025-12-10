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
from tqdm import tqdm

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
    multi_step_nrmse_per_dim,
    chamfer_distance_phase,
    chamfer_distance_full_state,
    spectral_radius as compute_spectral_radius,
    long_horizon_divergence_rate,
    long_horizon_divergence_rate_per_dim,
    reconstruction_error
)


# ==============================================================================
# Training Configuration
# ==============================================================================

# Dense training horizons: enforce Koopman linearity at every step from 1 to 100
TRAINING_HORIZONS = list(range(1, 101))  # [1, 2, 3, ..., 100]

# Evaluation horizons: test extrapolation beyond training
EVAL_HORIZONS = [100, 500, 1000]


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


def prepare_training_data(trajs, horizons=None, hankel_seq_len=16):
    """Prepare training data with multi-step horizons (vectorized for speed)"""
    if horizons is None:
        horizons = TRAINING_HORIZONS  # Dense: [1, 2, 3, ..., 100]
    n_traj, n_timesteps, n_x = trajs.shape
    max_horizon = max(horizons)
    
    print(f"  Creating training pairs for {n_traj} trajectories, {len(horizons)} horizons...")
    
    # Use only t=0 as initial condition (one sample per trajectory)
    # This gives diverse ICs without sliding window correlation
    result = {'x0': torch.tensor(trajs[:, 0, :], dtype=torch.float32)}
    
    # For each horizon, get the state at that timestep
    for h in horizons:
        result[f'x{h}'] = torch.tensor(trajs[:, h, :], dtype=torch.float32)
    
    print(f"  Training samples: {len(result['x0'])} (one IC per trajectory)")
    
    # Hankel sequences (subsample for memory efficiency)
    print(f"  Creating Hankel sequences...")
    max_hankel_trajs = min(1000, n_traj)  # Limit Hankel to 1000 trajectories
    sequences = []
    for traj in trajs[:max_hankel_trajs]:
        if n_timesteps >= hankel_seq_len:
            # Take only a few sequences per trajectory
            for start in range(0, min(100, n_timesteps - hankel_seq_len), hankel_seq_len):
                seq = traj[start:start + hankel_seq_len]
                if len(seq) == hankel_seq_len:
                    sequences.append(seq)
    
    if sequences:
        result['sequences'] = torch.tensor(np.stack(sequences, axis=0), dtype=torch.float32)
        print(f"  Hankel sequences: {len(sequences)}")
    
    return result


# ==============================================================================
# Evaluation Functions
# ==============================================================================

def evaluate_predictions(model_name, true_trajs, pred_trajs, n_x, dt):
    """
    Evaluate predictions using metrics from evaluation.py
    
    Returns dict of metrics with per-dimension breakdown
    """
    metrics = {'model': model_name, 'n_x': n_x}
    
    # Evaluation horizons: training (1-100) + extrapolation (500, 1000)
    eval_horizons = [1, 10, 20, 50, 100, 500, 1000]
    
    # Stack trajectories for batch evaluation: (n_test, n_steps+1, n_x)
    n_test = len(true_trajs)
    
    all_one_step_mse = []
    all_one_step_mse_per_dim = {d: [] for d in range(n_x)}
    all_nrmse_per_dim = {h: {d: [] for d in range(n_x)} for h in eval_horizons}
    all_chamfer_per_horizon = {h: [] for h in eval_horizons}  # Chamfer at each horizon
    all_divergence_per_horizon = {h: {d: [] for d in range(n_x)} for h in eval_horizons}  # Divergence per dim per horizon
    all_recon = []
    all_recon_per_dim = {d: [] for d in range(n_x)}
    
    for i in range(n_test):
        true = np.asarray(true_trajs[i])
        pred = np.asarray(pred_trajs[i])
        
        # Strict shape assertions
        assert true.ndim == 2, f"Trajectory {i}: true must be 2D (n_steps, n_x), got shape {true.shape}"
        assert pred.ndim == 2, f"Trajectory {i}: pred must be 2D (n_steps, n_x), got shape {pred.shape}"
        assert true.shape[1] == n_x, f"Trajectory {i}: true has wrong n_x: expected {n_x}, got {true.shape[1]}"
        assert pred.shape[1] == n_x, f"Trajectory {i}: pred has wrong n_x: expected {n_x}, got {pred.shape[1]}"
        assert true.shape == pred.shape, f"Trajectory {i}: shape mismatch: true={true.shape}, pred={pred.shape}"
        assert true.shape[0] >= 2, f"Trajectory {i}: need at least 2 timesteps, got {true.shape[0]}"
        
        # Reshape for evaluation functions: (1, n_steps, n_x)
        true_3d = true[np.newaxis, :, :]
        pred_3d = pred[np.newaxis, :, :]
        assert true_3d.shape == (1, true.shape[0], n_x), f"true_3d shape wrong: {true_3d.shape}"
        assert pred_3d.shape == (1, pred.shape[0], n_x), f"pred_3d shape wrong: {pred_3d.shape}"
        
        # 1-step MSE (aggregate and per-dim)
        all_one_step_mse.append(one_step_mse(true_3d[:, :2, :], pred_3d[:, :2, :]))
        for d in range(n_x):
            mse_d = float(np.mean((true[1, d] - pred[1, d])**2))
            all_one_step_mse_per_dim[d].append(mse_d)
        
        # Multi-step NRMSE per dimension
        n_steps = true.shape[0] - 1
        horizons = [h for h in eval_horizons if h <= n_steps]
        if horizons:
            nrmse_per_dim, _ = multi_step_nrmse_per_dim(true_3d, pred_3d, horizons)
            for h, dim_values in nrmse_per_dim.items():
                for d, val in enumerate(dim_values):
                    all_nrmse_per_dim[h][d].append(val)
        
        # Chamfer distance and Divergence rate at each horizon (like NRMSE)
        for h in horizons:
            # Compute on trajectory up to step h
            true_h = true_3d[:, :h+1, :]
            pred_h = pred_3d[:, :h+1, :]
            
            # Chamfer
            chamfer_h = chamfer_distance_full_state(true_h, pred_h)
            all_chamfer_per_horizon[h].append(chamfer_h)
            
            # Divergence rate per dimension (need at least 2 points for linear fit)
            if h >= 1:
                slopes_per_dim, _ = long_horizon_divergence_rate_per_dim(true_h, pred_h)
                for d, slope in enumerate(slopes_per_dim):
                    all_divergence_per_horizon[h][d].append(slope)
        
        # Reconstruction error (per-dim)
        for d in range(n_x):
            all_recon_per_dim[d].append(float((true[0, d] - pred[0, d])**2))
        all_recon.append(float(np.mean((true[0] - pred[0])**2)))
    
    # ===== Aggregate metrics =====
    
    # 1-step MSE: per-dim and aggregate (mean of per-dim)
    for d in range(n_x):
        if all_one_step_mse_per_dim[d]:
            metrics[f'one_step_mse_dim{d}'] = np.mean(all_one_step_mse_per_dim[d])
    metrics['one_step_mse'] = np.mean(all_one_step_mse) if all_one_step_mse else np.nan
    
    # NRMSE: per-dim and aggregate (RMS of per-dim)
    for h in eval_horizons:
        per_dim_means = []
        for d in range(n_x):
            if all_nrmse_per_dim[h][d]:
                dim_mean = np.mean([v for v in all_nrmse_per_dim[h][d] if not np.isnan(v)])
                metrics[f'nrmse_{h}step_dim{d}'] = dim_mean
                per_dim_means.append(dim_mean)
            else:
                metrics[f'nrmse_{h}step_dim{d}'] = np.nan
        
        # RMS aggregate across dimensions
        valid_means = [m for m in per_dim_means if not np.isnan(m)]
        if valid_means:
            metrics[f'nrmse_{h}step'] = np.sqrt(np.mean(np.array(valid_means)**2))
        else:
            metrics[f'nrmse_{h}step'] = np.nan
    
    # Chamfer distance at each horizon - HONEST REPORTING (like NRMSE)
    # If ANY trajectory has inf Chamfer at a horizon → report inf for that horizon
    for h in eval_horizons:
        chamfer_vals = all_chamfer_per_horizon[h]
        if chamfer_vals:
            # If any inf, report inf (model diverged)
            if any(np.isinf(c) for c in chamfer_vals):
                metrics[f'chamfer_{h}step'] = np.inf
            else:
                metrics[f'chamfer_{h}step'] = np.mean(chamfer_vals)
        else:
            metrics[f'chamfer_{h}step'] = np.nan
    
    # Track divergence at max horizon for backward compatibility
    max_h = max([h for h in eval_horizons if all_chamfer_per_horizon[h]], default=100)
    chamfer_max = all_chamfer_per_horizon.get(max_h, [])
    n_total = len(chamfer_max)
    n_diverged = sum(1 for c in chamfer_max if np.isinf(c))
    metrics['n_valid'] = n_total - n_diverged
    metrics['n_total'] = n_total
    metrics['n_diverged'] = n_diverged
    
    # Divergence rate at each horizon per dimension (like NRMSE)
    for h in eval_horizons:
        per_dim_means = []
        for d in range(n_x):
            div_vals = all_divergence_per_horizon[h][d]
            if div_vals:
                valid_divs = [v for v in div_vals if not np.isnan(v)]
                if valid_divs:
                    dim_mean = np.mean(valid_divs)
                    metrics[f'divergence_{h}step_dim{d}'] = dim_mean
                    per_dim_means.append(dim_mean)
                else:
                    metrics[f'divergence_{h}step_dim{d}'] = np.nan
            else:
                metrics[f'divergence_{h}step_dim{d}'] = np.nan
        
        # RMS aggregate across dimensions
        valid_means = [m for m in per_dim_means if not np.isnan(m)]
        if valid_means:
            metrics[f'divergence_{h}step'] = np.sqrt(np.mean(np.array(valid_means)**2))
        else:
            metrics[f'divergence_{h}step'] = np.nan
    
    # Reconstruction error: per-dim and aggregate (mean of per-dim)
    for d in range(n_x):
        if all_recon_per_dim[d]:
            metrics[f'recon_error_dim{d}'] = np.mean(all_recon_per_dim[d])
    metrics['reconstruction_error'] = np.mean(all_recon) if all_recon else np.nan
    
    return metrics


def print_model_metrics(metrics, n_x):
    """Print detailed evaluation metrics for a model with per-dimension breakdown"""
    model_name = metrics['model']
    n_x = metrics.get('n_x', n_x)
    
    print(f"\n{'═'*60}")
    print(f"  {model_name} Evaluation Results")
    print(f"{'═'*60}")
    
    # 1-step MSE (same format as NRMSE)
    agg_mse = metrics.get('one_step_mse', np.nan)
    per_dim_mse = "  |  ".join([
        f"d{d}={metrics.get(f'one_step_mse_dim{d}', np.nan):.6f}" 
        for d in range(n_x)
    ])
    print(f"\n  1-step MSE: {agg_mse:.6f}  ({per_dim_mse})")
    
    # Multi-step NRMSE (aggregate + per-dim)
    print(f"\n  Multi-step Cumulative NRMSE (per-dim normalized, RMS aggregate):")
    eval_horizons = [1, 10, 20, 50, 100, 500, 1000]
    for h in eval_horizons:
        agg_val = metrics.get(f'nrmse_{h}step', np.nan)
        if np.isnan(agg_val):
            continue
        
        # Collect per-dim values
        per_dim_str = "  |  ".join([
            f"d{d}={metrics.get(f'nrmse_{h}step_dim{d}', np.nan):.4f}" 
            for d in range(n_x)
        ])
        print(f"    {h:4d}-step: {agg_val:.4f}  ({per_dim_str})")
    
    # Chamfer distance at each horizon (like NRMSE)
    print(f"\n  Chamfer Distance (full state, per horizon):")
    for h in eval_horizons:
        chamfer_h = metrics.get(f'chamfer_{h}step', np.nan)
        if np.isnan(chamfer_h):
            continue
        if np.isinf(chamfer_h):
            print(f"    {h:4d}-step: inf")
        else:
            print(f"    {h:4d}-step: {chamfer_h:.4f}")
    
    # Summary: diverged count at max horizon
    n_valid = metrics.get('n_valid', 0)
    n_total = metrics.get('n_total', 0)
    n_div = metrics.get('n_diverged', 0)
    if n_div > 0:
        print(f"    ⚠ {n_div}/{n_total} trajectories diverged")
    
    # Divergence rate at each horizon (per-dim + RMS aggregate)
    print(f"\n  Divergence Rate (per-dim, RMS aggregate):")
    for h in eval_horizons:
        agg_val = metrics.get(f'divergence_{h}step', np.nan)
        if np.isnan(agg_val):
            continue
        per_dim_str = "  |  ".join([
            f"d{d}={metrics.get(f'divergence_{h}step_dim{d}', np.nan):.6f}"
            for d in range(n_x)
        ])
        print(f"    {h:4d}-step: {agg_val:.6f}  ({per_dim_str})")
    
    # Reconstruction error (same format as NRMSE)
    agg_recon = metrics.get('reconstruction_error', np.nan)
    per_dim_recon = "  |  ".join([
        f"d{d}={metrics.get(f'recon_error_dim{d}', np.nan):.6f}" 
        for d in range(n_x)
    ])
    print(f"\n  Recon error: {agg_recon:.6f}  ({per_dim_recon})")
    
    print(f"{'═'*60}\n")


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
    # Ensure minimum lag of 1 (lag 0 = no prediction capability)
    optimal_lag = max(1, optimal_lag)
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
    
    pbar = tqdm(range(start_epoch, n_epochs), desc="KAE Base", unit="epoch")
    for epoch in pbar:
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
            
            loss = 2.0 * loss_rec + 2.0 * loss_pred + 12.0 * loss_lin
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # Save checkpoint every epoch for resumption
        save_checkpoint(model, optimizer, epoch, best_loss, patience_counter, checkpoint_path)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), best_model_path)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'best': '✓'})
        else:
            patience_counter += 1
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'pat': f'{patience_counter}/{patience}'})
            if patience_counter >= patience:
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    elif os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    return model


def precompute_latents(model, all_x_dict, device, chunk_size=2000000):
    """Precompute encoder outputs for all timesteps at epoch start.
    
    Returns dict: {k: z_k tensor} for k in [0] + TRAINING_HORIZONS
    Keeps tensors on GPU for fast access during training.
    """
    model.eval()
    z_cache = {}
    
    with torch.no_grad():
        # Stack all x values: collect x0, x1, ..., x100
        all_keys = ['x0'] + [f'x{k}' for k in TRAINING_HORIZONS]
        
        for key in all_keys:
            x_data = all_x_dict[key]
            n_samples = x_data.shape[0]
            
            # Process in chunks to avoid OOM
            z_chunks = []
            for i in range(0, n_samples, chunk_size):
                chunk = x_data[i:i+chunk_size].to(device)
                z_chunk = model.encoder(chunk)
                z_chunks.append(z_chunk)  # Keep on GPU for fast training access
            
            z_cache[key] = torch.cat(z_chunks, dim=0)
    
    model.train()
    return z_cache


def compute_kae_loss(model, data_batch, device, mse, z_cache=None, batch_indices=None):
    """Compute full loss for KAE model.
    
    If z_cache is provided, uses precomputed latents for zk_true (faster).
    z0 is always recomputed to allow gradients to flow to encoder.
    """
    x0 = data_batch['x0']
    x1 = data_batch['x1']
    
    # Forward pass - always compute (needs gradients)
    out = model(x0)
    loss_rec = mse(out['x_rec'], x0)
    
    # Prediction loss - z0 computed fresh for gradients
    z0 = model.encoder(x0)
    z1_pred = z0 @ model.A_f.T
    x1_pred = model.decoder(z1_pred)
    loss_pred = mse(x1_pred, x1)
    
    # Multi-step linearity
    loss_lin = 0
    A_k = model.A_f.clone()
    
    if z_cache is not None and batch_indices is not None:
        # Use precomputed latents (fast path) - already on GPU
        for k in TRAINING_HORIZONS:
            zk_true = z_cache[f'x{k}'][batch_indices]
            zk_pred = z0 @ A_k.T
            loss_lin += mse(zk_pred, zk_true)
            A_k = A_k @ model.A_f
    else:
        # Compute on the fly (slow path, used for validation)
        for k in TRAINING_HORIZONS:
            x_k = data_batch[f'x{k}']
            zk_true = model.encoder(x_k)
            zk_pred = z0 @ A_k.T
            loss_lin += mse(zk_pred, zk_true)
            A_k = A_k @ model.A_f
    
    loss_lin /= len(TRAINING_HORIZONS)
    
    # Bidirectional + Spectral
    I = torch.eye(model.n_z, device=device)
    loss_bi = (model.A_f @ model.A_b - I).norm()**2
    loss_spec = spectral_radius_penalty(model.A_f, iters=8, target=1.005, lower=0.995)
    
    # Sparsity
    loss_sparse = model.sparsity_loss(mode="l1")
    
    # Total loss
    loss = (2.0 * loss_rec + 2.0 * loss_pred + 12.0 * loss_lin +
           1.0 * loss_bi + 5.0 * loss_spec + 1e-4 * loss_sparse)
    
    return loss


def train_advanced_kae(train_loader, val_loader, n_x, n_z, device, n_epochs, patience, 
                       save_dir, train_data_dict=None, val_data_dict=None, resume=True):
    """Train Advanced KAE (Model 4) with all loss components and checkpointing.
    
    Uses precomputed latents for ~10x speedup on linearity loss.
    """
    print("\n" + "="*60)
    print("Training Advanced KAE (Model 4 - 1 Expert)")
    print("="*60)
    
    model = KoopmanAE(n_x=n_x, n_z=n_z).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    
    # Checkpoint paths
    checkpoint_path = os.path.join(save_dir, 'advanced_kae_checkpoint.pth')
    best_model_path = os.path.join(save_dir, 'advanced_kae_best.pth')
    
    horizons = TRAINING_HORIZONS  # Dense: [1, 2, 3, ..., 100]
    
    # Try to resume
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if resume:
        start_epoch, best_val_loss, patience_counter = load_checkpoint(
            model, optimizer, checkpoint_path, device
        )
    
    best_state = None
    n_train = len(train_data_dict['x0']) if train_data_dict else len(train_loader.dataset)
    batch_size = train_loader.batch_size
    
    pbar = tqdm(range(start_epoch, n_epochs), desc="Adv KAE", unit="epoch")
    for epoch in pbar:
        # === Precompute latents at epoch start (big speedup!) ===
        if train_data_dict is not None:
            z_cache = precompute_latents(model, train_data_dict, device)
        else:
            z_cache = None
        
        # === Training ===
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Generate random permutation for this epoch
        perm = torch.randperm(n_train)
        
        for batch_start in range(0, n_train, batch_size):
            batch_indices = perm[batch_start:batch_start + batch_size]
            
            # Get x0 and x1 (need x1 for prediction loss)
            data_batch = {
                'x0': train_data_dict['x0'][batch_indices].to(device),
                'x1': train_data_dict['x1'][batch_indices].to(device)
            }
            
            loss = compute_kae_loss(model, data_batch, device, mse, z_cache, batch_indices)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # === Validation (no precompute - smaller dataset, needs fresh encoder) ===
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                data_batch = {'x0': batch[0].to(device)}
                for idx, h in enumerate(horizons):
                    data_batch[f'x{h}'] = batch[idx + 1].to(device)
                
                loss = compute_kae_loss(model, data_batch, device, mse)
                val_loss += loss.item()
                n_val += 1
        
        avg_val_loss = val_loss / n_val if n_val > 0 else float('inf')
        
        # Save checkpoint every epoch for resumption
        save_checkpoint(model, optimizer, epoch, best_val_loss, patience_counter, checkpoint_path)
        
        # Update progress bar
        pbar.set_postfix({'train': f'{avg_loss:.4f}', 'val': f'{avg_val_loss:.4f}'})
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), best_model_path)
            pbar.set_postfix({'train': f'{avg_loss:.4f}', 'val': f'{avg_val_loss:.4f}', 'best': '✓'})
        else:
            patience_counter += 1
            pbar.set_postfix({'train': f'{avg_loss:.4f}', 'val': f'{avg_val_loss:.4f}', 'pat': f'{patience_counter}/{patience}'})
            if patience_counter >= patience:
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    elif os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    return model


def precompute_moe_latents(model, all_x_dict, device, chunk_size=2000000):
    """Precompute encoder outputs for MoE (shared encoder).
    
    Returns dict: {key: z tensor} - same format as KAE since encoder is shared.
    Keeps tensors on GPU for fast access during training.
    """
    model.eval()
    z_cache = {}
    
    with torch.no_grad():
        all_keys = ['x0'] + [f'x{k}' for k in TRAINING_HORIZONS]
        
        for key in all_keys:
            x_data = all_x_dict[key]
            n_samples = x_data.shape[0]
            
            # MoE has shared encoder - just one precomputation needed
            z_chunks = []
            for start in range(0, n_samples, chunk_size):
                chunk = x_data[start:start+chunk_size].to(device)
                z_chunk = model.encoder(chunk)
                z_chunks.append(z_chunk)  # Keep on GPU for fast training access
            z_cache[key] = torch.cat(z_chunks, dim=0)
    
    model.train()
    return z_cache


def compute_moe_loss(model, data_batch, device, mse, n_experts, z_cache=None, batch_indices=None):
    """Compute full loss for MoE model.
    
    MoE uses shared encoder/decoder with multiple Koopman operators K[i].
    If z_cache is provided, uses precomputed latents for zk_true (faster).
    """
    x0 = data_batch['x0']
    x1 = data_batch['x1']
    
    # Forward pass - always compute fresh (shared encoder)
    out = model(x0)
    weights0 = out['weights']  # Gating weights (batch, n_experts)
    loss_rec = mse(out['x_rec'], x0)
    
    # Prediction loss using MoE's predict_next (IC gating)
    x1_pred, _, _, _ = model.predict_next(x0)
    loss_pred = mse(x1_pred, x1)
    
    # Multi-step linearity per expert
    # z0 computed fresh for gradients (shared encoder)
    z0 = model.encoder(x0)
    loss_lin = 0
    
    if z_cache is not None and batch_indices is not None:
        # Fast path with precomputed latents - already on GPU
        for i in range(n_experts):
            K_i = model.K[i]  # (n_z, n_z)
            K_power = K_i.clone()  # K^1
            
            for k in TRAINING_HORIZONS:
                zk_true = z_cache[f'x{k}'][batch_indices]
                zk_pred = z0 @ K_power.T
                loss_lin += (weights0[:, i:i+1] * (zk_pred - zk_true)**2).mean()
                K_power = K_power @ K_i  # K^(k+1)
    else:
        # Slow path - compute on the fly
        for k in TRAINING_HORIZONS:
            x_k = data_batch[f'x{k}']
            zk_true = model.encoder(x_k)
            for i in range(n_experts):
                K_i = model.K[i]
                # Compute K^k
                K_power = K_i
                for _ in range(k - 1):
                    K_power = K_power @ K_i
                zk_pred = z0 @ K_power.T
                loss_lin += (weights0[:, i:i+1] * (zk_pred - zk_true)**2).mean()
    
    loss_lin /= len(TRAINING_HORIZONS)
    
    # Load balancing
    avg_weights = weights0.mean(dim=0)
    target_weight = 1.0 / n_experts
    loss_balance = ((avg_weights - target_weight)**2).sum()
    
    # Bidirectional + Spectral per expert Koopman operator
    loss_bi = 0
    loss_spec = 0
    I = torch.eye(model.n_z, device=device)
    for i in range(n_experts):
        loss_bi += (model.K[i] @ model.K_b[i] - I).norm()**2
        loss_spec += spectral_radius_penalty(model.K[i], iters=8, target=1.005, lower=0.995)
    loss_bi /= n_experts
    loss_spec /= n_experts
    
    # Sparsity
    loss_sparse = model.sparsity_loss(mode="l1")
    
    # Total loss
    loss = (2.0 * loss_rec + 2.0 * loss_pred + 12.0 * loss_lin +
           1.0 * loss_balance + 1.0 * loss_bi + 5.0 * loss_spec + 1e-4 * loss_sparse)
    
    return loss


def train_moe(train_loader, val_loader, n_x, n_z, n_experts, device, n_epochs, patience,
              save_dir, train_data_dict=None, val_data_dict=None, resume=True):
    """Train MoE Koopman (Model 5) with checkpointing for resumption.
    
    Uses precomputed latents for ~10x speedup on linearity loss.
    """
    print("\n" + "="*60)
    print(f"Training MoE Koopman ({n_experts} Experts)")
    print("="*60)
    
    model = KoopmanMoE(n_x=n_x, n_z=n_z, n_experts=n_experts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    
    # Checkpoint paths
    checkpoint_path = os.path.join(save_dir, f'moe_{n_experts}expert_checkpoint.pth')
    best_model_path = os.path.join(save_dir, f'moe_{n_experts}expert_best.pth')
    
    horizons = TRAINING_HORIZONS  # Dense: [1, 2, 3, ..., 100]
    
    # Try to resume
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if resume:
        start_epoch, best_val_loss, patience_counter = load_checkpoint(
            model, optimizer, checkpoint_path, device
        )
    
    best_state = None
    n_train = len(train_data_dict['x0']) if train_data_dict else len(train_loader.dataset)
    batch_size = train_loader.batch_size
    
    pbar = tqdm(range(start_epoch, n_epochs), desc=f"MoE-{n_experts}", unit="epoch")
    for epoch in pbar:
        # === Precompute latents at epoch start (big speedup!) ===
        if train_data_dict is not None:
            z_cache = precompute_moe_latents(model, train_data_dict, device)
        else:
            z_cache = None
        
        # === Training ===
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Generate random permutation for this epoch
        perm = torch.randperm(n_train)
        
        for batch_start in range(0, n_train, batch_size):
            batch_indices = perm[batch_start:batch_start + batch_size]
            
            # Get x0 and x1 (need x1 for prediction loss)
            data_batch = {
                'x0': train_data_dict['x0'][batch_indices].to(device),
                'x1': train_data_dict['x1'][batch_indices].to(device)
            }
            
            loss = compute_moe_loss(model, data_batch, device, mse, n_experts, z_cache, batch_indices)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # === Validation (no precompute - smaller dataset) ===
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                data_batch = {'x0': batch[0].to(device)}
                for idx, h in enumerate(horizons):
                    data_batch[f'x{h}'] = batch[idx + 1].to(device)
                
                loss = compute_moe_loss(model, data_batch, device, mse, n_experts)
                val_loss += loss.item()
                n_val += 1
        
        avg_val_loss = val_loss / n_val if n_val > 0 else float('inf')
        
        # Save checkpoint every epoch for resumption
        save_checkpoint(model, optimizer, epoch, best_val_loss, patience_counter, checkpoint_path)
        
        # Update progress bar
        pbar.set_postfix({'train': f'{avg_loss:.4f}', 'val': f'{avg_val_loss:.4f}'})
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), best_model_path)
            pbar.set_postfix({'train': f'{avg_loss:.4f}', 'val': f'{avg_val_loss:.4f}', 'best': '✓'})
        else:
            patience_counter += 1
            pbar.set_postfix({'train': f'{avg_loss:.4f}', 'val': f'{avg_val_loss:.4f}', 'pat': f'{patience_counter}/{patience}'})
            if patience_counter >= patience:
                break
    
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
    """Predict using PyTorch model. Returns array of shape (n_steps+1, n_x)."""
    x0 = np.asarray(x0)
    assert x0.ndim == 1, f"x0 must be 1D, got shape {x0.shape}"
    n_x = x0.shape[0]
    
    model.eval()
    with torch.no_grad():
        x0_tensor = torch.tensor(x0, dtype=torch.float32).unsqueeze(0).to(device)
        assert x0_tensor.shape == (1, n_x), f"x0_tensor shape mismatch: {x0_tensor.shape}"
        
        if is_moe:
            preds, _ = model.predict(x0_tensor, n_steps=n_steps)
            preds = preds.squeeze(1).cpu().numpy()
        elif hasattr(model, 'predict_sequence'):
            # eDMD and KoopmanAE models with predict_sequence method
            preds = model.predict_sequence(x0_tensor, n_steps)
            preds = preds.cpu().numpy()
            # Handle different output shapes:
            # - eDMD returns (batch, n_steps+1, n_x)
            # - KoopmanAE returns (n_steps+1, batch, n_x)
            if preds.shape[0] == 1 and preds.ndim == 3:
                # (1, n_steps+1, n_x) -> squeeze batch dim
                preds = preds.squeeze(0)
            elif preds.shape[1] == 1 and preds.ndim == 3:
                # (n_steps+1, 1, n_x) -> squeeze batch dim
                preds = preds.squeeze(1)
        elif hasattr(model, 'encoder') and hasattr(model, 'A_f'):
            # KoopmanAE models: z_{k+1} = A @ z_k
            z = model.encoder(x0_tensor)
            n_z = z.shape[1]
            
            preds = [x0]  # Start with initial condition
            for _ in range(n_steps):
                z = z @ model.A_f.T
                x = model.decoder(z)
                assert x.shape == (1, n_x), f"Decoder output shape mismatch: {x.shape}"
                preds.append(x.cpu().numpy()[0])
            preds = np.stack(preds, axis=0)
        else:
            raise ValueError(f"Unknown model type: {type(model)}. Must have predict_sequence or encoder+A_f+decoder.")
    
    assert preds.ndim == 2, f"preds must be 2D, got shape {preds.shape}"
    assert preds.shape == (n_steps + 1, n_x), f"preds shape mismatch: expected ({n_steps+1}, {n_x}), got {preds.shape}"
    
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
    parser.add_argument('--data_dir', type=str, default='generated_data',
                       help='Directory containing pre-generated data (default: generated_data)')
    parser.add_argument('--max_test_traj', type=int, default=None,
                       help='Maximum number of test trajectories to evaluate (default: all)')
    parser.add_argument('--eval_only', action='store_true',
                       help='Skip training and only evaluate existing models')
    parser.add_argument('--models', type=str, default='var,edmd,kae_baseline,advanced_kae,moe_2expert,moe_3expert,moe_4expert',
                       help='Comma-separated list of models to train/evaluate (default: all)')
    
    args = parser.parse_args()
    
    # Parse models list
    ALL_MODELS = ['var', 'edmd', 'kae_baseline', 'advanced_kae', 'moe_2expert', 'moe_3expert', 'moe_4expert']
    selected_models = [m.strip() for m in args.models.split(',')]
    for m in selected_models:
        if m not in ALL_MODELS:
            raise ValueError(f"Unknown model: {m}. Valid models: {ALL_MODELS}")
    print(f"Selected models: {selected_models}")
    
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
    
    # Load or generate data
    print(f"\n{'='*60}")
    data_file = os.path.join(args.data_dir, f"{args.system}.npz")
    if os.path.exists(data_file):
        print(f"Loading pre-generated {args.system} data from {data_file}")
        data = np.load(data_file)
        t = data['t']
        trajs = data['trajs']
        print(f"  Loaded {len(trajs)} trajectories, {trajs.shape[1]} timesteps each")
        # Get config based on system
        if args.system == 'duffing':
            config = {'name': 'Duffing Oscillator', 'n_x': 2, 'state_labels': ['x', 'xdot']}
        elif args.system == 'vanderpol':
            config = {'name': 'Van der Pol Oscillator', 'n_x': 2, 'state_labels': ['x', 'xdot']}
        elif args.system == 'lorenz':
            config = {'name': 'Lorenz Attractor', 'n_x': 3, 'state_labels': ['x', 'y', 'z']}
        elif args.system == 'double_pendulum':
            config = {'name': 'Double Pendulum', 'n_x': 4, 'state_labels': ['θ1', 'θ2', 'ω1', 'ω2']}
        # Override n_traj with actual loaded count
        args.n_traj = len(trajs)
    else:
        print(f"Generating {args.system} data: {args.n_traj} trajectories")
        t, trajs, config = generate_dataset(args.system, args.n_traj, args.T, args.dt)
    print(f"{'='*60}")
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
    
    horizons = TRAINING_HORIZONS  # Dense: [1, 2, 3, ..., 100]
    
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
    if args.max_test_traj is not None:
        n_eval_trajs = min(args.max_test_traj, len(trajs_test))
    else:
        n_eval_trajs = len(trajs_test)  # Use all test trajectories
    print(f"\nEvaluating on {n_eval_trajs} test trajectories")
    
    # ===========================================================================
    # 1. VAR (ARIMA) Baseline
    # ===========================================================================
    if 'var' in selected_models:
        try:
            var_dir = os.path.join(output_dir, 'var')
            os.makedirs(var_dir, exist_ok=True)
            
            # Check for eval_only mode - load existing model
            var_model_path = os.path.join(var_dir, 'var_model.pkl')
            if args.eval_only and os.path.exists(var_model_path):
                print("\n" + "="*60)
                print("Loading VAR Model (eval_only mode)")
                print("="*60)
                with open(var_model_path, 'rb') as f:
                    var_model = pickle.load(f)
                lag_order = var_model.k_ar
                print(f"Loaded VAR model with lag order: {lag_order}")
            else:
                # Flatten TRAINING data only for VAR
                # Use first 100 steps of 1000 trajectories (same training window as neural nets)
                # This allows fair comparison and extrapolation testing to 500, 1000 steps
                max_var_trajs = 1000
                train_steps = 101  # t=0 to t=100 (same as TRAINING_HORIZONS)
                n_var_trajs = min(max_var_trajs, len(trajs_train))
                train_flat = trajs_train[:n_var_trajs, :train_steps, :].reshape(-1, n_x)
                print(f"  VAR training data: {len(train_flat)} samples ({n_var_trajs} traj × {train_steps} steps)")
                var_model, lag_order = train_var_model(train_flat, var_dir)
            
            # Evaluate on UNSEEN TEST trajectories
            true_trajs = []
            pred_trajs = []
            for test_traj in tqdm(trajs_test[:n_eval_trajs], desc="VAR eval", leave=False):
                x0 = test_traj[0]
                n_steps = min(1000, len(test_traj) - 1)  # Full extrapolation to 1000 steps
                pred = predict_var(var_model, x0, n_steps, lag_order)
                true_trajs.append(test_traj[:n_steps+1])
                pred_trajs.append(pred[:n_steps+1])
            
            metrics = evaluate_predictions("VAR (ARIMA)", true_trajs, pred_trajs, n_x, args.dt)
            metrics['n_params'] = lag_order * n_x * n_x
            all_results.append(metrics)
            models_info['VAR'] = {'lag_order': lag_order}
            print_model_metrics(metrics, n_x)
        except Exception as e:
            print(f"VAR training failed: {e}")
    else:
        print("\n[Skipping VAR - not in selected models]")
    
    # ===========================================================================
    # 2. eDMD Baseline
    # ===========================================================================
    if 'edmd' in selected_models:
        try:
            edmd_dir = os.path.join(output_dir, 'edmd')
            os.makedirs(edmd_dir, exist_ok=True)
            
            # Check for eval_only mode - load existing model
            edmd_model_path = os.path.join(edmd_dir, 'edmd_model.pt')
            if args.eval_only and os.path.exists(edmd_model_path):
                print("\n" + "="*60)
                print("Loading eDMD Model (eval_only mode)")
                print("="*60)
                edmd_model = EDMDModel(n_x=n_x).to(device)
                edmd_model.load_state_dict(torch.load(edmd_model_path, map_location=device))
                edmd_model.eval()
                print(f"Loaded eDMD model from {edmd_model_path}")
            else:
                # Use TRAINING data only
                train_x0 = train_data_dict['x0'].to(device)
                train_x1 = train_data_dict['x1'].to(device)
                edmd_model = train_edmd_model(train_x0, train_x1, n_x, edmd_dir, device)
            
            # Evaluate on UNSEEN TEST trajectories
            true_trajs = []
            pred_trajs = []
            for test_traj in tqdm(trajs_test[:n_eval_trajs], desc="eDMD eval", leave=False):
                x0 = test_traj[0]
                n_steps = min(1000, len(test_traj) - 1)  # Full extrapolation to 1000 steps
                pred = predict_pytorch_model(edmd_model, x0, n_steps, device)
                true_trajs.append(test_traj[:n_steps+1])
                pred_trajs.append(pred[:n_steps+1])
            
            metrics = evaluate_predictions("eDMD", true_trajs, pred_trajs, n_x, args.dt)
            metrics['n_params'] = edmd_model.K.numel()
            all_results.append(metrics)
            print_model_metrics(metrics, n_x)
        except Exception as e:
            print(f"eDMD training failed: {e}")
    else:
        print("\n[Skipping eDMD - not in selected models]")
    
    # ===========================================================================
    # 3. KAE Baseline (Simplified)
    # ===========================================================================
    if 'kae_baseline' in selected_models:
        try:
            kae_base_dir = os.path.join(output_dir, 'kae_baseline')
            os.makedirs(kae_base_dir, exist_ok=True)
            
            # Check for eval_only mode - load existing model
            kae_model_path = os.path.join(kae_base_dir, 'best_model.pt')
            if args.eval_only and os.path.exists(kae_model_path):
                print("\n" + "="*60)
                print("Loading KAE Baseline Model (eval_only mode)")
                print("="*60)
                kae_baseline = KoopmanAE(n_x=n_x, n_z=n_z).to(device)
                kae_baseline.load_state_dict(torch.load(kae_model_path, map_location=device))
                kae_baseline.eval()
                print(f"Loaded KAE Baseline from {kae_model_path}")
            else:
                kae_baseline = train_kae_baseline(simple_train, n_x, n_z, device, 
                                                  args.n_epochs, args.patience, kae_base_dir, resume=resume)
            
            # Evaluate on UNSEEN TEST trajectories
            true_trajs = []
            pred_trajs = []
            for test_traj in tqdm(trajs_test[:n_eval_trajs], desc="KAE-B eval", leave=False):
                x0 = test_traj[0]
                n_steps = min(1000, len(test_traj) - 1)  # Full extrapolation to 1000 steps
                pred = predict_pytorch_model(kae_baseline, x0, n_steps, device)
                true_trajs.append(test_traj[:n_steps+1])
                pred_trajs.append(pred[:n_steps+1])
            
            metrics = evaluate_predictions("KAE Baseline", true_trajs, pred_trajs, n_x, args.dt)
            metrics['n_params'] = sum(p.numel() for p in kae_baseline.parameters())
            all_results.append(metrics)
            print_model_metrics(metrics, n_x)
        except Exception as e:
            print(f"KAE Baseline training failed: {e}")
    else:
        print("\n[Skipping KAE Baseline - not in selected models]")
    
    # ===========================================================================
    # 4. Advanced KAE (Model 4 - 1 Expert)
    # ===========================================================================
    if 'advanced_kae' in selected_models:
        try:
            adv_kae_dir = os.path.join(output_dir, 'advanced_kae')
            os.makedirs(adv_kae_dir, exist_ok=True)
            
            # Check for eval_only mode - load existing model
            adv_model_path = os.path.join(adv_kae_dir, 'best_model.pt')
            if args.eval_only and os.path.exists(adv_model_path):
                print("\n" + "="*60)
                print("Loading Advanced KAE Model (eval_only mode)")
                print("="*60)
                adv_kae = KoopmanAE(n_x=n_x, n_z=n_z).to(device)
                adv_kae.load_state_dict(torch.load(adv_model_path, map_location=device))
                adv_kae.eval()
                print(f"Loaded Advanced KAE from {adv_model_path}")
            else:
                adv_kae = train_advanced_kae(train_loader, val_loader, n_x, n_z, device,
                                             args.n_epochs, args.patience, adv_kae_dir, 
                                             train_data_dict=train_data_dict, val_data_dict=val_data_dict, resume=resume)
            
            # Evaluate on UNSEEN TEST trajectories
            true_trajs = []
            pred_trajs = []
            for test_traj in tqdm(trajs_test[:n_eval_trajs], desc="Adv-KAE eval", leave=False):
                x0 = test_traj[0]
                n_steps = min(1000, len(test_traj) - 1)  # Full extrapolation to 1000 steps
                pred = predict_pytorch_model(adv_kae, x0, n_steps, device)
                true_trajs.append(test_traj[:n_steps+1])
                pred_trajs.append(pred[:n_steps+1])
            
            metrics = evaluate_predictions("Advanced KAE (1 Expert)", true_trajs, pred_trajs, n_x, args.dt)
            metrics['n_params'] = sum(p.numel() for p in adv_kae.parameters())
            metrics['spectral_radius'] = float(compute_spectral_radius(adv_kae.A_f.detach().cpu().numpy())[0])
            all_results.append(metrics)
            print_model_metrics(metrics, n_x)
        except Exception as e:
            print(f"Advanced KAE training failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[Skipping Advanced KAE - not in selected models]")
    
    # ===========================================================================
    # 5. MoE with 2, 3, 4 Experts
    # ===========================================================================
    for n_experts in [2, 3, 4]:
        model_key = f'moe_{n_experts}expert'
        if model_key not in selected_models:
            print(f"\n[Skipping MoE {n_experts} Expert - not in selected models]")
            continue
            
        try:
            moe_dir = os.path.join(output_dir, f'moe_{n_experts}expert')
            os.makedirs(moe_dir, exist_ok=True)
            
            # Check for eval_only mode - load existing model
            moe_model_path = os.path.join(moe_dir, 'best_model.pt')
            if args.eval_only and os.path.exists(moe_model_path):
                print("\n" + "="*60)
                print(f"Loading MoE {n_experts} Expert Model (eval_only mode)")
                print("="*60)
                moe_model = KoopmanMoE(n_x=n_x, n_z=n_z, n_experts=n_experts).to(device)
                moe_model.load_state_dict(torch.load(moe_model_path, map_location=device))
                moe_model.eval()
                print(f"Loaded MoE model from {moe_model_path}")
            else:
                moe_model = train_moe(train_loader, val_loader, n_x, n_z, n_experts, device,
                                      args.n_epochs, args.patience, moe_dir,
                                      train_data_dict=train_data_dict, val_data_dict=val_data_dict, resume=resume)
            
            # Evaluate on UNSEEN TEST trajectories
            true_trajs = []
            pred_trajs = []
            for test_traj in tqdm(trajs_test[:n_eval_trajs], desc=f"MoE-{n_experts} eval", leave=False):
                x0 = test_traj[0]
                n_steps = min(1000, len(test_traj) - 1)  # Full extrapolation to 1000 steps
                pred = predict_pytorch_model(moe_model, x0, n_steps, device, is_moe=True)
                true_trajs.append(test_traj[:n_steps+1])
                pred_trajs.append(pred[:n_steps+1])
            
            metrics = evaluate_predictions(f"MoE ({n_experts} Experts)", true_trajs, pred_trajs, n_x, args.dt)
            metrics['n_params'] = sum(p.numel() for p in moe_model.parameters())
            
            # Max spectral radius across experts (K is shape [n_experts, n_z, n_z])
            max_rho = 0
            for i in range(n_experts):
                K_i = moe_model.K[i].detach().cpu().numpy()
                rho, _ = compute_spectral_radius(K_i)
                max_rho = max(max_rho, rho)
            metrics['spectral_radius'] = max_rho
            
            all_results.append(metrics)
            print_model_metrics(metrics, n_x)
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
    
    # Reorder columns for readability
    # Priority: model info, aggregate metrics, then per-dim/per-horizon details
    eval_horizons = [1, 10, 20, 50, 100, 500, 1000]
    n_x = all_results[0].get('n_x', 4) if all_results else 4
    
    cols_order = ['model', 'n_params', 'n_x', 'n_valid', 'n_total', 'n_diverged']
    
    # 1-step MSE (aggregate + per-dim)
    cols_order += ['one_step_mse'] + [f'one_step_mse_dim{d}' for d in range(n_x)]
    
    # NRMSE at each horizon (aggregate + per-dim)
    for h in eval_horizons:
        cols_order += [f'nrmse_{h}step'] + [f'nrmse_{h}step_dim{d}' for d in range(n_x)]
    
    # Chamfer at each horizon
    cols_order += [f'chamfer_{h}step' for h in eval_horizons]
    
    # Divergence at each horizon (aggregate + per-dim)
    for h in eval_horizons:
        cols_order += [f'divergence_{h}step'] + [f'divergence_{h}step_dim{d}' for d in range(n_x)]
    
    # Reconstruction error (aggregate + per-dim)
    cols_order += ['reconstruction_error'] + [f'recon_error_dim{d}' for d in range(n_x)]
    
    # Spectral radius
    cols_order += ['spectral_radius']
    
    # Apply ordering (keep only existing columns, append any extras)
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
    if len(results_df) > 0 and 'model' in results_df.columns:
        # Select key columns for summary
        summary_cols = ['model', 'n_params', 'one_step_mse', 
                       'nrmse_100step', 'nrmse_500step', 'nrmse_1000step',
                       'chamfer_100step', 'chamfer_1000step',
                       'divergence_100step', 'n_diverged']
        available_cols = [c for c in summary_cols if c in results_df.columns]
        if available_cols:
            print(results_df[available_cols].to_string(index=False))
        else:
            print("No summary columns available")
        print(f"\nFull results saved to CSV with {len(results_df.columns)} columns")
    else:
        print("No results to display")
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

