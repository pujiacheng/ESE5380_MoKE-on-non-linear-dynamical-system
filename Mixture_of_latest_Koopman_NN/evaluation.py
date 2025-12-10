# evaluation.py 
# -----------------------------------------------------------

import torch
import numpy as np


# -----------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------

def to_tensor(x, device=None, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def flatten_batch_time(x):
    if x.dim() == 2:
        return x
    elif x.dim() == 3:
        n, t, d = x.shape
        return x.reshape(n * t, d)
    else:
        raise ValueError("Input must be 2D or 3D tensor.")


# -----------------------------------------------------------
# Prediction metrics
# -----------------------------------------------------------

def one_step_mse(x_true, x_pred, plot=False):
    """
    Compute 1-step MSE.
    If plot=True → draw per-dimension MSE bar chart.
    """
    x_true = to_tensor(x_true)
    x_pred = to_tensor(x_pred)

    x_t = flatten_batch_time(x_true)
    x_p = flatten_batch_time(x_pred)
    diff = x_t - x_p
    sq_err = torch.sum(diff * diff, dim=-1)
    mse = torch.mean(sq_err)

    # ------------------- Plot -------------------
    if plot:
        import matplotlib.pyplot as plt
        per_dim_mse = torch.mean((x_t - x_p) ** 2, dim=0).cpu().numpy()
        dims = np.arange(len(per_dim_mse))

        plt.figure(figsize=(6,4))
        plt.bar(dims, per_dim_mse)
        plt.xlabel("State Dimension")
        plt.ylabel("MSE")
        plt.title("1-step MSE per Dimension")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return float(mse)


def multi_step_nrmse(x_true, x_pred, horizons, sigma_per_dim=None, plot=False):
    """
    Compute CUMULATIVE NRMSE(T) for multiple horizons with per-dimension normalization.
    
    NRMSE(T) = sqrt(mean over t in [1..T], batch, dims of ((x_true - x_pred) / sigma_dim)^2)
    
    This is cumulative: measures average error from step 1 up to step T.
    Normalization is per-dimension to handle different scales (e.g., angles vs velocities).
    
    Args:
        x_true: Ground truth trajectories (n_batch, n_steps, n_dims)
        x_pred: Predicted trajectories (n_batch, n_steps, n_dims)
        horizons: List of time horizons to evaluate
        sigma_per_dim: Optional per-dimension std for normalization (n_dims,)
        plot: If True, plot NRMSE vs horizon
    
    Returns:
        results: Dict mapping horizon T to cumulative NRMSE value
                 Returns np.nan for horizons that exceed trajectory length
    """
    x_true = to_tensor(x_true)
    x_pred = to_tensor(x_pred)

    # Validate shapes
    assert x_true.shape == x_pred.shape, \
        f"Shape mismatch: x_true={x_true.shape}, x_pred={x_pred.shape}"
    assert x_true.dim() == 3, \
        f"Expected 3D tensor (batch, time, dims), got {x_true.dim()}D"

    n, t_max, d = x_true.shape

    # Per-dimension normalization: compute std for each dimension separately
    if sigma_per_dim is None:
        # Shape: (d,) - one std per dimension
        sigma_per_dim = torch.std(x_true, dim=(0, 1)) + 1e-8
        
        # Warn if any dimension has near-zero variance
        if torch.any(sigma_per_dim < 1e-6):
            import warnings
            zero_var_dims = torch.where(sigma_per_dim < 1e-6)[0].tolist()
            warnings.warn(f"Near-zero variance in dimensions {zero_var_dims}. NRMSE may be unreliable.")

    results = {}
    for T in horizons:
        if T >= t_max:
            # Explicitly return nan for horizons beyond trajectory length
            results[T] = float('nan')
            continue
        
        # CUMULATIVE: error from step 1 up to step T (inclusive)
        # Shape: (n, T, d)
        err = x_true[:, 1:T+1, :] - x_pred[:, 1:T+1, :]
        
        # Normalize each dimension by its own std
        # Shape: (n, T, d)
        err_normalized = err / sigma_per_dim.unsqueeze(0).unsqueeze(0)
        
        # NRMSE: sqrt of mean squared normalized error across all dims, times, batches
        nrmse_T = torch.sqrt(torch.mean(err_normalized ** 2))
        results[T] = float(nrmse_T.item())

    # ------------------- Plot -------------------
    if plot and len(results) > 0:
        import matplotlib.pyplot as plt
        H = sorted(results.keys())
        V = [results[h] for h in H]

        plt.figure(figsize=(6,4))
        plt.plot(H, V, marker='o', linewidth=2)
        plt.xlabel("Horizon T")
        plt.ylabel("Cumulative NRMSE(T)")
        plt.title("Cumulative NRMSE vs Horizon (Per-Dim Normalized)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return results


def multi_step_nrmse_per_dim(x_true, x_pred, horizons, sigma_per_dim=None, plot=False):
    """
    Compute CUMULATIVE NRMSE(T) separately for each dimension.
    
    Returns per-dimension NRMSE values for detailed analysis.
    
    Args:
        x_true: Ground truth trajectories (n_batch, n_steps, n_dims)
        x_pred: Predicted trajectories (n_batch, n_steps, n_dims)
        horizons: List of time horizons to evaluate
        sigma_per_dim: Optional per-dimension std for normalization (n_dims,)
        plot: If True, plot NRMSE vs horizon for each dimension
    
    Returns:
        results: Dict mapping horizon T to array of per-dimension NRMSE values
                 Returns [nan, nan, ...] for horizons that exceed trajectory length
        aggregate: Dict mapping horizon T to aggregate (mean across dims) NRMSE
    """
    x_true = to_tensor(x_true)
    x_pred = to_tensor(x_pred)

    # Validate shapes
    assert x_true.shape == x_pred.shape, \
        f"Shape mismatch: x_true={x_true.shape}, x_pred={x_pred.shape}"
    assert x_true.dim() == 3, \
        f"Expected 3D tensor (batch, time, dims), got {x_true.dim()}D"

    n, t_max, d = x_true.shape

    # Per-dimension normalization
    if sigma_per_dim is None:
        sigma_per_dim = torch.std(x_true, dim=(0, 1)) + 1e-8

    results = {}
    aggregate = {}
    
    for T in horizons:
        if T >= t_max:
            # Explicitly return nan for horizons beyond trajectory length
            results[T] = [float('nan')] * d
            aggregate[T] = float('nan')
            continue
        
        # CUMULATIVE: error from step 1 up to step T
        err = x_true[:, 1:T+1, :] - x_pred[:, 1:T+1, :]
        
        # Per-dimension NRMSE
        per_dim_nrmse = []
        for dim in range(d):
            err_dim = err[:, :, dim] / sigma_per_dim[dim]
            nrmse_dim = torch.sqrt(torch.mean(err_dim ** 2))
            per_dim_nrmse.append(float(nrmse_dim.item()))
        
        results[T] = per_dim_nrmse
        aggregate[T] = float(np.mean(per_dim_nrmse))

    # ------------------- Plot -------------------
    if plot and len(results) > 0:
        import matplotlib.pyplot as plt
        H = sorted(results.keys())
        
        plt.figure(figsize=(8, 5))
        for dim in range(d):
            V = [results[h][dim] for h in H]
            plt.plot(H, V, marker='o', linewidth=2, label=f'Dim {dim}')
        
        # Also plot aggregate
        V_agg = [aggregate[h] for h in H]
        plt.plot(H, V_agg, 'k--', linewidth=2, label='Mean')
        
        plt.xlabel("Horizon T")
        plt.ylabel("Cumulative NRMSE(T)")
        plt.title("Per-Dimension Cumulative NRMSE vs Horizon")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return results, aggregate


# -----------------------------------------------------------
# Phase-portrait fidelity
# -----------------------------------------------------------

def chamfer_distance_phase(x_true, x_pred, dims=None, plot=False):
    """
    Compute Chamfer distance in phase space.
    
    Args:
        x_true: Ground truth trajectories (n_batch, n_steps, n_dims) or (n_steps, n_dims)
        x_pred: Predicted trajectories (same shape as x_true)
        dims: Dimensions to use for Chamfer distance. 
              If None, uses ALL dimensions (full state space).
              If tuple/list, uses only specified dimensions.
        plot: If True and dims has 2 elements, draw phase portrait
    
    Returns:
        chamfer: Chamfer distance value (inf if predictions contain NaN/Inf)
    """
    x_true_t = to_tensor(x_true)
    x_pred_t = to_tensor(x_pred)
    
    # Validate shapes match (after potential broadcasting)
    assert x_true_t.shape[-1] == x_pred_t.shape[-1], \
        f"Dimension mismatch: x_true has {x_true_t.shape[-1]} dims, x_pred has {x_pred_t.shape[-1]}"
    
    x_true_flat = flatten_batch_time(x_true_t)
    x_pred_flat = flatten_batch_time(x_pred_t)
    
    # Use all dimensions if not specified
    if dims is None:
        dims = list(range(x_true_flat.shape[-1]))
    
    # Validate dims are valid
    n_dims = x_true_flat.shape[-1]
    assert all(0 <= d < n_dims for d in dims), \
        f"Invalid dims {dims} for data with {n_dims} dimensions"
    
    x_true_ = x_true_flat[:, list(dims)]
    x_pred_ = x_pred_flat[:, list(dims)]

    # Check for NaN/Inf - return inf with warning
    if torch.any(torch.isnan(x_pred_)) or torch.any(torch.isinf(x_pred_)):
        import warnings
        warnings.warn("Predictions contain NaN/Inf, returning inf for Chamfer distance")
        return float('inf')

    dist = torch.cdist(x_true_, x_pred_, p=2)
    d12 = torch.mean(torch.min(dist, dim=1).values)
    d21 = torch.mean(torch.min(dist, dim=0).values)
    chamfer = float((d12 + d21).item())

    # ------------------- Plot -------------------
    if plot and len(dims) == 2:
        import matplotlib.pyplot as plt
        x_t = x_true_.cpu().numpy()
        x_p = x_pred_.cpu().numpy()

        plt.figure(figsize=(5,5))
        plt.plot(x_t[:,0], x_t[:,1], '-o', markersize=3, alpha=0.7, label="True")
        plt.plot(x_p[:,0], x_p[:,1], '-x', markersize=3, alpha=0.7, label="Pred")
        plt.xlabel(f"Dim {dims[0]}")
        plt.ylabel(f"Dim {dims[1]}")
        plt.title("Phase Portrait: True vs Predicted")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return chamfer


def chamfer_distance_full_state(x_true, x_pred):
    """
    Convenience function: Chamfer distance using ALL state dimensions.
    
    This is the mathematically correct way to measure phase space fidelity
    for systems with more than 2 state variables.
    """
    return chamfer_distance_phase(x_true, x_pred, dims=None, plot=False)


# -----------------------------------------------------------
# Stability metrics
# -----------------------------------------------------------

def spectral_radius(K):
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()

    eigvals = np.linalg.eigvals(K)
    rho = np.max(np.abs(eigvals))
    return float(rho), eigvals


def long_horizon_divergence_rate(x_true, x_pred, plot=False):
    """
    Fit log(error_t) = a t + b.
    If plot=True → draw log(error) vs time.
    """
    x_true = to_tensor(x_true)
    x_pred = to_tensor(x_pred)

    n, T, d = x_true.shape

    errors = torch.sqrt(torch.sum((x_true - x_pred)**2, dim=-1)).mean(0)
    errors = errors + 1e-8
    log_err = torch.log(errors)

    t = torch.arange(T, dtype=torch.float32)
    A = torch.stack([t, torch.ones_like(t)], dim=1)
    b = log_err.unsqueeze(1)

    lstsq_result = torch.linalg.lstsq(A, b)
    slope = lstsq_result.solution[0,0].item()

    # ------------------- Plot -------------------
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,4))
        plt.plot(t, log_err.cpu(), '-o', markersize=3)
        plt.xlabel("Time step")
        plt.ylabel("log(error)")
        plt.title("Long-horizon Error Growth")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return float(slope), errors.cpu().numpy()


def long_horizon_divergence_rate_per_dim(x_true, x_pred):
    """
    Compute divergence rate (exponential growth rate) per dimension.
    
    Fits log(|error_t|) = a*t + b for each dimension separately.
    
    Args:
        x_true: (batch, T, d) true trajectories
        x_pred: (batch, T, d) predicted trajectories
    
    Returns:
        slopes: list of d floats, divergence rate per dimension
        aggregate_slope: float, RMS of per-dim slopes
    """
    x_true = to_tensor(x_true)
    x_pred = to_tensor(x_pred)
    
    n, T, d = x_true.shape
    
    t = torch.arange(T, dtype=torch.float32)
    A = torch.stack([t, torch.ones_like(t)], dim=1)
    
    slopes = []
    for dim in range(d):
        # Per-dimension absolute error, averaged over batch
        errors_dim = torch.abs(x_true[:, :, dim] - x_pred[:, :, dim]).mean(0)  # (T,)
        errors_dim = errors_dim + 1e-8
        log_err = torch.log(errors_dim)
        
        b = log_err.unsqueeze(1)
        lstsq_result = torch.linalg.lstsq(A, b)
        slope = lstsq_result.solution[0, 0].item()
        slopes.append(slope)
    
    # RMS aggregate
    aggregate_slope = np.sqrt(np.mean(np.array(slopes)**2))
    
    return slopes, aggregate_slope


# -----------------------------------------------------------
# Reconstruction metrics
# -----------------------------------------------------------

def reconstruction_error(x_true, x_rec, plot=False):
    """
    Compute ||x - x_rec||².
    If plot=True → plot true vs reconstructed trajectory (first dimension).
    """
    x_true = flatten_batch_time(to_tensor(x_true))
    x_rec = flatten_batch_time(to_tensor(x_rec))

    diff = x_true - x_rec
    rec_err = torch.mean(torch.sum(diff * diff, dim=-1)).item()

    # ------------------- Plot -------------------
    if plot:
        import matplotlib.pyplot as plt
        x_t = x_true.cpu().numpy()
        x_r = x_rec.cpu().numpy()

        plt.figure(figsize=(6,4))
        plt.plot(x_t[:,0], label="True", alpha=0.7)
        plt.plot(x_r[:,0], label="Reconstructed", alpha=0.7)
        plt.xlabel("Time step")
        plt.ylabel("State[0]")
        plt.title("Reconstruction Quality (dim 0)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return float(rec_err)
