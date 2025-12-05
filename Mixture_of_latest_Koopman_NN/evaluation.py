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


def multi_step_nrmse(x_true, x_pred, horizons, sigma_test=None, plot=False):
    """
    Compute NRMSE(T) for multiple horizons.
    If plot=True → draw NRMSE vs Horizon curve.
    """
    x_true = to_tensor(x_true)
    x_pred = to_tensor(x_pred)

    n, t_max, d = x_true.shape

    if sigma_test is None:
        sigma_test = torch.std(x_true.reshape(n * t_max, d)) + 1e-8

    results = {}
    for T in horizons:
        if T >= t_max:
            continue
        err = x_true[:, T, :] - x_pred[:, T, :]
        mse_T = torch.mean(torch.sum(err * err, dim=-1))
        rmse_T = torch.sqrt(mse_T)
        results[T] = float((rmse_T / sigma_test).item())

    # ------------------- Plot -------------------
    if plot and len(results) > 0:
        import matplotlib.pyplot as plt
        H = sorted(results.keys())
        V = [results[h] for h in H]

        plt.figure(figsize=(6,4))
        plt.plot(H, V, marker='o', linewidth=2)
        plt.xlabel("Horizon T")
        plt.ylabel("NRMSE(T)")
        plt.title("Multi-step NRMSE vs Horizon")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return results



# -----------------------------------------------------------
# Phase-portrait fidelity
# -----------------------------------------------------------

def chamfer_distance_phase(x_true, x_pred, dims=(0, 1), plot=False):
    """
    Compute Chamfer distance.
    If plot=True → draw true vs predicted phase portrait.
    """
    x_true_ = flatten_batch_time(to_tensor(x_true))[:, list(dims)]
    x_pred_ = flatten_batch_time(to_tensor(x_pred))[:, list(dims)]

    dist = torch.cdist(x_true_, x_pred_, p=2)
    d12 = torch.mean(torch.min(dist, dim=1).values)
    d21 = torch.mean(torch.min(dist, dim=0).values)
    chamfer = float((d12 + d21).item())

    # ------------------- Plot -------------------
    if plot:
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
