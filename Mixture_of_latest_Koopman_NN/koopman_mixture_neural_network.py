"""
Koopman Autoencoder: Hankel + Bidirectional Optimization

This module implements a Koopman Autoencoder combining ideas from 
Hankel/HAVOK and a bidirectional factorization of the linear latent operator.

Features:
- Nonlinear encoder/decoder (deep MLP with BatchNorm)
- Learnable forward A_f and backward A_b matrices with A_f A_b ≈ I constraint
- Hankel (delay) stacking of latent z to create delay coordinates and enforce linearity
- Sparsity regularization on encoder/decoder weights
- Loss: reconstruction + latent linearity + Hankel linearity + bidirectional constraint + spectral penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPEncoder(nn.Module):
    """Encoder network: maps state x to latent representation z"""

    def __init__(self, n_in=2, n_latent=6, hidden=128):
        super().__init__()
        self.n_in = n_in
        self.n_latent = n_latent
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_latent)
        )

    def forward(self, x):
        return self.net(x)


class MLPDecoder(nn.Module):
    """Decoder network: maps latent representation z back to state x"""

    def __init__(self, n_latent=6, n_out=2, hidden=128):
        super().__init__()
        self.n_latent = n_latent
        self.n_out = n_out
        self.net = nn.Sequential(
            nn.Linear(n_latent, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out)
        )
    
    def forward(self, z):
        return self.net(z)


class KoopmanAE(nn.Module):
    """
    Koopman Autoencoder model
    
    Architecture:
    - Encoder: x → z (nonlinear mapping to latent space)
    - Linear dynamics: z_{t+1} = A_f @ z_t (Koopman operator)
    - Decoder: z → x (nonlinear mapping back to state space)
    - Bidirectional: A_b for backward dynamics, A_f @ A_b ≈ I
    
    Args:
        n_x: dimension of state space
        n_z: dimension of latent space
        hidden: hidden layer size for encoder/decoder
        expert_id: optional ID for use in MoE (default: None)
    """
    def __init__(self, n_x=2, n_z=20, hidden=128, expert_id=None):
        super().__init__()
        self.n_x = n_x
        self.n_z = n_z
        self.expert_id = expert_id
        
        self.encoder = MLPEncoder(n_in=n_x, n_latent=n_z, hidden=hidden)
        self.decoder = MLPDecoder(n_latent=n_z, n_out=n_x, hidden=hidden)
        
        # Bidirectional linear maps in latent space
        self.A_f = nn.Parameter(torch.eye(n_z) + 0.01*torch.randn(n_z, n_z))
        self.A_b = nn.Parameter(torch.eye(n_z) + 0.01*torch.randn(n_z, n_z))
    
    def forward(self, x):
        """
        Forward pass: encode and reconstruct
        
        Args:
            x: input state tensor of shape (batch_size, n_x)
        
        Returns:
            dict with keys:
                - z: latent representation
                - x_rec: reconstructed state
        """
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return dict(z=z, x_rec=x_rec)
    
    def predict_next(self, x):
        """
        Predict next state using Koopman operator
        
        Args:
            x: current state (batch_size, n_x)
        
        Returns:
            x_next: predicted next state (batch_size, n_x)
        """
        z = self.encoder(x)
        z_next = z @ self.A_f.T
        x_next = self.decoder(z_next)
        return x_next
    
    def predict_sequence(self, x0, n_steps):
        """
        Predict sequence of states using Koopman operator
        
        Args:
            x0: initial state (batch_size, n_x)
            n_steps: number of steps to predict
        
        Returns:
            sequence: tensor of shape (n_steps+1, batch_size, n_x)
        """
        predictions = [x0]
        z = self.encoder(x0)
        
        for _ in range(n_steps):
            z = z @ self.A_f.T
            x = self.decoder(z)
            predictions.append(x)
        
        return torch.stack(predictions)

    def sparsity_loss(self, mode: str = "l1"):
        """
        Compute sparsity-promoting penalty over encoder/decoder weights.

        Args:
            mode: 'l1' (default) or 'l2'
        
        Returns:
            penalty value (scalar tensor)
        """
        penalty = torch.zeros(1, device=self.A_f.device)
        modules = [self.encoder, self.decoder]
        
        for module in modules:
            for name, param in module.named_parameters():
                if "weight" not in name:
                    continue
                if mode == "l2":
                    penalty = penalty + torch.sum(param**2)
                else:
                    penalty = penalty + torch.sum(param.abs())
        return penalty.squeeze()


def spectral_radius_penalty(A, iters=10, target=1.005, lower=0.995):
    """
    Compute spectral radius penalty using power iteration
    
    Penalizes BOTH extremes:
    - ρ(A) > target: predictions will explode (unstable)
    - ρ(A) < lower: predictions will decay to zero (over-damped)
    
    For conservative systems (like Duffing), ideal ρ ≈ 1.0
    
    Default bounds [0.995, 1.005] chosen for 100-step predictions:
    - 0.995^100 ≈ 0.61 (39% decay - acceptable)
    - 1.005^100 ≈ 1.65 (65% growth - acceptable)
    
    Args:
        A: matrix to compute spectral radius of
        iters: number of power iteration steps
        target: upper bound for spectral radius (default: 1.005)
        lower: lower bound for spectral radius (default: 0.995)
    
    Returns:
        penalty value (penalizes if ρ outside [lower, target])
    """
    v = torch.randn(A.shape[0], 1, device=A.device)
    v = v / (v.norm() + 1e-9)
    for _ in range(iters):
        v = A @ v
        v = v / (v.norm() + 1e-12)
    Av = A @ v
    rho = (v.squeeze() * Av.squeeze()).sum()
    
    # Penalize if ρ > target (too high → explosion)
    penalty_high = F.relu(rho - target)**2
    
    # Penalize if ρ < lower (too low → decay to zero)
    penalty_low = F.relu(lower - rho)**2
    
    return penalty_high + penalty_low


def hankel_stack_batch(z_seq, L):
    """
    Build Hankel matrix from sequence of latent states
    
    Args:
        z_seq: tensor of shape (B, T, n_z) where B is batch, T is time steps, n_z is latent dim
        L: window length for Hankel stacking
    
    Returns:
        Hankel matrix of shape (B, cols, L*n_z) where cols = T - L + 1
    """
    B, T, nz = z_seq.shape
    cols = T - L + 1
    H = []
    for i in range(cols):
        block = z_seq[:, i:i+L, :].reshape(B, -1)
        H.append(block)
    H = torch.stack(H, dim=1)
    return H


def compute_hankel_svd(H):
    """
    Compute SVD of Hankel matrix
    
    Args:
        H: Hankel matrix of shape (B, cols, d)
    
    Returns:
        U, S, Vt from SVD decomposition
    """
    B, cols, d = H.shape
    M = H.reshape(B*cols, d).cpu().numpy()
    M_centered = M - M.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
    return U, S, Vt


def compute_hankel_linearity_loss(z_seq, L=4, r=8, device='cpu'):
    """
    Compute Hankel-based linearity loss
    
    Projects latent sequences onto low-rank Hankel coordinates and
    enforces linear dynamics in that space.
    
    Args:
        z_seq: tensor of shape (B, T, n_z) - sequence of latent states
        L: Hankel window length
        r: rank for SVD truncation
        device: torch device
    
    Returns:
        loss: Hankel linearity loss (scalar)
    """
    H = hankel_stack_batch(z_seq, L=L)  # (B, cols, L*n_z)
    U, S, Vt = compute_hankel_svd(H)
    
    # Take top r components (low-rank)
    r = min(r, Vt.shape[0])
    V_r = Vt[:r].T  # shape (L*n_z, r)
    
    # Project Hankel columns to v coords
    Hmat = H.reshape(-1, H.shape[-1]).cpu().numpy()  # (B*cols, d)
    Vcoords = (Hmat @ V_r).reshape(H.shape[0], H.shape[1], r)
    Vcoords = torch.tensor(Vcoords, dtype=torch.float32, device=device)
    
    # Define v_t as column 0..cols-2 and v_t+1 as 1..cols-1
    v_t = Vcoords[:, :-1, :].reshape(-1, r)
    v_tp1 = Vcoords[:, 1:, :].reshape(-1, r)
    
    # Learn small A_v on the fly by ridge regression closed-form
    reg = 1e-6
    vt = v_t.detach().cpu().numpy()
    vtp1 = v_tp1.detach().cpu().numpy()
    G = vt.T @ vt + reg * np.eye(r)
    A_v = (vtp1.T @ vt) @ np.linalg.inv(G)  # shape (r, r)
    A_v = torch.tensor(A_v, dtype=torch.float32, device=device)
    
    # Enforce v_tp1 ≈ A_v @ v_t
    v_pred = v_t @ A_v.T
    loss_hankel = F.mse_loss(v_pred, v_tp1)
    
    return loss_hankel
