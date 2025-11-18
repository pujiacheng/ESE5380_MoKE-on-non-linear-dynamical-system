"""
Koopman Autoencoder: eDMD + Hankel + Bidirectional Optimization

This module implements a Koopman Autoencoder combining ideas from eDMD, 
Hankel/HAVOK, and a bidirectional factorization of the linear latent operator.

Features:
- Nonlinear encoder/decoder (deep MLP)
- Learnable forward A_f and backward A_b matrices with A_f A_b ≈ I constraint
- eDMD observables network g(x) with learnable linear map A_g
- Hankel (delay) stacking of latent z to create delay coordinates and enforce linearity
- Loss: reconstruction + latent linearity (1-step & multi-step) + eDMD regression + 
        Hankel linearity + bidirectional constraint + spectral penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPEncoder(nn.Module):
    """Encoder network: maps state x to latent representation z"""
    def __init__(self, n_in=2, n_latent=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_latent)
        )
    
    def forward(self, x):
        return self.net(x)


class MLPDecoder(nn.Module):
    """Decoder network: maps latent representation z back to state x"""
    def __init__(self, n_latent=6, n_out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_latent, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_out)
        )
    
    def forward(self, z):
        return self.net(z)


class ObservablesNet(nn.Module):
    """Observables network: maps state x to observables g(x) for eDMD"""
    def __init__(self, n_in=2, p=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.ReLU(),
            nn.Linear(64, p)
        )
    
    def forward(self, x):
        return self.net(x)


class KoopmanAE(nn.Module):
    """
    Koopman Autoencoder model
    
    Args:
        n_x: dimension of state space
        n_z: dimension of latent space
        p: dimension of observables space
    """
    def __init__(self, n_x=2, n_z=6, p=20):
        super().__init__()
        self.encoder = MLPEncoder(n_in=n_x, n_latent=n_z)
        self.decoder = MLPDecoder(n_latent=n_z, n_out=n_x)
        self.obs = ObservablesNet(n_in=n_x, p=p)
        
        # Bidirectional linear maps in latent space
        self.A_f = nn.Parameter(torch.eye(n_z) + 0.01*torch.randn(n_z, n_z))
        self.A_b = nn.Parameter(torch.eye(n_z) + 0.01*torch.randn(n_z, n_z))
        
        # eDMD linear map on observables (learnable)
        self.A_g = nn.Parameter(torch.eye(p))
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: input state tensor of shape (batch_size, n_x)
        
        Returns:
            dict with keys:
                - z: latent representation
                - x_rec: reconstructed state
                - g: observables
        """
        z = self.encoder(x)
        x_rec = self.decoder(z)
        g = self.obs(x)
        return dict(z=z, x_rec=x_rec, g=g)


def spectral_radius_penalty(A, iters=10, target=1.05):
    """
    Compute spectral radius penalty using power iteration
    
    Args:
        A: matrix to compute spectral radius of
        iters: number of power iteration steps
        target: target spectral radius (penalize if > target)
    
    Returns:
        penalty value
    """
    A = A.detach() if not A.requires_grad else A
    v = torch.randn(A.shape[0], 1, device=A.device)
    v = v / (v.norm() + 1e-9)
    for _ in range(iters):
        v = A @ v
        v = v / (v.norm() + 1e-12)
    Av = A @ v
    rho = (v.squeeze() * Av.squeeze()).sum()
    penalty = F.relu(rho - target)**2
    return penalty


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

