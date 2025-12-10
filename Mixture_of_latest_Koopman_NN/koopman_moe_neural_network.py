"""
Mixture of Koopman Experts (MoKE) with TRUE Linear Dynamics

Architecture:
- Single SHARED encoder φ: x → z
- Single SHARED decoder ψ: z → x  
- Multiple Koopman operators {K_1, K_2, ..., K_M}
- Gating network π(z_0): computes weights from INITIAL latent state ONLY

KEY INSIGHT: Initial-Condition (IC) Gating
============================================
The gating weights π are computed ONCE from z_0 and held FIXED for the 
entire trajectory. This gives us a fixed effective operator:

    K_eff = Σ π_k(z_0) · K_k   (FIXED for trajectory)

Then dynamics are truly LINEAR:
    z_{t+1} = K_eff · z_t
    z_t = K_eff^t · z_0   ← Closed-form multi-step prediction!

This preserves Koopman structure and enables efficient long-horizon prediction
via matrix powers instead of iterative simulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import base components
from koopman_mixture_neural_network import (
    MLPEncoder,
    MLPDecoder,
    spectral_radius_penalty,
    hankel_stack_batch,
    compute_hankel_svd,
    compute_hankel_linearity_loss
)


class LatentGatingNetwork(nn.Module):
    """
    Gating network: computes expert weights from latent state.
    
    IMPORTANT: This is called ONCE on z_0, not at every time step!
    The weights are then fixed for the entire trajectory.
    
    Args:
        n_z: dimension of latent space
        n_experts: number of Koopman operators
        temperature: softmax temperature (lower = sharper selection)
    """
    def __init__(self, n_z=20, n_experts=4, temperature=1.0):
        super().__init__()
        self.n_experts = n_experts
        self.temperature = temperature
        
        self.net = nn.Sequential(
            nn.Linear(n_z, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_experts)
        )
    
    def forward(self, z, temperature=None):
        """
        Compute gating weights.
        
        Args:
            z: latent state (batch, n_z) - should be z_0!
            temperature: optional temperature override
        
        Returns:
            pi: gating weights (batch, n_experts), sum to 1
        """
        temp = temperature if temperature is not None else self.temperature
        logits = self.net(z)
        pi = F.softmax(logits / temp, dim=-1)
        return pi


class KoopmanMoE(nn.Module):
    """
    Mixture of Koopman Experts with TRUE Linear Dynamics
    
    Architecture:
    - Single shared encoder φ: x → z
    - Single shared decoder ψ: z → x
    - Gating π(z_0): computes weights from INITIAL state only
    - Multiple Koopman operators K_1, ..., K_M
    
    Key Equation (IC Gating):
        K_eff = Σ π_k(z_0) · K_k     (computed ONCE)
        z_t = K_eff^t · z_0           (true linear dynamics!)
    
    Args:
        n_x: dimension of state space
        n_z: dimension of latent space
        n_experts: number of Koopman operators
        hidden: hidden layer size for encoder/decoder
    """
    def __init__(self, n_x=3, n_z=20, n_experts=4, hidden=128):
        super().__init__()
        self.n_x = n_x
        self.n_z = n_z
        self.n_experts = n_experts
        
        # Single shared encoder: x → z
        self.encoder = MLPEncoder(n_in=n_x, n_latent=n_z, hidden=hidden)
        
        # Single shared decoder: z → x
        self.decoder = MLPDecoder(n_latent=n_z, n_out=n_x, hidden=hidden)
        
        # Gating network (called ONCE on z_0)
        self.gating = LatentGatingNetwork(n_z=n_z, n_experts=n_experts)
        
        # Multiple Koopman operators (all act on same latent space)
        # Initialize near identity for stability
        self.K = nn.ParameterList([
            nn.Parameter(torch.eye(n_z) + 0.01 * torch.randn(n_z, n_z))
            for _ in range(n_experts)
        ])
        
        # Backward operators for bidirectional constraint
        self.K_b = nn.ParameterList([
            nn.Parameter(torch.eye(n_z) + 0.01 * torch.randn(n_z, n_z))
            for _ in range(n_experts)
        ])
    
    def compute_effective_K(self, z0):
        """
        Compute effective Koopman operator from initial latent state.
        
        This is the KEY method: K_eff is computed ONCE from z_0 and then
        used for ALL future time steps, preserving linear dynamics.
        
        K_eff = Σ π_k(z_0) · K_k
        
        Args:
            z0: initial latent state (batch, n_z)
        
        Returns:
            K_eff: effective Koopman operator (batch, n_z, n_z)
            pi: gating weights (batch, n_experts)
        """
        # Gating weights from initial state ONLY
        pi = self.gating(z0)  # (batch, n_experts)
        
        # Build K_eff = Σ π_k K_k
        batch_size = z0.shape[0]
        K_eff = torch.zeros(batch_size, self.n_z, self.n_z, device=z0.device)
        
        for k in range(self.n_experts):
            # pi[:, k] is (batch,), K[k] is (n_z, n_z)
            # K_eff[b] += pi[b, k] * K[k]
            K_eff = K_eff + pi[:, k].unsqueeze(-1).unsqueeze(-1) * self.K[k].unsqueeze(0)
        
        return K_eff, pi
    
    def forward(self, x):
        """
        Forward pass: reconstruction (autoencoder).
        
        Args:
            x: input state (batch, n_x)
        
        Returns:
            dict with x_rec, z, weights
        """
        z = self.encoder(x)
        x_rec = self.decoder(z)
        
        # Compute gating (for monitoring, not used in reconstruction)
        _, pi = self.compute_effective_K(z)
        
        return {
            'x_rec': x_rec,
            'z': z,
            'weights': pi
        }
    
    def predict_next(self, x):
        """
        One-step prediction using IC gating.
        
        z_1 = K_eff(z_0) · z_0
        
        Args:
            x: current state (batch, n_x)
        
        Returns:
            x_next: predicted next state
            pi: gating weights
            z0: current latent
            z1: next latent
        """
        z0 = self.encoder(x)
        K_eff, pi = self.compute_effective_K(z0)
        
        # z1 = K_eff @ z0 (batched matrix-vector multiplication)
        z1 = torch.bmm(K_eff, z0.unsqueeze(-1)).squeeze(-1)
        
        x_next = self.decoder(z1)
        return x_next, pi, z0, z1
    
    def predict_latent_next(self, z, K_eff):
        """
        One-step latent prediction with pre-computed K_eff.
        
        Args:
            z: current latent (batch, n_z)
            K_eff: effective Koopman operator (batch, n_z, n_z)
        
        Returns:
            z_next: next latent state
        """
        return torch.bmm(K_eff, z.unsqueeze(-1)).squeeze(-1)
    
    def matrix_power_batch(self, K, t):
        """
        Compute K^t for batched matrices using repeated squaring.
        
        O(log t) matrix multiplications instead of O(t).
        This enables efficient long-horizon prediction!
        
        Args:
            K: batched matrices (batch, n, n)
            t: power to raise to
        
        Returns:
            K^t: (batch, n, n)
        """
        if t == 0:
            batch, n, _ = K.shape
            return torch.eye(n, device=K.device).unsqueeze(0).expand(batch, -1, -1).clone()
        elif t == 1:
            return K.clone()
        
        batch, n, _ = K.shape
        result = torch.eye(n, device=K.device).unsqueeze(0).expand(batch, -1, -1).clone()
        base = K.clone()
        
        while t > 0:
            if t % 2 == 1:
                result = torch.bmm(result, base)
            base = torch.bmm(base, base)
            t //= 2
        
        return result
    
    def predict(self, x0, n_steps=100):
        """
        Multi-step prediction with TRUE linear dynamics.
        
        K_eff is computed ONCE from z_0, then held fixed:
            z_t = K_eff · z_{t-1}   (iterative)
        
        Args:
            x0: initial state (batch, n_x)
            n_steps: number of steps to predict
        
        Returns:
            predictions: tensor (n_steps+1, batch, n_x)
            weights: gating weights (batch, n_experts)
        """
        self.eval()
        
        # Encode initial state
        z0 = self.encoder(x0)
        
        # Compute effective K ONCE from initial condition
        K_eff, pi = self.compute_effective_K(z0)
        
        # Iterative prediction with FIXED K_eff
        predictions = [x0]
        z = z0
        
        with torch.no_grad():
            for _ in range(n_steps):
                z = self.predict_latent_next(z, K_eff)
                predictions.append(self.decoder(z))
        
        return torch.stack(predictions), pi
    
    def predict_at_horizons(self, x0, horizons):
        """
        Predict at SPECIFIC time steps using matrix powers.
        
        For step t: z_t = K_eff^t · z_0
        
        This is efficient for sparse prediction horizons!
        E.g., predict at t=[1, 10, 100, 1000] directly.
        
        Args:
            x0: initial state (batch, n_x)
            horizons: list of time steps to predict at
        
        Returns:
            predictions: dict {t: x_t} for each t in horizons
            pi: gating weights
            z0: initial latent
        """
        self.eval()
        
        z0 = self.encoder(x0)
        K_eff, pi = self.compute_effective_K(z0)
        
        predictions = {}
        latents = {}
        
        with torch.no_grad():
            for t in horizons:
                # Compute K_eff^t
                K_power = self.matrix_power_batch(K_eff, t)
                
                # z_t = K^t @ z_0
                z_t = torch.bmm(K_power, z0.unsqueeze(-1)).squeeze(-1)
                
                predictions[t] = self.decoder(z_t)
                latents[t] = z_t
        
        return predictions, latents, pi, z0
    
    def get_spectral_radius(self, K_eff):
        """
        Compute spectral radius of effective K operator.
        
        Args:
            K_eff: (batch, n_z, n_z)
        
        Returns:
            rho: spectral radius per batch element
        """
        # For small batches, compute eigenvalues directly
        eigenvalues = torch.linalg.eigvals(K_eff)
        rho = eigenvalues.abs().max(dim=-1).values
        return rho
    
    def sparsity_loss(self, mode: str = "l1"):
        """
        Compute sparsity penalty on encoder/decoder weights.
        """
        penalty = torch.zeros(1, device=self.K[0].device)
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
    
    def get_latent_sequence(self, x_sequence):
        """
        Encode a sequence of states to latent space.
        
        Args:
            x_sequence: (batch, T, n_x)
        
        Returns:
            z_sequence: (batch, T, n_z)
        """
        batch, T, n_x = x_sequence.shape
        x_flat = x_sequence.reshape(-1, n_x)
        z_flat = self.encoder(x_flat)
        return z_flat.reshape(batch, T, -1)
    
    def hankel_linearity_loss(self, x_sequence, L=4, r=8, device='cpu'):
        """Compute Hankel-based linearity loss."""
        z_seq = self.get_latent_sequence(x_sequence)
        return compute_hankel_linearity_loss(z_seq, L=L, r=r, device=device)


# ============================================================================
# Legacy compatibility aliases
# ============================================================================
GatingNetwork = LatentGatingNetwork


class NeuralBlendingNetwork(nn.Module):
    """DEPRECATED: Kept for backward compatibility only."""
    def __init__(self, n_x=2, n_experts=8):
        super().__init__()
        print("WARNING: NeuralBlendingNetwork is deprecated.")
        self.net = nn.Linear(n_experts * n_x + n_experts, n_x)
    
    def forward(self, expert_predictions, gating_weights):
        preds_concat = torch.cat(expert_predictions, dim=-1)
        combined = torch.cat([preds_concat, gating_weights], dim=-1)
        return self.net(combined)
