"""
Mixture of Experts (MoE) Koopman Neural Network

Architecture:
- 8 fully separate experts (each with Encoder + Koopman + Decoder)
- Input-based soft gating (recomputed at each step)
- Neural network blending function (learned combination in output space)
- No ObservablesNet (removed for simplicity)

Each expert learns its own coordinate system and dynamics.
Experts specialize through load balancing and diversity losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPEncoder(nn.Module):
    """Encoder network: maps state x to latent representation z"""
    def __init__(self, n_in=2, n_latent=6, expert_id=0):
        super().__init__()
        self.expert_id = expert_id
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
    def __init__(self, n_latent=6, n_out=2, expert_id=0):
        super().__init__()
        self.expert_id = expert_id
        self.net = nn.Sequential(
            nn.Linear(n_latent, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_out)
        )
    
    def forward(self, z):
        return self.net(z)


class GatingNetwork(nn.Module):
    """
    Gating network: decides which experts to activate based on input state
    
    Args:
        n_in: dimension of input state
        n_experts: number of experts
        gating_type: 'soft' (default), 'hard', or 'topk'
    """
    def __init__(self, n_in=2, n_experts=8, gating_type='soft', k=2):
        super().__init__()
        self.n_experts = n_experts
        self.gating_type = gating_type
        self.k = k  # For top-k gating
        
        self.net = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_experts)
        )
    
    def forward(self, x):
        """
        Args:
            x: input state (batch_size, n_in)
        
        Returns:
            weights: gating weights (batch_size, n_experts)
        """
        logits = self.net(x)
        
        if self.gating_type == 'soft':
            # Soft mixture: all experts contribute
            weights = F.softmax(logits, dim=-1)
        
        elif self.gating_type == 'hard':
            # Hard selection: winner-takes-all
            max_idx = logits.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(logits)
            weights.scatter_(1, max_idx, 1.0)
        
        elif self.gating_type == 'topk':
            # Top-k: only k experts active
            topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)
            weights = torch.zeros_like(logits)
            topk_weights = F.softmax(topk_vals, dim=-1)
            weights.scatter_(1, topk_idx, topk_weights)
        
        return weights


class NeuralBlendingNetwork(nn.Module):
    """
    Neural network for blending expert predictions
    
    Instead of simple weighted sum, learns non-linear combination
    
    Args:
        n_x: dimension of state space
        n_experts: number of experts
    """
    def __init__(self, n_x=2, n_experts=8):
        super().__init__()
        self.n_experts = n_experts
        self.n_x = n_x
        
        # Input: expert predictions + gating weights
        # Shape: (batch, n_experts * n_x + n_experts)
        input_dim = n_experts * n_x + n_experts
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_x)  # Output final state
        )
    
    def forward(self, expert_predictions, gating_weights):
        """
        Args:
            expert_predictions: list of tensors, each (batch, n_x)
            gating_weights: tensor (batch, n_experts)
        
        Returns:
            blended_output: tensor (batch, n_x)
        """
        # Concatenate all expert predictions
        # Shape: (batch, n_experts * n_x)
        preds_concat = torch.cat(expert_predictions, dim=-1)
        
        # Concatenate with gating weights
        # Shape: (batch, n_experts * n_x + n_experts)
        combined = torch.cat([preds_concat, gating_weights], dim=-1)
        
        # Neural network learns the blending
        blended = self.net(combined)
        
        return blended


class CompleteExpert(nn.Module):
    """
    Single complete expert: Encoder + Koopman Operator + Decoder
    
    Each expert learns its own coordinate system and linear dynamics
    """
    def __init__(self, n_x=2, n_z=6, expert_id=0):
        super().__init__()
        self.expert_id = expert_id
        self.n_z = n_z
        
        # Encoder: x → z
        self.encoder = MLPEncoder(n_in=n_x, n_latent=n_z, expert_id=expert_id)
        
        # Decoder: z → x
        self.decoder = MLPDecoder(n_latent=n_z, n_out=n_x, expert_id=expert_id)
        
        # Forward Koopman operator: z_t+1 = A_f @ z_t
        self.A_f = nn.Parameter(torch.eye(n_z) + 0.01*torch.randn(n_z, n_z))
        
        # Backward Koopman operator (for bidirectional constraint)
        self.A_b = nn.Parameter(torch.eye(n_z) + 0.01*torch.randn(n_z, n_z))
    
    def forward(self, x):
        """
        Encode and reconstruct
        
        Args:
            x: input state (batch, n_x)
        
        Returns:
            dict with 'z' and 'x_rec'
        """
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return {'z': z, 'x_rec': x_rec}
    
    def predict_next(self, x):
        """
        Full prediction: x → z → A@z → x_next
        
        Args:
            x: current state (batch, n_x)
        
        Returns:
            x_next: predicted next state (batch, n_x)
        """
        z = self.encoder(x)
        z_next = z @ self.A_f.T
        x_next = self.decoder(z_next)
        return x_next


class KoopmanMoE(nn.Module):
    """
    Mixture of Experts Koopman Autoencoder
    
    Architecture:
    - 8 fully separate experts
    - Input-based gating (recomputed dynamically)
    - Neural network blending in output space
    - No ObservablesNet
    
    Args:
        n_x: dimension of state space
        n_z: dimension of latent space (same for all experts)
        n_experts: number of experts
    """
    def __init__(self, n_x=2, n_z=6, n_experts=8):
        super().__init__()
        self.n_x = n_x
        self.n_z = n_z
        self.n_experts = n_experts
        
        # Gating network: x → weights
        self.gating = GatingNetwork(n_in=n_x, n_experts=n_experts, gating_type='soft')
        
        # 8 complete experts
        self.experts = nn.ModuleList([
            CompleteExpert(n_x=n_x, n_z=n_z, expert_id=i)
            for i in range(n_experts)
        ])
        
        # Neural blending network
        self.blending = NeuralBlendingNetwork(n_x=n_x, n_experts=n_experts)
    
    def forward(self, x):
        """
        Forward pass: reconstruction
        
        Args:
            x: input state (batch, n_x)
        
        Returns:
            dict with:
                - x_rec: blended reconstruction
                - weights: gating weights
                - expert_recs: individual expert reconstructions
                - expert_latents: individual expert latent states
        """
        # Step 1: Compute gating weights
        weights = self.gating(x)
        
        # Step 2: Each expert encodes and reconstructs
        expert_outputs = []
        expert_recs = []
        expert_latents = []
        
        for expert in self.experts:
            output = expert(x)
            expert_outputs.append(output)
            expert_recs.append(output['x_rec'])
            expert_latents.append(output['z'])
        
        # Step 3: Neural blending of reconstructions
        x_rec = self.blending(expert_recs, weights)
        
        return {
            'x_rec': x_rec,
            'weights': weights,
            'expert_recs': expert_recs,
            'expert_latents': expert_latents
        }
    
    def predict_next(self, x):
        """
        One-step prediction: x_t → x_{t+1}
        
        Args:
            x: current state (batch, n_x)
        
        Returns:
            x_next: predicted next state (batch, n_x)
            weights: gating weights used (batch, n_experts)
        """
        # Step 1: Compute gating weights
        weights = self.gating(x)
        
        # Step 2: Each expert predicts
        expert_preds = []
        for expert in self.experts:
            x_next_i = expert.predict_next(x)
            expert_preds.append(x_next_i)
        
        # Step 3: Neural blending of predictions
        x_next = self.blending(expert_preds, weights)
        
        return x_next, weights
    
    def predict(self, x0, n_steps=100):
        """
        Multi-step prediction with dynamic gating
        
        Args:
            x0: initial state (batch, n_x)
            n_steps: number of steps to predict
        
        Returns:
            predictions: tensor (n_steps+1, batch, n_x)
            weights_history: tensor (n_steps, batch, n_experts)
        """
        self.eval()
        
        predictions = [x0]
        weights_history = []
        
        x = x0
        
        with torch.no_grad():
            for step in range(n_steps):
                # Dynamic gating: recompute at each step
                x_next, weights = self.predict_next(x)
                
                predictions.append(x_next)
                weights_history.append(weights)
                x = x_next
        
        predictions = torch.stack(predictions)
        weights_history = torch.stack(weights_history)
        
        return predictions, weights_history


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
        z_seq: tensor of shape (B, T, n_z)
        L: window length for Hankel stacking
    
    Returns:
        Hankel matrix of shape (B, cols, L*n_z)
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


