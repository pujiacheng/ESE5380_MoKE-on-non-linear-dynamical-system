"""
Mixture of Experts (MoE) Koopman Neural Network

Architecture:
- N KoopmanAE experts (each with Encoder + Koopman A_f/A_b + Decoder + BatchNorm)
- Input-based soft gating (recomputed at each step)
- Neural network blending function (learned combination in output space)
- Hankel-based linearity constraint (inherited from base KoopmanAE)
- Sparsity regularization (inherited from base KoopmanAE)

Each expert is a full KoopmanAE that learns its own coordinate system and dynamics.
Experts specialize through load balancing loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import base KoopmanAE as building block
from koopman_mixture_neural_network import (
    KoopmanAE,
    spectral_radius_penalty,
    hankel_stack_batch,
    compute_hankel_svd,
    compute_hankel_linearity_loss
)


class GatingNetwork(nn.Module):
    """
    Gating network: decides which experts to activate based on input state
    
    Args:
        n_in: dimension of input state
        n_experts: number of experts
        gating_type: 'soft' (default), 'hard', or 'topk'
        k: number of experts for top-k gating
    """
    def __init__(self, n_in=2, n_experts=8, gating_type='soft', k=2):
        super().__init__()
        self.n_experts = n_experts
        self.gating_type = gating_type
        self.k = k  # For top-k gating
        
        self.net = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, n_experts)
        )
    
    def forward(self, x):
        """
        Compute gating weights for each expert
        
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
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, n_x)  # Output final state
        )
    
    def forward(self, expert_predictions, gating_weights):
        """
        Blend expert predictions using learned non-linear combination
        
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


class KoopmanMoE(nn.Module):
    """
    Mixture of Experts Koopman Autoencoder
    
    Architecture:
    - N KoopmanAE experts (full encoder + Koopman + decoder with BatchNorm)
    - Input-based gating (recomputed dynamically)
    - Neural network blending in output space
    - Hankel-based linearity constraint
    - Sparsity regularization
    
    Args:
        n_x: dimension of state space
        n_z: dimension of latent space (same for all experts)
        n_experts: number of experts
        hidden: hidden layer size for encoder/decoder
    """
    def __init__(self, n_x=2, n_z=6, n_experts=8, hidden=128):
        super().__init__()
        self.n_x = n_x
        self.n_z = n_z
        self.n_experts = n_experts
        
        # Gating network: x → weights
        self.gating = GatingNetwork(n_in=n_x, n_experts=n_experts, gating_type='soft')
        
        # N KoopmanAE experts (each is a complete KoopmanAE from Model 4)
        self.experts = nn.ModuleList([
            KoopmanAE(n_x=n_x, n_z=n_z, hidden=hidden, expert_id=i)
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
    
    def sparsity_loss(self, mode: str = "l1"):
        """
        Compute total sparsity loss across all experts
        
        Args:
            mode: 'l1' (default) or 'l2'
        
        Returns:
            total sparsity penalty (scalar)
        """
        total_sparsity = 0
        for expert in self.experts:
            total_sparsity += expert.sparsity_loss(mode=mode)
        return total_sparsity / self.n_experts
    
    def get_expert_latent_sequences(self, x_sequence):
        """
        Get latent sequences for each expert (for Hankel loss computation)
        
        Args:
            x_sequence: tensor of shape (batch, T, n_x) - sequence of states
        
        Returns:
            list of tensors, each (batch, T, n_z) - latent sequences per expert
        """
        batch, T, n_x = x_sequence.shape
        expert_z_seqs = []
        
        for expert in self.experts:
            # Encode all time steps for this expert
            x_flat = x_sequence.reshape(-1, n_x)
            z_flat = expert.encoder(x_flat)
            z_seq = z_flat.reshape(batch, T, -1)
            expert_z_seqs.append(z_seq)
        
        return expert_z_seqs
    
    def hankel_linearity_loss(self, x_sequence, L=4, r=8, device='cpu'):
        """
        Compute Hankel-based linearity loss for all experts
        
        Args:
            x_sequence: tensor of shape (batch, T, n_x) - sequence of states
            L: Hankel window length
            r: rank for SVD truncation
            device: torch device
        
        Returns:
            average Hankel linearity loss across experts
        """
        expert_z_seqs = self.get_expert_latent_sequences(x_sequence)
        
        total_hankel_loss = 0
        for z_seq in expert_z_seqs:
            total_hankel_loss += compute_hankel_linearity_loss(z_seq, L=L, r=r, device=device)
        
        return total_hankel_loss / self.n_experts
