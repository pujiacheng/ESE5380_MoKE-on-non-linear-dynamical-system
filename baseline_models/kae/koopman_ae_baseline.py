"""
Simplified Koopman Autoencoder Baseline Model

This is a simplified version of the Koopman Autoencoder that only uses:
- Encoder: maps state x to latent representation z
- Decoder: maps latent z back to state x
- Linear operator A_f: enforces linear dynamics in latent space (z(t+1) = A_f @ z(t))

No observables network, no Hankel, no bidirectional constraints, etc.
Only the core Koopman operator structure.
"""

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """Encoder network: maps state x to latent representation z"""

    def __init__(self, n_in=2, n_latent=20, hidden=128):
        super().__init__()
        # Using LayerNorm instead of BatchNorm for train/eval consistency
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_latent)
        )

    def forward(self, x):
        return self.net(x)


class MLPDecoder(nn.Module):
    """Decoder network: maps latent representation z back to state x"""

    def __init__(self, n_latent=20, n_out=2, hidden=128):
        super().__init__()
        # Using LayerNorm instead of BatchNorm for train/eval consistency
        self.net = nn.Sequential(
            nn.Linear(n_latent, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out)
        )
    
    def forward(self, z):
        return self.net(z)


class KoopmanAEBaseline(nn.Module):
    """
    Simplified Koopman Autoencoder Baseline Model
    
    Only includes:
    - Encoder: x -> z
    - Decoder: z -> x
    - Linear operator A_f: z(t+1) = A_f @ z(t)
    
    Args:
        n_x: dimension of state space
        n_z: dimension of latent space
    """
    def __init__(self, n_x=2, n_z=20):
        super().__init__()
        self.n_x = n_x
        self.n_z = n_z
        self.encoder = MLPEncoder(n_in=n_x, n_latent=n_z)
        self.decoder = MLPDecoder(n_latent=n_z, n_out=n_x)
        
        # Forward linear operator in latent space
        # Initialize near identity
        self.A_f = nn.Parameter(torch.eye(n_z) + 0.01*torch.randn(n_z, n_z))
    
    def forward(self, x):
        """
        Forward pass
        
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
        Predict next state using linear operator in latent space
        
        Args:
            x: current state tensor of shape (batch_size, n_x)
        
        Returns:
            x_next: predicted next state of shape (batch_size, n_x)
        """
        z = self.encoder(x)
        z_next = z @ self.A_f.T
        x_next = self.decoder(z_next)
        return x_next
    
    def predict_sequence(self, x0, n_steps):
        """
        Predict a sequence of states starting from initial state
        
        Args:
            x0: initial state tensor of shape (batch_size, n_x)
            n_steps: number of steps to predict
        
        Returns:
            sequence: tensor of shape (batch_size, n_steps+1, n_x)
        """
        batch_size = x0.shape[0]
        sequence = [x0]
        z = self.encoder(x0)
        
        for _ in range(n_steps):
            z = z @ self.A_f.T
            x = self.decoder(z)
            sequence.append(x)
        
        return torch.stack(sequence, dim=1)

