"""
eDMD (extended Dynamic Mode Decomposition) Baseline Model

Uses dictionary functions (observables) instead of neural network encoder.
The dictionary functions are:
    φ(x) = [x, xdot, x^2, x*xdot, xdot^2, x^3, sin(x), cos(x), sin(xdot), cos(xdot)]

The model learns a linear operator K such that:
    φ(x(t+1)) ≈ K @ φ(x(t))
"""

import torch
import torch.nn as nn
import numpy as np


class DictionaryFunctions(nn.Module):
    """
    Dictionary functions (observables) for eDMD
    
    Maps state x to observables φ(x) using polynomial and trigonometric functions
    """
    
    def __init__(self, n_x=2):
        """
        Args:
            n_x: dimension of state space (default: 2 for [x, xdot])
        """
        super().__init__()
        self.n_x = n_x
        
        # For 2D system: [x, xdot]
        # Dictionary: [x, xdot, x^2, x*xdot, xdot^2, x^3, sin(x), cos(x), sin(xdot), cos(xdot)]
        # Total: 10 observables
        self.n_obs = 10
    
    def forward(self, x):
        """
        Compute dictionary functions φ(x)
        
        Args:
            x: state tensor of shape (batch_size, n_x)
               For 2D: x[:, 0] = x, x[:, 1] = xdot
        
        Returns:
            phi: observables tensor of shape (batch_size, n_obs)
        """
        batch_size = x.shape[0]
        
        # Extract state variables
        x_var = x[:, 0]  # x
        xdot_var = x[:, 1]  # xdot
        
        # Compute dictionary functions
        phi_list = []
        
        # Linear terms
        phi_list.append(x_var)           # x
        phi_list.append(xdot_var)        # xdot
        
        # Quadratic terms
        phi_list.append(x_var ** 2)      # x^2
        phi_list.append(x_var * xdot_var)  # x*xdot
        phi_list.append(xdot_var ** 2)   # xdot^2
        
        # Cubic terms
        phi_list.append(x_var ** 3)      # x^3
        
        # Trigonometric terms
        phi_list.append(torch.sin(x_var))    # sin(x)
        phi_list.append(torch.cos(x_var))    # cos(x)
        phi_list.append(torch.sin(xdot_var)) # sin(xdot)
        phi_list.append(torch.cos(xdot_var)) # cos(xdot)
        
        # Stack into tensor
        phi = torch.stack(phi_list, dim=1)  # (batch_size, n_obs)
        
        return phi


class EDMDModel(nn.Module):
    """
    eDMD Model with dictionary functions
    
    Learns linear operator K: φ(x(t+1)) ≈ K @ φ(x(t))
    """
    
    def __init__(self, n_x=2, n_obs=10):
        """
        Args:
            n_x: dimension of state space
            n_obs: dimension of observables space (default: 10 for dictionary)
        """
        super().__init__()
        self.n_x = n_x
        self.n_obs = n_obs
        
        # Dictionary functions
        self.dictionary = DictionaryFunctions(n_x=n_x)
        
        # Linear operator K: φ(x(t+1)) = K @ φ(x(t))
        # Initialize near identity
        self.K = nn.Parameter(torch.eye(n_obs) + 0.01*torch.randn(n_obs, n_obs))
    
    def forward(self, x):
        """
        Forward pass: compute observables
        
        Args:
            x: state tensor of shape (batch_size, n_x)
        
        Returns:
            phi: observables tensor of shape (batch_size, n_obs)
        """
        return self.dictionary(x)
    
    def predict_next_observables(self, x):
        """
        Predict next observables using linear operator
        
        Args:
            x: current state tensor of shape (batch_size, n_x)
        
        Returns:
            phi_next: predicted next observables of shape (batch_size, n_obs)
        """
        phi = self.dictionary(x)
        phi_next = phi @ self.K.T
        return phi_next
    
    def predict_next_state(self, x):
        """
        Predict next state by predicting observables and extracting first n_x components
        
        Note: This is an approximation - we only use the first n_x observables (x, xdot)
        to reconstruct the state. For a full reconstruction, we'd need an inverse map.
        
        Args:
            x: current state tensor of shape (batch_size, n_x)
        
        Returns:
            x_next: predicted next state of shape (batch_size, n_x)
        """
        phi_next = self.predict_next_observables(x)
        # Extract first n_x observables (x, xdot) as predicted state
        x_next = phi_next[:, :self.n_x]
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
        current = x0
        
        for _ in range(n_steps):
            current = self.predict_next_state(current)
            sequence.append(current)
        
        return torch.stack(sequence, dim=1)


def compute_koopman_operator_analytical(phi_t, phi_t1, reg=1e-6):
    """
    Compute Koopman operator K analytically using least squares
    
    Solves: phi_t1 = phi_t @ K.T
    i.e., K = (phi_t1.T @ phi_t) @ (phi_t.T @ phi_t + reg*I)^(-1)
    
    Args:
        phi_t: observables at time t, shape (n_samples, n_obs)
        phi_t1: observables at time t+1, shape (n_samples, n_obs)
        reg: regularization parameter
    
    Returns:
        K: Koopman operator, shape (n_obs, n_obs)
    """
    # Convert to numpy if tensors
    if isinstance(phi_t, torch.Tensor):
        phi_t = phi_t.detach().cpu().numpy()
    if isinstance(phi_t1, torch.Tensor):
        phi_t1 = phi_t1.detach().cpu().numpy()
    
    # Compute K using least squares with regularization
    G = phi_t.T @ phi_t + reg * np.eye(phi_t.shape[1])
    A = phi_t1.T @ phi_t
    K = A @ np.linalg.inv(G)
    
    return torch.tensor(K, dtype=torch.float32)

