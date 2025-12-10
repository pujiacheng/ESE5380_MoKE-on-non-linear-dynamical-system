"""
eDMD (extended Dynamic Mode Decomposition) Baseline Model

Uses dictionary functions (observables) instead of neural network encoder.
Following standard EDMD practice (Williams et al. 2015, Li et al. 2017), 
the dictionary consists of:
    - All monomials up to degree d (complete polynomial basis)
    - Trigonometric functions: sin(x_i), cos(x_i) for all state variables

For n_x variables and max degree d, the number of monomials is C(n_x + d, d).

Example for n_x=2, d=3:
    Degree 0: 1
    Degree 1: x, y  
    Degree 2: x², xy, y²
    Degree 3: x³, x²y, xy², y³
    Total: 10 monomials

The model learns a linear operator K such that:
    φ(x(t+1)) ≈ K @ φ(x(t))

References:
    - Williams, Kevrekidis, Rowley (2015) "A data-driven approximation of the Koopman operator"
    - Li, Dietrich, Bollt, Kevrekidis (2017) "Extended DMD with dictionary learning"
"""

import torch
import torch.nn as nn
import numpy as np
from itertools import combinations_with_replacement


def count_monomials(n_vars, max_degree):
    """
    Count total number of monomials up to max_degree in n_vars variables.
    This is C(n_vars + max_degree, max_degree) = (n_vars + max_degree)! / (n_vars! * max_degree!)
    """
    from math import comb
    return comb(n_vars + max_degree, max_degree)


def generate_monomial_powers(n_vars, max_degree):
    """
    Generate all monomial power tuples up to max_degree.
    
    For n_vars=2, max_degree=3, returns:
        [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3)]
    
    Each tuple (p1, p2, ..., pn) represents x1^p1 * x2^p2 * ... * xn^pn
    """
    powers_list = []
    
    for degree in range(max_degree + 1):
        # Generate all ways to distribute 'degree' among n_vars variables
        # This is equivalent to combinations with replacement
        for combo in combinations_with_replacement(range(n_vars), degree):
            # Count occurrences of each variable index
            powers = [0] * n_vars
            for var_idx in combo:
                powers[var_idx] += 1
            powers_list.append(tuple(powers))
    
    return powers_list


class DictionaryFunctions(nn.Module):
    """
    Dictionary functions (observables) for eDMD
    
    Generates a complete polynomial basis up to specified degree,
    plus trigonometric terms. This follows standard EDMD practice.
    """
    
    def __init__(self, n_x=2, max_poly_degree=5, include_trig=True):
        """
        Args:
            n_x: dimension of state space
            max_poly_degree: maximum polynomial degree (default: 5)
            include_trig: whether to include sin/cos terms (default: True)
        """
        super().__init__()
        self.n_x = n_x
        self.max_poly_degree = max_poly_degree
        self.include_trig = include_trig
        
        # Generate all monomial powers up to max_degree
        self.monomial_powers = generate_monomial_powers(n_x, max_poly_degree)
        n_monomials = len(self.monomial_powers)
        
        # Trigonometric terms: sin(x_i), cos(x_i) for each variable
        n_trig = 2 * n_x if include_trig else 0
        
        self.n_monomials = n_monomials
        self.n_trig = n_trig
        self.n_obs = n_monomials + n_trig
        
        # Store info for printing
        self._degree_counts = {}
        for powers in self.monomial_powers:
            d = sum(powers)
            self._degree_counts[d] = self._degree_counts.get(d, 0) + 1
    
    def forward(self, x):
        """
        Compute dictionary functions φ(x)
        
        Args:
            x: state tensor of shape (batch_size, n_x)
        
        Returns:
            phi: observables tensor of shape (batch_size, n_obs)
        """
        batch_size = x.shape[0]
        phi_list = []
        
        # 1. Monomial terms: x1^p1 * x2^p2 * ... * xn^pn for all power tuples
        for powers in self.monomial_powers:
            # Compute product: x[:, 0]^powers[0] * x[:, 1]^powers[1] * ...
            monomial = torch.ones(batch_size, device=x.device, dtype=x.dtype)
            for var_idx, power in enumerate(powers):
                if power > 0:
                    monomial = monomial * (x[:, var_idx] ** power)
            phi_list.append(monomial)
        
        # 2. Trigonometric terms (if enabled)
        if self.include_trig:
            for i in range(self.n_x):
                phi_list.append(torch.sin(x[:, i]))
            for i in range(self.n_x):
                phi_list.append(torch.cos(x[:, i]))
        
        # Stack into tensor
        phi = torch.stack(phi_list, dim=1)  # (batch_size, n_obs)
        
        return phi
    
    def describe(self):
        """Return a description of the dictionary"""
        desc = f"Dictionary for n_x={self.n_x}, max_degree={self.max_poly_degree}\n"
        desc += f"Monomials by degree:\n"
        for d in sorted(self._degree_counts.keys()):
            desc += f"  Degree {d}: {self._degree_counts[d]} terms\n"
        desc += f"Total monomials: {self.n_monomials}\n"
        if self.include_trig:
            desc += f"Trigonometric: {self.n_trig} terms (sin/cos for each variable)\n"
        desc += f"Total observables: {self.n_obs}"
        return desc


class EDMDModel(nn.Module):
    """
    eDMD Model with dictionary functions
    
    Learns linear operator K: φ(x(t+1)) ≈ K @ φ(x(t))
    
    The dictionary uses a complete polynomial basis up to specified degree.
    """
    
    def __init__(self, n_x=2, n_obs=None, max_poly_degree=5, include_trig=True):
        """
        Args:
            n_x: dimension of state space
            n_obs: dimension of observables space (auto-computed if None)
            max_poly_degree: maximum polynomial degree for dictionary
            include_trig: whether to include trigonometric terms
        """
        super().__init__()
        self.n_x = n_x
        
        # Dictionary functions (computes n_obs automatically)
        self.dictionary = DictionaryFunctions(
            n_x=n_x, 
            max_poly_degree=max_poly_degree,
            include_trig=include_trig
        )
        
        # Use dictionary's n_obs if not specified
        if n_obs is None:
            n_obs = self.dictionary.n_obs
        self.n_obs = n_obs
        
        # Linear operator K: φ(x(t+1)) = K @ φ(x(t))
        # Initialize near identity
        self.K = nn.Parameter(torch.eye(n_obs) + 0.01 * torch.randn(n_obs, n_obs))
    
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
        Predict next state by predicting observables and extracting state components
        
        Note: The first n_x observables after the constant term (degree 0) are the 
        linear terms [x_0, x_1, ..., x_{n_x-1}]. We extract these for state prediction.
        
        Args:
            x: current state tensor of shape (batch_size, n_x)
        
        Returns:
            x_next: predicted next state of shape (batch_size, n_x)
        """
        phi_next = self.predict_next_observables(x)
        # The first observable is the constant (degree 0), 
        # then next n_x are the linear terms (degree 1)
        x_next = phi_next[:, 1:1 + self.n_x]
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


# Utility function to print dictionary info
def print_dictionary_info(n_x, max_degree=5):
    """Print information about the dictionary for given parameters"""
    dict_fn = DictionaryFunctions(n_x=n_x, max_poly_degree=max_degree, include_trig=True)
    print(dict_fn.describe())
    return dict_fn.n_obs
