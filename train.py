"""
Main training script for Koopman Autoencoder on Duffing oscillator data

This script:
1. Generates Duffing oscillator trajectories
2. Prepares data for training
3. Trains the Koopman Autoencoder model
4. Evaluates and visualizes predictions
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from data_simulation import generate_duffing_dataset
from koopman_mixture_neural_network import (
    KoopmanAE, 
    spectral_radius_penalty, 
    hankel_stack_batch, 
    compute_hankel_svd
)


def prepare_data_from_trajectories(trajs, device='cpu'):
    """
    Convert trajectory data to training triplets (x_t, x_{t+1}, x_{t+2})
    
    Args:
        trajs: array of shape (n_traj, n_steps, n_x)
        device: device to place tensors on
    
    Returns:
        DataLoader with triplets
    """
    n_traj, n_steps, n_x = trajs.shape
    
    # Flatten trajectories and create triplets
    all_xt = []
    all_xt1 = []
    all_xt2 = []
    
    for traj in trajs:
        # For each trajectory, create triplets
        if n_steps >= 3:
            xt = traj[:-2]
            xt1 = traj[1:-1]
            xt2 = traj[2:]
            all_xt.append(xt)
            all_xt1.append(xt1)
            all_xt2.append(xt2)
    
    # Concatenate all trajectories
    xt = np.concatenate(all_xt, axis=0)
    xt1 = np.concatenate(all_xt1, axis=0)
    xt2 = np.concatenate(all_xt2, axis=0)
    
    # Convert to tensors
    xt_t = torch.tensor(xt, dtype=torch.float32)
    xt1_t = torch.tensor(xt1, dtype=torch.float32)
    xt2_t = torch.tensor(xt2, dtype=torch.float32)
    
    return xt_t, xt1_t, xt2_t


def sample_sequence_batch(all_X, batch_size, Tseq=8, device='cpu'):
    """
    Sample random contiguous sequences from the dataset
    
    Args:
        all_X: tensor of all states, shape (N, n_x)
        batch_size: number of sequences to sample
        Tseq: length of each sequence
        device: device to place tensors on
    
    Returns:
        tensor of shape (batch_size, Tseq, n_x)
    """
    max_start = all_X.shape[0] - Tseq
    starts = np.random.randint(0, max_start, size=batch_size)
    seqs = [all_X[s:s+Tseq] for s in starts]
    return torch.stack(seqs, dim=0).to(device)


def train_model(model, train_loader, all_X, device, n_epochs=40, 
                batch_size=256, hankel_batch_size=64, hankel_Tseq=8, L=4):
    """
    Train the Koopman Autoencoder model
    
    Args:
        model: KoopmanAE model instance
        train_loader: DataLoader with training triplets
        all_X: all state data for Hankel sampling
        device: device to train on
        n_epochs: number of training epochs
        batch_size: batch size for main training
        hankel_batch_size: batch size for Hankel computation
        hankel_Tseq: sequence length for Hankel computation
        L: Hankel window length
    """
    # Hyperparameters for loss weights
    lam_rec, lam_lin, lam_ms = 1.0, 10.0, 2.0
    lam_edmd, lam_hankel = 1.0, 1.0
    lam_bi, lam_spec, lam_sparse = 1.0, 1.0, 1e-4
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    
    log = []
    n_x = model.encoder.net[0].in_features
    n_z = model.encoder.net[-1].out_features
    
    for ep in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        
        # Iterate over dataloader for 1-step and 2-step losses
        for x0, x1, x2 in train_loader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            
            out0 = model(x0)
            out1 = model(x1)
            out2 = model(x2)
            
            z0 = out0['z']
            z1 = out1['z']
            z2 = out2['z']
            xrec0 = out0['x_rec']
            
            # Reconstruction loss
            loss_rec = mse(xrec0, x0)
            
            # Latent linearity (1-step)
            z_pred = z0 @ model.A_f.T
            loss_lin = mse(z_pred, z1)
            
            # Multi-step (2-step)
            z_pred2 = z_pred @ model.A_f.T
            loss_ms = mse(z_pred2, z2)
            
            # eDMD observables regression (learnable A_g)
            g0 = out0['g']
            g1 = out1['g']
            g_pred = g0 @ model.A_g.T
            loss_edmd = mse(g_pred, g1)
            
            # Bidirectional constraint
            Id = torch.eye(model.A_f.shape[0], device=device)
            loss_bi = (model.A_f @ model.A_b - Id).norm()**2 + (model.A_b @ model.A_f - Id).norm()**2
            
            # Hankel term: sample sequences, encode, build Hankel, SVD to get low-rank coords
            seqs = sample_sequence_batch(all_X, batch_size=hankel_batch_size, 
                                        Tseq=hankel_Tseq, device=device)
            
            # Encode sequence to z_seq
            with torch.no_grad():
                # Don't update encoder while computing SVD basis (alternating style)
                z_seq = model.encoder(seqs.reshape(-1, n_x)).reshape(
                    seqs.shape[0], seqs.shape[1], n_z
                )  # (B, Tseq, n_z)
            
            H = hankel_stack_batch(z_seq, L=L)  # (B, cols, L*n_z)
            U, S, Vt = compute_hankel_svd(H)  # CPU SVD
            
            # Take top r components (low-rank)
            r = min(8, Vt.shape[0])
            V_r = Vt[:r].T  # shape (L*n_z, r)
            
            # Project Hankel columns to v coords
            Hmat = H.reshape(-1, H.shape[-1]).cpu().numpy()  # (B*cols, d)
            Vr = V_r  # (d, r)
            Vcoords = Hmat @ Vr  # (B*cols, r)
            Vcoords = torch.tensor(Vcoords, dtype=torch.float32, device=device).reshape(
                H.shape[0], H.shape[1], r
            )  # (B, cols, r)
            
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
            
            # Enforce v_tp1 ≈ A_v v_t
            v_pred = v_t @ A_v.T
            loss_hankel = mse(v_pred, v_tp1)
            
            # Spectral penalty (approx)
            loss_spec = spectral_radius_penalty(model.A_f, iters=8, target=1.1)
            
            # Sparsity on observables net weights
            sparsity_term = 0.0
            for name, param in model.obs.named_parameters():
                sparsity_term += param.norm(1)
            
            # Total loss
            loss = (lam_rec * loss_rec + lam_lin * loss_lin + lam_ms * loss_ms +
                    lam_edmd * loss_edmd + lam_hankel * loss_hankel + lam_bi * loss_bi +
                    lam_spec * loss_spec + lam_sparse * sparsity_term)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        log.append(avg_loss)
        
        if ep % 5 == 0:
            print(f"Epoch {ep:03d} | loss {avg_loss:.6f} | rec {loss_rec.item():.6f} | "
                  f"lin {loss_lin.item():.6f} | edmd {loss_edmd.item():.6f} | "
                  f"hankel {loss_hankel.item():.6f} | bi {loss_bi.item():.6f}")
    
    return log


def evaluate_model(model, test_traj, device, n_steps=100):
    """
    Evaluate model by predicting forward in time
    
    Args:
        model: trained KoopmanAE model
        test_traj: test trajectory array of shape (n_steps, n_x)
        device: device to run on
        n_steps: number of steps to predict
    
    Returns:
        true_traj: true trajectory
        pred_traj: predicted trajectory
    """
    model.eval()
    with torch.no_grad():
        # Take initial condition from test trajectory
        x0 = torch.tensor(test_traj[0], dtype=torch.float32).unsqueeze(0).to(device)
        z0 = model.encoder(x0)
        
        zs = [z0]
        z = z0
        for i in range(n_steps):
            z = z @ model.A_f.T
            zs.append(z)
        
        zs = torch.cat(zs, dim=0)  # (n_steps+1, n_z)
        preds = model.decoder(zs).cpu().numpy()  # (n_steps+1, 2)
        
        true = test_traj[:n_steps+1]
        
    return true, preds


def main():
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Generate Duffing oscillator data
    print("\nGenerating Duffing oscillator trajectories...")
    n_traj = 100
    T = 10.0
    dt = 0.01
    noise_std = 0.0
    
    t, trajs = generate_duffing_dataset(n_traj=n_traj, T=T, dt=dt, noise_std=noise_std)
    print(f"Generated {n_traj} trajectories, each with {trajs.shape[1]} time steps")
    
    # Prepare training data
    print("\nPreparing training data...")
    xt_t, xt1_t, xt2_t = prepare_data_from_trajectories(trajs, device=device)
    
    # Flatten all trajectories for Hankel sampling
    all_X = torch.tensor(trajs.reshape(-1, trajs.shape[-1]), dtype=torch.float32)
    
    batch_size = 256
    train_loader = DataLoader(
        TensorDataset(xt_t, xt1_t, xt2_t), 
        batch_size=batch_size, 
        shuffle=True
    )
    print(f"Training batches: {len(train_loader)}")
    
    # Create model
    n_x = 2  # Duffing oscillator state dimension
    n_z = 6  # Latent dimension
    p = 20   # Observables dimension
    
    model = KoopmanAE(n_x=n_x, n_z=n_z, p=p).to(device)
    print(f"\nModel created: n_x={n_x}, n_z={n_z}, p={p}")
    
    # Train model
    print("\nStarting training...")
    log = train_model(
        model=model,
        train_loader=train_loader,
        all_X=all_X,
        device=device,
        n_epochs=40,
        batch_size=batch_size,
        hankel_batch_size=64,
        hankel_Tseq=8,
        L=4
    )
    
    # Evaluate on a test trajectory
    print("\nEvaluating model...")
    test_traj = trajs[0]  # Use first trajectory as test
    true, preds = evaluate_model(model, test_traj, device, n_steps=100)
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Phase space plot
    plt.subplot(1, 2, 1)
    plt.plot(true[:, 0], true[:, 1], '-o', label='True', markersize=3, alpha=0.7)
    plt.plot(preds[:, 0], preds[:, 1], '-x', label='Koopman Pred', markersize=3, alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('xdot')
    plt.title('Phase Space: True vs Koopman-predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Time series plot
    plt.subplot(1, 2, 2)
    time_axis = np.arange(len(true)) * dt
    plt.plot(time_axis, true[:, 0], '-o', label='True x', markersize=2, alpha=0.7)
    plt.plot(time_axis, preds[:, 0], '-x', label='Pred x', markersize=2, alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('x')
    plt.title('Time Series: Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('koopman_prediction_results.png', dpi=150)
    print("Results saved to 'koopman_prediction_results.png'")
    plt.show()
    
    # Save model
    torch.save(model.state_dict(), 'koopman_model.pth')
    print("Model saved to 'koopman_model.pth'")


if __name__ == "__main__":
    main()

