#!/bin/bash
# Train MoE Koopman on all 4 dynamical systems
# With MULTI-STEP LINEARITY LOSS (1, 10, 20, ..., 100 steps)
# Enforces Koopman linearity A^k @ z_0 = z_k across multiple horizons

echo "=================================================================================================="
echo "              TRAINING MoE KOOPMAN WITH MULTI-STEP LINEARITY"
echo "=================================================================================================="
echo "  Architecture      : 4 experts (system-agnostic)"
echo "  Latent dimension  : 5× state dimension"
echo "  Max epochs        : 100"
echo "  Early stopping    : Based on TOTAL validation loss (patience = 20 epochs)"
echo "  Linearity horizons: 1, 10, 20, 30, 40, 50 steps"
echo "  Horizon weights   : 1.0 → 0.2 (decay, near-term more important)"
echo "  Validation split  : 10%"
echo "=================================================================================================="

# Create results directory
mkdir -p results_moe_comparison

# System 1: Duffing Oscillator
echo ""
echo ""
echo "=================================================================================================="
echo "                               [1/4] DUFFING OSCILLATOR (2D)"
echo "=================================================================================================="
python3 train_moe.py \
    --system duffing \
    --n_traj 10000 \
    --T 10.0 \
    --dt 0.01 \
    --n_epochs 100 \
    --batch_size 256 \
    --early_stopping \
    --patience 5 \
    --save_prefix "results_moe_comparison2/"

# System 2: Van der Pol Oscillator
echo ""
echo ""
echo "=================================================================================================="
echo "                            [2/4] VAN DER POL OSCILLATOR (2D)"
echo "=================================================================================================="
python3 train_moe.py \
    --system vanderpol \
    --n_traj 10000 \
    --T 20.0 \
    --dt 0.01 \
    --n_epochs 100 \
    --batch_size 256 \
    --early_stopping \
    --patience 5 \
    --save_prefix "results_moe_comparison2/"

# System 3: Lorenz Attractor
echo ""
echo ""
echo "=================================================================================================="
echo "                                [3/4] LORENZ ATTRACTOR (3D)"
echo "=================================================================================================="
python3 train_moe.py \
    --system lorenz \
    --n_traj 10000 \
    --T 20.0 \
    --dt 0.01 \
    --n_epochs 100 \
    --batch_size 256 \
    --early_stopping \
    --patience 20 \
    --save_prefix "results_moe_comparison/"

# System 4: Double Pendulum
echo ""
echo ""
echo "=================================================================================================="
echo "                                [4/4] DOUBLE PENDULUM (4D)"
echo "=================================================================================================="
python3 train_moe.py \
    --system double_pendulum \
    --n_traj 10000 \
    --T 10.0 \
    --dt 0.01 \
    --n_epochs 100 \
    --batch_size 256 \
    --early_stopping \
    --patience 20 \
    --save_prefix "results_moe_comparison/"

echo ""
echo ""
echo "=================================================================================================="
echo "                                    TRAINING COMPLETE!"
echo "=================================================================================================="
echo ""
echo "  Results directory: results_moe_comparison/"
echo ""
echo "  Generated files for each system:"
echo "    • {system}_moe_results.png    - Summary (overlaid trajectories, loss, expert usage)"
echo "    • {system}_{state}_grid.png   - Grid of 10 ICs for each state variable (predictions)"
echo "    • {system}_expert_usage.png   - Expert activation patterns over time"
echo "    • {system}_moe_model.pth      - Trained model weights"
echo ""
echo "=================================================================================================="

