#!/bin/bash
# Train MoE Koopman on all 4 dynamical systems
# Uses default architecture (8 experts, 3x latent dimension)
# No system-specific assumptions - architecture is fully data-driven

echo "=================================================================================================="
echo "                           TRAINING MoE KOOPMAN ON ALL 4 SYSTEMS"
echo "=================================================================================================="
echo "  Architecture      : 4 experts (system-agnostic)"
echo "  Latent dimension  : 10× state dimension"
echo "  Max epochs        : 100"
echo "  Early stopping    : Multi-criteria (patience = 20 epochs)"
echo "    - Metric        : 0.7 × Val_MS(8) + 0.3 × Val_Linearity"
echo "    - Balances      : Prediction accuracy + Koopman linearity"
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
    --n_traj 100 \
    --T 10.0 \
    --dt 0.01 \
    --n_epochs 100 \
    --batch_size 256 \
    --early_stopping \
    --patience 20 \
    --save_prefix "results_moe_comparison/"

# System 2: Van der Pol Oscillator
echo ""
echo ""
echo "=================================================================================================="
echo "                            [2/4] VAN DER POL OSCILLATOR (2D)"
echo "=================================================================================================="
python3 train_moe.py \
    --system vanderpol \
    --n_traj 100 \
    --T 20.0 \
    --dt 0.01 \
    --n_epochs 100 \
    --batch_size 256 \
    --early_stopping \
    --patience 20 \
    --save_prefix "results_moe_comparison/"

# System 3: Lorenz Attractor
echo ""
echo ""
echo "=================================================================================================="
echo "                                [3/4] LORENZ ATTRACTOR (3D)"
echo "=================================================================================================="
python3 train_moe.py \
    --system lorenz \
    --n_traj 100 \
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
    --n_traj 100 \
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
echo "    • {system}_moe_results.png    - Phase space, time series, loss curves, expert usage"
echo "    • {system}_expert_usage.png   - Expert activation patterns over time"
echo "    • {system}_moe_model.pth      - Trained model weights"
echo ""
echo "  To generate comparison report:"
echo "    python3 compare_all_systems.py --results_dir results_moe_comparison"
echo ""
echo "=================================================================================================="

