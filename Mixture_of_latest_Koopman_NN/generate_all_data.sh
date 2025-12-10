#!/bin/bash
# Generate data for all dynamical systems
# Run this ONCE before training to pre-generate all datasets
#
# Usage: ./generate_all_data.sh

set -e  # Exit on error

# Configuration - should match train_all_systems.sh
N_TRAJ=100000  # 100K trajectories for diverse IC coverage
T=10
DT=0.01
NOISE_STD=0.0
OUTPUT_DIR="generated_data"
SEED=42

echo "=============================================="
echo "Generating datasets for all systems"
echo "=============================================="
echo ""
echo "Parameters:"
echo "  Trajectories: $N_TRAJ"
echo "  T: $T"
echo "  dt: $DT"
echo "  Noise std: $NOISE_STD"
echo "  Seed: $SEED"
echo "  Output: $OUTPUT_DIR/"
echo ""

python generate_data.py \
    --n_traj $N_TRAJ \
    --T $T \
    --dt $DT \
    --noise_std $NOISE_STD \
    --output_dir $OUTPUT_DIR \
    --seed $SEED

echo ""
echo "=============================================="
echo "Data generation complete!"
echo "=============================================="
echo ""
echo "Now run training with:"
echo "  ./train_all_systems.sh"

