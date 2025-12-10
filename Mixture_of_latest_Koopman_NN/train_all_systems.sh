#!/bin/bash
# Train all models on all dynamical systems
# Usage: ./train_all_systems.sh
#
# Output structure:
#   comparison_results/
#   ├── duffing/
#   │   ├── var/
#   │   ├── edmd/
#   │   ├── kae_baseline/
#   │   ├── advanced_kae/
#   │   ├── moe_2expert/
#   │   ├── moe_3expert/
#   │   ├── moe_4expert/
#   │   ├── comparison_results.csv
#   │   ├── config.txt
#   │   └── data_split.npz
#   ├── vanderpol/
#   │   └── ...
#   ├── lorenz/
#   │   └── ...
#   └── double_pendulum/
#       └── ...

set -e  # Exit on error

# =============================================================================
# Configuration - All Arguments
# =============================================================================

# Data generation (must match generate_all_data.sh)
T=10                    # Simulation time per trajectory (10 sec = 1001 timesteps)
DT=0.01                 # Time step for simulation
N_TRAJ=100000           # Number of trajectories (100K for diverse IC coverage)

# Training
N_EPOCHS=150            # Maximum training epochs
PATIENCE=20            # Early stopping patience
BATCH_SIZE=2048         # Training batch size (GH200 98GB = max batch size = fastest)

# Data split (must sum to 1.0)
TRAIN_SPLIT=0.8         # Fraction for training (80%)
VAL_SPLIT=0.1           # Fraction for validation (10%)
TEST_SPLIT=0.1          # Fraction for testing (10%)

# Evaluation
MAX_TEST_TRAJ=10        # Max test trajectories for evaluation (set to "" to use all)

# Output directory structure
OUTPUT_DIR="comparison_results"
USE_TIMESTAMP=false     # Set to true for timestamped subdirectories

# Pre-generated data (set to empty to generate on-the-fly)
DATA_DIR="generated_data"  # Directory with pre-generated .npz files

# Resume options
RESUME=false            # Set to true to resume from checkpoints
RUN_DIR=""              # Specific run directory to resume (leave empty for new run)
EVAL_ONLY=false         # Set to true to skip training and only evaluate existing models

# Systems to train
# SYSTEMS=("duffing" "vanderpol" "lorenz" "double_pendulum")
SYSTEMS=("double_pendulum")

# Models to train
# Available: var, edmd, kae_baseline, advanced_kae, moe_2expert, moe_3expert, moe_4expert
# MODELS=("var" "edmd" "kae_baseline" "advanced_kae" "moe_2expert" "moe_3expert" "moe_4expert")
MODELS=("advanced_kae" "moe_2expert" "moe_3expert" "moe_4expert")
# =============================================================================
# Print Configuration
# =============================================================================

echo "=============================================="
echo "Training all models on all systems"
echo "=============================================="
echo ""
echo "Output Structure:"
echo "  $OUTPUT_DIR/"
for sys in "${SYSTEMS[@]}"; do
    echo "  ├── $sys/"
    echo "  │   ├── var/"
    echo "  │   ├── edmd/"
    echo "  │   ├── kae_baseline/"
    echo "  │   ├── advanced_kae/"
    echo "  │   ├── moe_2expert/"
    echo "  │   ├── moe_3expert/"
    echo "  │   ├── moe_4expert/"
    echo "  │   └── comparison_results.csv"
done
echo ""
echo "Data Generation:"
echo "  T (simulation time): $T"
echo "  dt (time step): $DT"
echo "  Trajectories: $N_TRAJ"
echo ""
echo "Training:"
echo "  Max epochs: $N_EPOCHS"
echo "  Early stop patience: $PATIENCE"
echo "  Batch size: $BATCH_SIZE"
echo ""
echo "Data Split:"
echo "  Train: $TRAIN_SPLIT ($(awk "BEGIN {printf \"%.0f\", $TRAIN_SPLIT * 100}")%)"
echo "  Validation: $VAL_SPLIT ($(awk "BEGIN {printf \"%.0f\", $VAL_SPLIT * 100}")%)"
echo "  Test: $TEST_SPLIT ($(awk "BEGIN {printf \"%.0f\", $TEST_SPLIT * 100}")%)"
echo ""
echo "Data:"
if [ -n "$DATA_DIR" ]; then
    echo "  Using pre-generated data from: $DATA_DIR"
else
    echo "  Generating data on-the-fly"
fi
echo ""
echo "Options:"
echo "  Use timestamp: $USE_TIMESTAMP"
echo "  Resume: $RESUME"
echo "  Eval only: $EVAL_ONLY"
if [ -n "$RUN_DIR" ]; then
    echo "  Run dir: $RUN_DIR"
fi
if [ -n "$MAX_TEST_TRAJ" ]; then
    echo "  Max test traj: $MAX_TEST_TRAJ"
else
    echo "  Max test traj: all"
fi
echo ""
echo "Systems: ${SYSTEMS[*]}"
echo "Models:  ${MODELS[*]}"
echo "=============================================="
echo ""

# =============================================================================
# Training Loop
# =============================================================================

for system in "${SYSTEMS[@]}"; do
    echo ""
    echo "======================================================"
    echo "  Starting training for: $system"
    echo "  Output: $OUTPUT_DIR/$system/"
    echo "======================================================"
    echo ""
    
    # Build command with all arguments
    CMD="python train_all_models.py \
        --system $system \
        --T $T \
        --dt $DT \
        --n_traj $N_TRAJ \
        --n_epochs $N_EPOCHS \
        --patience $PATIENCE \
        --batch_size $BATCH_SIZE \
        --train_split $TRAIN_SPLIT \
        --val_split $VAL_SPLIT \
        --test_split $TEST_SPLIT \
        --output_dir $OUTPUT_DIR"
    
    # Add data_dir if specified (use pre-generated data)
    if [ -n "$DATA_DIR" ]; then
        CMD="$CMD --data_dir $DATA_DIR"
    fi
    
    # Add timestamp flag if enabled
    if [ "$USE_TIMESTAMP" = true ]; then
        CMD="$CMD --use_timestamp"
    fi
    
    # Add resume flag if enabled
    if [ "$RESUME" = true ]; then
        CMD="$CMD --resume"
    fi
    
    # Add eval_only flag if enabled
    if [ "$EVAL_ONLY" = true ]; then
        CMD="$CMD --eval_only"
    fi
    
    # Add run_dir if specified
    if [ -n "$RUN_DIR" ]; then
        CMD="$CMD --run_dir $RUN_DIR"
    fi
    
    # Add max_test_traj if specified (for faster evaluation)
    if [ -n "$MAX_TEST_TRAJ" ]; then
        CMD="$CMD --max_test_traj $MAX_TEST_TRAJ"
    fi
    
    # Add models to train (convert array to comma-separated string)
    MODELS_STR=$(IFS=,; echo "${MODELS[*]}")
    CMD="$CMD --models $MODELS_STR"
    
    # Execute
    echo "Running: $CMD"
    echo ""
    eval $CMD
    
    echo ""
    echo "======================================================"
    echo "  Completed training for: $system"
    echo "  Results: $OUTPUT_DIR/$system/"
    echo "======================================================"
    echo ""
done

echo ""
echo "=============================================="
echo "All systems training complete!"
echo ""
echo "Results structure:"
echo "  $OUTPUT_DIR/"
for sys in "${SYSTEMS[@]}"; do
    echo "  └── $sys/comparison_results.csv"
done
echo "=============================================="
