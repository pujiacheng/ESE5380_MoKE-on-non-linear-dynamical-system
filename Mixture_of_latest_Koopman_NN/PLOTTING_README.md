# Plotting and Visualization Guide

This directory contains scripts for generating publication-quality plots and tables from model comparison results.

## Scripts

### 1. `plot_comparison_results.py`

Main plotting script that generates comprehensive visualizations from `comparison_results.csv` files.

#### Usage

```bash
# Generate plots for all systems
python plot_comparison_results.py --results_dir final_results_0 --output_dir plots

# Generate plots for specific systems
python plot_comparison_results.py --results_dir final_results_0 --output_dir plots --systems duffing lorenz

# Also generate combined multi-system plots
python plot_comparison_results.py --results_dir final_results_0 --output_dir plots --combined
```

#### Generated Plots (per system)

1. **`{system}_metrics_vs_horizon.png`**
   - 3-panel figure showing:
     - NRMSE vs Horizon (log-log scale)
     - Chamfer Distance vs Horizon (log-log scale)
     - Divergence Rate vs Horizon (log scale)
   - Vertical dashed line at 100 steps separates training (≤100) from extrapolation (>100)
   - All models shown with distinct colors and markers

2. **`{system}_nrmse_comparison.png`**
   - Bar charts comparing NRMSE at key horizons: 100, 500, 1000 steps
   - Shows training performance (100) vs extrapolation (500, 1000)

3. **`{system}_per_dimension_nrmse.png`** (for multi-dimensional systems)
   - Per-dimension NRMSE breakdown at horizons 100, 500, 1000
   - Useful for understanding which state dimensions are harder to predict

4. **`{system}_divergence_summary.png`**
   - Bar chart showing number of valid vs diverged trajectories per model
   - Helps identify models with stability issues

5. **`{system}_chamfer_comparison.png`**
   - Bar charts comparing Chamfer distance at horizons 100, 500, 1000
   - Models with diverged trajectories (inf Chamfer) are marked with ∞

6. **`{system}_short_term_metrics.png`**
   - Comparison of one-step MSE and reconstruction error
   - Shows short-term prediction and reconstruction quality

#### Combined Plots (if `--combined` flag used)

- **`combined/combined_nrmse_{horizon}step.png`**
  - 2×2 grid comparing NRMSE across all 4 systems at specific horizons
  - Useful for cross-system comparison

### 2. `generate_summary_tables.py`

Generates LaTeX and CSV summary tables for paper inclusion.

#### Usage

```bash
# Generate all tables
python generate_summary_tables.py --results_dir final_results_0 --output_dir tables

# Generate tables for specific systems
python generate_summary_tables.py --results_dir final_results_0 --output_dir tables --systems duffing lorenz
```

#### Generated Tables

1. **`summary_table.csv` / `summary_table.tex`**
   - Cross-system summary with key metrics:
     - Model name, parameter count
     - 1-step MSE
     - NRMSE at 100, 500, 1000 steps
     - Chamfer distance at 100, 1000 steps
     - Number of diverged trajectories
     - Spectral radius (if available)

2. **`{system}_detailed_table.csv` / `{system}_detailed_table.tex`**
   - Detailed per-system tables with all available metrics
   - Includes all horizons and per-dimension breakdowns

## Understanding the Metrics

### NRMSE (Normalized Root Mean Squared Error)
- **Cumulative metric**: Measures average error from step 1 up to horizon T
- **Per-dimension normalization**: Each dimension normalized by its own standard deviation
- **Lower is better**: Values closer to 0 indicate better predictions
- **Training vs Extrapolation**: 
  - ≤100 steps: Training region (models trained on these horizons)
  - >100 steps: Extrapolation region (models tested beyond training)

### Chamfer Distance
- **Phase space fidelity**: Measures how well predicted trajectories match true trajectories in full state space
- **Lower is better**: Smaller values indicate better phase space reconstruction
- **Inf values**: Indicate model divergence (predictions contain NaN/Inf)

### Divergence Rate
- **Stability metric**: Exponential growth rate of prediction error
- **Negative/zero is better**: Indicates stable or decaying errors
- **Positive values**: Indicate exponentially growing errors (unstable)

### One-Step MSE
- **Short-term accuracy**: Error in predicting one step ahead
- **Lower is better**: Measures immediate prediction quality

### Reconstruction Error
- **Encoder-decoder quality**: Error in reconstructing input from latent representation
- **Lower is better**: Measures autoencoder reconstruction fidelity

## Model Abbreviations

- **VAR**: VAR (ARIMA) - Vector Autoregression baseline
- **eDMD**: Extended Dynamic Mode Decomposition baseline
- **KAE-B**: Koopman Autoencoder Baseline
- **KAE-A**: Advanced Koopman Autoencoder (1 Expert)
- **MoE-2/3/4**: Mixture of Experts with 2/3/4 experts

## Edge Cases Handled

1. **Infinity values**: Models that diverge show `inf` for Chamfer distance, marked with ∞ in plots
2. **NaN values**: Missing data (e.g., horizons beyond trajectory length) shown as `---` in tables
3. **Diverged trajectories**: Tracked separately and shown in divergence summary plots
4. **Different state dimensions**: Per-dimension plots only generated for systems with >2 dimensions

## Plot Customization

The plotting scripts use a consistent color scheme and style:
- Each model has a unique color and marker
- Log scales used for metrics that span multiple orders of magnitude
- Training/extrapolation boundary clearly marked
- Publication-quality settings (300 DPI, tight layout)

To customize:
- Edit `MODEL_STYLE` dictionary in `plot_comparison_results.py` for colors/markers
- Modify `HORIZONS` list to change which horizons are plotted
- Adjust `TRAINING_MAX_HORIZON` to change training/extrapolation boundary

## Example Workflow

```bash
# 1. Generate all plots
python plot_comparison_results.py --results_dir final_results_0 --output_dir plots --combined

# 2. Generate summary tables
python generate_summary_tables.py --results_dir final_results_0 --output_dir tables

# 3. Review plots in plots/ directory
# 4. Use tables/*.tex files in LaTeX paper
```

## File Structure

```
Mixture_of_latest_Koopman_NN/
├── plot_comparison_results.py      # Main plotting script
├── generate_summary_tables.py      # Table generation script
├── final_results_0/                # Results directory
│   ├── duffing/
│   │   └── duffing_YYYYMMDD_HHMMSS/
│   │       └── comparison_results.csv
│   ├── vanderpol/
│   ├── lorenz/
│   └── double_pendulum/
├── plots/                          # Generated plots (created by script)
│   ├── duffing_*.png
│   ├── vanderpol_*.png
│   ├── lorenz_*.png
│   ├── double_pendulum_*.png
│   └── combined/
└── tables/                         # Generated tables (created by script)
    ├── summary_table.csv
    ├── summary_table.tex
    └── {system}_detailed_table.{csv,tex}
```

## Notes

- All plots use log scales where appropriate for better visualization
- Training region (≤100 steps) vs extrapolation region (>100 steps) is clearly marked
- Models with diverged trajectories are handled gracefully (marked with ∞ or special indicators)
- Per-dimension breakdowns help identify which state variables are harder to predict
- Tables are formatted for easy inclusion in LaTeX papers

