# Results Analysis & Visualization Plan

## Data Overview

### Systems & Dimensions
- **Duffing**: 2D (x, ẋ)
- **Van der Pol**: 2D (x, ẋ)
- **Lorenz**: 3D (x, y, z)
- **Double Pendulum**: 4D (θ₁, θ₂, ω₁, ω₂)

### Models Compared
1. VAR (ARIMA) - baseline
2. eDMD - Koopman baseline
3. KAE Baseline - simple autoencoder
4. Advanced KAE (1 Expert) - Model 4
5. MoE (2/3/4 Experts) - Model 5 variants

### Evaluation Horizons
- **Training**: 1-100 steps (dense training on all these horizons)
- **Extrapolation**: 500, 1000 steps (beyond training)
- **Reported**: 1, 10, 20, 50, 100, 500, 1000

## Key Metrics Explained

### 1. NRMSE (Normalized Root Mean Squared Error)
- **Cumulative**: Measures average error from step 1 to horizon T
- **Per-dimension normalized**: Each dimension scaled by its own std
- **Lower is better**: 0 = perfect, >1 = poor
- **Use**: Primary accuracy metric, shows degradation over time

### 2. Chamfer Distance
- **Phase space fidelity**: How well trajectories match in full state space
- **Lower is better**: Small = good reconstruction
- **inf = diverged**: Model predictions contain NaN/inf
- **Use**: Overall trajectory quality, especially for chaotic systems

### 3. Divergence Rate
- **Exponential growth**: Fitted from log(error) vs time
- **Negative/zero is good**: Stable or decaying errors
- **Positive = unstable**: Errors grow exponentially
- **Use**: Long-term stability indicator

### 4. One-Step MSE
- **Short-term accuracy**: Next-step prediction error
- **Lower is better**: Immediate prediction quality
- **Use**: Model's instantaneous prediction capability

### 5. Reconstruction Error
- **Autoencoder quality**: x → encode → decode → x
- **Lower is better**: Information preservation
- **Use**: Latent representation quality

### 6. Spectral Radius
- **Koopman operator stability**: max|eigenvalue|
- **~1.0 is ideal**: Preserves energy
- **>1 = unstable, <1 = dissipative**
- **Use**: Theoretical stability check

### 7. Diverged Trajectories
- **Counts**: n_valid vs n_diverged out of n_total
- **Critical for reliability**: Shows model robustness
- **Use**: Identify failure modes

## Critical Edge Cases Found

### 1. eDMD Divergence
- **Duffing**: 2/10 trajectories diverged (inf Chamfer at 500+ steps)
- **Double Pendulum**: 1/10 diverged
- Shows baseline weakness for long horizons

### 2. Training/Extrapolation Boundary
- **100 steps**: Last training horizon
- **500, 1000 steps**: True extrapolation test
- **Critical to visualize separately**

### 3. Per-Dimension Analysis
- **Lorenz dim 2 (z)**: Often higher error (chaotic vertical)
- **Double Pendulum ω dims**: Velocities harder than angles
- **Important for understanding failure modes**

## Visualization Strategy

### PLOTS (Figures for Paper)

#### 1. Metrics vs Horizon (3-panel subplot) ✓ CURRENT
- NRMSE, Chamfer, Divergence Rate vs horizon
- Log-log scale, vertical line at horizon=100
- Shows training→extrapolation transition
- **Status**: Good, keep

#### 2. NRMSE Comparison Bars ✓ CURRENT
- Bar charts at horizons 100, 500, 1000
- Direct model comparison
- **Status**: Good, keep

#### 3. Per-Dimension NRMSE ✓ CURRENT
- Multi-dimensional systems only
- Grouped bars by dimension
- **Status**: Good, keep

#### 4. Chamfer Comparison ✓ CURRENT
- Bar charts with inf handling
- **Status**: Good, keep

#### 5. Short-Term Metrics ✓ CURRENT
- One-step MSE + Reconstruction error
- **Status**: Good, keep

#### 6. **NEW: Lyapunov Time Estimation**
- Plot: NRMSE vs horizon, mark when NRMSE=1.0
- Estimates predictability horizon
- **Input needed**: NRMSE data (already have!)
- **Computation**: Interpolate to find T where NRMSE(T)=1.0

#### 7. **NEW: Spectral Radius Comparison**
- Bar chart: spectral radius by model
- Reference line at ρ=1.0
- Only for neural models (VAR/eDMD don't have)

#### 8. **NEW: Combined Cross-System NRMSE Heatmap**
- Heatmap: models (rows) × systems (cols)
- Color = NRMSE at horizon 1000
- Quick overview of all results

### TABLES (LaTeX for Paper)

#### 1. **Divergence Summary Table** (instead of plot)
- Columns: Model | System | n_valid | n_diverged | n_total | % Success
- More compact than bar charts
- **Change from current**: divergence_summary.png → table

#### 2. Summary Metrics Table ✓ CURRENT
- Key metrics across all systems
- **Status**: Good, keep

#### 3. **NEW: Horizon-specific NRMSE Table**
- Rows: Models
- Cols: NRMSE@100, NRMSE@500, NRMSE@1000 for each system
- Highlights extrapolation performance

#### 4. **NEW: Best Model per System Table**
- Winner for each metric + system combination
- Bold best values

## Lyapunov Time Computation

### Definition
Time horizon where predictions become unreliable (NRMSE ≥ 1.0)

### Computation from Existing Data
```python
def compute_lyapunov_time(horizons, nrmse_values, threshold=1.0):
    """
    Find horizon T where NRMSE(T) crosses threshold.

    Args:
        horizons: [1, 10, 20, 50, 100, 500, 1000]
        nrmse_values: NRMSE at each horizon
        threshold: 1.0 = prediction as bad as mean

    Returns:
        lyapunov_time: interpolated horizon value
    """
    # Find crossing point
    for i in range(len(horizons)-1):
        if nrmse_values[i] < threshold <= nrmse_values[i+1]:
            # Linear interpolation
            h0, h1 = horizons[i], horizons[i+1]
            n0, n1 = nrmse_values[i], nrmse_values[i+1]
            t_lyap = h0 + (threshold - n0) * (h1 - h0) / (n1 - n0)
            return t_lyap

    # Not crossed yet or already above
    if nrmse_values[0] >= threshold:
        return 0  # Already unreliable
    else:
        return float('inf')  # Never crossed (very good!)
```

### Visualization
- Plot NRMSE curves with horizontal line at y=1.0
- Annotate crossing points
- Bar chart: Lyapunov time by model

## Implementation Priorities

### Phase 1: Enhanced Existing Scripts
1. ✓ Keep all current plots
2. ✓ Add Lyapunov time plot
3. ✓ Add spectral radius comparison
4. ✓ Add cross-system heatmap
5. ✓ Convert divergence_summary.png → table

### Phase 2: New Tables
1. ✓ Horizon-specific NRMSE table
2. ✓ Best model table
3. ✓ Divergence table

### Phase 3: Enhanced Features
1. Statistical significance testing (if multiple runs available)
2. Confidence intervals (if available)
3. Failure mode analysis (which ICs diverge?)

## File Organization

```
final_results_0/
├── duffing/duffing_*/comparison_results.csv
├── vanderpol/vanderpol_*/comparison_results.csv
├── lorenz/lorenz_*/comparison_results.csv
└── double_pendulum/double_pendulum_*/comparison_results.csv

plots/  (output)
├── {system}_metrics_vs_horizon.png
├── {system}_nrmse_comparison.png
├── {system}_chamfer_comparison.png
├── {system}_per_dimension_nrmse.png  (3D/4D only)
├── {system}_short_term_metrics.png
├── {system}_lyapunov_time.png  ← NEW
├── {system}_spectral_radius.png  ← NEW
└── combined/
    ├── combined_nrmse_heatmap.png  ← NEW
    └── combined_nrmse_{horizon}step.png

tables/  (output)
├── summary_table.{csv,tex}
├── divergence_table.{csv,tex}  ← NEW (was plot)
├── horizon_nrmse_table.{csv,tex}  ← NEW
├── best_models_table.{csv,tex}  ← NEW
└── {system}_detailed_table.{csv,tex}
```

## Notes for Paper

### Main Results to Highlight
1. **MoE outperforms baselines** at long horizons (500, 1000 steps)
2. **Extrapolation capability**: Performance beyond training (100→1000)
3. **System-specific performance**: Different models excel on different systems
4. **Stability**: MoE has zero diverged trajectories vs eDMD
5. **Lyapunov time**: Longer prediction horizons for MoE models

### Key Comparisons
- MoE vs KAE Baseline: Multi-expert benefit
- MoE vs Advanced KAE: Multiple operators vs single
- 2/3/4 experts: Diminishing returns? Optimal complexity?
- Spectral radius: How close to 1.0? Stability correlation?

### Statistical Tests (if needed)
- Paired t-test on NRMSE across test trajectories
- Wilcoxon signed-rank (non-parametric alternative)
- Effect size: Cohen's d

## Color Scheme (for consistency)

```python
MODEL_COLORS = {
    'VAR (ARIMA)': '#1f77b4',       # Blue
    'eDMD': '#ff7f0e',              # Orange
    'KAE Baseline': '#2ca02c',      # Green
    'Advanced KAE (1 Expert)': '#d62728',  # Red
    'MoE (2 Experts)': '#9467bd',   # Purple
    'MoE (3 Experts)': '#8c564b',   # Brown
    'MoE (4 Experts)': '#e377c2',   # Pink
}
```
