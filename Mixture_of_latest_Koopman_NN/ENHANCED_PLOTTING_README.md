# Enhanced Plotting and Visualization Guide

## Overview

This document describes the **enhanced** plotting and table generation scripts for the Koopman MoE paper. These scripts build upon the original versions with new features specifically designed to highlight key results.

## New Scripts

### 1. `plot_comparison_results_enhanced.py`
Enhanced plotting with 3 major new features:
- **Lyapunov time estimation**: When do predictions become unreliable?
- **Spectral radius visualization**: Koopman operator stability
- **Cross-system heatmaps**: Overview of all results

### 2. `generate_summary_tables_enhanced.py`
Enhanced tables with 5 different table types:
- **Divergence summary table** (replaces divergence_summary.png)
- **Horizon-specific NRMSE table**: Focus on extrapolation
- **Best models table**: Winner for each metric/system
- **Spectral radius table**: Stability analysis
- **Enhanced summary table**: Comprehensive overview

## Usage

### Quick Start

```bash
# Generate all enhanced plots
python plot_comparison_results_enhanced.py \
    --results_dir Mixture_of_latest_Koopman_NN/final_results_0 \
    --output_dir plots_enhanced \
    --heatmap

# Generate all enhanced tables
python generate_summary_tables_enhanced.py \
    --results_dir Mixture_of_latest_Koopman_NN/final_results_0 \
    --output_dir tables_enhanced
```

### Advanced Usage

```bash
# Only specific systems
python plot_comparison_results_enhanced.py \
    --results_dir final_results_0 \
    --output_dir plots_enhanced \
    --systems duffing lorenz

# Custom horizons for heatmap
# (edit HORIZONS in script: [100, 500, 1000])
```

## New Visualizations

### 1. Lyapunov Time Estimation

**What it shows**: Time horizon where predictions become unreliable (NRMSE ≥ 1.0)

**Files generated**:
- `{system}_lyapunov_time.png`: 2-panel figure
  - Left: NRMSE curves with threshold line
  - Right: Bar chart of Lyapunov times

**Interpretation**:
- **Longer Lyapunov time = better**: Model remains accurate longer
- **∞ symbol**: Model never crosses threshold (excellent!)
- **0**: Model unreliable from the start (poor)
- **Training horizon line**: Shows if model can extrapolate

**Key insights**:
- MoE models should have longer Lyapunov times than baselines
- Chaotic systems (Lorenz) will have shorter times than oscillators
- Helps answer: "How far ahead can we trust predictions?"

### 2. Spectral Radius Comparison

**What it shows**: Stability of learned Koopman operators

**Files generated**:
- `{system}_spectral_radius.png`: Bar chart with ρ=1.0 reference line

**Interpretation**:
- **ρ ≈ 1.0**: Ideal (energy-preserving)
- **ρ > 1.0**: Unstable (errors grow)
- **ρ < 1.0**: Dissipative (errors decay)

**Key insights**:
- Neural models should cluster near 1.0
- Spectral radius regularization effectiveness
- Correlation with long-term accuracy

**Note**: Only neural models have spectral radius (VAR/eDMD excluded)

### 3. Cross-System NRMSE Heatmap

**What it shows**: Overview of all models × systems at specific horizon

**Files generated**:
- `combined_nrmse_heatmap_100step.png`: Training performance
- `combined_nrmse_heatmap_500step.png`: Mid extrapolation
- `combined_nrmse_heatmap_1000step.png`: Far extrapolation

**Interpretation**:
- **Color intensity**: Red = poor, Yellow = good (log scale)
- **White cells**: Missing/diverged data
- **Values shown**: NRMSE at each (model, system) pair

**Key insights**:
- Quickly identify which models excel on which systems
- Extrapolation degradation (compare 100 vs 1000)
- System difficulty ranking

## Enhanced Tables

### 1. Divergence Summary Table
**Replaces**: `{system}_divergence_summary.png`

**Why table instead of plot**:
- More compact for multiple systems
- Exact values visible
- Better for LaTeX inclusion

**Columns**:
- System, Model, Valid, Diverged, Total, Success (%)

**Usage in paper**:
```latex
\input{tables_enhanced/divergence_summary_table.tex}
```

**Key insights**:
- eDMD has divergence issues (Duffing: 2/10, Double Pendulum: 1/10)
- All MoE models: 0 diverged trajectories (100% success!)
- Baseline reliability comparison

### 2. Horizon-Specific NRMSE Table

**What it shows**: NRMSE at critical horizons (100, 500, 1000) across all systems

**Format**:
- Rows: Models
- Columns: System (H=100), System (H=500), System (H=1000)

**Key insights**:
- Training vs extrapolation performance
- Which models generalize best
- System-specific difficulties

### 3. Best Models Table

**What it shows**: Winner for each metric and system combination

**Format**:
- Rows: Metrics (One-Step MSE, NRMSE@100, NRMSE@500, etc.)
- Columns: Systems
- Cells: Best model name + value

**Key insights**:
- No single model dominates all metrics
- MoE variants win most long-horizon metrics
- Baselines may win short-term metrics

### 4. Spectral Radius Table

**What it shows**: Koopman operator stability across systems

**Columns**:
- System, Model, Spectral Radius (ρ), Δ from 1.0, Status

**Status definitions**:
- **Stable**: |ρ - 1.0| < 0.01
- **Unstable**: ρ > 1.01
- **Dissipative**: ρ < 0.99

**Key insights**:
- How well spectral radius penalty works
- Which systems require more stable operators
- Correlation with extrapolation performance

### 5. Enhanced Summary Table

**What it shows**: Comprehensive metrics for all models and systems

**Columns**:
- System, Model, Params
- 1-step MSE, NRMSE-100/500/1000
- Chamfer-100/1000, Div-100
- Diverged, ρ (spectral radius)

**Usage**: Main results table for paper

## What to Plot vs What to Table

### Plots (Visualizations) ✓
1. **Metrics vs Horizon** (3-panel): Shows trends, extrapolation boundary
2. **NRMSE Comparison Bars**: Direct model comparison at key horizons
3. **Chamfer Comparison**: Phase space fidelity
4. **Per-Dimension NRMSE**: Dimension-specific analysis (3D/4D only)
5. **Short-Term Metrics**: One-step MSE + reconstruction
6. **Lyapunov Time**: Predictability horizon
7. **Spectral Radius**: Stability visualization
8. **Cross-System Heatmap**: Overview

### Tables (LaTeX) ✓
1. **Divergence Summary**: Exact counts, compact
2. **Horizon NRMSE**: Numerical comparisons
3. **Best Models**: Quick reference
4. **Spectral Radius**: Precise values matter
5. **Summary Table**: Comprehensive reference

## Edge Cases Handled

### 1. Infinity Values (Chamfer Distance)
- **In plots**: Marked with × or omitted from line plots
- **In tables**: Shown as ∞ symbol
- **In heatmaps**: Masked (white cells)

### 2. NaN Values (Missing Data)
- **In plots**: Gaps in line plots
- **In tables**: Shown as ---
- **In heatmaps**: Masked (white cells)

### 3. Training/Extrapolation Boundary (100 steps)
- **All time-series plots**: Vertical dashed line at horizon=100
- **Annotations**: "Extrapolation →" label
- **Lyapunov plot**: Red reference line

### 4. Per-Dimension Metrics
- **Automatically detected**: Only for n_x > 2
- **Duffing/VanderPol**: 2D (x, ẋ)
- **Lorenz**: 3D (x, y, z) - z dimension often highest error
- **Double Pendulum**: 4D (θ₁, θ₂, ω₁, ω₂) - velocities harder

### 5. Model Name Consistency
- **Full names** in data: "Advanced KAE (1 Expert)"
- **Short names** in plots/tables: "KAE-A"
- **Consistent colors** across all visualizations

## Lyapunov Time Computation

### Mathematical Definition

Lyapunov time $T_L$ is the horizon where predictions become as bad as guessing the mean:

$$T_L = \min\{T : \text{NRMSE}(T) \geq 1.0\}$$

### Implementation

```python
def compute_lyapunov_time(horizons, nrmse_values, threshold=1.0):
    """
    Find horizon T where NRMSE(T) crosses threshold.
    Uses linear interpolation between evaluation points.
    """
    for i in range(len(horizons)-1):
        if nrmse_values[i] < threshold <= nrmse_values[i+1]:
            # Linear interpolation
            h0, h1 = horizons[i], horizons[i+1]
            n0, n1 = nrmse_values[i], nrmse_values[i+1]
            t_lyap = h0 + (threshold - n0) * (h1 - h0) / (n1 - n0)
            return t_lyap

    # Edge cases
    if nrmse_values[0] >= threshold:
        return 0  # Already unreliable
    else:
        return float('inf')  # Never crossed
```

### Inputs Required

**Already available in your data**:
- NRMSE values at horizons [1, 10, 20, 50, 100, 500, 1000]
- From columns: `nrmse_1step`, `nrmse_10step`, ..., `nrmse_1000step`

**No additional computation needed!**

### Expected Results

For typical chaotic/oscillatory systems:
- **Duffing/VanderPol**: $T_L \approx$ 500-1000 steps (periodic)
- **Lorenz**: $T_L \approx$ 50-200 steps (chaotic)
- **Double Pendulum**: $T_L \approx$ 100-500 steps (quasi-periodic)

**Model comparison**:
- VAR: shortest $T_L$ (linear approximation)
- eDMD: moderate $T_L$ (fixed basis)
- MoE: longest $T_L$ (adaptive dynamics)

## File Organization

```
final_results_0/
├── duffing/duffing_*/comparison_results.csv
├── vanderpol/vanderpol_*/comparison_results.csv
├── lorenz/lorenz_*/comparison_results.csv
└── double_pendulum/double_pendulum_*/comparison_results.csv

plots_enhanced/  (NEW output directory)
├── duffing_metrics_vs_horizon.png
├── duffing_lyapunov_time.png  ← NEW
├── duffing_spectral_radius.png  ← NEW
├── duffing_nrmse_comparison.png
├── duffing_chamfer_comparison.png
├── duffing_per_dimension_nrmse.png
├── duffing_short_term_metrics.png
├── combined_nrmse_heatmap_100step.png  ← NEW
├── combined_nrmse_heatmap_500step.png  ← NEW
├── combined_nrmse_heatmap_1000step.png  ← NEW
└── (same for vanderpol, lorenz, double_pendulum)

tables_enhanced/  (NEW output directory)
├── summary_table.{csv,tex}
├── divergence_summary_table.{csv,tex}  ← NEW (was plot)
├── horizon_nrmse_table.{csv,tex}  ← NEW
├── best_models_table.{csv,tex}  ← NEW
└── spectral_radius_table.{csv,tex}  ← NEW
```

## Comparison: Original vs Enhanced

### Original Scripts
- ✓ Basic metrics vs horizon
- ✓ Model comparison bars
- ✓ Per-dimension analysis
- ✓ Divergence summary plot
- ✓ Basic summary table

### Enhanced Scripts (New Features)
- ✅ **Lyapunov time estimation** (predictability horizon)
- ✅ **Spectral radius visualization** (stability analysis)
- ✅ **Cross-system heatmaps** (overview)
- ✅ **Divergence table** (replaces plot, more compact)
- ✅ **Horizon-specific NRMSE table** (focus on key horizons)
- ✅ **Best models table** (quick reference)
- ✅ **Spectral radius table** (detailed stability)
- ✅ **Better inf/nan handling** (robust edge cases)

## Recommended Workflow

### For Paper

1. **Run enhanced scripts**:
```bash
cd Mixture_of_latest_Koopman_NN
python plot_comparison_results_enhanced.py --results_dir final_results_0 --output_dir plots_enhanced --heatmap
python generate_summary_tables_enhanced.py --results_dir final_results_0 --output_dir tables_enhanced
```

2. **Main figures** (include in paper):
   - `{system}_metrics_vs_horizon.png`: Shows training→extrapolation
   - `{system}_lyapunov_time.png`: Predictability analysis
   - `combined_nrmse_heatmap_1000step.png`: Cross-system overview
   - `{system}_per_dimension_nrmse.png`: Dimension-specific (for 3D/4D)

3. **Main tables** (include in paper):
   - `summary_table.tex`: Comprehensive results
   - `divergence_summary_table.tex`: Stability/reliability
   - `best_models_table.tex`: Quick summary
   - `horizon_nrmse_table.tex`: Extrapolation focus

4. **Supplementary** (appendix):
   - All other plots and tables
   - Per-system detailed tables

### For Presentation

1. **Key slides**:
   - Cross-system heatmap: Overview of results
   - Lyapunov time bars: Predictability comparison
   - NRMSE vs horizon: Training→extrapolation trend
   - Divergence table: Reliability comparison

## Key Messages to Highlight

### 1. Extrapolation Capability
- **Plot**: NRMSE vs horizon (vertical line at 100)
- **Table**: Horizon NRMSE table (compare 100 vs 1000)
- **Message**: "MoE maintains <X% error even at 10× extrapolation"

### 2. Zero Divergence
- **Table**: Divergence summary
- **Message**: "MoE: 0/10 diverged vs eDMD: 3/40 total"

### 3. Longer Predictability
- **Plot**: Lyapunov time bars
- **Message**: "MoE predicts 2× farther than baselines"

### 4. Stable Operators
- **Plot**: Spectral radius bars
- **Table**: Spectral radius table
- **Message**: "All MoE variants achieve ρ ≈ 1.00 ± 0.01"

### 5. System Generalization
- **Plot**: Cross-system heatmap
- **Table**: Best models table
- **Message**: "MoE-3 wins on 3/4 systems at horizon=1000"

## Statistical Significance (Future Work)

If you have multiple runs or want to add error bars:

```python
# Modify scripts to include:
# 1. Standard deviation bars on plots
# 2. Confidence intervals in tables
# 3. Paired t-tests for model comparisons
# 4. Effect sizes (Cohen's d)
```

Currently, scripts assume single run per model/system.

## Troubleshooting

### Issue: "No data loaded"
- Check `--results_dir` path is correct
- Ensure CSV files exist in `{system}/{timestamp}/comparison_results.csv`

### Issue: "Spectral radius data not available"
- Normal for VAR/eDMD (they don't have Koopman operators)
- Check neural models have `spectral_radius` column

### Issue: Plots have gaps/missing data
- Normal if NRMSE=nan at some horizons (trajectory too short)
- Chamfer=inf indicates divergence (expected for some models)

### Issue: LaTeX table formatting
- Use `\usepackage{booktabs}` for better tables
- Manually adjust column widths if needed
- Consider rotating wide tables: `\begin{sidewaystable}`

## Citation

If you use these enhanced scripts, please cite:
```bibtex
@article{your_paper_2025,
  title={Mixture of Koopman Experts for Nonlinear Dynamical Systems},
  author={Your Name et al.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Contact

For questions or issues:
- Check `ANALYSIS_NOTES.md` for detailed analysis
- See original `PLOTTING_README.md` for basic usage
- Raise issue on GitHub repository
