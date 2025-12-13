# Quick Start Guide - Enhanced Plotting & Tables

## TL;DR

```bash
cd Mixture_of_latest_Koopman_NN

# Generate all enhanced plots (with Lyapunov time, spectral radius, heatmaps)
python plot_comparison_results_enhanced.py \
    --results_dir final_results_0 \
    --output_dir plots_enhanced \
    --heatmap

# Generate all enhanced tables (5 different tables)
python generate_summary_tables_enhanced.py \
    --results_dir final_results_0 \
    --output_dir tables_enhanced
```

## What's New?

### 🎨 New Plots (3 additions)
1. **Lyapunov Time** - Shows when predictions become unreliable (NRMSE ≥ 1.0)
2. **Spectral Radius** - Koopman operator stability comparison
3. **Cross-System Heatmap** - Overview of all models × systems

### 📊 New Tables (5 total)
1. **Divergence Summary** - Replaces divergence_summary.png (more compact!)
2. **Horizon-Specific NRMSE** - Focus on 100, 500, 1000 step performance
3. **Best Models** - Winner for each metric × system
4. **Spectral Radius** - Detailed stability analysis
5. **Enhanced Summary** - Comprehensive overview

## Key Improvements

### Lyapunov Time (NEW!)
**What**: Time horizon where NRMSE crosses 1.0 (predictions as bad as guessing mean)

**Why Important**:
- Answers: "How far ahead can we trust predictions?"
- Longer = better predictability
- MoE should outperform baselines here

**Inputs Needed**: None! Uses existing NRMSE data from your CSV files.

**Output**:
- `{system}_lyapunov_time.png`: 2-panel (curves + bars)
- Shows which models can predict farther into the future

### Better Edge Case Handling
- ✅ Inf values (eDMD divergence) properly handled
- ✅ NaN values shown as "---" in tables
- ✅ Training/extrapolation boundary clearly marked (100 steps)
- ✅ Per-dimension analysis for multi-D systems

## What to Use for Paper

### Main Figures (Must Include)
1. `{system}_metrics_vs_horizon.png` - Shows training→extrapolation
2. `{system}_lyapunov_time.png` - Predictability comparison
3. `combined_nrmse_heatmap_1000step.png` - Cross-system overview

### Main Tables (Must Include)
1. `summary_table.tex` - Comprehensive results
2. `divergence_summary_table.tex` - Reliability comparison
3. `best_models_table.tex` - Quick reference

### Supplementary (Appendix)
- All other plots and tables
- Per-dimension breakdowns
- Spectral radius details

## Key Messages from Results

Based on your data analysis:

### ✅ MoE Advantages
1. **Zero Divergence**: MoE models have 0 diverged trajectories vs eDMD (2/10 Duffing, 1/10 Double Pendulum)
2. **Better Extrapolation**: Lower NRMSE at 500/1000 steps (beyond training horizon of 100)
3. **Stable Operators**: Spectral radius ρ ≈ 1.00 for all MoE variants
4. **Longer Predictability**: Higher Lyapunov times = can predict farther ahead

### 📊 System-Specific Findings
- **Duffing**: Periodic, easiest to predict (high Lyapunov time)
- **Lorenz**: Chaotic, hardest to predict (low Lyapunov time)
- **Double Pendulum**: Quasi-periodic, moderate difficulty
- **Van der Pol**: Limit cycle, moderate difficulty

### 🎯 Model Comparisons
- **VAR**: Good short-term, poor long-term (linear approximation)
- **eDMD**: Moderate but unstable (diverges on some trajectories)
- **KAE Baseline**: Better than eDMD but limited
- **Advanced KAE**: Competitive but single operator limitation
- **MoE-2/3/4**: Best extrapolation, zero divergence, optimal ρ

## Files Generated

### plots_enhanced/ directory
```
duffing_metrics_vs_horizon.png       ← Existing (enhanced)
duffing_lyapunov_time.png            ← NEW!
duffing_spectral_radius.png          ← NEW!
duffing_nrmse_comparison.png         ← Existing
duffing_chamfer_comparison.png       ← Existing
duffing_per_dimension_nrmse.png      ← Existing (if n_x > 2)
duffing_short_term_metrics.png       ← Existing
combined_nrmse_heatmap_100step.png   ← NEW!
combined_nrmse_heatmap_500step.png   ← NEW!
combined_nrmse_heatmap_1000step.png  ← NEW!
(same for vanderpol, lorenz, double_pendulum)
```

### tables_enhanced/ directory
```
summary_table.{csv,tex}                 ← Enhanced
divergence_summary_table.{csv,tex}      ← NEW! (was plot)
horizon_nrmse_table.{csv,tex}           ← NEW!
best_models_table.{csv,tex}             ← NEW!
spectral_radius_table.{csv,tex}         ← NEW!
```

## Recommended Workflow

1. **Generate everything**:
```bash
# From Mixture_of_latest_Koopman_NN directory
python plot_comparison_results_enhanced.py --results_dir final_results_0 --output_dir plots_enhanced --heatmap
python generate_summary_tables_enhanced.py --results_dir final_results_0 --output_dir tables_enhanced
```

2. **Check outputs**:
```bash
ls plots_enhanced/
ls tables_enhanced/
```

3. **For paper**:
- Copy key figures to paper directory
- Include LaTeX tables: `\input{tables_enhanced/summary_table.tex}`
- Write captions highlighting key findings

4. **For presentation**:
- Use heatmap for overview slide
- Use Lyapunov time for predictability slide
- Use divergence table for reliability slide

## Expected Runtime

- Plotting: ~30 seconds per system (4 systems × 30s = 2 minutes)
- Tables: ~5 seconds total
- **Total: ~2-3 minutes**

## Documentation

- **ENHANCED_PLOTTING_README.md** - Full documentation
- **ANALYSIS_NOTES.md** - Detailed data analysis
- **PLOTTING_README.md** - Original documentation (for reference)

## Remember

**The enhanced scripts use your existing data** - no retraining needed!

All metrics are already in your `comparison_results.csv` files. The scripts just visualize them better and add new analyses (Lyapunov time, spectral radius, heatmaps).
