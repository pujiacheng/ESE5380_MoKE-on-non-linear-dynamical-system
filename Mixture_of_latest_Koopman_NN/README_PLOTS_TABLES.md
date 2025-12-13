# All Plots and Tables - Ready for Paper!

## ✅ Generated Successfully

All plots and tables have been generated from `final_results_1` data and are ready for your paper!

### 📊 Plots Location: `paper_plots/`
- 15 high-resolution PNG files (300 DPI)
- Ready to include in LaTeX/Word

### 📄 Tables Location: `paper_tables/`
- 5 CSV files (for spreadsheet viewing)
- 5 LaTeX .tex files (for paper inclusion)

## Quick Reference

### What to Include in Main Paper

**Figures (pick 3-4)**:
1. `combined_nrmse_heatmap_1000step.png` - Overview of all results
2. `duffing_metrics_vs_horizon.png` - Shows training→extrapolation
3. `duffing_lyapunov_time.png` - Predictability analysis
4. (Optional) `duffing_spectral_radius.png` - Stability

**Tables (pick 2-3)**:
1. `divergence_summary_table.tex` - **Critical finding: eDMD diverges!**
2. `summary_table.tex` - Comprehensive results
3. `horizon_nrmse_table.tex` - Extrapolation focus

### What to Put in Supplementary

- All remaining plots (12 plot files)
- All remaining tables (3 table files)
- Detailed per-system analysis

## Key Findings to Highlight

### 🚨 Critical Discovery: eDMD Reliability Issue

**eDMD diverged on 31/200 test trajectories (15.5%)**:
- Duffing: 16/100 failed (84% success)
- Double Pendulum: 15/100 failed (85% success)

**All MoE models: 0/700 diverged (100% success)**

**This is your killer result!** eDMD has better average metrics but catastrophically fails on 15% of cases. MoE is more reliable.

### 📈 Performance Insights

**Best models vary by system**:
- Van der Pol: eDMD (when doesn't diverge)
- Duffing: MoE-4 (training), eDMD (extrapolation, but unreliable)
- Lorenz: eDMD (but all models struggle - chaotic)
- Double Pendulum: MoE-2 (reliable choice)

**Spectral radius**: All MoE achieve ρ ≈ 1.00 ± 0.01 (excellent stability)

## How to Use in LaTeX

### Include Figures

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{paper_plots/combined_nrmse_heatmap_1000step.png}
\caption{Cross-system NRMSE comparison at 1000-step extrapolation (10× training horizon). 
eDMD achieves lowest average error but diverges on 15\% of test trajectories (white cells). 
MoE models maintain 100\% success rate with competitive performance.}
\label{fig:heatmap}
\end{figure}
```

### Include Tables

```latex
\begin{table}[htbp]
\centering
\caption{Trajectory stability comparison showing success rate across all test trajectories.}
\label{tab:divergence}
\input{paper_tables/divergence_summary_table.tex}
\end{table}
```

## File Sizes

All plots are high-quality (300 DPI):
- Individual system plots: ~120-550 KB each
- Heatmaps: ~200 KB each
- Tables: <5 KB each

**Total size**: ~4.5 MB for all plots + tables

## Viewing the Results

### Plots (PNG files)
- Open with any image viewer
- Preview in Finder (Mac)
- Open in browser

### Tables
**CSV files**:
```bash
# View in terminal
column -s, -t < paper_tables/best_models_table.csv

# Or open in Excel/Google Sheets
```

**LaTeX files**:
- Ready to `\input{}` in your LaTeX paper
- No modifications needed (already formatted)

## Regenerating (if needed)

If you need to regenerate with different settings:

```bash
cd Mixture_of_latest_Koopman_NN

# Regenerate plots
python plot_comparison_results_enhanced.py \
    --results_dir final_results_1 \
    --output_dir paper_plots \
    --heatmap

# Regenerate tables
python generate_summary_tables_enhanced.py \
    --results_dir final_results_1 \
    --output_dir paper_tables
```

## Documentation

- `RESULTS_SUMMARY.md` - Detailed analysis of findings
- `ENHANCED_PLOTTING_README.md` - Full plotting documentation
- `QUICK_START.md` - Quick reference guide
- `ANALYSIS_NOTES.md` - Data analysis notes

## Next Steps

1. ✅ Review all plots and tables
2. ✅ Select figures/tables for main paper vs supplementary
3. ⬜ Write figure captions emphasizing key findings
4. ⬜ Write results section referring to tables/figures
5. ⬜ Create architecture/method diagrams
6. ⬜ Prepare for submission

---

**All ready for your paper! 🎉**
