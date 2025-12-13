"""
Enhanced Comprehensive Plotting Script for Model Comparison Results

This script creates publication-quality plots from comparison_results.csv files
for all 4 dynamical systems: duffing, vanderpol, lorenz, double_pendulum.

New features:
1. Lyapunov time estimation and visualization
2. Spectral radius comparison
3. Cross-system NRMSE heatmap
4. Better inf/nan handling
5. Statistical annotations

Usage:
    python plot_comparison_results_enhanced.py --results_dir final_results_0 --output_dir plots_enhanced
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from scipy import interpolate
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Set publication-quality style
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    try:
        plt.style.use('seaborn-paper')
    except OSError:
        plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


# ==============================================================================
# Color and Style Configuration
# ==============================================================================

# Model colors and markers
MODEL_STYLE = {
    'VAR (ARIMA)': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'VAR'},
    'eDMD': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-', 'label': 'eDMD'},
    'KAE Baseline': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-', 'label': 'KAE-B'},
    'Advanced KAE (1 Expert)': {'color': '#d62728', 'marker': 'v', 'linestyle': '-', 'label': 'KAE-A'},
    'MoE (2 Experts)': {'color': '#9467bd', 'marker': 'D', 'linestyle': '-', 'label': 'MoE-2'},
    'MoE (3 Experts)': {'color': '#8c564b', 'marker': 'p', 'linestyle': '-', 'label': 'MoE-3'},
    'MoE (4 Experts)': {'color': '#e377c2', 'marker': 'h', 'linestyle': '-', 'label': 'MoE-4'},
}

# System names for titles
SYSTEM_NAMES = {
    'duffing': 'Duffing Oscillator',
    'vanderpol': 'Van der Pol Oscillator',
    'lorenz': 'Lorenz Attractor',
    'double_pendulum': 'Double Pendulum'
}

# State dimension labels
STATE_LABELS = {
    'duffing': ['x', 'ẋ'],
    'vanderpol': ['x', 'ẋ'],
    'lorenz': ['x', 'y', 'z'],
    'double_pendulum': ['θ₁', 'θ₂', 'ω₁', 'ω₂']
}

# Evaluation horizons
HORIZONS = [1, 10, 20, 50, 100, 500, 1000]
TRAINING_MAX_HORIZON = 100  # Training goes up to 100 steps


# ==============================================================================
# Data Loading and Processing
# ==============================================================================

def load_results(results_dir, system):
    """Load comparison results CSV for a system"""
    system_dir = os.path.join(results_dir, system)
    if not os.path.exists(system_dir):
        raise FileNotFoundError(f"System directory not found: {system_dir}")

    subdirs = [d for d in os.listdir(system_dir) if os.path.isdir(os.path.join(system_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {system_dir}")

    csv_path = os.path.join(system_dir, subdirs[0], 'comparison_results.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def get_metric_values(df, metric_prefix, horizons):
    """Extract metric values for given horizons"""
    values = {}
    for h in horizons:
        col = f'{metric_prefix}_{h}step'
        if col in df.columns:
            values[h] = df[col].values
        else:
            values[h] = np.full(len(df), np.nan)
    return values


def handle_inf_values(values, replace_with=None):
    """Handle inf values in metric arrays"""
    values_arr = np.array(values)
    if replace_with is None:
        finite_vals = values_arr[np.isfinite(values_arr)]
        if len(finite_vals) > 0:
            replace_with = np.max(finite_vals) * 10
        else:
            replace_with = 1e6
    return np.where(np.isinf(values_arr), replace_with, values_arr)


# ==============================================================================
# Lyapunov Time Estimation
# ==============================================================================

def compute_lyapunov_time(horizons, nrmse_values, threshold=1.0):
    """
    Estimate Lyapunov time: horizon where NRMSE crosses threshold.

    Args:
        horizons: List of evaluation horizons
        nrmse_values: NRMSE values at each horizon
        threshold: NRMSE threshold (1.0 = prediction as bad as mean)

    Returns:
        lyapunov_time: Interpolated horizon value, inf if never crossed, 0 if always above
    """
    # Remove nans and infs
    valid_idx = np.isfinite(nrmse_values)
    if not np.any(valid_idx):
        return np.nan

    h = np.array(horizons)[valid_idx]
    n = np.array(nrmse_values)[valid_idx]

    if len(h) == 0:
        return np.nan

    # Already above threshold at first horizon
    if n[0] >= threshold:
        return 0.0

    # Find crossing point
    for i in range(len(h)-1):
        if n[i] < threshold <= n[i+1]:
            # Linear interpolation
            t_lyap = h[i] + (threshold - n[i]) * (h[i+1] - h[i]) / (n[i+1] - n[i])
            return t_lyap

    # Never crossed
    return np.inf


def plot_lyapunov_time(ax, df, system):
    """
    Plot Lyapunov time estimation for all models.

    Shows NRMSE curves with horizontal line at threshold=1.0,
    and marks the crossing points.
    """
    horizons = HORIZONS
    threshold = 1.0

    lyapunov_times = []
    model_names = []

    for model_name in df['model'].values:
        style = MODEL_STYLE.get(model_name, {'color': 'gray', 'marker': 'o', 'linestyle': '-', 'label': model_name})

        nrmse_values = []
        valid_horizons = []

        for h in horizons:
            col = f'nrmse_{h}step'
            if col in df.columns:
                val = df[df['model'] == model_name][col].values[0]
                if not np.isnan(val) and not np.isinf(val):
                    nrmse_values.append(val)
                    valid_horizons.append(h)

        if len(valid_horizons) > 0:
            ax.plot(valid_horizons, nrmse_values,
                   color=style['color'], marker=style['marker'],
                   linestyle=style['linestyle'], label=style['label'],
                   linewidth=2, markersize=6, alpha=0.8)

            # Compute Lyapunov time
            t_lyap = compute_lyapunov_time(valid_horizons, nrmse_values, threshold)
            lyapunov_times.append(t_lyap)
            model_names.append(style['label'])

            # Mark crossing point if finite
            if np.isfinite(t_lyap) and t_lyap > 0:
                ax.plot(t_lyap, threshold, 'x', color=style['color'],
                       markersize=10, markeredgewidth=3, zorder=10)

    # Horizontal line at threshold
    ax.axhline(y=threshold, color='black', linestyle='--',
              linewidth=1.5, alpha=0.7, label=f'NRMSE = {threshold}')

    # Vertical line at training boundary
    ax.axvline(x=TRAINING_MAX_HORIZON, color='gray', linestyle=':',
              linewidth=1.5, alpha=0.5)

    ax.set_xlabel('Horizon (steps)', fontweight='bold')
    ax.set_ylabel('NRMSE', fontweight='bold')
    ax.set_title(f'{SYSTEM_NAMES[system]} - Lyapunov Time Estimation', fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', framealpha=0.9, ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    return lyapunov_times, model_names


def plot_lyapunov_time_bars(ax, df, system):
    """Bar chart of Lyapunov times for all models"""
    horizons = HORIZONS
    threshold = 1.0

    models = df['model'].values
    lyapunov_times = []

    for model_name in models:
        nrmse_values = []
        valid_horizons = []

        for h in horizons:
            col = f'nrmse_{h}step'
            if col in df.columns:
                val = df[df['model'] == model_name][col].values[0]
                if not np.isnan(val) and not np.isinf(val):
                    nrmse_values.append(val)
                    valid_horizons.append(h)

        t_lyap = compute_lyapunov_time(valid_horizons, nrmse_values, threshold)
        lyapunov_times.append(t_lyap)

    # Plot bars
    colors = [MODEL_STYLE.get(m, {}).get('color', 'gray') for m in models]
    x = np.arange(len(models))

    # Handle inf values for plotting
    plot_values = np.array(lyapunov_times)
    has_inf = np.isinf(plot_values)
    max_finite = np.max(plot_values[np.isfinite(plot_values)]) if np.any(np.isfinite(plot_values)) else 1000
    plot_values = np.where(has_inf, max_finite * 1.3, plot_values)
    plot_values = np.where(plot_values == 0, 1, plot_values)  # Replace 0 with small value for log scale

    bars = ax.bar(x, plot_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    # Mark inf values
    for i, (bar, is_inf, orig_val) in enumerate(zip(bars, has_inf, lyapunov_times)):
        if is_inf:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                   '∞', ha='center', va='bottom', fontsize=14, fontweight='bold')
        elif orig_val == 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                   '0', ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_STYLE.get(m, {}).get('label', m) for m in models],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Lyapunov Time (steps)', fontweight='bold')
    ax.set_title(f'{SYSTEM_NAMES[system]} - Predictability Horizon', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=TRAINING_MAX_HORIZON, color='red', linestyle='--',
              alpha=0.5, linewidth=1, label='Training horizon')
    ax.legend(fontsize=8)


# ==============================================================================
# Spectral Radius Visualization
# ==============================================================================

def plot_spectral_radius(ax, df, system):
    """Plot spectral radius comparison (neural models only)"""
    if 'spectral_radius' not in df.columns:
        ax.text(0.5, 0.5, 'Spectral radius data not available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        return

    models = df['model'].values
    spec_rad = df['spectral_radius'].values

    # Filter out nan/inf values
    valid_mask = np.isfinite(spec_rad) & (spec_rad > 0)
    models = models[valid_mask]
    spec_rad = spec_rad[valid_mask]

    if len(models) == 0:
        ax.text(0.5, 0.5, 'No valid spectral radius data',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        return

    colors = [MODEL_STYLE.get(m, {}).get('color', 'gray') for m in models]
    x = np.arange(len(models))

    bars = ax.bar(x, spec_rad, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    # Reference line at ρ=1.0
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ρ = 1.0 (ideal)')

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_STYLE.get(m, {}).get('label', m) for m in models],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Spectral Radius (ρ)', fontweight='bold')
    ax.set_title(f'{SYSTEM_NAMES[system]} - Koopman Operator Stability', fontweight='bold')
    ax.set_ylim(bottom=max(0, min(spec_rad)*0.9), top=max(spec_rad)*1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)


# ==============================================================================
# Cross-System Heatmap
# ==============================================================================

def create_cross_system_heatmap(results_dir, output_dir, systems, horizon=1000):
    """Create heatmap of NRMSE across all systems and models"""
    all_data = {}
    for system in systems:
        try:
            all_data[system] = load_results(results_dir, system)
        except Exception as e:
            print(f"Error loading {system}: {e}")
            continue

    if not all_data:
        print("No data loaded for heatmap!")
        return

    # Collect NRMSE values
    model_order = ['VAR (ARIMA)', 'eDMD', 'KAE Baseline', 'Advanced KAE (1 Expert)',
                   'MoE (2 Experts)', 'MoE (3 Experts)', 'MoE (4 Experts)']
    system_order = systems

    # Create matrix
    matrix = []
    valid_models = []

    for model in model_order:
        row = []
        model_found = False
        for system in system_order:
            if system in all_data:
                df = all_data[system]
                if model in df['model'].values:
                    col = f'nrmse_{horizon}step'
                    val = df[df['model'] == model][col].values[0]
                    row.append(val if np.isfinite(val) else np.nan)
                    model_found = True
                else:
                    row.append(np.nan)
            else:
                row.append(np.nan)

        if model_found:
            matrix.append(row)
            valid_models.append(model)

    matrix = np.array(matrix)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use log scale for colormap
    vmin = np.nanmin(matrix[matrix > 0])
    vmax = np.nanmax(matrix[np.isfinite(matrix)])

    # Create masked array for nan values
    masked_matrix = np.ma.masked_invalid(matrix)

    im = ax.imshow(masked_matrix, cmap='YlOrRd', aspect='auto',
                   norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))

    # Set ticks
    ax.set_xticks(np.arange(len(system_order)))
    ax.set_yticks(np.arange(len(valid_models)))
    ax.set_xticklabels([SYSTEM_NAMES[s] for s in system_order], fontsize=10)
    ax.set_yticklabels([MODEL_STYLE[m]['label'] for m in valid_models], fontsize=10)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add values as text
    for i in range(len(valid_models)):
        for j in range(len(system_order)):
            val = matrix[i, j]
            if np.isfinite(val):
                text = ax.text(j, i, f'{val:.2f}',
                             ha="center", va="center", color="black" if val < vmax/2 else "white",
                             fontsize=9, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'NRMSE at {horizon} steps', rotation=270, labelpad=20, fontweight='bold')

    ax.set_title(f'Cross-System NRMSE Comparison (Horizon = {horizon} steps)',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'combined_nrmse_heatmap_{horizon}step.png'), dpi=300)
    plt.close()
    print(f"  Saved: combined_nrmse_heatmap_{horizon}step.png")


# ==============================================================================
# Import existing plotting functions (from original script)
# ==============================================================================

def plot_nrmse_vs_horizon(ax, df, system, show_training_split=True):
    """Plot NRMSE vs horizon for all models"""
    horizons = HORIZONS

    for model_name in df['model'].values:
        style = MODEL_STYLE.get(model_name, {'color': 'gray', 'marker': 'o', 'linestyle': '-', 'label': model_name})

        nrmse_values = []
        valid_horizons = []

        for h in horizons:
            col = f'nrmse_{h}step'
            if col in df.columns:
                val = df[df['model'] == model_name][col].values[0]
                if not np.isnan(val) and not np.isinf(val):
                    nrmse_values.append(val)
                    valid_horizons.append(h)

        if len(valid_horizons) > 0:
            ax.plot(valid_horizons, nrmse_values,
                   color=style['color'], marker=style['marker'],
                   linestyle=style['linestyle'], label=style['label'],
                   linewidth=2, markersize=6)

    if show_training_split:
        ax.axvline(x=TRAINING_MAX_HORIZON, color='gray', linestyle='--',
                  linewidth=1.5, alpha=0.7)
        ylim = ax.get_ylim()
        ax.text(TRAINING_MAX_HORIZON * 1.2, ylim[1] * 0.5,
               'Extrapolation →', fontsize=9, alpha=0.7, rotation=90)

    ax.set_xlabel('Horizon (steps)', fontweight='bold')
    ax.set_ylabel('NRMSE', fontweight='bold')
    ax.set_title(f'{SYSTEM_NAMES[system]} - NRMSE vs Horizon', fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', framealpha=0.9, ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3, which='both')


def plot_chamfer_vs_horizon(ax, df, system, show_training_split=True):
    """Plot Chamfer distance vs horizon for all models"""
    horizons = HORIZONS

    for model_name in df['model'].values:
        style = MODEL_STYLE.get(model_name, {'color': 'gray', 'marker': 'o', 'linestyle': '-', 'label': model_name})

        chamfer_values = []
        valid_horizons = []
        has_inf = False

        for h in horizons:
            col = f'chamfer_{h}step'
            if col in df.columns:
                val = df[df['model'] == model_name][col].values[0]
                if np.isinf(val):
                    has_inf = True
                elif not np.isnan(val):
                    chamfer_values.append(val)
                    valid_horizons.append(h)

        if len(valid_horizons) > 0:
            ax.plot(valid_horizons, chamfer_values,
                   color=style['color'], marker=style['marker'],
                   linestyle=style['linestyle'], label=style['label'],
                   linewidth=2, markersize=6)

    if show_training_split:
        ax.axvline(x=TRAINING_MAX_HORIZON, color='gray', linestyle='--',
                  linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Horizon (steps)', fontweight='bold')
    ax.set_ylabel('Chamfer Distance', fontweight='bold')
    ax.set_title(f'{SYSTEM_NAMES[system]} - Chamfer Distance vs Horizon', fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', framealpha=0.9, ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3, which='both')


def plot_divergence_vs_horizon(ax, df, system, show_training_split=True):
    """Plot divergence rate vs horizon for all models"""
    horizons = HORIZONS

    for model_name in df['model'].values:
        style = MODEL_STYLE.get(model_name, {'color': 'gray', 'marker': 'o', 'linestyle': '-', 'label': model_name})

        div_values = []
        valid_horizons = []

        for h in horizons:
            col = f'divergence_{h}step'
            if col in df.columns:
                val = df[df['model'] == model_name][col].values[0]
                if not np.isnan(val) and not np.isinf(val):
                    div_values.append(val)
                    valid_horizons.append(h)

        if len(valid_horizons) > 0:
            ax.plot(valid_horizons, div_values,
                   color=style['color'], marker=style['marker'],
                   linestyle=style['linestyle'], label=style['label'],
                   linewidth=2, markersize=6)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    if show_training_split:
        ax.axvline(x=TRAINING_MAX_HORIZON, color='gray', linestyle='--',
                  linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Horizon (steps)', fontweight='bold')
    ax.set_ylabel('Divergence Rate', fontweight='bold')
    ax.set_title(f'{SYSTEM_NAMES[system]} - Divergence Rate vs Horizon', fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='best', framealpha=0.9, ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3, which='both')


# ==============================================================================
# Main Plotting Function
# ==============================================================================

def create_all_plots(results_dir, output_dir, systems=None):
    """Create all plots for specified systems"""
    if systems is None:
        systems = ['duffing', 'vanderpol', 'lorenz', 'double_pendulum']

    os.makedirs(output_dir, exist_ok=True)

    for system in systems:
        print(f"\nProcessing {system}...")

        try:
            df = load_results(results_dir, system)
            print(f"  Loaded {len(df)} models")
        except Exception as e:
            print(f"  Error loading {system}: {e}")
            continue

        # ======================================================================
        # Figure 1: Main Metrics vs Horizon (3-panel subplot)
        # ======================================================================
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        plot_nrmse_vs_horizon(axes[0], df, system)
        plot_chamfer_vs_horizon(axes[1], df, system)
        plot_divergence_vs_horizon(axes[2], df, system)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{system}_metrics_vs_horizon.png'), dpi=300)
        plt.close()
        print(f"  Saved: {system}_metrics_vs_horizon.png")

        # ======================================================================
        # Figure 2: Lyapunov Time (2-panel: curve + bars)
        # ======================================================================
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        plot_lyapunov_time(axes[0], df, system)
        plot_lyapunov_time_bars(axes[1], df, system)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{system}_lyapunov_time.png'), dpi=300)
        plt.close()
        print(f"  Saved: {system}_lyapunov_time.png")

        # ======================================================================
        # Figure 3: Spectral Radius
        # ======================================================================
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plot_spectral_radius(ax, df, system)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{system}_spectral_radius.png'), dpi=300)
        plt.close()
        print(f"  Saved: {system}_spectral_radius.png")

    print(f"\n✓ All plots saved to {output_dir}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Enhanced plotting script for comparison results')
    parser.add_argument('--results_dir', type=str, default='final_results_0',
                       help='Directory containing results (default: final_results_0)')
    parser.add_argument('--output_dir', type=str, default='plots_enhanced',
                       help='Output directory for plots (default: plots_enhanced)')
    parser.add_argument('--systems', type=str, nargs='+',
                       default=['duffing', 'vanderpol', 'lorenz', 'double_pendulum'],
                       help='Systems to plot (default: all)')
    parser.add_argument('--heatmap', action='store_true',
                       help='Also create cross-system NRMSE heatmap')

    args = parser.parse_args()

    print("="*70)
    print("Enhanced Comparison Plotting")
    print("="*70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Systems: {args.systems}")
    print("="*70)

    # Create individual system plots
    create_all_plots(args.results_dir, args.output_dir, args.systems)

    # Create cross-system heatmap if requested
    if args.heatmap:
        print("\n" + "="*70)
        print("Creating Cross-System NRMSE Heatmap")
        print("="*70)
        for horizon in [100, 500, 1000]:
            create_cross_system_heatmap(args.results_dir, args.output_dir, args.systems, horizon)

    print("\n" + "="*70)
    print("✓ Enhanced plotting complete!")
    print("="*70)


if __name__ == "__main__":
    main()
