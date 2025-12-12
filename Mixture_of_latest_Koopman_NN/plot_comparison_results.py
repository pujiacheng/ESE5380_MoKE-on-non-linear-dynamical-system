"""
Comprehensive Plotting Script for Model Comparison Results

This script creates publication-quality plots from comparison_results.csv files
for all 4 dynamical systems: duffing, vanderpol, lorenz, double_pendulum.

Plots created:
1. NRMSE vs Horizon (with training/extrapolation regions)
2. Chamfer Distance vs Horizon
3. Divergence Rate vs Horizon
4. Model comparison bar charts
5. Per-dimension breakdowns (for multi-dimensional systems)

Usage:
    python plot_comparison_results.py --results_dir final_results_0 --output_dir plots
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

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
    'VAR (ARIMA)': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'VAR (ARIMA)'},
    'eDMD': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-', 'label': 'eDMD'},
    'KAE Baseline': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-', 'label': 'KAE Baseline'},
    'Advanced KAE (1 Expert)': {'color': '#d62728', 'marker': 'v', 'linestyle': '-', 'label': 'Advanced KAE'},
    'MoE (2 Experts)': {'color': '#9467bd', 'marker': 'D', 'linestyle': '-', 'label': 'MoE (2 Experts)'},
    'MoE (3 Experts)': {'color': '#8c564b', 'marker': 'p', 'linestyle': '-', 'label': 'MoE (3 Experts)'},
    'MoE (4 Experts)': {'color': '#e377c2', 'marker': 'h', 'linestyle': '-', 'label': 'MoE (4 Experts)'},
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
    # Find the CSV file (handle different timestamp directories)
    system_dir = os.path.join(results_dir, system)
    if not os.path.exists(system_dir):
        raise FileNotFoundError(f"System directory not found: {system_dir}")
    
    # Find the most recent timestamp directory
    subdirs = [d for d in os.listdir(system_dir) if os.path.isdir(os.path.join(system_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {system_dir}")
    
    # Use the first subdirectory (assuming there's only one)
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
    if replace_with is None:
        # Replace inf with a large value for plotting (will be shown as special marker)
        replace_with = np.nanmax([v for v in values if not np.isinf(v) and not np.isnan(v)]) * 10
    return np.where(np.isinf(values), replace_with, values)


# ==============================================================================
# Plotting Functions
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
    
    # Add vertical line to separate training from extrapolation
    if show_training_split:
        ax.axvline(x=TRAINING_MAX_HORIZON, color='gray', linestyle='--', 
                  linewidth=1.5, alpha=0.7, label='Training/Extrapolation')
        ax.text(TRAINING_MAX_HORIZON + 20, ax.get_ylim()[1] * 0.95, 
               'Extrapolation', fontsize=9, alpha=0.7, rotation=90)
    
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
                    # Mark inf with a special value for visualization
                    chamfer_values.append(np.nan)
                    valid_horizons.append(h)
                elif not np.isnan(val):
                    chamfer_values.append(val)
                    valid_horizons.append(h)
        
        if len(valid_horizons) > 0:
            # Plot regular values
            mask = ~np.isnan(chamfer_values)
            if np.any(mask):
                ax.plot(np.array(valid_horizons)[mask], np.array(chamfer_values)[mask],
                       color=style['color'], marker=style['marker'],
                       linestyle=style['linestyle'], label=style['label'],
                       linewidth=2, markersize=6)
            
            # Mark inf values with special marker
            if has_inf:
                inf_horizons = [h for h, v in zip(valid_horizons, chamfer_values) if np.isnan(v)]
                if inf_horizons:
                    ax.scatter(inf_horizons, [ax.get_ylim()[1] * 0.95] * len(inf_horizons),
                             color=style['color'], marker='x', s=100, linewidths=3,
                             zorder=10, label=f"{style['label']} (diverged)" if not has_inf else None)
    
    # Add vertical line to separate training from extrapolation
    if show_training_split:
        ax.axvline(x=TRAINING_MAX_HORIZON, color='gray', linestyle='--',
                  linewidth=1.5, alpha=0.7)
        ax.text(TRAINING_MAX_HORIZON + 20, ax.get_ylim()[1] * 0.95,
               'Extrapolation', fontsize=9, alpha=0.7, rotation=90)
    
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
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add vertical line to separate training from extrapolation
    if show_training_split:
        ax.axvline(x=TRAINING_MAX_HORIZON, color='gray', linestyle='--',
                  linewidth=1.5, alpha=0.7)
        ax.text(TRAINING_MAX_HORIZON + 20, ax.get_ylim()[1] * 0.95,
               'Extrapolation', fontsize=9, alpha=0.7, rotation=90)
    
    ax.set_xlabel('Horizon (steps)', fontweight='bold')
    ax.set_ylabel('Divergence Rate', fontweight='bold')
    ax.set_title(f'{SYSTEM_NAMES[system]} - Divergence Rate vs Horizon', fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='best', framealpha=0.9, ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3, which='both')


def plot_model_comparison_bars(ax, df, system, metric_col, title, ylabel, log_scale=False):
    """Create bar chart comparing models on a specific metric"""
    models = df['model'].values
    values = df[metric_col].values
    
    # Handle inf values
    has_inf = np.isinf(values)
    values_plot = np.where(has_inf, np.nanmax(values[~has_inf]) * 1.5, values)
    
    # Get colors for each model
    colors = [MODEL_STYLE.get(m, {}).get('color', 'gray') for m in models]
    
    bars = ax.bar(range(len(models)), values_plot, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Mark inf values
    for i, (bar, is_inf) in enumerate(zip(bars, has_inf)):
        if is_inf:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                   '∞', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_STYLE.get(m, {}).get('label', m) for m in models],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    if log_scale:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')


def plot_per_dimension_nrmse(ax, df, system, horizon):
    """Plot per-dimension NRMSE for a specific horizon"""
    n_dims = df['n_x'].iloc[0]
    dim_labels = STATE_LABELS.get(system, [f'Dim {i}' for i in range(n_dims)])
    
    x = np.arange(len(df))
    width = 0.8 / n_dims
    
    for dim in range(n_dims):
        col = f'nrmse_{horizon}step_dim{dim}'
        if col in df.columns:
            values = df[col].values
            offset = (dim - n_dims/2 + 0.5) * width
            ax.bar(x + offset, values, width, 
                  label=dim_labels[dim] if dim < len(dim_labels) else f'Dim {dim}',
                  alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_STYLE.get(m, {}).get('label', m) for m in df['model'].values],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(f'NRMSE at {horizon} steps', fontweight='bold')
    ax.set_title(f'{SYSTEM_NAMES[system]} - Per-Dimension NRMSE (Horizon={horizon})', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')


def plot_divergence_summary(ax, df, system):
    """Plot summary of diverged trajectories"""
    models = df['model'].values
    n_total = df['n_total'].values
    n_diverged = df['n_diverged'].values
    n_valid = df['n_valid'].values
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, n_valid, width, label='Valid', color='green', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, n_diverged, width, label='Diverged', color='red', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_STYLE.get(m, {}).get('label', m) for m in models],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Number of Trajectories', fontweight='bold')
    ax.set_title(f'{SYSTEM_NAMES[system]} - Trajectory Stability', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')


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
        # Figure 1: Main Metrics vs Horizon (3 subplots)
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
        # Figure 2: Model Comparison at Key Horizons
        # ======================================================================
        key_horizons = [100, 500, 1000]  # Training, mid extrapolation, far extrapolation
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, h in enumerate(key_horizons):
            col = f'nrmse_{h}step'
            if col in df.columns:
                plot_model_comparison_bars(axes[idx], df, system, col,
                                         f'NRMSE at {h} steps',
                                         'NRMSE', log_scale=True)
            else:
                axes[idx].text(0.5, 0.5, f'No data for {h} steps',
                             ha='center', va='center', transform=axes[idx].transAxes)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{system}_nrmse_comparison.png'), dpi=300)
        plt.close()
        print(f"  Saved: {system}_nrmse_comparison.png")
        
        # ======================================================================
        # Figure 3: Per-Dimension Breakdown (for multi-dimensional systems)
        # ======================================================================
        n_dims = df['n_x'].iloc[0]
        if n_dims > 2:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            key_horizons = [100, 500, 1000]
            
            for idx, h in enumerate(key_horizons):
                plot_per_dimension_nrmse(axes[idx], df, system, h)
            
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{system}_per_dimension_nrmse.png'), dpi=300)
            plt.close()
            print(f"  Saved: {system}_per_dimension_nrmse.png")
        
        # ======================================================================
        # Figure 4: Divergence Summary
        # ======================================================================
        if 'n_diverged' in df.columns:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            plot_divergence_summary(ax, df, system)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{system}_divergence_summary.png'), dpi=300)
            plt.close()
            print(f"  Saved: {system}_divergence_summary.png")
        
        # ======================================================================
        # Figure 5: Chamfer Distance Comparison at Key Horizons
        # ======================================================================
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        key_horizons = [100, 500, 1000]
        
        for idx, h in enumerate(key_horizons):
            col = f'chamfer_{h}step'
            if col in df.columns:
                plot_model_comparison_bars(axes[idx], df, system, col,
                                         f'Chamfer Distance at {h} steps',
                                         'Chamfer Distance', log_scale=True)
            else:
                axes[idx].text(0.5, 0.5, f'No data for {h} steps',
                             ha='center', va='center', transform=axes[idx].transAxes)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{system}_chamfer_comparison.png'), dpi=300)
        plt.close()
        print(f"  Saved: {system}_chamfer_comparison.png")
        
        # ======================================================================
        # Figure 6: One-step MSE and Reconstruction Error
        # ======================================================================
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if 'one_step_mse' in df.columns:
            plot_model_comparison_bars(axes[0], df, system, 'one_step_mse',
                                     'One-Step MSE', 'MSE', log_scale=True)
        
        if 'reconstruction_error' in df.columns:
            plot_model_comparison_bars(axes[1], df, system, 'reconstruction_error',
                                     'Reconstruction Error', 'MSE', log_scale=True)
        
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{system}_short_term_metrics.png'), dpi=300)
        plt.close()
        print(f"  Saved: {system}_short_term_metrics.png")
    
    print(f"\n✓ All plots saved to {output_dir}")


# ==============================================================================
# Combined Multi-System Plots
# ==============================================================================

def create_combined_plots(results_dir, output_dir, systems=None):
    """Create combined plots comparing all systems"""
    if systems is None:
        systems = ['duffing', 'vanderpol', 'lorenz', 'double_pendulum']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all data
    all_data = {}
    for system in systems:
        try:
            all_data[system] = load_results(results_dir, system)
        except Exception as e:
            print(f"Error loading {system}: {e}")
            continue
    
    if not all_data:
        print("No data loaded!")
        return
    
    # ======================================================================
    # Combined NRMSE at key horizons across all systems
    # ======================================================================
    key_horizons = [100, 500, 1000]
    
    for horizon in key_horizons:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, system in enumerate(systems):
            if system not in all_data:
                axes[idx].axis('off')
                continue
            
            df = all_data[system]
            col = f'nrmse_{horizon}step'
            
            if col in df.columns:
                models = df['model'].values
                values = df[col].values
                colors = [MODEL_STYLE.get(m, {}).get('color', 'gray') for m in models]
                
                bars = axes[idx].bar(range(len(models)), values, color=colors, 
                                    alpha=0.7, edgecolor='black', linewidth=1)
                axes[idx].set_xticks(range(len(models)))
                axes[idx].set_xticklabels([MODEL_STYLE.get(m, {}).get('label', m) for m in models],
                                         rotation=45, ha='right', fontsize=8)
                axes[idx].set_ylabel('NRMSE', fontweight='bold')
                axes[idx].set_title(f'{SYSTEM_NAMES[system]}', fontweight='bold')
                axes[idx].set_yscale('log')
                axes[idx].grid(True, alpha=0.3, axis='y')
            else:
                axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center',
                             transform=axes[idx].transAxes)
        
        fig.suptitle(f'NRMSE Comparison at {horizon} Steps (All Systems)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'combined_nrmse_{horizon}step.png'), dpi=300)
        plt.close()
        print(f"  Saved: combined_nrmse_{horizon}step.png")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Plot comparison results')
    parser.add_argument('--results_dir', type=str, default='final_results_0',
                       help='Directory containing results (default: final_results_0)')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Output directory for plots (default: plots)')
    parser.add_argument('--systems', type=str, nargs='+',
                       default=['duffing', 'vanderpol', 'lorenz', 'double_pendulum'],
                       help='Systems to plot (default: all)')
    parser.add_argument('--combined', action='store_true',
                       help='Also create combined multi-system plots')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Creating Comparison Plots")
    print("="*70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Systems: {args.systems}")
    print("="*70)
    
    # Create individual system plots
    create_all_plots(args.results_dir, args.output_dir, args.systems)
    
    # Create combined plots if requested
    if args.combined:
        print("\n" + "="*70)
        print("Creating Combined Multi-System Plots")
        print("="*70)
        combined_dir = os.path.join(args.output_dir, 'combined')
        create_combined_plots(args.results_dir, combined_dir, args.systems)
    
    print("\n" + "="*70)
    print("✓ Plotting complete!")
    print("="*70)


if __name__ == "__main__":
    main()

