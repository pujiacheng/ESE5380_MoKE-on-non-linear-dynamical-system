"""
Enhanced Table Generation Script for Model Comparison Results

Creates LaTeX and CSV summary tables for paper inclusion.

New features:
1. Divergence summary table (replaces divergence_summary.png)
2. Horizon-specific NRMSE table
3. Best model per metric table
4. Statistical comparisons
5. Better inf/nan handling

Usage:
    python generate_summary_tables_enhanced.py --results_dir final_results_0 --output_dir tables_enhanced
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


SYSTEM_NAMES = {
    'duffing': 'Duffing',
    'vanderpol': 'Van der Pol',
    'lorenz': 'Lorenz',
    'double_pendulum': 'Double Pendulum'
}

MODEL_SHORT_NAMES = {
    'VAR (ARIMA)': 'VAR',
    'eDMD': 'eDMD',
    'KAE Baseline': 'KAE-B',
    'Advanced KAE (1 Expert)': 'KAE-A',
    'MoE (2 Experts)': 'MoE-2',
    'MoE (3 Experts)': 'MoE-3',
    'MoE (4 Experts)': 'MoE-4',
}


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


def format_value(val, fmt='.4f', nan_str='---', inf_str='∞'):
    """Format a value for table display"""
    if pd.isna(val):
        return nan_str
    if np.isinf(val):
        return inf_str
    return f"{val:{fmt}}"


def bold_best(values, lower_is_better=True):
    """Return index of best value (finite only)"""
    finite_vals = [(i, v) for i, v in enumerate(values) if np.isfinite(v)]
    if not finite_vals:
        return None

    if lower_is_better:
        best_idx = min(finite_vals, key=lambda x: x[1])[0]
    else:
        best_idx = max(finite_vals, key=lambda x: x[1])[0]

    return best_idx


# ==============================================================================
# Table 1: Divergence Summary (replaces plot)
# ==============================================================================

def create_divergence_table(results_dir, output_dir, systems=None):
    """Create divergence summary table showing trajectory stability"""
    if systems is None:
        systems = ['duffing', 'vanderpol', 'lorenz', 'double_pendulum']

    os.makedirs(output_dir, exist_ok=True)

    all_rows = []

    for system in systems:
        try:
            df = load_results(results_dir, system)

            for _, row in df.iterrows():
                model = MODEL_SHORT_NAMES.get(row['model'], row['model'])

                n_valid = int(row.get('n_valid', 0))
                n_diverged = int(row.get('n_diverged', 0))
                n_total = int(row.get('n_total', 0))
                success_rate = 100 * n_valid / n_total if n_total > 0 else 0

                all_rows.append({
                    'System': SYSTEM_NAMES[system],
                    'Model': model,
                    'Valid': n_valid,
                    'Diverged': n_diverged,
                    'Total': n_total,
                    'Success (%)': f'{success_rate:.1f}'
                })
        except Exception as e:
            print(f"Error processing {system}: {e}")
            continue

    # Create DataFrame
    div_df = pd.DataFrame(all_rows)

    # Save CSV
    csv_path = os.path.join(output_dir, 'divergence_summary_table.csv')
    div_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save LaTeX table
    latex_path = os.path.join(output_dir, 'divergence_summary_table.tex')
    with open(latex_path, 'w') as f:
        latex = div_df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.1f}" if not pd.isna(x) else "---")
        f.write(latex)
    print(f"Saved LaTeX: {latex_path}")

    return div_df


# ==============================================================================
# Table 2: Horizon-Specific NRMSE Table
# ==============================================================================

def create_horizon_nrmse_table(results_dir, output_dir, systems=None, horizons=None):
    """Create table showing NRMSE at specific horizons across all systems"""
    if systems is None:
        systems = ['duffing', 'vanderpol', 'lorenz', 'double_pendulum']
    if horizons is None:
        horizons = [100, 500, 1000]

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
        return None

    # Get unique models
    all_models = set()
    for df in all_data.values():
        all_models.update(df['model'].values)
    all_models = sorted(all_models)

    # Build table
    rows = []
    for model in all_models:
        row = {'Model': MODEL_SHORT_NAMES.get(model, model)}

        for system in systems:
            if system not in all_data:
                for h in horizons:
                    row[f'{SYSTEM_NAMES[system]} (H={h})'] = '---'
                continue

            df = all_data[system]
            if model not in df['model'].values:
                for h in horizons:
                    row[f'{SYSTEM_NAMES[system]} (H={h})'] = '---'
                continue

            model_row = df[df['model'] == model].iloc[0]
            for h in horizons:
                col = f'nrmse_{h}step'
                val = model_row.get(col, np.nan)
                row[f'{SYSTEM_NAMES[system]} (H={h})'] = format_value(val, '.3f')

        rows.append(row)

    table_df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(output_dir, 'horizon_nrmse_table.csv')
    table_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save LaTeX
    latex_path = os.path.join(output_dir, 'horizon_nrmse_table.tex')
    with open(latex_path, 'w') as f:
        latex = table_df.to_latex(index=False, escape=False)
        f.write(latex)
    print(f"Saved LaTeX: {latex_path}")

    return table_df


# ==============================================================================
# Table 3: Best Model Per Metric
# ==============================================================================

def create_best_models_table(results_dir, output_dir, systems=None):
    """Create table showing best model for each metric and system"""
    if systems is None:
        systems = ['duffing', 'vanderpol', 'lorenz', 'double_pendulum']

    os.makedirs(output_dir, exist_ok=True)

    metrics_to_check = [
        ('one_step_mse', 'One-Step MSE', True),
        ('nrmse_100step', 'NRMSE (H=100)', True),
        ('nrmse_500step', 'NRMSE (H=500)', True),
        ('nrmse_1000step', 'NRMSE (H=1000)', True),
        ('chamfer_100step', 'Chamfer (H=100)', True),
        ('chamfer_1000step', 'Chamfer (H=1000)', True),
        ('divergence_100step', 'Div Rate (H=100)', True),
        ('reconstruction_error', 'Recon Error', True),
    ]

    all_rows = []

    for metric_col, metric_name, lower_is_better in metrics_to_check:
        row = {'Metric': metric_name}

        for system in systems:
            try:
                df = load_results(results_dir, system)

                if metric_col not in df.columns:
                    row[SYSTEM_NAMES[system]] = '---'
                    continue

                values = df[metric_col].values
                best_idx = bold_best(values, lower_is_better)

                if best_idx is not None:
                    best_model = df.iloc[best_idx]['model']
                    best_val = values[best_idx]
                    row[SYSTEM_NAMES[system]] = f"{MODEL_SHORT_NAMES.get(best_model, best_model)} ({best_val:.3f})"
                else:
                    row[SYSTEM_NAMES[system]] = '---'

            except Exception as e:
                row[SYSTEM_NAMES[system]] = '---'
                print(f"Error processing {system}: {e}")

        all_rows.append(row)

    best_df = pd.DataFrame(all_rows)

    # Save CSV
    csv_path = os.path.join(output_dir, 'best_models_table.csv')
    best_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save LaTeX
    latex_path = os.path.join(output_dir, 'best_models_table.tex')
    with open(latex_path, 'w') as f:
        latex = best_df.to_latex(index=False, escape=False)
        f.write(latex)
    print(f"Saved LaTeX: {latex_path}")

    return best_df


# ==============================================================================
# Table 4: Enhanced Summary Table (from original)
# ==============================================================================

def create_summary_table(results_dir, output_dir, systems=None):
    """Create comprehensive summary table with key metrics"""
    if systems is None:
        systems = ['duffing', 'vanderpol', 'lorenz', 'double_pendulum']

    os.makedirs(output_dir, exist_ok=True)

    all_rows = []

    for system in systems:
        try:
            df = load_results(results_dir, system)

            for _, row in df.iterrows():
                model = MODEL_SHORT_NAMES.get(row['model'], row['model'])

                metrics = {
                    'System': SYSTEM_NAMES[system],
                    'Model': model,
                    'Params': int(row['n_params']) if not pd.isna(row['n_params']) else '---',
                    '1-step MSE': format_value(row.get('one_step_mse', np.nan), '.6f'),
                    'NRMSE-100': format_value(row.get('nrmse_100step', np.nan), '.4f'),
                    'NRMSE-500': format_value(row.get('nrmse_500step', np.nan), '.4f'),
                    'NRMSE-1000': format_value(row.get('nrmse_1000step', np.nan), '.4f'),
                    'Chamfer-100': format_value(row.get('chamfer_100step', np.nan), '.4f'),
                    'Chamfer-1000': format_value(row.get('chamfer_1000step', np.nan), '.4f'),
                    'Div-100': format_value(row.get('divergence_100step', np.nan), '.4f'),
                    'Diverged': f"{int(row.get('n_diverged', 0))}/{int(row.get('n_total', 0))}",
                }

                if 'spectral_radius' in row and not pd.isna(row['spectral_radius']):
                    metrics['ρ'] = format_value(row['spectral_radius'], '.4f')

                all_rows.append(metrics)
        except Exception as e:
            print(f"Error processing {system}: {e}")
            continue

    summary_df = pd.DataFrame(all_rows)

    # Save CSV
    csv_path = os.path.join(output_dir, 'summary_table.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save LaTeX
    latex_path = os.path.join(output_dir, 'summary_table.tex')
    with open(latex_path, 'w') as f:
        latex = summary_df.to_latex(index=False, escape=False)
        f.write(latex)
    print(f"Saved LaTeX: {latex_path}")

    return summary_df


# ==============================================================================
# Table 5: Spectral Radius Comparison
# ==============================================================================

def create_spectral_radius_table(results_dir, output_dir, systems=None):
    """Create table comparing spectral radius across systems"""
    if systems is None:
        systems = ['duffing', 'vanderpol', 'lorenz', 'double_pendulum']

    os.makedirs(output_dir, exist_ok=True)

    all_rows = []

    for system in systems:
        try:
            df = load_results(results_dir, system)

            if 'spectral_radius' not in df.columns:
                continue

            for _, row in df.iterrows():
                model = MODEL_SHORT_NAMES.get(row['model'], row['model'])
                spec_rad = row.get('spectral_radius', np.nan)

                if not pd.isna(spec_rad) and spec_rad > 0:
                    all_rows.append({
                        'System': SYSTEM_NAMES[system],
                        'Model': model,
                        'Spectral Radius (ρ)': format_value(spec_rad, '.4f'),
                        'Δ from 1.0': format_value(abs(spec_rad - 1.0), '.4f'),
                        'Status': 'Stable' if abs(spec_rad - 1.0) < 0.01 else 'Unstable' if spec_rad > 1.01 else 'Dissipative'
                    })
        except Exception as e:
            print(f"Error processing {system}: {e}")
            continue

    if not all_rows:
        print("No spectral radius data found")
        return None

    spec_df = pd.DataFrame(all_rows)

    # Save CSV
    csv_path = os.path.join(output_dir, 'spectral_radius_table.csv')
    spec_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save LaTeX
    latex_path = os.path.join(output_dir, 'spectral_radius_table.tex')
    with open(latex_path, 'w') as f:
        latex = spec_df.to_latex(index=False, escape=False)
        f.write(latex)
    print(f"Saved LaTeX: {latex_path}")

    return spec_df


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Enhanced table generation for comparison results')
    parser.add_argument('--results_dir', type=str, default='final_results_0',
                       help='Directory containing results')
    parser.add_argument('--output_dir', type=str, default='tables_enhanced',
                       help='Output directory for tables')
    parser.add_argument('--systems', type=str, nargs='+',
                       default=['duffing', 'vanderpol', 'lorenz', 'double_pendulum'],
                       help='Systems to process')

    args = parser.parse_args()

    print("="*70)
    print("Enhanced Table Generation")
    print("="*70)

    # Table 1: Summary table
    print("\n" + "="*70)
    print("Creating Summary Table")
    print("="*70)
    create_summary_table(args.results_dir, args.output_dir, args.systems)

    # Table 2: Divergence summary (replaces plot)
    print("\n" + "="*70)
    print("Creating Divergence Summary Table (replaces plot)")
    print("="*70)
    create_divergence_table(args.results_dir, args.output_dir, args.systems)

    # Table 3: Horizon-specific NRMSE
    print("\n" + "="*70)
    print("Creating Horizon-Specific NRMSE Table")
    print("="*70)
    create_horizon_nrmse_table(args.results_dir, args.output_dir, args.systems)

    # Table 4: Best models
    print("\n" + "="*70)
    print("Creating Best Models Table")
    print("="*70)
    create_best_models_table(args.results_dir, args.output_dir, args.systems)

    # Table 5: Spectral radius
    print("\n" + "="*70)
    print("Creating Spectral Radius Table")
    print("="*70)
    create_spectral_radius_table(args.results_dir, args.output_dir, args.systems)

    print("\n" + "="*70)
    print("✓ Enhanced table generation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
