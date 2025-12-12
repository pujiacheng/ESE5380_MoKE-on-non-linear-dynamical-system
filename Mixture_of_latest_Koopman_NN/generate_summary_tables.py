"""
Generate Summary Tables from Comparison Results

Creates LaTeX and CSV summary tables for paper inclusion.

Usage:
    python generate_summary_tables.py --results_dir final_results_0 --output_dir tables
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


def create_summary_table(results_dir, output_dir, systems=None):
    """Create summary table with key metrics"""
    if systems is None:
        systems = ['duffing', 'vanderpol', 'lorenz', 'double_pendulum']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all data
    all_rows = []
    
    for system in systems:
        try:
            df = load_results(results_dir, system)
            
            for _, row in df.iterrows():
                model = MODEL_SHORT_NAMES.get(row['model'], row['model'])
                
                # Extract key metrics
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
                    'Diverged': f"{int(row.get('n_diverged', 0))}/{int(row.get('n_total', 0))}",
                }
                
                if 'spectral_radius' in row:
                    metrics['Spectral Radius'] = format_value(row['spectral_radius'], '.4f')
                
                all_rows.append(metrics)
        except Exception as e:
            print(f"Error processing {system}: {e}")
            continue
    
    # Create DataFrame
    summary_df = pd.DataFrame(all_rows)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'summary_table.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Save LaTeX table
    latex_path = os.path.join(output_dir, 'summary_table.tex')
    with open(latex_path, 'w') as f:
        f.write(summary_df.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.4f}" if not pd.isna(x) else "---"))
    print(f"Saved LaTeX: {latex_path}")
    
    return summary_df


def create_per_system_tables(results_dir, output_dir, systems=None):
    """Create detailed tables for each system"""
    if systems is None:
        systems = ['duffing', 'vanderpol', 'lorenz', 'double_pendulum']
    
    os.makedirs(output_dir, exist_ok=True)
    
    for system in systems:
        try:
            df = load_results(results_dir, system)
            
            # Select key columns
            key_cols = ['model', 'n_params', 'one_step_mse']
            
            # Add NRMSE columns
            for h in [1, 10, 20, 50, 100, 500, 1000]:
                col = f'nrmse_{h}step'
                if col in df.columns:
                    key_cols.append(col)
            
            # Add Chamfer columns
            for h in [100, 500, 1000]:
                col = f'chamfer_{h}step'
                if col in df.columns:
                    key_cols.append(col)
            
            # Add divergence info
            if 'n_diverged' in df.columns:
                key_cols.extend(['n_valid', 'n_total', 'n_diverged'])
            
            # Filter to existing columns
            key_cols = [c for c in key_cols if c in df.columns]
            
            # Create table
            table_df = df[key_cols].copy()
            table_df['model'] = table_df['model'].map(MODEL_SHORT_NAMES).fillna(table_df['model'])
            
            # Format numeric columns
            for col in table_df.columns:
                if col != 'model' and table_df[col].dtype in [np.float64, np.float32]:
                    table_df[col] = table_df[col].apply(lambda x: format_value(x, '.4f'))
            
            # Save
            csv_path = os.path.join(output_dir, f'{system}_detailed_table.csv')
            table_df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")
            
            # LaTeX
            latex_path = os.path.join(output_dir, f'{system}_detailed_table.tex')
            with open(latex_path, 'w') as f:
                f.write(table_df.to_latex(index=False, escape=False))
            print(f"Saved: {latex_path}")
            
        except Exception as e:
            print(f"Error processing {system}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='Generate summary tables')
    parser.add_argument('--results_dir', type=str, default='final_results_0',
                       help='Directory containing results')
    parser.add_argument('--output_dir', type=str, default='tables',
                       help='Output directory for tables')
    parser.add_argument('--systems', type=str, nargs='+',
                       default=['duffing', 'vanderpol', 'lorenz', 'double_pendulum'],
                       help='Systems to process')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Generating Summary Tables")
    print("="*70)
    
    # Create summary table
    create_summary_table(args.results_dir, args.output_dir, args.systems)
    
    # Create per-system detailed tables
    print("\n" + "="*70)
    print("Generating Per-System Detailed Tables")
    print("="*70)
    create_per_system_tables(args.results_dir, args.output_dir, args.systems)
    
    print("\n" + "="*70)
    print("✓ Table generation complete!")
    print("="*70)


if __name__ == "__main__":
    main()

