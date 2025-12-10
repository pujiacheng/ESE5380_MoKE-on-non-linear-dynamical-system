"""
Pre-generate datasets for all dynamical systems.

This script generates and saves trajectory data for all systems so that
training runs can reuse the same data without regenerating it.

Usage:
    python generate_data.py --n_traj 20000 --T 100 --dt 0.01 --output_dir generated_data

Output structure:
    generated_data/
    ├── duffing.npz
    ├── vanderpol.npz
    ├── lorenz.npz
    └── double_pendulum.npz

Each .npz file contains:
    - t: time array (n_timesteps,)
    - trajs: trajectory array (n_traj, n_timesteps, n_x)
    - config: system configuration dict
    - generation_params: parameters used to generate the data
"""

import os
import argparse
import numpy as np
import time
from datetime import datetime

from data_simulation import (
    generate_duffing_dataset,
    generate_vanderpol_dataset,
    generate_lorenz_dataset,
    generate_double_pendulum_dataset
)


SYSTEMS = {
    'duffing': {
        'generator': generate_duffing_dataset,
        'n_x': 2,
        'state_labels': ['x', 'xdot'],
        'name': 'Duffing Oscillator'
    },
    'vanderpol': {
        'generator': generate_vanderpol_dataset,
        'n_x': 2,
        'state_labels': ['x', 'xdot'],
        'name': 'Van der Pol Oscillator'
    },
    'lorenz': {
        'generator': generate_lorenz_dataset,
        'n_x': 3,
        'state_labels': ['x', 'y', 'z'],
        'name': 'Lorenz Attractor'
    },
    'double_pendulum': {
        'generator': generate_double_pendulum_dataset,
        'n_x': 4,
        'state_labels': ['theta1', 'theta1_dot', 'theta2', 'theta2_dot'],
        'name': 'Double Pendulum'
    }
}


def generate_and_save_dataset(system_name, n_traj, T, dt, noise_std, output_dir, seed=42):
    """
    Generate dataset for a system and save to disk.
    
    Args:
        system_name: Name of the system
        n_traj: Number of trajectories
        T: Simulation time
        dt: Time step
        noise_std: Noise standard deviation
        output_dir: Output directory
        seed: Random seed for reproducibility
    
    Returns:
        Path to saved file
    """
    if system_name not in SYSTEMS:
        raise ValueError(f"Unknown system: {system_name}. Available: {list(SYSTEMS.keys())}")
    
    system_info = SYSTEMS[system_name]
    
    print(f"\n{'='*60}")
    print(f"Generating {system_info['name']} data")
    print(f"{'='*60}")
    print(f"  Trajectories: {n_traj}")
    print(f"  T: {T}, dt: {dt}")
    print(f"  Noise std: {noise_std}")
    print(f"  Seed: {seed}")
    
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Generate data with timing
    start_time = time.time()
    t, trajs = system_info['generator'](n_traj=n_traj, T=T, dt=dt, noise_std=noise_std, show_progress=True)
    elapsed = time.time() - start_time
    
    print(f"  Generated shape: {trajs.shape}")
    print(f"  Time steps: {len(t)}")
    print(f"  Generation time: {elapsed:.1f}s ({elapsed/n_traj*1000:.2f}ms per trajectory)")
    
    # Prepare metadata
    generation_params = {
        'n_traj': n_traj,
        'T': T,
        'dt': dt,
        'noise_std': noise_std,
        'seed': seed,
        'generated_at': datetime.now().isoformat()
    }
    
    config = {
        'name': system_info['name'],
        'n_x': system_info['n_x'],
        'state_labels': system_info['state_labels']
    }
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{system_name}.npz")
    
    np.savez(
        output_path,
        t=t,
        trajs=trajs,
        config=config,
        generation_params=generation_params
    )
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved to: {output_path} ({file_size_mb:.1f} MB)")
    
    return output_path


def load_dataset(path):
    """
    Load a pre-generated dataset from disk.
    
    Args:
        path: Path to .npz file
    
    Returns:
        t: time array
        trajs: trajectory array
        config: system configuration dict
        generation_params: generation parameters dict
    """
    data = np.load(path, allow_pickle=True)
    
    t = data['t']
    trajs = data['trajs']
    config = data['config'].item()  # Convert from 0-d array to dict
    generation_params = data['generation_params'].item()
    
    return t, trajs, config, generation_params


def main():
    parser = argparse.ArgumentParser(description='Pre-generate datasets for all systems')
    parser.add_argument('--n_traj', type=int, default=100,
                       help='Number of trajectories per system (default: 20000)')
    parser.add_argument('--T', type=float, default=100.0,
                       help='Simulation time per trajectory (default: 100.0)')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step (default: 0.01)')
    parser.add_argument('--noise_std', type=float, default=0.0,
                       help='Noise standard deviation (default: 0.0)')
    parser.add_argument('--output_dir', type=str, default='generated_data',
                       help='Output directory (default: generated_data)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--systems', type=str, nargs='+', 
                       default=['duffing', 'vanderpol', 'lorenz', 'double_pendulum'],
                       help='Systems to generate (default: all)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Dataset Generation")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Systems: {args.systems}")
    print(f"Parameters: n_traj={args.n_traj}, T={args.T}, dt={args.dt}")
    
    saved_files = []
    system_times = []
    total_start = time.time()
    
    for idx, system in enumerate(args.systems):
        sys_start = time.time()
        path = generate_and_save_dataset(
            system_name=system,
            n_traj=args.n_traj,
            T=args.T,
            dt=args.dt,
            noise_std=args.noise_std,
            output_dir=args.output_dir,
            seed=args.seed
        )
        saved_files.append(path)
        sys_elapsed = time.time() - sys_start
        system_times.append(sys_elapsed)
        
        # Estimate remaining time
        if idx < len(args.systems) - 1:
            avg_time = sum(system_times) / len(system_times)
            remaining = avg_time * (len(args.systems) - idx - 1)
            print(f"  Estimated time remaining: {remaining/60:.1f} min")
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60)
    print(f"Generated {len(saved_files)} datasets:")
    for path, sys_time in zip(saved_files, system_times):
        print(f"  - {path} ({sys_time:.1f}s)")
    
    # Print total size and time
    total_size = sum(os.path.getsize(p) for p in saved_files)
    print(f"\nTotal size: {total_size / (1024*1024):.1f} MB")
    print(f"Total time: {total_elapsed/60:.1f} min ({total_elapsed:.1f}s)")
    
    print(f"\nTo use in training, run:")
    print(f"  python train_all_models.py --data_dir {args.output_dir} --system <system_name>")


if __name__ == "__main__":
    main()

