"""
Data Simulation for Dynamical Systems

Supports four types of systems:
1. Duffing oscillator (bistable potential)
2. Van der Pol oscillator
3. Lorenz attractor
4. Double pendulum

Generates CSV output with trajectory number, time step, state variables, and derivatives.
"""

import numpy as np
import pandas as pd
import argparse


# ============================================================================
# 4th-order Runge-Kutta integrator
# ============================================================================
def rk4_step(f, t, y, dt):
    """4th-order Runge-Kutta integration step"""
    k1 = f(t, y)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = f(t + dt, y + dt*k3)
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


# ============================================================================
# System 1: Duffing Oscillator (bistable potential)
# x'' = x - x^3
# State: [x, xdot]
# ============================================================================
def duffing_rhs(t, y, alpha=1.0, beta=1.0, delta=0.0, gamma=0.0, omega=0.0):
    """
    Duffing oscillator: x'' = -delta*x' - alpha*x - beta*x^3 + gamma*cos(omega*t)
    Default: x'' = x - x^3 (conservative, bistable)
    """
    x, xdot = y
    xddot = -delta*xdot - alpha*x - beta*(x**3) + gamma*np.cos(omega*t)
    return np.array([xdot, xddot])


# ============================================================================
# System 2: Van der Pol Oscillator
# x'' = mu*(1 - x^2)*x' - x
# State: [x, xdot]
# ============================================================================
def vanderpol_rhs(t, y, mu=1.0):
    """
    Van der Pol oscillator: x'' = mu*(1 - x^2)*x' - x
    """
    x, xdot = y
    xddot = mu*(1 - x**2)*xdot - x
    return np.array([xdot, xddot])


# ============================================================================
# System 3: Lorenz Attractor
# dx/dt = sigma*(y - x)
# dy/dt = x*(rho - z) - y
# dz/dt = x*y - beta*z
# State: [x, y, z]
# ============================================================================
def lorenz_rhs(t, y, sigma=10.0, rho=28.0, beta=8/3):
    """
    Lorenz attractor system
    """
    x, y, z = y
    dxdt = sigma*(y - x)
    dydt = x*(rho - z) - y
    dzdt = x*y - beta*z
    return np.array([dxdt, dydt, dzdt])


# ============================================================================
# System 4: Double Pendulum
# State: [theta1, theta1_dot, theta2, theta2_dot]
# ============================================================================
def double_pendulum_rhs(t, y, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
    """
    Double pendulum system
    theta1, theta1_dot: angle and angular velocity of first pendulum
    theta2, theta2_dot: angle and angular velocity of second pendulum
    """
    theta1, theta1_dot, theta2, theta2_dot = y
    
    # Intermediate calculations
    delta = theta2 - theta1
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)
    sin_theta1 = np.sin(theta1)
    sin_theta2 = np.sin(theta2)
    
    # Denominators
    denom1 = (m1 + m2)*L1 - m2*L1*cos_delta**2
    denom2 = (L2/L1)*denom1
    
    # Angular accelerations
    theta1_ddot = (m2*L1*theta1_dot**2*sin_delta*cos_delta +
                   m2*g*sin_theta2*cos_delta +
                   m2*L2*theta2_dot**2*sin_delta -
                   (m1 + m2)*g*sin_theta1) / denom1
    
    theta2_ddot = (-m2*L2*theta2_dot**2*sin_delta*cos_delta +
                   (m1 + m2)*g*sin_theta1*cos_delta -
                   (m1 + m2)*L1*theta1_dot**2*sin_delta -
                   (m1 + m2)*g*sin_theta2) / denom2
    
    return np.array([theta1_dot, theta1_ddot, theta2_dot, theta2_ddot])


# ============================================================================
# Unified simulation function
# ============================================================================
def simulate_system(system_type, initial_conditions, T=10.0, dt=0.01, 
                    noise_std=0.0, **system_params):
    """
    Simulate a dynamical system
    
    Args:
        system_type: 'duffing', 'vanderpol', 'lorenz', or 'double_pendulum'
        initial_conditions: array of initial conditions (shape depends on system)
        T: total simulation time
        dt: time step
        noise_std: standard deviation of noise to add
        **system_params: system-specific parameters
    
    Returns:
        t: time array
        traj: trajectory array (n_steps, n_dim)
        derivatives: derivative array (n_steps, n_dim) if available
    """
    # Select RHS function
    if system_type == 'duffing':
        rhs_func = lambda t, y: duffing_rhs(t, y, **system_params)
        n_dim = 2
    elif system_type == 'vanderpol':
        rhs_func = lambda t, y: vanderpol_rhs(t, y, **system_params)
        n_dim = 2
    elif system_type == 'lorenz':
        rhs_func = lambda t, y: lorenz_rhs(t, y, **system_params)
        n_dim = 3
    elif system_type == 'double_pendulum':
        rhs_func = lambda t, y: double_pendulum_rhs(t, y, **system_params)
        n_dim = 4
    else:
        raise ValueError(f"Unknown system type: {system_type}")
    
    # Initialize
    steps = int(T/dt) + 1
    t = np.linspace(0, T, steps)
    traj = np.zeros((steps, n_dim))
    derivatives = np.zeros((steps, n_dim))
    
    y = np.array(initial_conditions, dtype=float)
    
    # Simulate
    for i, ti in enumerate(t):
        traj[i] = y
        deriv = rhs_func(ti, y)
        derivatives[i] = deriv
        
        y = rk4_step(rhs_func, ti, y, dt)
        
        if noise_std > 0:
            y += noise_std * np.random.randn(n_dim)
    
    return t, traj, derivatives


# ============================================================================
# Generate dataset with multiple trajectories
# ============================================================================
def generate_dataset(system_type, n_traj=100, T=10.0, dt=0.01, noise_std=0.0,
                     ic_range=None, **system_params):
    """
    Generate multiple trajectories for a system
    
    Args:
        system_type: 'duffing', 'vanderpol', 'lorenz', or 'double_pendulum'
        n_traj: number of trajectories
        T: simulation time per trajectory
        dt: time step
        noise_std: noise standard deviation
        ic_range: dict with ranges for initial conditions (defaults provided)
        **system_params: system-specific parameters
    
    Returns:
        t: time array
        trajs: array of trajectories (n_traj, n_steps, n_dim)
        all_derivatives: array of derivatives (n_traj, n_steps, n_dim)
    """
    # Default initial condition ranges
    if ic_range is None:
        if system_type == 'duffing':
            ic_range = {'x': (-2, 2), 'xdot': (-2, 2)}
        elif system_type == 'vanderpol':
            ic_range = {'x': (-3, 3), 'xdot': (-3, 3)}
        elif system_type == 'lorenz':
            ic_range = {'x': (-20, 20), 'y': (-20, 20), 'z': (0, 50)}
        elif system_type == 'double_pendulum':
            ic_range = {'theta1': (-np.pi, np.pi), 'theta1_dot': (-2, 2),
                       'theta2': (-np.pi, np.pi), 'theta2_dot': (-2, 2)}
    
    trajs = []
    all_derivatives = []
    
    for _ in range(n_traj):
        # Generate random initial conditions
        if system_type == 'duffing':
            ic = [np.random.uniform(*ic_range['x']),
                  np.random.uniform(*ic_range['xdot'])]
        elif system_type == 'vanderpol':
            ic = [np.random.uniform(*ic_range['x']),
                  np.random.uniform(*ic_range['xdot'])]
        elif system_type == 'lorenz':
            ic = [np.random.uniform(*ic_range['x']),
                  np.random.uniform(*ic_range['y']),
                  np.random.uniform(*ic_range['z'])]
        elif system_type == 'double_pendulum':
            ic = [np.random.uniform(*ic_range['theta1']),
                  np.random.uniform(*ic_range['theta1_dot']),
                  np.random.uniform(*ic_range['theta2']),
                  np.random.uniform(*ic_range['theta2_dot'])]
        
        t, traj, derivs = simulate_system(system_type, ic, T=T, dt=dt,
                                         noise_std=noise_std, **system_params)
        trajs.append(traj)
        all_derivatives.append(derivs)
    
    trajs = np.stack(trajs)
    all_derivatives = np.stack(all_derivatives)
    
    return t, trajs, all_derivatives


# ============================================================================
# Export to CSV
# ============================================================================
def export_to_csv(t, trajs, derivatives, system_type, output_path):
    """
    Export trajectories to CSV with trajectory number, time, states, and derivatives
    
    Args:
        t: time array
        trajs: trajectory array (n_traj, n_steps, n_dim)
        derivatives: derivative array (n_traj, n_steps, n_dim)
        system_type: system type string
        output_path: path to save CSV
    """
    n_traj, n_steps, n_dim = trajs.shape
    
    # Define column names based on system type
    if system_type == 'duffing':
        state_names = ['x', 'xdot']
        deriv_names = ['xdot', 'xddot']
    elif system_type == 'vanderpol':
        state_names = ['x', 'xdot']
        deriv_names = ['xdot', 'xddot']
    elif system_type == 'lorenz':
        state_names = ['x', 'y', 'z']
        deriv_names = ['xdot', 'ydot', 'zdot']
    elif system_type == 'double_pendulum':
        state_names = ['theta1', 'theta1_dot', 'theta2', 'theta2_dot']
        deriv_names = ['theta1_dot', 'theta1_ddot', 'theta2_dot', 'theta2_ddot']
    else:
        state_names = [f'state_{i}' for i in range(n_dim)]
        deriv_names = [f'deriv_{i}' for i in range(n_dim)]
    
    # Build DataFrame
    data_list = []
    for traj_id in range(n_traj):
        for step in range(n_steps):
            row = {
                'traj_id': traj_id,
                'time': t[step],
                'time_step': step
            }
            # Add state variables
            for i, name in enumerate(state_names):
                row[name] = trajs[traj_id, step, i]
            # Add derivatives
            for i, name in enumerate(deriv_names):
                row[name] = derivatives[traj_id, step, i]
            data_list.append(row)
    
    df = pd.DataFrame(data_list)
    df.to_csv(output_path, index=False)
    print(f"Exported {n_traj} trajectories to {output_path}")
    print(f"CSV shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


# ============================================================================
# Main function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Simulate dynamical systems and export to CSV')
    parser.add_argument('--system', type=str, required=True,
                       choices=['duffing', 'vanderpol', 'lorenz', 'double_pendulum'],
                       help='System type to simulate')
    parser.add_argument('--n_traj', type=int, default=100,
                       help='Number of trajectories (default: 100)')
    parser.add_argument('--T', type=float, default=10.0,
                       help='Simulation time per trajectory (default: 10.0)')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step (default: 0.01)')
    parser.add_argument('--noise_std', type=float, default=0.0,
                       help='Noise standard deviation (default: 0.0)')
    parser.add_argument('--output', type=str, default='simulated_data.csv',
                       help='Output CSV file path (default: simulated_data.csv)')
    
    # System-specific parameters
    # Duffing
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Duffing: alpha parameter (default: 1.0)')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Duffing: beta parameter (default: 1.0)')
    parser.add_argument('--delta', type=float, default=0.0,
                       help='Duffing: damping parameter (default: 0.0)')
    parser.add_argument('--gamma', type=float, default=0.0,
                       help='Duffing: forcing amplitude (default: 0.0)')
    parser.add_argument('--omega', type=float, default=0.0,
                       help='Duffing: forcing frequency (default: 0.0)')
    
    # Van der Pol
    parser.add_argument('--mu', type=float, default=1.0,
                       help='Van der Pol: mu parameter (default: 1.0)')
    
    # Lorenz
    parser.add_argument('--sigma', type=float, default=10.0,
                       help='Lorenz: sigma parameter (default: 10.0)')
    parser.add_argument('--rho', type=float, default=28.0,
                       help='Lorenz: rho parameter (default: 28.0)')
    parser.add_argument('--beta_lorenz', type=float, default=8/3,
                       help='Lorenz: beta parameter (default: 8/3)')
    
    # Double pendulum
    parser.add_argument('--L1', type=float, default=1.0,
                       help='Double pendulum: length of first pendulum (default: 1.0)')
    parser.add_argument('--L2', type=float, default=1.0,
                       help='Double pendulum: length of second pendulum (default: 1.0)')
    parser.add_argument('--m1', type=float, default=1.0,
                       help='Double pendulum: mass of first pendulum (default: 1.0)')
    parser.add_argument('--m2', type=float, default=1.0,
                       help='Double pendulum: mass of second pendulum (default: 1.0)')
    parser.add_argument('--g', type=float, default=9.81,
                       help='Double pendulum: gravitational constant (default: 9.81)')
    
    args = parser.parse_args()
    
    # Prepare system parameters
    if args.system == 'duffing':
        system_params = {
            'alpha': args.alpha,
            'beta': args.beta,
            'delta': args.delta,
            'gamma': args.gamma,
            'omega': args.omega
        }
    elif args.system == 'vanderpol':
        system_params = {'mu': args.mu}
    elif args.system == 'lorenz':
        system_params = {
            'sigma': args.sigma,
            'rho': args.rho,
            'beta': args.beta_lorenz
        }
    elif args.system == 'double_pendulum':
        system_params = {
            'L1': args.L1,
            'L2': args.L2,
            'm1': args.m1,
            'm2': args.m2,
            'g': args.g
        }
    
    # Generate data
    print(f"Generating {args.n_traj} trajectories for {args.system} system...")
    t, trajs, derivatives = generate_dataset(
        system_type=args.system,
        n_traj=args.n_traj,
        T=args.T,
        dt=args.dt,
        noise_std=args.noise_std,
        **system_params
    )
    
    print(f"Generated trajectories shape: {trajs.shape}")
    print(f"Time steps: {len(t)}")
    
    # Export to CSV
    df = export_to_csv(t, trajs, derivatives, args.system, args.output)
    
    print(f"\nSimulation complete!")
    print(f"System: {args.system}")
    print(f"Parameters: {system_params}")
    print(f"Output saved to: {args.output}")


# ============================================================================
# Backward compatibility functions
# ============================================================================
def simulate_duffing(x0, xdot0, T=10.0, dt=0.01, noise_std=0.0):
    """Backward compatibility wrapper for Duffing simulation"""
    t, traj, _ = simulate_system('duffing', [x0, xdot0], T=T, dt=dt, noise_std=noise_std)
    return t, traj


def generate_duffing_dataset(n_traj=100, T=10.0, dt=0.01, noise_std=0.0):
    """Backward compatibility wrapper for Duffing dataset generation"""
    t, trajs, _ = generate_dataset('duffing', n_traj=n_traj, T=T, dt=dt, noise_std=noise_std)
    return t, trajs


if __name__ == "__main__":
    main()
