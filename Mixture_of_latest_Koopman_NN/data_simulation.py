import numpy as np


# ------------------------------------------------------
# Duffing ODE (conservative double-well)
#   x'' = x - x^3
# State vector y = [x, xdot]
# ------------------------------------------------------
def duffing_rhs(t, y):
    x, xdot = y
    xddot = x - x**3
    return np.array([xdot, xddot])


# ------------------------------------------------------
# 4th-order Runge-Kutta integrator
# ------------------------------------------------------
def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = f(t + dt, y + dt*k3)
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


# ------------------------------------------------------
# Simulate one trajectory
# ------------------------------------------------------
def simulate_duffing(x0, xdot0, T=10.0, dt=0.01, noise_std=0.0):
    steps = int(T/dt) + 1
    t = np.linspace(0, T, steps)

    traj = np.zeros((steps, 2))
    y = np.array([x0, xdot0], dtype=float)

    for i, ti in enumerate(t):
        traj[i] = y
        y = rk4_step(duffing_rhs, ti, y, dt)

        if noise_std > 0:
            y += noise_std * np.random.randn(2)

    return t, traj


# ------------------------------------------------------
# Generate many trajectories (dataset)
# ------------------------------------------------------
def generate_duffing_dataset(n_traj=100, T=10.0, dt=0.01, noise_std=0.0):
    trajs = []
    initial_conditions = np.random.uniform(-2, 2, size=(n_traj, 2))

    for x0, xdot0 in initial_conditions:
        t, traj = simulate_duffing(x0, xdot0, T=T, dt=dt, noise_std=noise_std)
        trajs.append(traj)

    trajs = np.stack(trajs)   # shape (n_traj, steps, 2)
    return t, trajs


# Example usage:
if __name__ == "__main__":
    t, trajs = generate_duffing_dataset(n_traj=5)

    print("Time vector:", t.shape)
    print("Trajectories:", trajs.shape)
    print("One example state series:")
    print(trajs[0][:5])

