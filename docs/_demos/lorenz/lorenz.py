#!/usr/bin/env python3
# ruff: noqa: D100 D103

import numpy as np
from tqdm import tqdm

import tecio

# Lorenz system parameters
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0


def lorenz(state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


# Time integration (RK4)
dt = 0.01
n_steps = 5000

traj = np.zeros((n_steps, 3))
state = np.array([1.0, 1.0, 1.0])

# Set consistent prog bar format
bf = "{desc:15} {percentage:3.0f}% |{bar:50}| {n_fmt:>5} / {total_fmt:<5} [{elapsed}<{remaining} {rate_fmt}]"

for i in tqdm(range(n_steps), desc="RK4 Integration", bar_format=bf):
    traj[i] = state

    k1 = lorenz(state)
    k2 = lorenz(state + 0.5 * dt * k1)
    k3 = lorenz(state + 0.5 * dt * k2)
    k4 = lorenz(state + dt * k3)

    state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Split coordinates
x = traj[:, 0]
y = traj[:, 1]
z = traj[:, 2]

# Write Tecplot file
with tecio.open("lorenz.szplt", "w") as szl:
    # Zone 1: full trajectory as ordered line
    szl.write_ordered_zone(
        title="Lorenz Attractor",
        variables=["x", "y", "z", "t", "tau"],
        data=[x, y, z],
        passive_vars=["t", "tau"],
        strand_id=0,
    )

    # Zone 2+: trajectory animation (every 10th step)
    for i in tqdm(range(0, n_steps, 10), desc="Writing zones", bar_format=bf):
        szl.write_ordered_zone(
            title="Trajectory",
            variables=["x", "y", "z", "t", "tau"],
            data=[
                x[: i + 1],
                y[: i + 1],
                z[: i + 1],
                np.arange(i + 1) * dt,
                (np.arange(i + 1) - i - 1) * dt,
            ],
            strand_id=1,
            solution_time=i * dt,
        )
    for i in tqdm(range(0, n_steps, 10), desc="Writing zones", bar_format=bf):
        szl.write_ordered_zone(
            title="Particle",
            variables=["x", "y", "z", "t", "tau"],
            data=[x[i : i + 1], y[i : i + 1], z[i : i + 1]],
            passive_vars=["t", "tau"],
            strand_id=2,
            solution_time=i * dt,
        )

    print("Finalizing output file")
print("Done")
