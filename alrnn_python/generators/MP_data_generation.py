"""
Generate matching-pennies trajectories for AL-RNN.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "datasets" / "MP_data"


nash_policy = np.array([0.5, 0.5], dtype=np.float64)
nash_observation = np.zeros(2, dtype=np.float64)

# settings
n_trajectories = 80
n_steps = 1500
train_fraction = 0.8
dt = 0.05
seed = 11
angular_frequency = 4.0

initial_min_radius = 0.18
initial_max_radius = 0.42
initial_eps = 0.03



def rotation_matrix(dt):
    """Exact discrete-time update for matching-pennies gradient learning.

    For centered coordinates y = (p1 - 0.5, p2 - 0.5), the continuous-time
    system is

        dx1/dt =  4 x2
        dx2/dt = -4 x1

    so one step is a pure rotation by angle -4 dt. This preserves radius
    around the mixed equilibrium exactly up to floating-point precision.
    """
    theta = angular_frequency * dt
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def sample_initial_states(rng):
    """Sample initial joint strategies away from the mixed equilibrium."""
    initial_states = []

    while len(initial_states) < n_trajectories:
        # constraint: [eps, 1-eps], avoid extreme strategies.
        state = rng.uniform(initial_eps, 1.0 - initial_eps, size=2)
        
        # distance constraint: [initial_min_radius, initial_max_radius]
        radius = np.linalg.norm(state - nash_policy)
        if initial_min_radius <= radius <= initial_max_radius:
            initial_states.append(state)

    return np.array(initial_states, dtype=np.float64)


def simulate_trajectory(initial_state):
    """Simulate one centered joint-strategy trajectory."""
    trajectory = np.empty((n_steps, 2), dtype=np.float64)

    centered_state = initial_state - nash_policy
    trajectory[0] = centered_state

    update = rotation_matrix(dt)
    for t in range(1, n_steps):
        centered_state = update @ centered_state
        trajectory[t] = centered_state

    return trajectory


def zscore_from_train(train, test):
    """Normalize train/test trajectories"""
    train_flat = train.reshape(-1, train.shape[-1])
    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)

    train_norm = (train - mean) / std
    test_norm = (test - mean) / std

    stats = {
        "method": "zscore_from_train",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "equilibrium_norm": ((nash_observation - mean) / std).tolist(),
    }
    return train_norm, test_norm, stats



def main():
    rng = np.random.default_rng(seed)
    initial_states = sample_initial_states(rng)
    trajectories = np.array([simulate_trajectory(x0) for x0 in initial_states])

    n_train = int(train_fraction * n_trajectories)
    train = trajectories[:n_train]
    test = trajectories[n_train:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_norm, test_norm, norm_stats = zscore_from_train(train, test)

    # Primary AL-RNN inputs: each initial condition remains an independent
    np.save(OUTPUT_DIR / "train_trajectories_norm.npy", train_norm.astype(np.float32))
    np.save(OUTPUT_DIR / "test_trajectories_norm.npy", test_norm.astype(np.float32))
    np.save(OUTPUT_DIR / "train_trajectories_raw.npy", train.astype(np.float32))
    np.save(OUTPUT_DIR / "test_trajectories_raw.npy", test.astype(np.float32))
    np.save(OUTPUT_DIR / "initial_states.npy", (initial_states - nash_policy).astype(np.float32))

    metadata = {
        "game": "matching_pennies",
        "state": ["p1", "p2"],
        "state_representation": "nash_centered_observation",
        "equilibrium": nash_observation.tolist(),
        "nash_observation": nash_observation.tolist(),
        "nash_policy": nash_policy.tolist(),
        "normalization": norm_stats,
        "n_train_trajectories": int(train.shape[0]),
        "n_test_trajectories": int(test.shape[0]),
        "n_steps_trajectory": n_steps,
        "seed": seed,
        "initial_radius_min": initial_min_radius,
        "initial_radius_max": initial_max_radius,
        "train_trajectory_shape": list(train_norm.shape),
        "test_trajectory_shape": list(test_norm.shape),
    }

    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
