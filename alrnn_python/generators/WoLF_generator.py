"""
Generate 4D Nash-centered WoLF-PHC trajectories for AL-RNN.

The WoLF-PHC dynamics are simulated internally in the full 6D joint-policy
space:

    (p1_R, p1_P, p1_S, p2_R, p2_P, p2_S)

The observations saved for AL-RNN remove the simplex redundancy and center the
mixed Nash equilibrium at the origin:

    (p1_R - 1/3, p1_P - 1/3, p2_R - 1/3, p2_P - 1/3)

This keeps the WoLF-PHC update intact while making the saved dataset compatible
with the 4D RPS-style analysis pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "datasets" / "WoLF_4D_data_new"

ACTIONS = ["rock", "paper", "scissors"]
STATE_4D = [
    "p1_rock_minus_equilibrium",
    "p1_paper_minus_equilibrium",
    "p2_rock_minus_equilibrium",
    "p2_paper_minus_equilibrium",
]
FULL_POLICY_STATE = [
    "p1_rock",
    "p1_paper",
    "p1_scissors",
    "p2_rock",
    "p2_paper",
    "p2_scissors",
]

PAYOFF_1 = np.array(
    [
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
PAYOFF_2 = -PAYOFF_1

NASH_PLAYER = np.full(3, 1.0 / 3.0, dtype=np.float64)
NASH_POLICY = np.concatenate([NASH_PLAYER, NASH_PLAYER])
NASH_OBSERVATION = np.zeros(4, dtype=np.float64)

# Dataset settings.
N_TRAJECTORIES = 200
N_STEPS = 3000
TRAIN_FRACTION = 0.8
SEED = 23

# WoLF-PHC settings.
WIN_LR = 3e-4
LOSE_LR = 1.2e-3
MIN_PROB = 0.04

# Initial-state settings.
INITIAL_MIN_DISTANCE = 0.12
INITIAL_MAX_DISTANCE = 0.28
INITIAL_MIN_ACTION_PROB = 0.10


def full_policy_to_observation(policy_6d):
    """Convert full 6D joint policy to 4D Nash-centered observation."""
    policy_6d = np.asarray(policy_6d)
    obs = np.empty(policy_6d.shape[:-1] + (4,), dtype=policy_6d.dtype)

    obs[..., 0] = policy_6d[..., 0] - 1.0 / 3.0
    obs[..., 1] = policy_6d[..., 1] - 1.0 / 3.0
    obs[..., 2] = policy_6d[..., 3] - 1.0 / 3.0
    obs[..., 3] = policy_6d[..., 4] - 1.0 / 3.0
    return obs


def observation_to_full_policy(obs_4d):
    """Recover full 6D joint policy from 4D Nash-centered observation."""
    obs_4d = np.asarray(obs_4d)
    policy = np.empty(obs_4d.shape[:-1] + (6,), dtype=obs_4d.dtype)

    policy[..., 0] = obs_4d[..., 0] + 1.0 / 3.0
    policy[..., 1] = obs_4d[..., 1] + 1.0 / 3.0
    policy[..., 2] = 1.0 - policy[..., 0] - policy[..., 1]
    policy[..., 3] = obs_4d[..., 2] + 1.0 / 3.0
    policy[..., 4] = obs_4d[..., 3] + 1.0 / 3.0
    policy[..., 5] = 1.0 - policy[..., 3] - policy[..., 4]
    return policy


def project_simplex(policy, min_probability=MIN_PROB):
    """Keep a policy inside the probability simplex."""
    policy = np.maximum(policy, min_probability)
    return policy / policy.sum()


def sample_policy(rng):
    policy = rng.dirichlet(np.full(3, 3.0))
    policy = np.maximum(policy, INITIAL_MIN_ACTION_PROB)
    return policy / policy.sum()


def sample_initial_state(rng):
    """Sample a full 6D joint policy away from the Nash equilibrium."""
    for _ in range(10000):
        policy1 = sample_policy(rng)
        policy2 = sample_policy(rng)
        state = np.concatenate([policy1, policy2])
        distance = np.linalg.norm(state - NASH_POLICY)

        if INITIAL_MIN_DISTANCE <= distance <= INITIAL_MAX_DISTANCE:
            return state

    raise RuntimeError("Failed to sample an initial state with the requested distance.")


def random_argmax(values, rng, tol=1e-12):
    best = values.max()
    candidates = np.flatnonzero(values >= best - tol)
    return int(rng.choice(candidates))


def policy_value(policy, q_values):
    return float(policy @ q_values)


def wolf_phc_update(policy, average_policy, q_values, rng):
    """One WoLF-PHC current-policy update."""
    delta = WIN_LR if policy_value(policy, q_values) > policy_value(average_policy, q_values) else LOSE_LR
    greedy_action = random_argmax(q_values, rng)
    updated = policy.copy()
    moved = 0.0

    for action in range(3):
        if action == greedy_action:
            continue

        delta_action = min(max(updated[action] - MIN_PROB, 0.0), delta / 2.0)
        updated[action] -= delta_action
        moved += delta_action

    updated[greedy_action] += moved
    return project_simplex(updated)


def expected_q_values(policy1, policy2):
    q1 = PAYOFF_1 @ policy2
    q2 = PAYOFF_2.T @ policy1
    return q1, q2


def simulate_full_policy_trajectory(initial_state, rng):
    """Simulate one independent WoLF-PHC full joint-policy trajectory."""
    policy1 = project_simplex(initial_state[:3])
    policy2 = project_simplex(initial_state[3:])
    average_policy1 = policy1.copy()
    average_policy2 = policy2.copy()
    trajectory = np.empty((N_STEPS, 6), dtype=np.float64)

    for t in range(N_STEPS):
        trajectory[t] = np.concatenate([policy1, policy2])
        q1, q2 = expected_q_values(policy1, policy2)

        count = t + 1
        average_policy1 += (policy1 - average_policy1) / count
        average_policy2 += (policy2 - average_policy2) / count

        policy1 = wolf_phc_update(policy1, average_policy1, q1, rng)
        policy2 = wolf_phc_update(policy2, average_policy2, q2, rng)

    return trajectory


def distance_diagnostics(full_policy_trajectories):
    """Summarize endpoint distances to Nash in full policy space."""
    joint_distances = np.linalg.norm(
        full_policy_trajectories - NASH_POLICY[None, None, :],
        axis=2,
    )
    agent1_distances = np.linalg.norm(
        full_policy_trajectories[:, :, :3] - NASH_POLICY[:3][None, None, :],
        axis=2,
    )
    agent2_distances = np.linalg.norm(
        full_policy_trajectories[:, :, 3:] - NASH_POLICY[3:][None, None, :],
        axis=2,
    )
    joint_ratios = joint_distances[:, -1] / np.maximum(joint_distances[:, 0], 1e-10)
    min_prob = full_policy_trajectories.min(axis=(1, 2))

    return {
        "agent1_final_distance_mean": float(agent1_distances[:, -1].mean()),
        "agent1_final_distance_max": float(agent1_distances[:, -1].max()),
        "agent2_final_distance_mean": float(agent2_distances[:, -1].mean()),
        "agent2_final_distance_max": float(agent2_distances[:, -1].max()),
        "joint_initial_distance_mean": float(joint_distances[:, 0].mean()),
        "joint_final_distance_mean": float(joint_distances[:, -1].mean()),
        "joint_final_initial_ratio_mean": float(joint_ratios.mean()),
        "joint_final_initial_ratio_max": float(joint_ratios.max()),
        "min_probability_min": float(min_prob.min()),
        "min_probability_mean": float(min_prob.mean()),
    }


def generate_trajectories():
    rng = np.random.default_rng(SEED)
    full_trajectories = []
    initial_states = []

    while len(full_trajectories) < N_TRAJECTORIES:
        initial_state = sample_initial_state(rng)
        trajectory = simulate_full_policy_trajectory(initial_state, rng)

        full_trajectories.append(trajectory)
        initial_states.append(initial_state)

    return np.asarray(full_trajectories), np.asarray(initial_states)


def zscore_from_train(train, test):
    train_flat = train.reshape(-1, train.shape[-1])
    mean = train_flat.mean(axis=0)
    std = np.maximum(train_flat.std(axis=0), 1e-8)

    train_norm = (train - mean) / std
    test_norm = (test - mean) / std

    stats = {
        "method": "zscore_from_train",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "equilibrium_norm": ((NASH_OBSERVATION - mean) / std).tolist(),
    }
    return train_norm, test_norm, stats


def main():
    full_trajectories, initial_states = generate_trajectories()
    trajectories = full_policy_to_observation(full_trajectories)
    initial_observations = full_policy_to_observation(initial_states)

    n_train = int(TRAIN_FRACTION * N_TRAJECTORIES)
    train = trajectories[:n_train]
    test = trajectories[n_train:]
    train_initial = initial_observations[:n_train]
    test_initial = initial_observations[n_train:]

    train_full = full_trajectories[:n_train]
    test_full = full_trajectories[n_train:]

    train_norm, test_norm, norm_stats = zscore_from_train(train, test)
    train_diag = distance_diagnostics(train_full)
    test_diag = distance_diagnostics(test_full)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "train_trajectories_norm.npy", train_norm.astype(np.float32))
    np.save(OUTPUT_DIR / "test_trajectories_norm.npy", test_norm.astype(np.float32))
    np.save(OUTPUT_DIR / "train_trajectories_raw.npy", train.astype(np.float32))
    np.save(OUTPUT_DIR / "test_trajectories_raw.npy", test.astype(np.float32))
    np.save(OUTPUT_DIR / "train_initial_states.npy", train_initial.astype(np.float32))
    np.save(OUTPUT_DIR / "test_initial_states.npy", test_initial.astype(np.float32))
    np.save(OUTPUT_DIR / "train_full.npy", train_full.astype(np.float32))
    np.save(OUTPUT_DIR / "test_full.npy", test_full.astype(np.float32))

    metadata = {
        "game": "wolf_phc_rps",
        "state_representation": "nash_centered_4d_joint_policy",
        "actions": ACTIONS,
        "payoff_player_1": PAYOFF_1.tolist(),
        "payoff_player_2": PAYOFF_2.tolist(),
        "q_values": "expected_payoffs",
        "target": "convergent_wolf_phc_transient_toward_nash",
        "mixed_nash_observation": NASH_OBSERVATION.tolist(),
        "mixed_nash_policy": NASH_POLICY.tolist(),
        "normalization": norm_stats,
        "diagnostics": {"train": train_diag, "test": test_diag},
        "n_train_trajectories": int(train.shape[0]),
        "n_test_trajectories": int(test.shape[0]),
        "n_steps_per_trajectory": N_STEPS,
        "learning_rate_win": WIN_LR,
        "learning_rate_lose": LOSE_LR,
        "min_action_probability": MIN_PROB,
        "initial_distance_min": INITIAL_MIN_DISTANCE,
        "initial_distance_max": INITIAL_MAX_DISTANCE,
        "seed": SEED,
        "train_trajectory_shape": list(train_norm.shape),
        "test_trajectory_shape": list(test_norm.shape),
    }

    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
