"""Generate clean single-regime generalized RPS replicator datasets for AL-RNN.

The script creates three independent datasets from the same initial policies:

    inward_rps : S < 1, inward spiral toward the mixed Nash equilibrium
    center_rps : S = 1, neutral cycle around the mixed Nash equilibrium
    outward_rps: S > 1, outward spiral away from the mixed Nash equilibrium

Each player follows the same single-population generalized RPS replicator field.
The saved AL-RNN state is the non-redundant 4D Nash-centered observation:

    (p1_R - 1/3, p1_P - 1/3, p2_R - 1/3, p2_P - 1/3)

Each dataset is intended to train one independent AL-RNN model, then check
whether it recovers the mixed Nash equilibrium at the origin, the rotation
direction, and the expected radial behavior.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "RPS_data"

REGIMES = {
    "inward_rps": 0.85,
    "center_rps": 1.0,
    "outward_rps": 1.15,
}

ACTIONS = ["rock", "paper", "scissors"]
STATE_NAMES = [
    "p1_rock",
    "p1_paper",
    "p2_rock",
    "p2_paper",
]


# settings
N_TRAJECTORIES = 200
N_STEPS = 1200
DT = 0.035
TRAIN_FRACTION = 0.8
SEED = 11


INITIAL_RADIUS_RANGE = (0.05, 0.16)
MIN_ACTION_PROBABILITY = 0.02
MAX_INITIAL_ATTEMPTS = 10_000

SOLVER_RTOL = 1e-9
SOLVER_ATOL = 1e-11
PROBABILITY_TOL = 1e-8

# nash point
MIXED_NASH_POLICY = np.full(3, 1.0 / 3.0, dtype=np.float64)
MIXED_NASH_OBS = np.zeros(4, dtype=np.float64)

# coordinates
SIMPLEX_BASIS_1 = np.array([1.0, -1.0, 0.0], dtype=np.float64) / np.sqrt(2.0)
SIMPLEX_BASIS_2 = np.array([1.0, 1.0, -2.0], dtype=np.float64) / np.sqrt(6.0)


def project_simplex(policy: np.ndarray) -> np.ndarray:
    policy = np.maximum(policy, 0.0)
    total = policy.sum()
    if total <= 0.0:
        return MIXED_NASH_POLICY.copy()
    return policy / total


def project_joint_state(state: np.ndarray) -> np.ndarray:
    return np.concatenate([project_simplex(state[:3]), project_simplex(state[3:])])


def encode_joint_state(state: np.ndarray) -> np.ndarray:
    """
    keep 4-dimension state representation
    """
    state = project_joint_state(state)
    p1, p2 = state[:3], state[3:]
    return np.array([
        p1[0] - 1.0 / 3.0,
        p1[1] - 1.0 / 3.0,
        p2[0] - 1.0 / 3.0,
        p2[1] - 1.0 / 3.0,
    ], dtype=np.float64)


def encode_trajectory(full_trajectory: np.ndarray) -> np.ndarray:
    return np.asarray([encode_joint_state(state) for state in full_trajectory], dtype=np.float64)


def payoff_matrix(s: float) -> np.ndarray:
    return np.array(
        [
            [0.0, -s, 1.0],
            [1.0, 0.0, -s],
            [-s, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def replicator_rhs(policy: np.ndarray, payoff_vector: np.ndarray) -> np.ndarray:
    return policy * (payoff_vector - float(policy @ payoff_vector))


def rhs(_t: float, state: np.ndarray, s: float) -> np.ndarray:
    """
    No coupling
    """
    p1, p2 = state[:3], state[3:]
    A = payoff_matrix(s)
    return np.concatenate([
        replicator_rhs(p1, A @ p1),
        replicator_rhs(p2, A @ p2),
    ])


def simplex_orbit(theta: float, radius: float) -> np.ndarray:
    """
    In RPS probabilistic simplex, an initial policy point is generated centered 
    on the mixed Nash policy, according to a given angle theta and radius radius.
    """
    direction = np.cos(theta) * SIMPLEX_BASIS_1 + np.sin(theta) * SIMPLEX_BASIS_2
    return project_simplex(MIXED_NASH_POLICY + radius * direction)


def sample_initial_state(rng: np.random.Generator) -> np.ndarray:
    for _ in range(MAX_INITIAL_ATTEMPTS):
        theta1 = rng.uniform(0.0, 2.0 * np.pi)
        theta2 = rng.uniform(0.0, 2.0 * np.pi)
        radius1 = rng.uniform(*INITIAL_RADIUS_RANGE)
        radius2 = rng.uniform(*INITIAL_RADIUS_RANGE)

        p1 = simplex_orbit(theta1, radius1)
        p2 = simplex_orbit(theta2, radius2)

        state = np.concatenate([p1, p2])
        if float(state.min()) >= MIN_ACTION_PROBABILITY:
            return state

    raise RuntimeError("Failed to sample a valid initial joint policy.")


def sample_initial_states() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return np.asarray(
        [sample_initial_state(rng) for _ in range(N_TRAJECTORIES)],
        dtype=np.float64,
    )


def simulate_full_trajectory(initial_state: np.ndarray, s: float) -> np.ndarray:
    t_eval = np.arange(N_STEPS, dtype=np.float64) * DT
    solution = solve_ivp(
        fun=lambda t, y: rhs(t, y, s),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=project_joint_state(initial_state),
        t_eval=t_eval,
        rtol=SOLVER_RTOL,
        atol=SOLVER_ATOL,
    )
    if not solution.success:
        raise RuntimeError(f"solve_ivp failed for S={s}: {solution.message}")
    full = solution.y.T
    return np.asarray([project_joint_state(state) for state in full], dtype=np.float64)


def generate_trajectories(initial_states: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    full = np.asarray([simulate_full_trajectory(state, s) for state in initial_states], dtype=np.float64)
    observations = np.asarray([encode_trajectory(traj) for traj in full], dtype=np.float64)
    return observations, full


def split_train_test(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    n_train = int(TRAIN_FRACTION * len(arrays[0]))
    out = []
    for array in arrays:
        out.extend([array[:n_train], array[n_train:]])
    return tuple(out)


def normalize_from_train(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    flat = train.reshape(-1, train.shape[-1])
    mean = np.zeros(train.shape[-1], dtype=np.float64)
    std = np.maximum(flat.std(axis=0), 1e-8)
    return (train - mean) / std, (test - mean) / std, {
        "method": "scale_from_train_keep_mixed_nash_at_origin",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "equilibrium_norm": ((MIXED_NASH_OBS - mean) / std).tolist(),
    }


def expected_radial_behavior(regime_name: str) -> str:
    return {
        "center_rps": "approximately neutral radius",
        "inward_rps": "radius decreases toward the mixed Nash equilibrium",
        "outward_rps": "radius increases away from the mixed Nash equilibrium",
    }.get(regime_name, "custom generalized RPS radial behavior")


def sanity_check(observations: np.ndarray, full_policies: np.ndarray):
    if not np.isfinite(observations).all() or not np.isfinite(full_policies).all():
        raise RuntimeError("Trajectories contain NaN or Inf.")
    
    min_prob = float(full_policies.min())
    max_prob = float(full_policies.max())

    if min_prob < -PROBABILITY_TOL or max_prob > 1.0 + PROBABILITY_TOL:
        raise RuntimeError(f"Probability bounds failed: min={min_prob}, max={max_prob}.")
    
    player_sums = np.stack([full_policies[..., :3].sum(axis=-1), full_policies[..., 3:].sum(axis=-1)])

    if float(np.max(np.abs(player_sums - 1.0))) > 1e-7:
        raise RuntimeError("Projected policies do not sum to one.")


def save_regime_dataset(regime_name: str, s: float, initial_states: np.ndarray,) -> dict:
    output_dir = OUTPUT_ROOT / regime_name

    trajectories, full_policies = generate_trajectories(initial_states, s)
    sanity_check(trajectories, full_policies)

    labels = np.zeros(trajectories.shape[:2], dtype=np.int64)
    train, test, train_labels, test_labels = split_train_test(trajectories, labels)
    train_norm, test_norm, norm_stats = normalize_from_train(train, test)

    arrays = {
        "train_trajectories_norm": train_norm.astype(np.float32),
        "test_trajectories_norm": test_norm.astype(np.float32),
        "train_trajectories_raw": train.astype(np.float32),
        "test_trajectories_raw": test.astype(np.float32),
        "train_regime_labels": train_labels.astype(np.int64),
        "test_regime_labels": test_labels.astype(np.int64),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(output_dir / f"{name}.npy", array)

    metadata = {
        "game": "rps_replicator_seperate_model",
        "regime": regime_name,
        "s": s,
        "expected_radial_behavior": expected_radial_behavior(regime_name),
        "dynamics": "two independent generalized RPS replicator fields",
        "state_names": STATE_NAMES,
        "mixed_nash_observation": MIXED_NASH_OBS.tolist(),
        "n_trajectories": N_TRAJECTORIES,
        "nums_trajectories": N_TRAJECTORIES,
        "n_steps": N_STEPS,
        "dt": DT,
        "train_fraction": TRAIN_FRACTION,
        "normalization": norm_stats,
        "shapes": {name: list(array.shape) for name, array in arrays.items()},
        "seed": SEED,
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def save_all_datasets() -> dict:
    initial_states = sample_initial_states()
    return {
        regime_name: save_regime_dataset(regime_name, s, initial_states)
        for regime_name, s in REGIMES.items()
    }


def main() -> None:
    reports = save_all_datasets()
    print("Saved RPS datasets to:", OUTPUT_ROOT)
    for regime_name, metadata in reports.items():
        print(f"{regime_name}: S={metadata['s']}, train={metadata['shapes']['train_trajectories_norm']}")


if __name__ == "__main__":
    main()
