"""AL-RNN free-rollout policy validity checks.
"""

from __future__ import annotations

import numpy as np

from alrnn_eval_utils import rollout_on_test


def decode_centered_policy(raw_4d):
    """Transform four centred RPS coordinates into two 3D policies."""
    raw_4d = np.asarray(raw_4d)
    if raw_4d.shape[-1] != 4:
        raise ValueError(
            f"Expected four centred RPS coordinates, found {raw_4d.shape[-1]}"
        )

    policy = np.empty(raw_4d.shape[:-1] + (6,), dtype=raw_4d.dtype)
    policy[..., 0] = raw_4d[..., 0] + 1.0 / 3.0
    policy[..., 1] = raw_4d[..., 1] + 1.0 / 3.0
    policy[..., 2] = 1.0 - policy[..., 0] - policy[..., 1]

    policy[..., 3] = raw_4d[..., 2] + 1.0 / 3.0
    policy[..., 4] = raw_4d[..., 3] + 1.0 / 3.0
    policy[..., 5] = 1.0 - policy[..., 3] - policy[..., 4]
    return policy


def invalid_policy_summary(
    pred_raw,
    tol=1e-6,
):

    pred_raw = np.asarray(pred_raw)
    if pred_raw.ndim != 3:
        raise ValueError(
            "pred_raw must have shape (n_rollouts, n_steps, 4 or 6); "
            f"found {pred_raw.shape}"
        )
    
    if tol < 0:
        raise ValueError(f"tol must be non-negative, found {tol}")

    n_rollouts = pred_raw.shape[0]

    policy = decode_centered_policy(pred_raw) if pred_raw.shape[-1] == 4 else pred_raw
    player_policy = policy.reshape(policy.shape[:-1] + (2, 3))
    player_sum = player_policy.sum(axis=-1)
    sum_error = np.abs(player_sum - 1.0)

    invalid_player_step = (
        ~np.isfinite(player_policy).all(axis=-1)
        | (player_policy < -tol).any(axis=-1)
        | (player_policy > 1.0 + tol).any(axis=-1)
        | ~np.isfinite(sum_error)
        | (sum_error > tol)
    )
    invalid_step = invalid_player_step.any(axis=-1)

    invalid_rollout = invalid_step.any(axis=1)
    valid_probability = player_policy[~invalid_player_step]
    invalid_probability = player_policy[invalid_player_step]
    finite_invalid_probability = invalid_probability[
        np.isfinite(invalid_probability)
    ]

    invalid_events = []
    for rollout_position in np.flatnonzero(invalid_rollout):
        first_invalid_step = int(
            np.flatnonzero(invalid_step[rollout_position])[0]
        )
        players = (
            np.flatnonzero(
                invalid_player_step[rollout_position, first_invalid_step]
            )
            + 1
        )
        invalid_events.append(
            {
                "trajectory": int(rollout_position) + 1,
                "invalid_start_time": first_invalid_step + 1,
                "players": players.astype(int).tolist(),
            }
        )

    return {
        "n_invalid": int(invalid_rollout.sum()),
        "n_invalid_pred_rollouts": int(invalid_rollout.sum()),
        "invalid_pred_rollout_fraction": (
            float(invalid_rollout.mean()) if n_rollouts else 0.0
        ),
        "min_valid_pred_probability": (
            float(valid_probability.min()) if valid_probability.size else np.nan
        ),
        "max_valid_pred_probability": (
            float(valid_probability.max()) if valid_probability.size else np.nan
        ),
        "min_invalid_pred_probability": (
            float(finite_invalid_probability.min())
            if finite_invalid_probability.size
            else np.nan
        ),
        "max_invalid_pred_probability": (
            float(finite_invalid_probability.max())
            if finite_invalid_probability.size
            else np.nan
        ),
        "invalid_pred_events": invalid_events,
    }


def policy_validity_analysis(ctx, tol=1e-6):
    test = ctx["data"]["test_norm"]
    steps = test.shape[1] - 1
    rollout = rollout_on_test(ctx, steps=steps)

    validity = invalid_policy_summary(
        rollout["pred_raw"],
        tol=tol,
    )

    return {
        "n_test_rollouts": len(test),
        "steps": steps,
        "state_dim": ctx["N"],
        "tol": tol,
        "n_invalid": validity["n_invalid"],
        "n_invalid_pred_rollouts": validity["n_invalid_pred_rollouts"],
        "invalid_pred_rollout_fraction": validity[
            "invalid_pred_rollout_fraction"
        ],
        "min_valid_pred_probability": validity["min_valid_pred_probability"],
        "max_valid_pred_probability": validity["max_valid_pred_probability"],
        "min_invalid_pred_probability": validity[
            "min_invalid_pred_probability"
        ],
        "max_invalid_pred_probability": validity[
            "max_invalid_pred_probability"
        ],
        "invalid_pred_events": validity["invalid_pred_events"],
        "rollout": rollout,
    }
