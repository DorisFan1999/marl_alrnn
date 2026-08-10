"""Modified AL-RNN free-rollout policy validity checks."""

from __future__ import annotations

import numpy as np

try:
    from .modified_alrnn_eval_utils import rollout_on_test
except ImportError:
    from modified_alrnn_eval_utils import rollout_on_test


def invalid_policy_summary(pred_probability, N1_output=3, N2_output=3, tol=1e-6):
    """Record the first invalid time and players for each modified rollout."""
    probability = np.asarray(pred_probability)
    if probability.ndim != 3:
        raise ValueError(
            "pred_probability must have shape (n_rollouts, n_steps, N_output); "
            f"found {probability.shape}"
        )
    if probability.shape[-1] != N1_output + N2_output:
        raise ValueError(
            f"Expected probability dimension {N1_output + N2_output}, "
            f"found {probability.shape[-1]}"
        )
    if tol < 0:
        raise ValueError(f"tol must be non-negative, found {tol}")

    players = [
        probability[..., :N1_output],
        probability[..., N1_output:],
    ]
    invalid_players = []
    valid_probability_values = []
    invalid_probability_values = []

    for player_probability in players:
        sum_error = np.abs(player_probability.sum(axis=-1) - 1.0)
        invalid_player = (
            ~np.isfinite(player_probability).all(axis=-1)
            | (player_probability < -tol).any(axis=-1)
            | (player_probability > 1.0 + tol).any(axis=-1)
            | ~np.isfinite(sum_error)
            | (sum_error > tol)
        )
        invalid_players.append(invalid_player)
        valid_probability_values.append(player_probability[~invalid_player].reshape(-1))
        invalid_probability_values.append(player_probability[invalid_player].reshape(-1))

    invalid_player_step = np.stack(invalid_players, axis=-1)
    invalid_step = invalid_player_step.any(axis=-1)
    invalid_rollout = invalid_step.any(axis=1)

    valid_probability = np.concatenate(valid_probability_values)
    invalid_probability = np.concatenate(invalid_probability_values)
    finite_invalid_probability = invalid_probability[
        np.isfinite(invalid_probability)
    ]

    invalid_events = []
    for rollout_position in np.flatnonzero(invalid_rollout):
        first_invalid_step = int(
            np.flatnonzero(invalid_step[rollout_position])[0]
        )
        invalid_events.append(
            {
                "trajectory": int(rollout_position) + 1,
                "invalid_start_time": first_invalid_step + 1,
                "players": (
                    np.flatnonzero(
                        invalid_player_step[rollout_position, first_invalid_step]
                    )
                    + 1
                ).astype(int).tolist(),
            }
        )

    n_rollouts = probability.shape[0]
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
        rollout["pred_probability"],
        ctx["N1_output"],
        ctx["N2_output"],
        tol=tol,
    )

    return {
        "n_test_rollouts": len(test),
        "steps": steps,
        "state_dim": ctx["N_output"],
        "tol": tol,
        **validity,
        "rollout": rollout,
    }
