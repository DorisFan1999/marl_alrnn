"""AL-RNN free-rollout policy validity checks."""

from __future__ import annotations

import numpy as np

from alrnn_eval_utils import rollout_on_test


def decode_centered_policy(raw_4d):
    # decode, transform the 4d centered policy to 6d policy state
    policy = np.empty(raw_4d.shape[:-1] + (6,), dtype=raw_4d.dtype)
    policy[..., 0] = raw_4d[..., 0] + 1.0 / 3.0
    policy[..., 1] = raw_4d[..., 1] + 1.0 / 3.0
    policy[..., 2] = 1.0 - policy[..., 0] - policy[..., 1]

    policy[..., 3] = raw_4d[..., 2] + 1.0 / 3.0
    policy[..., 4] = raw_4d[..., 3] + 1.0 / 3.0
    policy[..., 5] = 1.0 - policy[..., 3] - policy[..., 4]
    return policy


def invalid_policy_summary(pred_raw, tol=1e-6):
    # tol is the tolerance for checking if the sum of probabilities is 1.0
    policy = decode_centered_policy(pred_raw) if pred_raw.shape[-1] == 4 else pred_raw # decode
    # reshape to (n_rollouts, T, 2, 3)
    player_policy = policy.reshape(policy.shape[:-1] + (2, 3))
    # sum_error: |p_rock + p_paper + p_scissors - 1.0|
    sum_error = np.abs(player_policy.sum(axis=-1) - 1.0)

    invalid_step = (
        ~np.isfinite(policy).all(axis=-1) # p is finite
        | (policy < -tol).any(axis=-1) # p>= 0
        | (policy > 1.0 + tol).any(axis=-1) # p<= 1
        | (~np.isfinite(sum_error)).any(axis=-1) # sum(p) is finite
        | (sum_error > tol).any(axis=-1) # sum(p) = 1.0
    )

    # If any single time step within a rollout is invalid, the entire rollout is marked as invalid.
    invalid_rollout = invalid_step.any(axis=1)

    # extract finite policy values for min/max computation
    finite_policy = policy[np.isfinite(policy)] 

    return {
        "n_invalid": int(invalid_rollout.sum()),
        "n_invalid_pred_rollouts": int(invalid_rollout.sum()),
        "invalid_pred_rollout_fraction": float(invalid_rollout.mean()),
        "min_pred_probability": float(finite_policy.min()) if finite_policy.size else np.nan,
        "max_pred_probability": float(finite_policy.max()) if finite_policy.size else np.nan,
        "invalid_pred_rollout_indices": np.flatnonzero(invalid_rollout).astype(int).tolist(),
    }


def policy_validity_analysis(ctx, tol=1e-6):
    test = ctx["data"]["test_norm"]
    indices = list(range(len(test)))
    steps = test.shape[1] - 1
    rollout = rollout_on_test(ctx, trajectory_indices=indices, steps=steps)

    validity = invalid_policy_summary(rollout["pred_raw"], tol=tol)

    return {
        "n_test_rollouts": len(indices),
        "steps": steps,
        "state_dim": ctx["N"],
        "tol": tol,
        "n_invalid": validity["n_invalid"],
        "n_invalid_pred_rollouts": validity["n_invalid_pred_rollouts"],
        "invalid_pred_rollout_fraction": validity["invalid_pred_rollout_fraction"],
        "min_pred_probability": validity["min_pred_probability"],
        "max_pred_probability": validity["max_pred_probability"],
        "invalid_pred_rollout_indices": validity["invalid_pred_rollout_indices"],
        "rollout": rollout,
    }
