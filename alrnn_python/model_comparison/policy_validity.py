"""Policy-validity checks for complete modified AL-RNN rollouts."""

from __future__ import annotations

import pandas as pd

from evaluation.modified_policy_validity import invalid_policy_summary


def evaluate_rollout_validity(rollouts, tol=1e-6):
    summary_rows = []
    event_rows = []

    for rollout in rollouts:
        validity = invalid_policy_summary(rollout["pred"], tol=tol)
        identifiers = {
            "P": rollout["P"],
            "model_seed": rollout["model_seed"],
            "run_id": rollout["run_id"],
        }
        summary_rows.append(
            {
                **identifiers,
                "n_trajectories": rollout["n_test_trajectories"],
                "n_invalid_trajectories": validity["n_invalid"],
                "invalid_trajectory_fraction": validity[
                    "invalid_pred_rollout_fraction"
                ],
            }
        )
        event_rows.extend(
            {**identifiers, **event}
            for event in validity["invalid_pred_events"]
        )

    event_columns = [
        "P",
        "model_seed",
        "run_id",
        "trajectory",
        "invalid_start_time",
        "players",
    ]
    return {
        "summary": pd.DataFrame(summary_rows),
        "invalid_events": pd.DataFrame(event_rows, columns=event_columns),
    }
