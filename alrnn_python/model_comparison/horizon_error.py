"""Fixed-horizon joint 6D path error."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_horizon_errors(pred, true, horizons=(300, 600, 900, 1199)):
    path_error = np.linalg.norm(pred - true, axis=-1)
    rows = []
    for horizon in horizons:
        endpoint = path_error[:, horizon - 1]
        q1, median, q3 = np.quantile(endpoint, [0.25, 0.5, 0.75])
        rows.append(
            {
                "horizon": int(horizon),
                "mean_error": float(endpoint.mean()),
                "median_error": float(median),
                "q1_error": float(q1),
                "q3_error": float(q3),
            }
        )
    return rows


def evaluate_horizon_errors(rollouts, horizons=(300, 600, 900, 1199)):
    rows = []
    for rollout in rollouts:
        identifiers = {
            "P": rollout["P"],
            "model_seed": rollout["model_seed"],
            "run_id": rollout["run_id"],
        }
        rows.extend(
            {**identifiers, **row}
            for row in calculate_horizon_errors(
                rollout["pred"],
                rollout["true"],
                horizons=horizons,
            )
        )
    return pd.DataFrame(rows)


def summarize_horizon_errors(per_seed):
    return (
        per_seed.groupby(["P", "horizon"])["median_error"]
        .agg(["mean", "std"])
        .reset_index()
    )
