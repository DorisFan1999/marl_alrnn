"""Load full-test free rollouts for one regime."""

from __future__ import annotations

from pathlib import Path

from evaluation.modified_alrnn_eval_utils import (
    load_eval_context,
    rollout_on_test,
)
from .models import check_seed_groups, find_regime_models


def load_regime_rollouts(
    model_root,
    data_root,
    regime,
    connection_mode,
    p_values=(2, 4, 8),
    expected_seeds=5,
):
    models = find_regime_models(
        model_root,
        regime,
        connection_mode,
        p_values=p_values,
    )
    check_seed_groups(models, p_values=p_values, expected_seeds=expected_seeds)

    rollouts = []

    for _, model_row in models.iterrows():
        ctx = load_eval_context(
            Path(data_root) / regime,
            Path(model_row["model_path"]),
        )
        test = ctx["data"]["test_norm"]
        rollout = rollout_on_test(
            ctx,
            trajectory_indices=list(range(len(test))),
            steps=test.shape[1] - 1,
        )
        pred = rollout["pred_probability"]
        true = rollout["target_probability"]
        rollouts.append(
            {
                "P": int(model_row["P"]),
                "model_seed": int(model_row["model_seed"]),
                "run_id": model_row["run_id"],
                "n_test_trajectories": len(test),
                "steps": rollout["steps"],
                "pred": pred,
                "true": true,
            }
        )

    return {
        "regime": regime,
        "models": models,
        "rollouts": rollouts,
    }
