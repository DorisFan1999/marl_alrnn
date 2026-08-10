"""Full-path deviation analysis for modified AL-RNN probability rollouts."""

from __future__ import annotations

from modified_alrnn_eval_utils import rollout_on_test
from path_deviation import (
    path_deviation_summary,
    plot_horizon_errors,
    plot_path_error_quartiles,
)


def full_path_deviation_analysis(
    ctx,
    trajectory_indices=None,
    steps=None,
    validity_tol=1e-6,
    horizons=None,
    plot=False,
):
    """Measure modified predictions against true joint 6D RPS policies."""
    if ctx["N1_output"] != 3 or ctx["N2_output"] != 3:
        raise ValueError("Modified RPS path deviation requires 3 outputs per player")

    test = ctx["data"]["test_norm"]
    trajectory_indices = (
        list(range(len(test)))
        if trajectory_indices is None
        else list(trajectory_indices)
    )
    rollout = rollout_on_test(
        ctx,
        trajectory_indices=trajectory_indices,
        steps=steps,
    )
    deviation = path_deviation_summary(
        rollout["pred_probability"],
        rollout["target_probability"],
        validity_tol=validity_tol,
        horizons=horizons,
    )

    if plot:
        plot_path_error_quartiles(deviation["path_error_quartiles"])
        plot_horizon_errors(
            deviation["error_by_time"],
            [row["horizon"] for row in deviation["horizon_summary"]],
        )

    return {
        "n_test_rollouts": len(trajectory_indices),
        "steps": rollout["steps"],
        "mean_path_error": deviation["mean_path_error"],
        "median_path_error": deviation["median_path_error"],
        "max_path_error": deviation["max_path_error"],
        "final_path_error": deviation["final_path_error"],
        "n_valid_trajectories_at_final_time": deviation[
            "n_valid_trajectories_at_final_time"
        ],
        "per_trajectory": deviation["per_trajectory"],
        "error_by_time": deviation["error_by_time"],
        "path_error_quartiles": deviation["path_error_quartiles"],
        "horizon_summary": deviation["horizon_summary"],
        "rollout": rollout,
    }


modified_path_deviation_analysis = full_path_deviation_analysis
