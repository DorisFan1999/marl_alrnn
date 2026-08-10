"""All-test long free-rollout metrics for modified AL-RNN."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from flow_geometry import add_trajectory_arrow
from modified_alrnn_eval_utils import (
    probability_attractor_metrics,
    probability_rollout_mse_summary,
    rollout_on_test,
)
from modified_policy_validity import invalid_policy_summary


def valid_plot_lengths(pred_probability, N1_output, N2_output, tol=1e-6):
    """Return valid prefix lengths for modified probability rollouts."""
    lengths = np.full(
        pred_probability.shape[0], pred_probability.shape[1], dtype=int
    )
    validity = invalid_policy_summary(
        pred_probability,
        N1_output=N1_output,
        N2_output=N2_output,
        tol=tol,
    )
    for event in validity["invalid_pred_events"]:
        lengths[event["trajectory"] - 1] = event["invalid_start_time"] - 1
    return lengths


def plot_rollout_error_cases(ctx, rollout, mse_summary, validity_tol=1e-6):
    """Plot median- and worst-error rollouts in each player's RPS plane."""
    true = rollout["target_probability"]
    pred = rollout["pred_probability"]
    plot_lengths = valid_plot_lengths(
        pred,
        N1_output=ctx["N1_output"],
        N2_output=ctx["N2_output"],
        tol=validity_tol,
    )
    panels = [
        (0, 1, "player 1"),
        (ctx["N1_output"], ctx["N1_output"] + 1, "player 2"),
    ]
    cases = [
        (
            "Median-error case",
            mse_summary["median_case_rollout"] - 1,
            mse_summary["median_case_mse"],
        ),
        (
            "Worst-MSE case",
            mse_summary["worst_case_rollout"] - 1,
            mse_summary["worst_rollout_mse"],
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.4), squeeze=False)
    for row, (case_name, rollout_index, case_mse) in enumerate(cases):
        plot_length = int(plot_lengths[rollout_index])
        true_case = true[rollout_index, :plot_length]
        pred_case = pred[rollout_index, :plot_length]
        if plot_length < true.shape[1]:
            validity_text = (
                f"valid through t={plot_length}; "
                f"invalid from t={plot_length + 1}"
            )
        else:
            validity_text = "valid through entire free rollout"

        for column, (i, j, player_name) in enumerate(panels):
            ax = axes[row, column]
            ax.plot(
                true_case[:, i],
                true_case[:, j],
                color="C0",
                linewidth=1.4,
                label="true",
            )
            ax.plot(
                pred_case[:, i],
                pred_case[:, j],
                color="C1",
                linestyle="--",
                linewidth=1.2,
                label="modified AL-RNN",
            )
            add_trajectory_arrow(
                ax,
                true_case[:, i],
                true_case[:, j],
                color="C0",
                position=0.58,
            )
            add_trajectory_arrow(
                ax,
                pred_case[:, i],
                pred_case[:, j],
                color="C1",
                position=0.72,
                linestyle="--",
            )
            ax.scatter([1.0 / 3.0], [1.0 / 3.0], color="black", marker="x")
            ax.set_title(
                f"{case_name}: trajectory {rollout_index + 1}, {player_name}\n"
                f"probability MSE={case_mse:.6g}\n({validity_text})",
                fontsize=9,
                pad=5,
            )
            ax.set_xlabel("rock probability", fontsize=9)
            ax.set_ylabel("paper probability", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.axis("equal")
            ax.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Modified AL-RNN long free-rollout reconstruction", fontsize=12)
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    plt.show()
    return fig, axes


def test_long_rollout_analysis(
    ctx,
    n_bins=6,
    smoothing=20,
    plot=False,
    validity_tol=1e-6,
):
    test = ctx["data"]["test_norm"]
    indices = list(range(len(test)))
    steps = test.shape[1] - 1
    rollout = rollout_on_test(ctx, trajectory_indices=indices, steps=steps)

    mse = probability_rollout_mse_summary(rollout)
    attractor = probability_attractor_metrics(
        rollout, n_bins=n_bins, smoothing=smoothing
    )
    if plot:
        plot_rollout_error_cases(
            ctx,
            rollout,
            mse_summary=mse,
            validity_tol=validity_tol,
        )

    return {
        "n_test_rollouts": len(indices),
        "steps": steps,
        "state_dim": ctx["N_output"],
        "target_is_probability": rollout["target_probability_validity"]["valid"],
        "mean_rollout_mse": mse["mean_rollout_mse"],
        "median_rollout_mse": mse["median_rollout_mse"],
        "worst_rollout_mse": mse["worst_rollout_mse"],
        "median_case_rollout": mse["median_case_rollout"],
        "median_case_mse": mse["median_case_mse"],
        "worst_case_rollout": mse["worst_case_rollout"],
        "state_space_divergence": attractor["state_space_divergence"],
        "power_spectrum_error_mean": attractor["power_spectrum_error_mean"],
        "power_spectrum_error_std": attractor["power_spectrum_error_std"],
        "per_rollout_mse": mse["per_rollout_mse"],
        "rollout": rollout,
    }
