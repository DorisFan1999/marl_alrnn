"""All-test long free-rollout error metrics.

This module evaluates the autonomous long rollout of the trained AL-RNN on every test trajectory. 

For each test trajectory, observations should be:

    X_i = (x_{i,0}, x_{i,1}, ..., x_{i,T-1}),

the model receives only the initial condition x_{i,0}. It then generates

    hat{X_i} = ( hat{x_{i,1}}, hat{x_{i,2}}, ..., hat{x_{i,T-1}})

by recursively feeding its own previous prediction back into the model. 
The prediction is compared time-point by time-point with the ground truth

    (x_{i,1}, x_{i,2}, ..., x_{i,T-1}).

For one trajectory, the rollout MSE is

    MSE_i = 1 / ((T - 1) D) * sum_{t=1}^{T-1} ||xhat_{i,t} - x_{i,t}||_2^2,

where D is the state dimension.

The mean value is also equal to rollout_on_test(ctx)["mse"] when all test trajectories are used. 
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from alrnn_eval_utils import rollout_on_test
from flow_geometry import add_trajectory_arrow
from policy_validity import invalid_policy_summary
from training.metrics import power_spectrum_error, state_space_divergence_binning


def rollout_mse_summary(rollout):
    pred = rollout["pred_norm"]
    true = rollout["target_norm"]
    per_rollout_mse = ((pred - true) ** 2).mean(axis=(1, 2)) # mean over time and state dimension for each rollout
    median_rollout_mse = float(np.median(per_rollout_mse))
    median_rollout_index = int(np.argmin(np.abs(per_rollout_mse - median_rollout_mse)))
    worst_rollout_index = int(np.argmax(per_rollout_mse))

    return {
        "mean_rollout_mse": float(rollout["mse"]), # rollout["mse"] is the mean over all test rollouts
        "median_rollout_mse": median_rollout_mse,
        "worst_rollout_mse": float(per_rollout_mse[worst_rollout_index]),
        "median_case_rollout": median_rollout_index + 1,
        "median_case_mse": float(per_rollout_mse[median_rollout_index]),
        "worst_case_rollout": worst_rollout_index + 1,
        "per_rollout_mse": per_rollout_mse.tolist(),
    }


def valid_plot_lengths(pred_raw, tol=1e-6):
    """Return the number of valid points to plot for each RPS rollout."""
    pred_raw = np.asarray(pred_raw)
    lengths = np.full(pred_raw.shape[0], pred_raw.shape[1], dtype=int)
    if pred_raw.shape[-1] != 4:
        return lengths

    validity = invalid_policy_summary(pred_raw, tol=tol)
    for event in validity["invalid_pred_events"]:
        lengths[event["trajectory"] - 1] = event["invalid_start_time"] - 1
    return lengths


def plot_rollout_error_cases(rollout, mse_summary=None, state_names=None, validity_tol=1e-6,):
    """Plot median and worst rollouts in the same planes as flow geometry."""
    if mse_summary is None:
        mse_summary = rollout_mse_summary(rollout)

    true = rollout["target_raw"]
    pred = rollout["pred_raw"]
    if true.shape[-1] < 2:
        raise ValueError("Flow-geometry plotting requires at least two state dimensions")

    if state_names is None or len(state_names) != true.shape[-1]:
        state_names = [f"state {dimension + 1}" for dimension in range(true.shape[-1])]

    plot_lengths = valid_plot_lengths(pred, tol=validity_tol)

    panels = [(0, 1, "player 1")]
    if true.shape[-1] >= 4:
        panels.append((2, 3, "player 2"))

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

    fig, axes = plt.subplots(len(cases), len(panels), figsize=(4 * len(panels), 6.4), squeeze=False)

    for row, (case_name, rollout_index, case_mse) in enumerate(cases):
        plot_length = int(plot_lengths[rollout_index])
        true_case = true[rollout_index, :plot_length]
        pred_case = pred[rollout_index, :plot_length]
        if plot_length < true.shape[1]:
            validity_text = (f"valid through t={plot_length}; invalid from t={plot_length + 1}")
        else:
            validity_text = "valid through entire free rollout"

        for column, (i, j, player_name) in enumerate(panels):
            ax = axes[row, column]
            ax.plot(true_case[:, i], true_case[:, j], color="C0", linewidth=1.4, label="true")
            ax.plot(pred_case[:, i], pred_case[:, j], color="C1", linestyle="--", linewidth=1.2,label="AL-RNN free rollout")

            add_trajectory_arrow(ax, true_case[:, i], true_case[:, j], color="C0", position=0.58)
            add_trajectory_arrow(ax, pred_case[:, i], pred_case[:, j], color="C1", position=0.72, linestyle="--")

            ax.scatter([0.0], [0.0], color="black", marker="x", zorder=5)
            ax.set_title(
                f"{case_name}: trajectory {rollout_index + 1}, {player_name}\n"
                f"free-rollout normalized MSE={case_mse:.6g}\n({validity_text})",
                fontsize=9,
                pad=5,
            )
            ax.set_xlabel(state_names[i], fontsize=9)
            ax.set_ylabel(state_names[j], fontsize=9)
            ax.tick_params(labelsize=8)
            ax.axis("equal")
            ax.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Long free-rollout reconstruction in flow geometry", fontsize=12, y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    plt.show()
    return fig, axes


def attractor_metrics(rollout, n_bins=6, smoothing=20):
    pred = rollout["pred_norm"]
    true = rollout["target_norm"]
    state_dim = true.shape[-1]

    pred_flat = pred.reshape(-1, state_dim)
    true_flat = true.reshape(-1, state_dim)
    finite = np.isfinite(pred_flat).all(axis=1) & np.isfinite(true_flat).all(axis=1)

    if finite.any():
        # put the finite predictions and true values into torch tensors for state space divergence computation
        dstsp = state_space_divergence_binning(
            torch.tensor(pred_flat[finite], dtype=torch.float32),
            torch.tensor(true_flat[finite], dtype=torch.float32),
            n_bins=n_bins,
        )
    else:
        dstsp = np.nan

    ps_errors = []
    for pred_i, true_i in zip(pred, true):
        if np.isfinite(pred_i).all() and np.isfinite(true_i).all():
            # compute power spectrum error for each rollout trajectory
            ps_errors.append(power_spectrum_error(pred_i, true_i, smoothing=smoothing))

    return {
        "state_space_divergence": float(dstsp),
        "power_spectrum_error_mean": float(np.mean(ps_errors)) if ps_errors else np.nan,
        "power_spectrum_error_std": float(np.std(ps_errors)) if ps_errors else np.nan,
    }


def test_long_rollout_analysis(ctx, n_bins=6, smoothing=20, plot=False):
    test = ctx["data"]["test_norm"]
    indices = list(range(len(test)))
    steps = test.shape[1] - 1
    rollout = rollout_on_test(ctx, trajectory_indices=indices, steps=steps) # mse of all test rollout trajectories

    mse = rollout_mse_summary(rollout) 
    attractor = attractor_metrics(rollout, n_bins=n_bins, smoothing=smoothing)

    if plot:
        plot_rollout_error_cases(
            rollout,
            mse_summary=mse,
            state_names=ctx["data"]["metadata"].get("state_names"),
        )

    return {
        "n_test_rollouts": len(indices),
        "steps": steps,
        "state_dim": ctx["N"],
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
