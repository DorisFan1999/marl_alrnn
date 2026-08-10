"""Full-path deviation between true and predicted RPS trajectories.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from alrnn_eval_utils import rollout_on_test
from policy_validity import decode_centered_policy, invalid_policy_summary


def as_joint_policy(states):
    """Convert centered 4D states to joint 6D policies when necessary."""
    states = np.asarray(states, dtype=float)
    if states.shape[-1] == 4:
        return decode_centered_policy(states)
    if states.shape[-1] == 6:
        return states
    raise ValueError("RPS states must have 4 centered coordinates or 6 probabilities")


def euclidean_path_error(pred_raw, true_raw):
    """Return the joint 6D Euclidean error at every rollout and time step."""
    pred_raw = np.asarray(pred_raw, dtype=float)
    true_raw = np.asarray(true_raw, dtype=float)
    if pred_raw.shape != true_raw.shape or pred_raw.ndim != 3:
        raise ValueError(
            "pred_raw and true_raw must have the same shape "
            "(n_rollouts, n_steps, 4 or 6)"
        )

    difference = as_joint_policy(pred_raw) - as_joint_policy(true_raw)
    return np.linalg.norm(difference, axis=-1)


def error_growth_per_step(error):
    """Fit error(t) = slope * t + intercept over one valid trajectory."""
    error = np.asarray(error, dtype=float)
    finite = np.isfinite(error)
    error = error[finite]
    if len(error) < 2:
        return np.nan

    time = np.arange(1, len(error) + 1, dtype=float)
    time -= time.mean()
    denominator = np.sum(time ** 2)
    return float(np.sum(time * (error - error.mean())) / denominator)


def path_error_quartiles(error_by_time, n_steps):
    """Calculate Q1, median, and Q3 across valid trajectories at each time."""
    q1 = []
    median = []
    q3 = []
    n_valid = []

    for time_index in range(n_steps):
        values = np.asarray(
            [error[time_index] for error in error_by_time if len(error) > time_index],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        n_valid.append(int(values.size))
        if values.size:
            quartiles = np.quantile(values, [0.25, 0.5, 0.75])
            q1.append(float(quartiles[0]))
            median.append(float(quartiles[1]))
            q3.append(float(quartiles[2]))
        else:
            q1.append(np.nan)
            median.append(np.nan)
            q3.append(np.nan)

    return {
        "time": list(range(1, n_steps + 1)),
        "q1": q1,
        "median": median,
        "q3": q3,
        "n_valid": n_valid,
    }


def plot_path_error_quartiles(quartiles):
    """Plot the median full-path error and its interquartile range."""
    time = np.asarray(quartiles["time"], dtype=int)
    q1 = np.asarray(quartiles["q1"], dtype=float)
    median = np.asarray(quartiles["median"], dtype=float)
    q3 = np.asarray(quartiles["q3"], dtype=float)
    finite = np.isfinite(q1) & np.isfinite(median) & np.isfinite(q3)

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.fill_between(
        time[finite], q1[finite], q3[finite], color="C0", alpha=0.22, label="Q1–Q3"
    )
    ax.plot(time[finite], median[finite], color="C0", linewidth=1.5, label="median")
    ax.set_title("Full-path Euclidean error (valid predictions only)", fontsize=10)
    ax.set_xlabel("rollout time", fontsize=9)
    ax.set_ylabel("joint 6D path error", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    return fig, ax


def horizon_error_summary(error_by_time, horizons):
    """Summarize endpoint error at fixed rollout horizons."""
    n_rollouts = len(error_by_time)
    rows = []

    for horizon in horizons:
        valid_errors = [error for error in error_by_time if len(error) >= horizon]
        endpoint_error = np.asarray(
            [error[horizon - 1] for error in valid_errors], dtype=float
        )
        endpoint_error = endpoint_error[np.isfinite(endpoint_error)]

        if endpoint_error.size:
            q1, median, q3 = np.quantile(endpoint_error, [0.25, 0.5, 0.75])
            mean_error = float(np.mean(endpoint_error))
        else:
            q1 = median = q3 = mean_error = np.nan

        rows.append(
            {
                "horizon": int(horizon),
                "n_valid": int(endpoint_error.size),
                "valid_fraction": (
                    float(endpoint_error.size / n_rollouts) if n_rollouts else np.nan
                ),
                "mean_error_at_horizon": mean_error,
                "median_error_at_horizon": float(median),
                "q1_error_at_horizon": float(q1),
                "q3_error_at_horizon": float(q3),
            }
        )

    return rows


def plot_horizon_errors(error_by_time, horizons):
    """Draw the endpoint-error distribution at each fixed horizon."""
    distributions = []
    labels = []
    for horizon in horizons:
        values = np.asarray(
            [error[horizon - 1] for error in error_by_time if len(error) >= horizon],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        distributions.append(values if values.size else np.asarray([np.nan]))
        labels.append(f"H={horizon}\n(n={values.size})")

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    boxes = ax.boxplot(
        distributions,
        positions=np.arange(len(horizons)),
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "C1", "linewidth": 1.4},
        boxprops={"edgecolor": "C0", "linewidth": 1.1},
        whiskerprops={"color": "C0", "linewidth": 1.0},
        capprops={"color": "C0", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "none",
            "markeredgecolor": "C0",
        },
    )
    for box in boxes["boxes"]:
        box.set_facecolor("C0")
        box.set_alpha(0.22)

    ax.set_title("Full-path error distributions at fixed horizons", fontsize=10)
    ax.set_xlabel("rollout horizon and valid trajectories", fontsize=9)
    ax.set_ylabel("joint 6D error", fontsize=9)
    ax.set_xticks(np.arange(len(horizons)), labels)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()
    return fig, ax


def path_deviation_summary(
    pred_raw,
    true_raw,
    validity_tol=1e-6,
    horizons=None,
):
    """Summarize path error before each rollout's first invalid prediction."""
    path_error = euclidean_path_error(pred_raw, true_raw)
    validity = invalid_policy_summary(pred_raw, tol=validity_tol)
    invalid_start_times = {
        event["trajectory"]: event["invalid_start_time"]
        for event in validity["invalid_pred_events"]
    }

    error_by_time = []
    per_trajectory = []
    all_valid_errors = []

    for rollout_index, rollout_error in enumerate(path_error):
        trajectory = rollout_index + 1
        invalid_start_time = invalid_start_times.get(trajectory)
        end = (
            invalid_start_time - 1
            if invalid_start_time is not None
            else len(rollout_error)
        )
        valid_error = rollout_error[:end]
        finite_error = valid_error[np.isfinite(valid_error)]
        growth = error_growth_per_step(valid_error)

        error_by_time.append(valid_error.tolist())
        all_valid_errors.extend(finite_error.tolist())
        per_trajectory.append(
            {
                "trajectory": trajectory,
                "invalid_start_time": invalid_start_time,
                "mean_path_error": (
                    float(np.mean(finite_error)) if finite_error.size else np.nan
                ),
                "max_path_error": (
                    float(np.max(finite_error)) if finite_error.size else np.nan
                ),
                "error_growth_per_step": growth,
            }
        )

    n_steps = path_error.shape[1]
    if horizons is None:
        horizons = [horizon for horizon in (300, 600, 900) if horizon <= n_steps]
        if n_steps not in horizons:
            horizons.append(n_steps)
    horizons = sorted(set(int(horizon) for horizon in horizons))
    if any(horizon < 1 or horizon > n_steps for horizon in horizons):
        raise ValueError(f"horizons must be between 1 and {n_steps}; found {horizons}")

    quartiles = path_error_quartiles(error_by_time, n_steps=n_steps)
    horizon_summary = horizon_error_summary(error_by_time, horizons=horizons)
    # Average only trajectories that remain valid at the requested final time.
    final_time_errors = np.asarray(
        [error[-1] for error in error_by_time if len(error) == n_steps],
        dtype=float,
    )
    final_time_errors = final_time_errors[np.isfinite(final_time_errors)]
    all_valid_errors = np.asarray(all_valid_errors, dtype=float)

    return {
        "mean_path_error": (
            float(np.mean(all_valid_errors)) if all_valid_errors.size else np.nan
        ),
        "median_path_error": (
            float(np.median(all_valid_errors)) if all_valid_errors.size else np.nan
        ),
        "max_path_error": (
            float(np.max(all_valid_errors)) if all_valid_errors.size else np.nan
        ),
        "final_path_error": (
            float(np.mean(final_time_errors)) if final_time_errors.size else np.nan
        ),
        "n_valid_trajectories_at_final_time": int(final_time_errors.size),
        "per_trajectory": per_trajectory,
        "error_by_time": error_by_time,
        "path_error_quartiles": quartiles,
        "horizon_summary": horizon_summary,
    }


def full_path_deviation_analysis(
    ctx,
    trajectory_indices=None,
    steps=None,
    validity_tol=1e-6,
    horizons=None,
    plot=False,
):
    """Run free rollouts and calculate their complete RPS path deviation."""
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
        rollout["pred_raw"],
        rollout["target_raw"],
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


rps_path_deviation_analysis = full_path_deviation_analysis
