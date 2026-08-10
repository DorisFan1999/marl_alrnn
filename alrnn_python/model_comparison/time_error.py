"""Joint 6D path error over the complete rollout time."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_time_error(pred, true):
    path_error = np.linalg.norm(pred - true, axis=-1)
    return np.median(path_error, axis=0)


def evaluate_time_errors(rollouts):
    rows = []
    for rollout in rollouts:
        median_error = calculate_time_error(
            rollout["pred"],
            rollout["true"],
        )
        rows.extend(
            {
                "P": rollout["P"],
                "model_seed": rollout["model_seed"],
                "run_id": rollout["run_id"],
                "time": time,
                "median_path_error": float(error),
            }
            for time, error in enumerate(median_error, start=1)
        )
    return pd.DataFrame(rows)


def summarize_time_error(per_seed):
    return (
        per_seed.groupby(["P", "time"])["median_path_error"]
        .agg(["mean", "std"])
        .reset_index()
    )


def _sample_plot_points(data, stride):
    data = data.sort_values("time")
    positions = np.arange(0, len(data), stride)
    if positions[-1] != len(data) - 1:
        positions = np.append(positions, len(data) - 1)
    return data.iloc[positions]


def plot_time_error(per_seed, regime, plot_stride=5):
    summary = summarize_time_error(per_seed)
    p_values = sorted(per_seed["P"].unique())
    seed_values = sorted(per_seed["model_seed"].unique())
    seed_colors = {
        seed: plt.get_cmap("tab10")(i)
        for i, seed in enumerate(seed_values)
    }
    fig, axes = plt.subplots(
        1,
        len(p_values),
        figsize=(10.5, 3.2),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for ax, P in zip(axes, p_values):
        p_seed = per_seed[per_seed["P"] == P]
        for seed, seed_data in p_seed.groupby("model_seed"):
            seed_data = _sample_plot_points(seed_data, plot_stride)
            ax.plot(
                seed_data["time"],
                seed_data["median_path_error"],
                color=seed_colors[seed],
                linewidth=0.8,
                alpha=0.75,
                label=f"seed {seed}",
            )

        p_summary = _sample_plot_points(
            summary[summary["P"] == P],
            plot_stride,
        )
        x = p_summary["time"].to_numpy()
        mean = p_summary["mean"].to_numpy()
        std = p_summary["std"].to_numpy()
        ax.fill_between(
            x,
            np.maximum(mean - std, 0.0),
            mean + std,
            color="0.4",
            alpha=0.14,
            label="±1 seed std",
        )
        ax.plot(x, mean, color="black", linewidth=1.8, label="seed mean")
        ax.set_title(f"P={P}", fontsize=9)
        ax.set_xlabel("rollout time", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("median joint 6D path error", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(labels),
        fontsize=7,
    )
    fig.suptitle(f"{regime}: full-path error over time", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    plt.show()
    return fig, axes
