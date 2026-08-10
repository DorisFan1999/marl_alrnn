"""Learned flow geometry checks for modified AL-RNN."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


from alrnn_eval_utils import normalize
from modified_alrnn_eval_utils import (
        predict_one_step,
        probability_to_centered_rps
        )
from flow_geometry import add_trajectory_arrow, unit_vectors


def learned_vector_field(ctx, player, grid_size=17):
    if ctx["N1_output"] != 3 or ctx["N2_output"] != 3:
        raise ValueError(
            "The RPS probability-simplex vector field requires three outputs "
            "per player"
        )
    if player not in (1, 2):
        raise ValueError("player must be 1 or 2")

    axis = np.linspace(0.0, 1.0, grid_size)
    X, Y = np.meshgrid(axis, axis)
    valid = X + Y <= 1.0
    probability = np.full(
        (grid_size * grid_size, ctx["N_output"]),
        1.0 / 3.0,
        dtype=np.float32,
    )

    start = 0 if player == 1 else ctx["N1_output"]
    probability[:, start] = X.ravel()
    probability[:, start + 1] = Y.ravel()
    probability[:, start + 2] = 1.0 - X.ravel() - Y.ravel()

    centered = probability_to_centered_rps(probability)
    observation = normalize(centered, ctx["data"]["metadata"])
    pred = predict_one_step(ctx["model"], observation)
    U = (pred[:, start] - probability[:, start]).reshape(X.shape)
    V = (pred[:, start + 1] - probability[:, start + 1]).reshape(Y.shape)
    U[~valid] = np.nan
    V[~valid] = np.nan
    return X, Y, U, V


def plot_flow_geometry(
    ctx,
    rollout,
    max_trajectories=6,
    grid_size=17,
    show_vector_field=False,
    zoom_to_trajectories=True,
    axis_padding=0.12,
):
    true = rollout["target_probability"]
    pred = rollout["pred_probability"]
    n = min(max_trajectories, len(true))

    panels = [
        (1, 0, 1, "player 1"),
        (
            2,
            ctx["N1_output"],
            ctx["N1_output"] + 1,
            "player 2",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (player, i, j, title) in zip(axes, panels):
        if show_vector_field:
            X, Y, U, V = learned_vector_field(
                ctx,
                player=player,
                grid_size=grid_size,
            )
            U_plot, V_plot = unit_vectors(U, V)
            ax.quiver(
                X,
                Y,
                U_plot,
                V_plot,
                angles="xy",
                scale_units="xy",
                scale=25,
                width=0.0035,
                color="black",
                alpha=0.35,
            )

        for k in range(n):
            ax.plot(
                true[k, :, i],
                true[k, :, j],
                color="C0",
                alpha=0.45,
                linewidth=1.2,
            )
            ax.plot(
                pred[k, :, i],
                pred[k, :, j],
                color="C1",
                linestyle="--",
                alpha=0.75,
                linewidth=1.3,
            )
            add_trajectory_arrow(
                ax,
                true[k, :, i],
                true[k, :, j],
                color="C0",
                position=0.58,
            )
            add_trajectory_arrow(
                ax,
                pred[k, :, i],
                pred[k, :, j],
                color="C1",
                position=0.72,
                linestyle="--",
            )

        n_actions = ctx["N1_output"] if player == 1 else ctx["N2_output"]
        equilibrium = 1.0 / n_actions
        ax.scatter([equilibrium], [equilibrium], color="black", marker="x")

        if zoom_to_trajectories:
            coordinates = np.concatenate(
                [
                    true[:n, :, [i, j]].reshape(-1),
                    pred[:n, :, [i, j]].reshape(-1),
                    np.array([equilibrium]),
                ]
            )
            lower = float(np.min(coordinates))
            upper = float(np.max(coordinates))
            span = max(upper - lower, 1e-6)
            padding = max(axis_padding * span, 0.01)
            lower = max(0.0, lower - padding)
            upper = min(1.0, upper + padding)
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
        else:
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)

        ax.set_title(title)
        ax.set_xlabel("rock probability")
        ax.set_ylabel("paper probability")
        ax.set_aspect("equal")

    axes[0].plot([], [], color="C0", label="dataset observation")
    axes[0].plot([], [], color="C1", linestyle="--", label="modified AL-RNN")
    axes[0].legend()
    fig.suptitle("Modified AL-RNN flow geometry")
    fig.tight_layout()
    plt.show()


def flow_geometry_analysis(
    ctx,
    rollout,
    plot=True,
    max_trajectories=6,
    grid_size=17,
    show_vector_field=False,
    zoom_to_trajectories=True,
    axis_padding=0.12,
):
    if ctx["N1_output"] < 2 or ctx["N2_output"] < 2:
        raise ValueError("Flow geometry requires at least two actions per player")

    if plot:
        plot_flow_geometry(
            ctx,
            rollout,
            max_trajectories=max_trajectories,
            grid_size=grid_size,
            show_vector_field=show_vector_field,
            zoom_to_trajectories=zoom_to_trajectories,
            axis_padding=axis_padding,
        )

    return {
        "expected_radial_behavior": (
            ctx["data"]["metadata"].get("expected_radial_behavior")
        ),
        "target_is_probability": rollout["target_probability_validity"]["valid"],
        "description": (
            "Player-specific action-probability trajectories for modified AL-RNN."
        ),
    }
