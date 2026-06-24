"""Learned flow geometry checks for AL-RNN."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from alrnn_eval_utils import normalize, predict_one_step, unnormalize


def learned_vector_field(ctx, dims, grid_lim, grid_size=17):
    x = np.linspace(-grid_lim, grid_lim, grid_size)
    y = np.linspace(-grid_lim, grid_lim, grid_size)
    X, Y = np.meshgrid(x, y)

    raw = np.zeros((grid_size * grid_size, ctx["N"]), dtype=np.float32)
    raw[:, dims[0]] = X.ravel()
    raw[:, dims[1]] = Y.ravel()

    norm = normalize(raw, ctx["data"]["metadata"])
    pred_norm = predict_one_step(ctx["model"], norm)
    pred_raw = unnormalize(pred_norm, ctx["data"]["metadata"])

    U = pred_raw[:, dims[0]] - raw[:, dims[0]]
    V = pred_raw[:, dims[1]] - raw[:, dims[1]]
    return X, Y, U.reshape(X.shape), V.reshape(Y.shape)


def unit_vectors(U, V, eps=1e-12):
    norm = np.sqrt(U ** 2 + V ** 2)
    return U / np.maximum(norm, eps), V / np.maximum(norm, eps)


def add_trajectory_arrow(ax, x, y, color, position=0.65, linestyle="-", scale=13):
    xy = np.column_stack([x, y])
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 3:
        return

    idx = int(np.clip(position * (len(xy) - 2), 0, len(xy) - 2))
    start = xy[idx]
    end = xy[idx + 1]
    if np.linalg.norm(end - start) <= 1e-12:
        return

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 1.4,
            "linestyle": linestyle,
            "mutation_scale": scale,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=6,
    )


def plot_flow_geometry(
    ctx,
    rollout,
    max_trajectories=6,
    grid_size=17,
    show_vector_field=False,
):
    true = rollout["target_raw"]
    pred = rollout["pred_raw"]
    n = min(max_trajectories, len(true))

    panels = [(0, 1, "player 1")]
    if ctx["N"] >= 4:
        panels.append((2, 3, "player 2"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    axes = np.atleast_1d(axes)

    for ax, (i, j, title) in zip(axes, panels):
        dims = (i, j)
        grid_lim = 1.15 * max(
            np.max(np.abs(true[:, :, dims])),
            np.max(np.abs(pred[:, :, dims])),
            1e-3,
        )
        if show_vector_field:
            X, Y, U, V = learned_vector_field(ctx, dims=dims, grid_lim=grid_lim, grid_size=grid_size)
            U_plot, V_plot = unit_vectors(U, V)
            ax.quiver(
                X,
                Y,
                U_plot,
                V_plot,
                angles="xy",
                scale_units="xy",
                scale=60,
                width=0.0035,
                color="black",
                alpha=0.35,
            )

        for k in range(n):
            ax.plot(true[k, :, i], true[k, :, j], color="C0", alpha=0.3)
            ax.plot(pred[k, :, i], pred[k, :, j], color="C1", linestyle="--", alpha=0.55)

            add_trajectory_arrow(ax, true[k, :, i], true[k, :, j], color="C0", position=0.58)
            add_trajectory_arrow(
                ax,
                pred[k, :, i],
                pred[k, :, j],
                color="C1",
                position=0.72,
                linestyle="--",
            )

        ax.scatter([0.0], [0.0], color="black", marker="x")
        ax.set_title(title)
        ax.axis("equal")

    axes[0].plot([], [], color="C0", label="true")
    axes[0].plot([], [], color="C1", linestyle="--", label="AL-RNN")
    axes[0].legend()
    fig.suptitle("Learned flow geometry")
    fig.tight_layout()
    plt.show()


def flow_geometry_analysis(ctx, rollout, plot=True):
    if plot:
        plot_flow_geometry(ctx, rollout)

    return {
        "expected_radial_behavior": ctx["data"]["metadata"].get("expected_radial_behavior"),
        "description": "Learned vector field plotted with true and recovered trajectories.",
    }
