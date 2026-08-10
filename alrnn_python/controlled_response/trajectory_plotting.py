"""Paired probability trajectories selected by median cross RMS."""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from evaluation.flow_geometry import add_trajectory_arrow

from .geometry import player_probability
from .plotting import select_runs


def _median_cross_rms_runs(result, source_player):
    groups = defaultdict(list)
    for run in select_runs(result, source_player):
        groups[(run.alpha, run.sign)].append(run)
    if not groups:
        raise ValueError("no matching valid runs")

    chosen = {}
    for condition, runs in groups.items():
        median_rms = float(np.median([run.cross.rms for run in runs]))
        chosen[condition] = min(
            runs,
            key=lambda run: (abs(run.cross.rms - median_rms), run.trajectory_id),
        )
    return chosen


def _through_window(values, window):
    if window is None:
        return values
    if window < 0:
        raise ValueError("window must be nonnegative")
    if window >= len(values):
        raise ValueError(
            f"window={window} exceeds available horizon={len(values) - 1}"
        )
    return values[:window + 1]


def _plot_pair(axis, run, observed_player, window):
    p_base = player_probability(run.p_base, observed_player)
    p_shocked = player_probability(run.p_shocked, observed_player)
    p_base = _through_window(p_base, window).detach().cpu().numpy()
    p_shocked = _through_window(p_shocked, window).detach().cpu().numpy()
    axis.plot(p_base[:, 0], p_base[:, 1], color="C0", linewidth=1.2, label="baseline")
    axis.plot(
        p_shocked[:, 0], p_shocked[:, 1], color="C1", linestyle="--",
        linewidth=1.2, label="shocked",
    )
    add_trajectory_arrow(axis, p_base[:, 0], p_base[:, 1], "C0", position=0.62)
    add_trajectory_arrow(
        axis, p_shocked[:, 0], p_shocked[:, 1], "C1",
        position=0.72, linestyle="--",
    )
    axis.scatter(p_base[0, 0], p_base[0, 1], color="C0", s=17, zorder=7)
    axis.scatter(p_shocked[0, 0], p_shocked[0, 1], color="C1", marker="x", s=24, zorder=7)
    axis.scatter(1 / 3, 1 / 3, color="black", marker="x", s=20)
    axis.set_title(
        f"α={run.alpha:g}, sign={run.sign:+d}, trajectory {run.trajectory_id}\n"
        f"cross RMS={run.cross.rms:.3g}",
        fontsize=8,
    )
    axis.set_xlabel("rock probability", fontsize=8)
    axis.set_ylabel("paper probability", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.set_aspect("equal", adjustable="box")


def plot_paired_probability_trajectories(result, source_player=1, response="own", window=None):
    """Plot one median-cross-RMS trajectory for every alpha and sign."""
    if response not in ("own", "cross"):
        raise ValueError("response must be 'own' or 'cross'")
    chosen = _median_cross_rms_runs(result, source_player)
    alphas = sorted({alpha for alpha, _ in chosen})
    signs = sorted({sign for _, sign in chosen}, reverse=True)
    fig, axes = plt.subplots(
        len(signs), len(alphas), squeeze=False,
        figsize=(3.1 * len(alphas), 2.8 * len(signs)),
    )
    for row, sign in enumerate(signs):
        for column, alpha in enumerate(alphas):
            axis = axes[row, column]
            run = chosen.get((alpha, sign))
            if run is None:
                axis.set_visible(False)
                continue
            observed_player = run.source_player if response == "own" else run.target_player
            _plot_pair(axis, run, observed_player, window)

    axes[0, 0].legend(fontsize=7)
    observed = source_player if response == "own" else 2 if source_player == 1 else 1
    final_horizon = window if window is not None else next(iter(chosen.values())).horizon
    fig.suptitle(
        f"Player {source_player} impulse: Player {observed} {response} trajectories "
        f"selected by median cross RMS, h=0..{final_horizon}",
        fontsize=10,
    )
    fig.tight_layout()
    return fig, axes
