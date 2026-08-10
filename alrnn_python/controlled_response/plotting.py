"""Small plotting helpers that return Matplotlib objects without side effects."""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch


def select_runs(result, source_player, alpha=None, sign=None):
    return [
        run for run in result.valid_runs()
        if run.source_player == source_player
        and (alpha is None or run.alpha == alpha)
        and (sign is None or run.sign == sign)
    ]


def _aggregate(tensors, statistic):
    stacked = torch.stack(list(tensors))
    if statistic == "median":
        values = torch.nanmedian(stacked, dim=0).values
    elif statistic == "mean":
        values = torch.nanmean(stacked, dim=0)
    else:
        raise ValueError("statistic must be 'mean' or 'median'")
    return values.detach().cpu().numpy()


def plot_response_norms(result, response="cross", source_player=1, sign=1, statistic="median"):
    """Plot aggregated own or cross distance curves across impulse magnitudes."""
    if response not in ("own", "cross"):
        raise ValueError("response must be 'own' or 'cross'")
    selected = select_runs(result, source_player, sign=sign)
    groups = defaultdict(list)
    for run in selected:
        groups[run.alpha].append(getattr(run, response).norm)
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    for alpha, values in sorted(groups.items()):
        curve = _aggregate(values, statistic)
        ax.plot(np.arange(len(curve)), curve, label=f"α={alpha:g}")
    ax.set(xlabel="response horizon h", ylabel=f"{response}_distance")
    ax.set_title(f"Player {source_player} impulse: {statistic} {response} distance")
    if groups:
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, ax


def plot_own_cosine(result, source_player=1, sign=1, statistic="median"):
    """Plot own-response direction relative to the actual h=0 impulse."""
    selected = select_runs(result, source_player, sign=sign)
    groups = defaultdict(list)
    for run in selected:
        groups[run.alpha].append(run.own.cosine)
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    for alpha, values in sorted(groups.items()):
        ax.plot(_aggregate(values, statistic), label=f"α={alpha:g}")
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set(xlabel="response horizon h", ylabel="own-response cosine", ylim=(-1.05, 1.05))
    ax.set_title(f"Player {source_player} impulse: {statistic} direction")
    if groups:
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, ax
