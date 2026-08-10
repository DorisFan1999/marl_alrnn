"""Linear-region transition checks for modified AL-RNN rollouts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .modified_linear_region_functions import (
    latent_region_bits,
    used_subregion_analysis,
)


def plot_transition_connectome(
    transition,
    title="Subregion transition frequency",
    show_coordinates=True,
):
    if not transition.get("enabled", False):
        return

    connectome = np.asarray(transition["connectome"])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(connectome, cmap="Blues", aspect="equal")
    ax.set_title(title)
    if show_coordinates:
        ax.set_xlabel("next subregion")
        ax.set_ylabel("current subregion")
        ax.set_xticks(range(len(transition["region_labels"])))
        ax.set_yticks(range(len(transition["region_labels"])))
        ax.set_xticklabels(
            transition["region_labels"],
            rotation=90,
            fontsize="x-small",
        )
        ax.set_yticklabels(transition["region_labels"], fontsize="x-small")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="fraction of all transitions")
    fig.tight_layout()
    plt.show()


def transition_probability_analysis(generated_latent, model, plot=True):
    """Compute pooled within-trajectory joint transition frequencies.

    This is the player-separated counterpart of the original AL-RNN
    transition analysis. The only model-specific change is how the ReLU
    coordinates are selected: the last P1 coordinates of player 1's latent
    block and the last P2 coordinates of player 2's block are concatenated
    into one joint region code.
    """
    bits = latent_region_bits(generated_latent, model)
    used = used_subregion_analysis(generated_latent, model)
    labels = used["region_labels"]
    if not labels:
        return {"enabled": False, "reason": "no used subregions found"}

    label_to_index = {label: i for i, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), dtype=float)

    for seq_bits in bits:
        seq_labels = ["".join(map(str, row.astype(int))) for row in seq_bits]
        for current_label, next_label in zip(seq_labels[:-1], seq_labels[1:]):
            counts[label_to_index[current_label], label_to_index[next_label]] += 1.0

    total = counts.sum()
    transition_frequency = counts / total if total > 0 else counts
    row_sums = counts.sum(axis=1, keepdims=True)
    row_stochastic_transition = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts),
        where=row_sums > 0,
    )

    result = {
        "enabled": True,
        "num_used_subregions": used["num_used_subregions"],
        "total_possible_subregions": used["total_possible_subregions"],
        "region_labels": labels,
        "region_frequencies": used["region_frequencies"],
        "transition_counts": counts,
        "connectome": transition_frequency,
        "transition_probabilities": transition_frequency,
        "row_stochastic_transition": row_stochastic_transition,
        "description": (
            "Pooled within-trajectory transitions over the joint "
            "modified AL-RNN regions."
        ),
    }

    if plot:
        plot_transition_connectome(
            result,
            title="Joint subregion transition frequency",
        )

    return result
