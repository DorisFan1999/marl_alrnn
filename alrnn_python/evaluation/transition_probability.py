"""Used-subregion and transition-probability checks for AL-RNN rollouts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

try:
    from .linear_region_functions import convert_to_bits
except ImportError:
    from linear_region_functions import convert_to_bits


def latent_region_bits(generated_latent, P):
    """Convert multi-trajectory latent rollouts to ReLU-region bit codes."""
    latent = np.asarray(generated_latent)
    if latent.ndim != 3:
        raise ValueError("generated_latent must have shape (n_trajectories, n_steps, M)")
    if P is None or P <= 0:
        raise ValueError("P must be a positive integer")
    if P > latent.shape[-1]:
        raise ValueError(f"P={P} cannot exceed latent dimension M={latent.shape[-1]}")

    relu_latent = latent[:, :, -P:]
    return np.asarray([convert_to_bits(seq) for seq in relu_latent], dtype=int)


def region_labels(bits):
    """Return sorted symbolic region labels visited by the rollout."""
    flat_bits = bits.reshape(-1, bits.shape[-1])
    labels = sorted({"".join(map(str, row.astype(int))) for row in flat_bits})
    return labels


def used_subregion_analysis(generated_latent, P):
    """Count visited AL-RNN linear subregions across all rollout trajectories."""
    bits = latent_region_bits(generated_latent, P)
    flat_bits = bits.reshape(-1, bits.shape[-1])
    labels = region_labels(bits)

    region_counts = np.array(
        [np.sum(["".join(map(str, row.astype(int))) == label for row in flat_bits]) for label in labels],
        dtype=float,
    )
    order = np.argsort(region_counts)[::-1]

    return {
        "bits": bits,
        "num_used_subregions": int(len(labels)),
        "total_possible_subregions": int(2 ** P),
        "region_labels": [labels[i] for i in order],
        "region_frequencies": region_counts[order],
    }


def transition_probability_analysis(generated_latent, P, plot=True):
    """Compute pooled within-trajectory transition frequencies.

    Transitions are counted only between consecutive states inside the same
    rollout trajectory, then pooled across trajectories. This avoids artificial
    transitions from the end of one rollout to the start of another.
    """
    bits = latent_region_bits(generated_latent, P)
    used = used_subregion_analysis(generated_latent, P)
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
        "description": "Pooled within-trajectory transition frequencies over active AL-RNN linear subregions.",
    }

    if plot:
        plot_transition_connectome(result)

    return result


def plot_transition_connectome(transition):
    if not transition.get("enabled", False):
        return

    connectome = np.asarray(transition["connectome"])

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(connectome, cmap="Blues", aspect="equal")
    ax.set_title("Linear subregion transition frequency")
    ax.set_xlabel("next subregion")
    ax.set_ylabel("current subregion")
    ax.set_xticks(range(len(transition["region_labels"])))
    ax.set_yticks(range(len(transition["region_labels"])))
    ax.set_xticklabels(transition["region_labels"], rotation=90, fontsize="x-small")
    ax.set_yticklabels(transition["region_labels"], fontsize="x-small")
    fig.colorbar(im, ax=ax, label="fraction of all transitions")
    fig.tight_layout()
    plt.show()
