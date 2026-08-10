"""Linear-region functions for modified AL-RNN latent states."""

from __future__ import annotations

import numpy as np


def convert_to_bits(data):
    return (data > 0).astype(int)


def validate_generated_latent(generated_latent):
    latent = np.asarray(generated_latent)
    if latent.ndim != 3:
        raise ValueError(
            "generated_latent must have shape "
            "(n_trajectories, n_steps, M)"
        )
    if latent.shape[0] == 0:
        raise ValueError("generated_latent must contain at least one trajectory")
    if not np.isfinite(latent).all():
        raise ValueError("generated_latent contains NaN or Inf")

    return latent


def region_labels(bits):
    if np.prod(bits.shape[:-1]) == 0:
        return []
    if bits.shape[-1] == 0:
        return [""]

    flat_bits = bits.reshape(-1, bits.shape[-1])
    return sorted({"".join(map(str, row.astype(int))) for row in flat_bits})


def nonlinear_latent_indices(model):
    indices = []

    if model.P1 > 0:
        indices.extend(range(model.M1 - model.P1, model.M1))

    if model.P2 > 0:
        start2 = model.M1 + model.M2 - model.P2
        indices.extend(range(start2, model.M1 + model.M2))

    return indices


def latent_region_bits(generated_latent, model):
    """Convert all ReLU variables to joint region bit codes."""
    latent = validate_generated_latent(generated_latent)
    if model.P1 < 0 or model.P2 < 0:
        raise ValueError("P1 and P2 must be non-negative")
    if model.P1 + model.P2 <= 0:
        raise ValueError("P1 + P2 must be a positive integer")
    if model.P1 > model.M1 or model.P2 > model.M2:
        raise ValueError("P1 <= M1 and P2 <= M2 must hold")

    expected_latent_dim = model.M1 + model.M2
    if latent.shape[-1] != expected_latent_dim:
        raise ValueError(
            f"Expected latent dimension {expected_latent_dim}, "
            f"found {latent.shape[-1]}"
        )

    indices = nonlinear_latent_indices(model)
    relu_latent = latent[..., indices]
    return np.asarray([convert_to_bits(seq) for seq in relu_latent], dtype=int)


def used_subregion_analysis(generated_latent, model):
    """Count joint linear subregions visited by a rollout."""
    bits = latent_region_bits(generated_latent, model)
    labels = region_labels(bits)

    if not labels:
        region_counts = np.array([], dtype=float)
    elif bits.shape[-1] == 0:
        region_counts = np.array([np.prod(bits.shape[:-1])], dtype=float)
    else:
        flat_bits = bits.reshape(-1, bits.shape[-1])
        region_counts = np.array(
            [
                np.sum(
                    [
                        "".join(map(str, row.astype(int))) == label
                        for row in flat_bits
                    ]
                )
                for label in labels
            ],
            dtype=float,
        )
    order = np.argsort(region_counts)[::-1]

    return {
        "bits": bits,
        "num_used_subregions": int(len(labels)),
        "total_possible_subregions": int(2 ** (model.P1 + model.P2)),
        "region_labels": [labels[i] for i in order],
        "region_frequencies": region_counts[order],
    }
