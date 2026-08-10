"""Full-test rollout MSE, state-space divergence, and power-spectrum error."""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.metrics import power_spectrum_error


def _sparse_histogram(values, minimum, maximum, n_bins):
    coordinates = (
        n_bins * (values - minimum) / (maximum - minimum)
    ).astype(np.int64)
    in_range = ((coordinates > 0) & (coordinates < n_bins)).all(axis=1)
    coordinates = coordinates[in_range]
    if not len(coordinates):
        return np.array([], dtype=np.int64), np.array([], dtype=float)

    linear = np.ravel_multi_index(
        coordinates.T,
        dims=(n_bins,) * values.shape[1],
    )
    labels, counts = np.unique(linear, return_counts=True)
    return labels, counts.astype(float)


def state_space_divergence_6d(pred, true, n_bins=30, alpha=1e-5):
    """Compute the existing binned Dstsp definition without a dense 6D array."""
    pred = np.asarray(pred, dtype=float).reshape(-1, 6)
    true = np.asarray(true, dtype=float).reshape(-1, 6)
    minimum = true.min(axis=0)
    maximum = true.max(axis=0)

    pred_labels, pred_counts = _sparse_histogram(
        pred, minimum, maximum, n_bins
    )
    true_labels, true_counts = _sparse_histogram(
        true, minimum, maximum, n_bins
    )
    labels = np.union1d(pred_labels, true_labels)

    pred_full = np.zeros(len(labels), dtype=float)
    true_full = np.zeros(len(labels), dtype=float)
    pred_full[np.searchsorted(labels, pred_labels)] = pred_counts
    true_full[np.searchsorted(labels, true_labels)] = true_counts

    total_bins = n_bins ** true.shape[1]
    pred_denominator = pred_counts.sum() + alpha * total_bins
    true_denominator = true_counts.sum() + alpha * total_bins
    pred_probability = (pred_full + alpha) / pred_denominator
    true_probability = (true_full + alpha) / true_denominator

    divergence = np.sum(
        true_probability * np.log(true_probability / pred_probability)
    )
    empty_bins = total_bins - len(labels)
    if empty_bins:
        pred_empty = alpha / pred_denominator
        true_empty = alpha / true_denominator
        divergence += empty_bins * true_empty * np.log(true_empty / pred_empty)
    return float(divergence)


def calculate_rollout_metrics(pred, true, n_bins=30, smoothing=20):
    dh = [
        power_spectrum_error(true_i, pred_i, smoothing=smoothing)
        for true_i, pred_i in zip(true, pred)
    ]
    return {
        "rollout_mse": float(np.mean((pred - true) ** 2)),
        "state_space_divergence": state_space_divergence_6d(
            pred, true, n_bins=n_bins
        ),
        "power_spectrum_error": float(np.mean(dh)),
    }


def evaluate_rollout_metrics(rollouts, n_bins=30, smoothing=20):
    rows = []
    for rollout in rollouts:
        rows.append(
            {
                "P": rollout["P"],
                "model_seed": rollout["model_seed"],
                "run_id": rollout["run_id"],
                "n_test_trajectories": rollout["n_test_trajectories"],
                "steps": rollout["steps"],
                **calculate_rollout_metrics(
                    rollout["pred"],
                    rollout["true"],
                    n_bins=n_bins,
                    smoothing=smoothing,
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_rollout_metrics(per_seed):
    metrics = [
        "rollout_mse",
        "state_space_divergence",
        "power_spectrum_error",
    ]
    return per_seed.groupby("P")[metrics].agg(["mean", "std"])
