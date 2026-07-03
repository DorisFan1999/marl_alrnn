"""All-test long free-rollout error metrics.

This module evaluates the autonomous long rollout of the trained AL-RNN on every test trajectory. 

For each test trajectory, observations should be:

    X_i = (x_{i,0}, x_{i,1}, ..., x_{i,T-1}),

the model receives only the initial condition x_{i,0}. It then generates

    hat{X_i} = ( hat{x_{i,1}}, hat{x_{i,2}}, ..., hat{x_{i,T-1}})

by recursively feeding its own previous prediction back into the model. 
The prediction is compared time-point by time-point with the ground truth

    (x_{i,1}, x_{i,2}, ..., x_{i,T-1}).

For one trajectory, the rollout MSE is

    MSE_i = 1 / ((T - 1) D) * sum_{t=1}^{T-1} ||xhat_{i,t} - x_{i,t}||_2^2,

where D is the state dimension.

The mean value is also equal to rollout_on_test(ctx)["mse"] when all test trajectories are used. 
"""

from __future__ import annotations

import numpy as np
import torch

from alrnn_eval_utils import rollout_on_test
from training.metrics import power_spectrum_error, state_space_divergence_binning


def rollout_mse_summary(rollout):
    pred = rollout["pred_norm"]
    true = rollout["target_norm"]
    per_rollout_mse = ((pred - true) ** 2).mean(axis=(1, 2)) # mean over time and state dimension for each rollout

    return {
        "mean_rollout_mse": float(rollout["mse"]), # rollout["mse"] is the mean over all test rollouts
        "median_rollout_mse": float(np.median(per_rollout_mse)),
        "worst_rollout_mse": float(np.max(per_rollout_mse)),
        "per_rollout_mse": per_rollout_mse.tolist(),
    }


def attractor_metrics(rollout, n_bins=6, smoothing=20):
    pred = rollout["pred_norm"]
    true = rollout["target_norm"]
    state_dim = true.shape[-1]

    pred_flat = pred.reshape(-1, state_dim)
    true_flat = true.reshape(-1, state_dim)
    finite = np.isfinite(pred_flat).all(axis=1) & np.isfinite(true_flat).all(axis=1)

    if finite.any():
        # put the finite predictions and true values into torch tensors for state space divergence computation
        dstsp = state_space_divergence_binning(
            torch.tensor(pred_flat[finite], dtype=torch.float32),
            torch.tensor(true_flat[finite], dtype=torch.float32),
            n_bins=n_bins,
        )
    else:
        dstsp = np.nan

    ps_errors = []
    for pred_i, true_i in zip(pred, true):
        if np.isfinite(pred_i).all() and np.isfinite(true_i).all():
            # compute power spectrum error for each rollout trajectory
            ps_errors.append(power_spectrum_error(pred_i, true_i, smoothing=smoothing))

    return {
        "state_space_divergence": float(dstsp),
        "power_spectrum_error_mean": float(np.mean(ps_errors)) if ps_errors else np.nan,
        "power_spectrum_error_std": float(np.std(ps_errors)) if ps_errors else np.nan,
    }


def test_long_rollout_analysis(ctx, n_bins=6, smoothing=20):
    test = ctx["data"]["test_norm"]
    indices = list(range(len(test)))
    steps = test.shape[1] - 1
    rollout = rollout_on_test(ctx, trajectory_indices=indices, steps=steps) # mse of all test rollout trajectories

    mse = rollout_mse_summary(rollout) 
    attractor = attractor_metrics(rollout, n_bins=n_bins, smoothing=smoothing)

    return {
        "n_test_rollouts": len(indices),
        "steps": steps,
        "state_dim": ctx["N"],
        "mean_rollout_mse": mse["mean_rollout_mse"],
        "median_rollout_mse": mse["median_rollout_mse"],
        "worst_rollout_mse": mse["worst_rollout_mse"],
        "state_space_divergence": attractor["state_space_divergence"],
        "power_spectrum_error_mean": attractor["power_spectrum_error_mean"],
        "power_spectrum_error_std": attractor["power_spectrum_error_std"],
        "per_rollout_mse": mse["per_rollout_mse"],
        "rollout": rollout,
    }
