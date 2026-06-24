from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from alrnn_eval_utils import predict_one_step, rollout_on_test
from training.metrics import power_spectrum_error, state_space_divergence_binning


def one_step_error(ctx, split="test"):
    trajectories = ctx["data"][f"{split}_norm"]
    current = trajectories[:, :-1, :].reshape(-1, ctx["N"])
    target = trajectories[:, 1:, :].reshape(-1, ctx["N"])

    pred = predict_one_step(ctx["model"], current)
    err = (pred - target) ** 2

    return {
        "mse": float(err.mean()),
        "per_dim_mse": err.mean(axis=0).tolist(),
    }


def predict_n_step(model, x, n, batch_size=10_000):
    device = next(model.parameters()).device
    preds = []

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            z = batch @ model.B
            z[:, :model.N] = batch

            for _ in range(n):
                z = model(z)

            preds.append(z[:, :model.N].cpu().numpy())

    return np.concatenate(preds, axis=0)


def n_step_error(ctx, n=10, split="test"):
    trajectories = ctx["data"][f"{split}_norm"]

    current = trajectories[:, :-n, :].reshape(-1, ctx["N"])
    target = trajectories[:, n:, :].reshape(-1, ctx["N"])

    pred = predict_n_step(ctx["model"], current, n=n)
    err = (pred - target) ** 2

    return {
        "n": int(n),
        "mse": float(err.mean()),
        "per_dim_mse": err.mean(axis=0).tolist(),
    }


def attractor_metrics(rollout, n_bins=6, smoothing=20):
    pred = rollout["pred_norm"]
    true = rollout["target_norm"]
    state_dim = true.shape[-1]

    pred_flat = torch.tensor(pred.reshape(-1, state_dim), dtype=torch.float32)
    true_flat = torch.tensor(true.reshape(-1, state_dim), dtype=torch.float32)
    dstsp = state_space_divergence_binning(pred_flat, true_flat, n_bins=n_bins)

    dh = []
    for i in range(len(pred)):
        dh.append(power_spectrum_error(pred[i], true[i], smoothing=smoothing))

    return {
        "state_space_divergence": round(float(dstsp), 6),
        #"state_space_bins": int(n_bins),
        "power_spectrum_error_mean": round(float(np.mean(dh)), 6),
        "power_spectrum_error_std": round(float(np.std(dh)), 6),
        #"power_spectrum_smoothing": smoothing,
    }

"""
def plot_trajectory_overlay(rollout, max_trajectories=6):
    true = rollout["target_raw"]
    pred = rollout["pred_raw"]
    n = min(max_trajectories, len(true))

    fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n), sharex=True)
    axes = np.atleast_1d(axes)

    for k, ax in enumerate(axes):
        for d in range(true.shape[-1]):
            ax.plot(true[k, :, d], color=f"C{d}", alpha=0.35)
            ax.plot(pred[k, :, d], color=f"C{d}", linestyle="--", alpha=0.85)
        ax.set_title(f"trajectory {rollout['trajectory_indices'][k]}")

    axes[-1].set_xlabel("step")
    fig.suptitle("Trajectory reconstruction")
    fig.tight_layout()
    plt.show()
"""

def trajectory_reconstruction_analysis(
    ctx,
    n_rollout=6,
    steps=None,
    prediction_horizon=10,
    plot=True,
    n_bins=6,
    smoothing=20,
):
    indices = list(range(min(n_rollout, len(ctx["data"]["test_norm"]))))
    one_step = one_step_error(ctx, split="test")
    n_step = n_step_error(ctx, n=prediction_horizon, split="test")
    rollout = rollout_on_test(ctx, trajectory_indices=indices, steps=steps)
    attractor = attractor_metrics(rollout, n_bins=n_bins, smoothing=smoothing)

    #if plot:
        #plot_trajectory_overlay(rollout)

    return {
        "one_step_mse": round(one_step["mse"], 6),
        "one_step_per_dim_mse": [round(x, 6) for x in one_step["per_dim_mse"]],
        "n_step_mse": round(n_step["mse"], 6),
        "n_step_per_dim_mse": [round(x, 6) for x in n_step["per_dim_mse"]],
        "prediction_horizon": n_step["n"],
        "rollout_mse": round(rollout["mse"], 6),
        "attractor_metrics": attractor,
        "rollout": rollout,
    }
