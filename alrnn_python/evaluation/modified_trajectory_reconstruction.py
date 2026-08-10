"""Trajectory reconstruction checks for modified AL-RNN."""

from __future__ import annotations

import numpy as np
import torch

from modified_alrnn_eval_utils import (
        initialize_latent_state,
        probability_attractor_metrics,
        predict_one_step,
        rollout_on_test,
    )


def one_step_error(ctx, split="test"):
    observations = ctx["data"][f"{split}_norm"]
    probabilities = ctx["data"][f"{split}_probability"]
    current = observations[:, :-1, :].reshape(-1, ctx["N_input"])
    target = probabilities[:, 1:, :].reshape(-1, ctx["N_output"])

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
            batch = torch.tensor(
                x[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            z = initialize_latent_state(model, batch)

            for _ in range(n):
                z = model(z)

            preds.append(model.linear_score(z).cpu().numpy())

    return np.concatenate(preds, axis=0)


def n_step_error(ctx, n=10, split="test"):
    observations = ctx["data"][f"{split}_norm"]
    probabilities = ctx["data"][f"{split}_probability"]

    current = observations[:, :-n, :].reshape(-1, ctx["N_input"])
    target = probabilities[:, n:, :].reshape(-1, ctx["N_output"])

    pred = predict_n_step(ctx["model"], current, n=n)
    err = (pred - target) ** 2

    return {
        "n": int(n),
        "mse": float(err.mean()),
        "per_dim_mse": err.mean(axis=0).tolist(),
    }


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
    attractor = probability_attractor_metrics(
        rollout,
        n_bins=n_bins,
        smoothing=smoothing,
    )

    return {
        "one_step_mse": round(one_step["mse"], 6),
        "one_step_per_dim_mse": [round(x, 6) for x in one_step["per_dim_mse"]],
        "n_step_mse": round(n_step["mse"], 6),
        "n_step_per_dim_mse": [round(x, 6) for x in n_step["per_dim_mse"]],
        "prediction_horizon": n_step["n"],
        "rollout_mse": round(rollout["probability_mse"], 6),
        "target_is_probability": rollout["target_probability_validity"]["valid"],
        "attractor_metrics": attractor,
        "rollout": rollout,
    }
