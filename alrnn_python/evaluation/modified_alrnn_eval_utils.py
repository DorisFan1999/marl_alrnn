"""Shared loading and rollout helpers for modified AL-RNN evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
PYTHON_DIR = THIS_DIR.parent
TRAINING_DIR = PYTHON_DIR / "training"

for path in (THIS_DIR, TRAINING_DIR, PYTHON_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from alrnn_eval_utils import load_dataset, unnormalize
from training.modified_alrnn_model import (
    Modified_AL_RNN,
    initialize_latent_state,
    predict_free_sequence,
)
from training.metrics import power_spectrum_error, state_space_divergence_binning


def centered_rps_to_probability(centered):
    """Convert 4D Nash-centered RPS coordinates to full 6D probabilities."""
    probability = np.empty(centered.shape[:-1] + (6,), dtype=centered.dtype)
    probability[..., 0] = centered[..., 0] + 1.0 / 3.0
    probability[..., 1] = centered[..., 1] + 1.0 / 3.0
    probability[..., 2] = 1.0 - probability[..., 0] - probability[..., 1]
    probability[..., 3] = centered[..., 2] + 1.0 / 3.0
    probability[..., 4] = centered[..., 3] + 1.0 / 3.0
    probability[..., 5] = 1.0 - probability[..., 3] - probability[..., 4]
    return probability


def normalized_rps_to_probability(normalized, metadata):
    """Undo normalization and recover full action probabilities."""
    centered = unnormalize(normalized, metadata)
    return centered_rps_to_probability(centered)


def probability_to_centered_rps(probability):
    """Map full 6D probabilities to the stored 4D centered coordinates."""
    centered = np.empty(probability.shape[:-1] + (4,), dtype=probability.dtype)
    centered[..., 0] = probability[..., 0] - 1.0 / 3.0
    centered[..., 1] = probability[..., 1] - 1.0 / 3.0
    centered[..., 2] = probability[..., 3] - 1.0 / 3.0
    centered[..., 3] = probability[..., 4] - 1.0 / 3.0
    return centered


def load_model(model_path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    train_config = checkpoint["train_config"]

    model = Modified_AL_RNN(
        M1=train_config["M1"],
        P1=train_config["P1"],
        N1=train_config["N1"],
        M2=train_config["M2"],
        P2=train_config["P2"],
        N2=train_config["N2"],
        use_W12=train_config["use_W12"],
        use_W21=train_config["use_W21"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train_config = dict(train_config)
    model.eval()
    return model


def load_eval_context(data_dir, model_path, device="cpu"):
    data = load_dataset(data_dir)
    model = load_model(model_path, device=device)

    data_dimension = data["test_norm"].shape[-1]
    if data_dimension != model.N:
        raise ValueError(
            f"Dataset observation dimension {data_dimension} does not match "
            f"the modified model input dimension N1 + N2 = {model.N}"
        )

    data["train_probability"] = normalized_rps_to_probability(data["train_norm"], data["metadata"])
    data["test_probability"] = normalized_rps_to_probability(data["test_norm"], data["metadata"])

    return {
        "data": data,
        "model": model,
        "model_path": Path(model_path),
        "M": model.M,
        "P": model.P,
        "M1": model.M1,
        "P1": model.P1,
        "M2": model.M2,
        "P2": model.P2,
        "N_input": model.N,
        "N1_input": model.N1,
        "N2_input": model.N2,
        "N_output": model.N_out,
        "N1_output": model.N1_out,
        "N2_output": model.N2_out,
        "use_W12": model.use_W12,
        "use_W21": model.use_W21,
        "train_config": model.train_config,
        "device": device,
    }


def probability_summary(values, N1_output=3, N2_output=3, tol=1e-6):
    values = np.asarray(values)
    if values.shape[-1] != N1_output + N2_output:
        raise ValueError(
            f"Expected probability dimension {N1_output + N2_output}, "
            f"found {values.shape[-1]}"
        )

    player1 = values[..., :N1_output]
    player2 = values[..., N1_output:]
    finite = np.isfinite(values)
    sum_error1 = np.abs(player1.sum(axis=-1) - 1.0)
    sum_error2 = np.abs(player2.sum(axis=-1) - 1.0)

    valid = (
        finite.all()
        and np.all(player1 >= -tol)
        and np.all(player1 <= 1.0 + tol)
        and np.all(player2 >= -tol)
        and np.all(player2 <= 1.0 + tol)
        and np.all(sum_error1 <= tol)
        and np.all(sum_error2 <= tol)
    )

    finite_values = values[finite]
    return {
        "valid": bool(valid),
        "min_probability": float(finite_values.min()) if finite_values.size else np.nan,
        "max_probability": float(finite_values.max()) if finite_values.size else np.nan,
        "max_sum_error_player1": float(np.max(sum_error1)),
        "max_sum_error_player2": float(np.max(sum_error2)),
    }


def probability_rollout_mse_summary(rollout):
    pred = rollout["pred_probability"]
    true = rollout["target_probability"]
    per_rollout_mse = ((pred - true) ** 2).mean(axis=(1, 2))
    median_rollout_mse = float(np.median(per_rollout_mse))
    median_rollout_index = int(
        np.argmin(np.abs(per_rollout_mse - median_rollout_mse))
    )
    worst_rollout_index = int(np.argmax(per_rollout_mse))

    return {
        "mean_rollout_mse": float(rollout["probability_mse"]),
        "median_rollout_mse": median_rollout_mse,
        "worst_rollout_mse": float(per_rollout_mse[worst_rollout_index]),
        "median_case_rollout": median_rollout_index + 1,
        "median_case_mse": float(per_rollout_mse[median_rollout_index]),
        "worst_case_rollout": worst_rollout_index + 1,
        "per_rollout_mse": per_rollout_mse.tolist(),
    }


def probability_attractor_metrics(rollout, n_bins=6, smoothing=20):
    pred = rollout["pred_probability"]
    true = rollout["target_probability"]
    state_dim = true.shape[-1]

    pred_flat = pred.reshape(-1, state_dim)
    true_flat = true.reshape(-1, state_dim)
    finite = np.isfinite(pred_flat).all(axis=1) & np.isfinite(true_flat).all(axis=1)

    if finite.any():
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
            ps_errors.append(
                power_spectrum_error(
                    pred_i,
                    true_i,
                    smoothing=smoothing,
                )
            )

    return {
        "state_space_divergence": float(dstsp),
        "power_spectrum_error_mean": (float(np.mean(ps_errors)) if ps_errors else np.nan),
        "power_spectrum_error_std": (float(np.std(ps_errors)) if ps_errors else np.nan)
    }


def one_step_readout(model, x):
    if x.ndim == 1:
        x = x.unsqueeze(0)

    z = initialize_latent_state(model, x)
    z = model(z)
    return model.linear_score(z)


def predict_one_step(model, x, batch_size=10_000):
    device = next(model.parameters()).device
    preds = []

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(
                x[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            pred = one_step_readout(model, batch)
            preds.append(pred.cpu().numpy())

    return np.concatenate(preds, axis=0)


def rollout_on_test(ctx, trajectory_indices=None, steps=None):
    test_observation = ctx["data"]["test_norm"]
    test_probability = ctx["data"]["test_probability"]

    if trajectory_indices is None:
        trajectory_indices = list(range(len(test_observation)))

    selected_observation = test_observation[trajectory_indices]
    selected_probability = test_probability[trajectory_indices]
    steps = (
        selected_observation.shape[1] - 1
        if steps is None
        else min(steps, selected_observation.shape[1] - 1)
    )

    device = next(ctx["model"].parameters()).device
    x0 = torch.tensor(
        selected_observation[:, 0, :],
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        latent_tensor = predict_free_sequence(ctx["model"], x0, steps)
        pred_tensor = ctx["model"].linear_score(latent_tensor)

    latent = latent_tensor.cpu().numpy()
    pred_probability = pred_tensor.cpu().numpy()
    target_probability = selected_probability[:, 1:steps + 1, :]

    pred_validity = probability_summary(
        pred_probability,
        ctx["N1_output"],
        ctx["N2_output"],
    )
    target_validity = probability_summary(
        target_probability,
        ctx["N1_output"],
        ctx["N2_output"],
    )

    if not target_validity["valid"]:
        raise RuntimeError("Recovered dataset targets are invalid action probabilities")

    return {
        "trajectory_indices": trajectory_indices,
        "steps": steps,
        "pred_probability": pred_probability,
        "target_probability": target_probability,
        "latent": latent,
        "probability_mse": float(
            np.mean((pred_probability - target_probability) ** 2)
        ),
        "pred_probability_validity": pred_validity,
        "target_probability_validity": target_validity
    }
