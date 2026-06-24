"""Shared loading, rollout, and saving helpers for AL-RNN evaluation."""

from __future__ import annotations

import json
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

from training.alrnn_model import AL_RNN, predict_free_sequence


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    with (data_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    train_norm = np.load(data_dir / "train_trajectories_norm.npy").astype(np.float32)
    test_norm = np.load(data_dir / "test_trajectories_norm.npy").astype(np.float32)

    train_raw_path = data_dir / "train_trajectories_raw.npy"
    test_raw_path = data_dir / "test_trajectories_raw.npy"
    train_raw = (
        np.load(train_raw_path).astype(np.float32)
        if train_raw_path.exists()
        else unnormalize(train_norm, metadata)
    )
    test_raw = (
        np.load(test_raw_path).astype(np.float32)
        if test_raw_path.exists()
        else unnormalize(test_norm, metadata)
    )

    return {
        "data_dir": data_dir,
        "metadata": metadata,
        "train_norm": train_norm,
        "test_norm": test_norm,
        "train_raw": train_raw,
        "test_raw": test_raw,
    }


def load_model(model_path, M, P, N, device="cpu"):
    model = AL_RNN(M=M, P=P, N=N).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_eval_context(data_dir, model_path, M, P, device="cpu"):
    data = load_dataset(data_dir)
    # Determine the dimensions
    N = data["test_norm"].shape[-1]
    model = load_model(model_path, M=M, P=P, N=N, device=device)

    return {
        "data": data,
        "model": model,
        "model_path": Path(model_path),
        "M": M,
        "P": P,
        "N": N,
        "device": device,
    }


def normalization_arrays(metadata, state_dim):
    norm = metadata.get("normalization", {})
    mean = np.asarray(norm.get("mean", np.zeros(state_dim)), dtype=np.float32)
    std = np.asarray(norm.get("std", np.ones(state_dim)), dtype=np.float32)
    return mean, std


def normalize(x, metadata):
    mean, std = normalization_arrays(metadata, x.shape[-1])
    return (x - mean) / std


def unnormalize(x, metadata):
    mean, std = normalization_arrays(metadata, x.shape[-1])
    return x * std + mean


def one_step_readout(model, x):
    if x.ndim == 1:
        x = x.unsqueeze(0)

    z = x @ model.B
    z[:, :model.N] = x
    return model(z)[:, :model.N]


def predict_one_step(model, x, batch_size=10_000):
    device = next(model.parameters()).device
    preds = []

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            pred = one_step_readout(model, batch)
            preds.append(pred.cpu().numpy())

    return np.concatenate(preds, axis=0)


def rollout_on_test(ctx, trajectory_indices=None, steps=None):
    test_norm = ctx["data"]["test_norm"]

    if trajectory_indices is None:
        trajectory_indices = list(range(len(test_norm)))

    selected = test_norm[trajectory_indices]
    # should start from t=0
    steps = selected.shape[1] - 1 if steps is None else min(steps, selected.shape[1] - 1)

    device = next(ctx["model"].parameters()).device
    x0 = torch.tensor(selected[:, 0, :], dtype=torch.float32, device=device)

    with torch.no_grad():
        latent = predict_free_sequence(ctx["model"], x0, steps).cpu().numpy()

    pred_norm = latent[:, :, :ctx["N"]]
    target_norm = selected[:, 1:steps + 1, :]
    pred_raw = unnormalize(pred_norm, ctx["data"]["metadata"])
    target_raw = ctx["data"]["test_raw"][trajectory_indices, 1:steps + 1, :]

    return {
        "trajectory_indices": trajectory_indices,
        "steps": steps,
        "pred_norm": pred_norm,
        "target_norm": target_norm,
        "pred_raw": pred_raw,
        "target_raw": target_raw,
        "latent": latent,
        "mse": float(np.mean((pred_norm - target_norm) ** 2)),
    }
