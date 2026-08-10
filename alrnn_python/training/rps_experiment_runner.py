from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset3d_compatible import TimeSeriesDataset
from modified_alrnn_model import Modified_AL_RNN, train_sh


def centered_rps_to_probability(centered):
    """
    convert centered RPS targets to full action probability targets
    """
    probability = np.empty(centered.shape[:-1] + (6,), dtype=centered.dtype)
    probability[..., 0] = centered[..., 0] + 1.0 / 3.0
    probability[..., 1] = centered[..., 1] + 1.0 / 3.0
    probability[..., 2] = 1.0 - probability[..., 0] - probability[..., 1]
    probability[..., 3] = centered[..., 2] + 1.0 / 3.0
    probability[..., 4] = centered[..., 3] + 1.0 / 3.0
    probability[..., 5] = 1.0 - probability[..., 3] - probability[..., 4]

    player1_probability = probability[..., :3]
    player2_probability = probability[..., 3:]

    # check that the probabilities are valid
    if not np.all((player1_probability >= 0) & (player1_probability <= 1)):
        raise ValueError("Player 1 action probabilities must satisfy 0 <= p <= 1")
    if not np.all((player2_probability >= 0) & (player2_probability <= 1)):
        raise ValueError("Player 2 action probabilities must satisfy 0 <= p <= 1")

    return probability


def load_training_arrays(experiment):
    train_data_path = Path(experiment["train_data_path"])
    train_observation = np.load(train_data_path).astype(np.float32,copy=False)

    metadata_path = train_data_path.parent / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    mean = np.asarray(metadata["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(metadata["normalization"]["std"], dtype=np.float32)
    nash_centered_target = train_observation * std + mean
    train_target = centered_rps_to_probability(nash_centered_target).astype(np.float32, copy=False)

    train_target_path = train_data_path
    target_representation = "decoded_from_normalized_nash_centered_4d"
    return (train_observation, train_target, train_target_path, metadata_path, target_representation)


def train_current_model(train_norm, train_target, model_path, config, use_W12, use_W21):
    seed = int(config["seed"])
    model_seed = int(config.get("model_seed", seed))
    batch_seed = int(config.get("batch_seed", seed))
    deterministic = bool(config.get("deterministic", True))
    torch_num_threads = int(config.get("torch_num_threads", 1))

    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.set_num_threads(torch_num_threads)

    N = train_norm.shape[-1]

    ssi = int(config.get("ssi", 25))
    checkpoint_interval = int(config.get("checkpoint_interval", 25))
    state_space_bins = int(config.get("state_space_bins", 30))
    state_loss_weight = float(config.get("state_loss_weight", 1.0))
    probability_loss_weight = float(config.get("probability_loss_weight", 1.0))
    progress_label = config.get("progress_label")
    use_best_model = True
    effective_config = {**dict(config),
        "M": int(config["M1"] + config["M2"]),
        "P": int(config["P1"] + config["P2"]),
        "N": int(N),
        "N_out": 6,
        "N1_out": 3,
        "N2_out": 3,
        "model_class": "Modified_AL_RNN",
        "seed": seed,
        "model_seed": model_seed,
        "batch_seed": batch_seed,
        "deterministic": deterministic,
        "torch_num_threads": torch_num_threads,
        "use_W12": use_W12,
        "use_W21": use_W21,
        "train_data_shape": list(train_norm.shape),
        "train_target_shape": list(train_target.shape),
        "target_representation": config.get(
            "target_representation",
            "decoded_from_normalized_nash_centered_4d",
        ),
        "train_target_path": config.get("train_target_path"),
        "metadata_path": config.get("metadata_path"),
        "ssi": ssi,
        "checkpoint_interval": checkpoint_interval,
        "state_space_bins": state_space_bins,
        "state_loss_weight": state_loss_weight,
        "probability_loss_weight": probability_loss_weight,
        "checkpoint_state_space_dimensions": 4,
        "checkpoint_state_space_representation": (
            "nonredundant_player_probabilities"
        ),
        "checkpoint_selection_metric": "state_space_divergence_binning",
        "use_best_model": use_best_model,
        "optimizer": "RAdam",
        "scheduler": "ExponentialLR",
        "loss_function": "state_MSE_plus_probability_MSE",
        "state_loss_function": "MSELoss",
        "probability_loss_function": "MSELoss",
    }
    model = Modified_AL_RNN(M1=config["M1"], P1=config["P1"], N1=config["N1"], M2=config["M2"], P2=config["P2"], N2=config["N2"], use_W12=use_W12, use_W21=use_W21)
    dataset = TimeSeriesDataset(train_norm, target_data=train_target, sequence_length=config["sequence_length"], batch_size=config["batch_size"])
    random.seed(batch_seed)
    np.random.seed(batch_seed)
    torch.manual_seed(batch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(batch_seed)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=config["start_learning_rate"])
    gamma = (config["end_learning_rate"] / config["start_learning_rate"]) ** (1 / config["num_epochs"])
    effective_config["scheduler_gamma"] = float(gamma)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_path.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    def save_checkpoint(
        completed_epochs,
        current_model,
        best_model,
        losses,
        state_losses,
        probability_losses,
        klx,
        dh,
    ):
        selected_model = best_model if use_best_model else current_model
        checkpoint = {
            "checkpoint_version": 7,
            "epoch": completed_epochs,
            "model_class": "Modified_AL_RNN",
            "model_state_dict": selected_model.state_dict(),
            "current_model_state_dict": current_model.state_dict(),
            "best_model_state_dict": best_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_config": effective_config,
            "training_metrics": {
                "losses": list(losses),
                "state_losses": list(state_losses),
                "probability_losses": list(probability_losses),
                "state_space_divergence": list(klx),
                "power_spectrum_error": list(dh),
            },
            "runtime": {
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
            },
        }
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{completed_epochs:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, model_path)
        np.save(model_path.parent / "losses.npy", np.asarray(losses, dtype=np.float32))
        with (model_path.parent / "config.json").open("w", encoding="utf-8") as f:
            json.dump(effective_config, f, indent=2)

    metrics = train_sh(
        model,
        dataset,
        optimizer,
        scheduler,
        loss_fn,
        num_epochs=config["num_epochs"],
        alpha=config["alpha"],
        n_interleave=config["n_interleave"],
        batches_per_epoch=config["batches_per_epoch"],
        ssi=ssi,
        use_best_model=use_best_model,
        checkpoint_interval=checkpoint_interval,
        checkpoint_callback=save_checkpoint,
        progress_label=progress_label,
        state_space_bins=state_space_bins,
        state_loss_weight=state_loss_weight,
        probability_loss_weight=probability_loss_weight,
    )

    return model, metrics


def format_run_id(run_id):
    run_id = int(run_id)
    if not 0 < run_id < 1000:
        raise ValueError("run_id must satisfy 0 < run_id < 1000")
    return f"{run_id:03d}"


def build_model_path(model_root, config, experiment):
    P = int(config["P1"] + config["P2"])
    return (
        Path(model_root)
        / experiment["dataset_type"]
        / experiment["regime"]
        / experiment["model_variant"]
        / experiment["connection_mode"]
        / f"P_{P}"
        / format_run_id(experiment["run_id"])
        / "model.pt"
    )


def run_experiment(model_root, base_config, experiment):
    config = {**dict(base_config), **dict(experiment.get("config", {}))}
    config.update({
        "experiment_name": experiment["name"],
        "dataset_type": experiment["dataset_type"],
        "regime": experiment["regime"],
        "model_variant": experiment["model_variant"],
        "connection_mode": experiment["connection_mode"],
        "run_id": int(experiment["run_id"]),
        "train_data_path": experiment["train_data_path"],
    })
    train_norm, train_target, train_target_path, metadata_path, target_representation = load_training_arrays(experiment)
    model_path = build_model_path(model_root, config, experiment)
    config["model_path"] = str(model_path)
    config["train_target_path"] = str(train_target_path)
    config["metadata_path"] = str(metadata_path)
    config["target_representation"] = target_representation

    _, metrics = train_current_model(
        train_norm,
        train_target,
        model_path,
        config,
        use_W12=experiment["use_W12"],
        use_W21=experiment["use_W21"],
    )

    return {
        "name": experiment["name"],
        "model_path": str(model_path),
        "dataset_type": experiment["dataset_type"],
        "regime": experiment["regime"],
        "model_variant": experiment["model_variant"],
        "connection_mode": experiment["connection_mode"],
        "P": config["P1"] + config["P2"],
        "run_id": format_run_id(experiment["run_id"]),
        "seed": config["seed"],
        "model_seed": config.get("model_seed", config["seed"]),
        "batch_seed": config.get("batch_seed", config["seed"]),
        "use_W12": experiment["use_W12"],
        "use_W21": experiment["use_W21"],
        "final_loss": metrics[0][-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    base_config = json.loads(args.base_config)
    experiment = json.loads(args.experiment)
    experiment["config"] = {
        **dict(experiment.get("config", {})),
        "progress_label": experiment["name"],
    }

    result = run_experiment(
        args.model_root,
        base_config,
        experiment,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
