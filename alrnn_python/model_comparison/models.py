"""Find trained models for one regime and connection mode."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def find_regime_models(
    model_root,
    regime,
    connection_mode,
    p_values=(2, 4, 8),
    model_variant="player_separate",
):
    rows = []
    group_root = (
        Path(model_root)
        / regime
        / model_variant
        / connection_mode
    )

    for P in p_values:
        for config_path in sorted((group_root / f"P_{P}").glob("*/config.json")):
            model_path = config_path.parent / "model.pt"
            if not model_path.exists():
                continue

            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)

            rows.append(
                {
                    "P": int(config["P1"] + config["P2"]),
                    "model_seed": int(config["model_seed"]),
                    "batch_seed": int(config["batch_seed"]),
                    "run_id": config_path.parent.name,
                    "model_path": str(model_path),
                }
            )

    models = pd.DataFrame(rows)
    if models.empty:
        raise FileNotFoundError(
            f"No models found for {regime}/{connection_mode} under {model_root}"
        )

    return models.sort_values(["P", "model_seed"]).reset_index(drop=True)


def check_seed_groups(models, p_values=(2, 4, 8), expected_seeds=5):
    counts = models.groupby("P").size()
    missing = [P for P in p_values if P not in counts.index]
    if missing:
        raise RuntimeError(f"Missing model groups for P={missing}")

    wrong = counts[counts != expected_seeds]
    if not wrong.empty:
        raise RuntimeError(
            f"Expected {expected_seeds} seeds per P; found {wrong.to_dict()}"
        )

