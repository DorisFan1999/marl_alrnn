"""Across-seed summaries for controlled-response experiments."""

from pathlib import Path

import pandas as pd

from evaluation.modified_alrnn_eval_utils import load_eval_context

from .adapters import ModifiedALRNNAdapter
from .experiment import run_controlled_response_experiment


def seed_result_table(result, P, run_id, use_W12, use_W21):
    """Summarize trajectories within one seed for every impulse condition."""
    frame = result.summary_dataframe()
    frame["analysis_valid"] = (
        frame["intervention_valid"] & frame["full_horizon_valid"]
    )
    frame["P"] = P
    frame["run_id"] = run_id
    frame["use_W12"] = use_W12
    frame["use_W21"] = use_W21
    grouping = [
        "P", "run_id", "regime", "model_seed", "use_W12", "use_W21",
        "horizon", "source_player", "target_player", "alpha", "sign", "direction",
    ]
    return (
        frame.groupby(grouping, as_index=False, dropna=False)
        .agg(
            n_trajectories=("trajectory_id", "size"),
            valid_trajectories=("analysis_valid", "sum"),
            median_own_rms=("own_rms", "median"),
            median_cross_rms=("cross_rms", "median"),
            max_cross_rms=("cross_rms", "max"),
        )
    )


def run_seed_response_analysis(model_paths, data_dir, config, regime, device="cpu"):
    """Run all supplied seed checkpoints and return Table 1."""
    paths = [Path(path) for path in model_paths]
    if not paths:
        raise ValueError("model_paths must not be empty")
    tables = []
    settings = set()
    for model_path in paths:
        context = load_eval_context(data_dir, model_path, device=device)
        settings.add((context["P"], context["use_W12"], context["use_W21"]))
        result = run_controlled_response_experiment(
            adapter=ModifiedALRNNAdapter(context["model"]),
            initial_conditions=context["data"]["test_norm"][:, 0, :],
            config=config,
            regime=regime,
            model_seed=context["train_config"]["model_seed"],
        )
        tables.append(seed_result_table(
            result=result,
            P=context["P"],
            run_id=model_path.parent.name,
            use_W12=context["use_W12"],
            use_W21=context["use_W21"],
        ))
    if len(settings) != 1:
        raise ValueError("all seed models must share P and W12/W21 settings")
    return pd.concat(tables, ignore_index=True).sort_values(
        ["model_seed", "source_player", "alpha", "sign"]
    ).reset_index(drop=True)


def summarize_across_seeds(seed_table):
    """Aggregate seed-level medians into Table 2."""
    grouping = [
        "P", "regime", "use_W12", "use_W21", "horizon",
        "source_player", "target_player", "alpha", "sign", "direction",
    ]
    return (
        seed_table.groupby(grouping, as_index=False, dropna=False)
        .agg(
            n_seeds=("model_seed", "nunique"),
            min_valid_trajectories=("valid_trajectories", "min"),
            median_seed_own_rms=("median_own_rms", "median"),
            median_seed_cross_rms=("median_cross_rms", "median"),
            max_seed_cross_rms=("median_cross_rms", "max"),
        )
    )
