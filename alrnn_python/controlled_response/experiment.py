"""Experiment orchestration without plotting or file output."""

import torch

from .config import ExperimentConfig
from .geometry import player_probability
from .metrics import response_metrics
from .rollout import paired_free_rollout
from .types import (
    ExperimentResult,
    ExperimentRun,
    InterventionDiagnostics,
)


def _diagnostics(intervention, trajectory_index):
    return InterventionDiagnostics(
        feasible=bool(intervention.feasible[trajectory_index]),
        valid=bool(intervention.valid[trajectory_index]),
        invalid_reason=intervention.reasons[trajectory_index],
        delta_hat=intervention.delta_hat[trajectory_index].detach().clone(),
        input_norm_ratio=float(intervention.input_norm_ratio[trajectory_index]),
        input_cosine=float(intervention.input_cosine[trajectory_index]),
    )


def _build_runs(paired, config, source_player, alpha, sign, trajectory_ids, regime, model_seed): 
    target_player = 2 if source_player == 1 else 1
    runs = []

    for trajectory_index, trajectory_id in enumerate(trajectory_ids):
        diagnostics = _diagnostics(paired.intervention, trajectory_index)
        full_horizon = bool(paired.full_horizon_valid[trajectory_index])
        own = cross = None
        if diagnostics.valid and full_horizon:
            r_own = (
                player_probability(paired.p_shocked[trajectory_index], source_player)
                - player_probability(paired.p_base[trajectory_index], source_player)
            )
            r_cross = (
                player_probability(paired.p_shocked[trajectory_index], target_player)
                - player_probability(paired.p_base[trajectory_index], target_player)
            )

            own = response_metrics(
                r_own, diagnostics.delta_hat, config.cosine_eps, True
            )
            cross = response_metrics(
                r_cross, diagnostics.delta_hat, config.cosine_eps, False
            )

        runs.append(ExperimentRun(
            regime=regime,
            model_seed=model_seed,
            trajectory_id=int(trajectory_id),
            source_player=source_player,
            target_player=target_player,
            alpha=alpha,
            sign=sign,
            direction=config.direction,
            horizon=config.horizon,
            diagnostics=diagnostics,
            full_horizon_valid=full_horizon,
            failure_horizon=paired.failure_horizons[trajectory_index],
            failure_branch=paired.failure_branches[trajectory_index],
            p_base=paired.p_base[trajectory_index].detach().clone(),
            p_shocked=paired.p_shocked[trajectory_index].detach().clone(),
            own=own,
            cross=cross,
        ))

    return runs


def run_controlled_response_experiment(adapter, initial_conditions, config=None, regime="unspecified", model_seed=None, trajectory_ids=None):
    """Run every configured paired experiment and return in-memory results"""
    config = config or ExperimentConfig()
    count = len(initial_conditions)
    ids = tuple(range(1, count + 1)) if trajectory_ids is None else tuple(trajectory_ids)

    if len(ids) != count:
        raise ValueError("trajectory_ids must match initial_conditions")
    
    runs = []

    for source_player in config.source_players:
        for alpha in config.alphas:
            for sign in config.signs:
                delta = torch.tensor(config.delta(alpha, sign))
                paired = paired_free_rollout(adapter, initial_conditions, source_player, delta, config)
                runs.extend(_build_runs(paired, config, source_player, alpha, sign, ids, regime, model_seed))
                    
    return ExperimentResult(runs)
