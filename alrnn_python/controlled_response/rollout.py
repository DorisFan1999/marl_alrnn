"""Paired h=0 intervention and deterministic free-rollout engine."""

from contextlib import contextmanager

import torch

from .geometry import probability_validity
from .types import PairedRollout


@contextmanager
def _evaluation_mode(adapter):
    model = getattr(adapter, "model", None)
    was_training = bool(model.training) if model is not None else False
    if model is not None:
        model.eval()
    try:
        # close grad during rollout
        with torch.no_grad():
            yield
    finally:
        if model is not None:
            model.train(was_training)


def _branch_validity(adapter, z, p, config):
    # check every horizon
    z = adapter.state_tensor(z)
    finite = torch.isfinite(z).reshape(z.shape[0], -1)
    finite = finite.all(dim=-1)
    return finite & probability_validity(p, config.probability_atol)


def _record_failures(h, base_valid, shocked_valid, failure_horizons, failure_branches):
    for index in range(len(base_valid)):
        if failure_horizons[index] is not None:
            continue
        base_failed = not bool(base_valid[index])
        shocked_failed = not bool(shocked_valid[index])
        if not base_failed and not shocked_failed:
            continue
        failure_horizons[index] = h
        if base_failed and shocked_failed:
            failure_branches[index] = "both"
        elif base_failed:
            failure_branches[index] = "baseline"
        else:
            failure_branches[index] = "shocked"


def paired_free_rollout(adapter, initial_conditions, source_player, delta, config):
    """
    Initialize once, intervene at h=0, then freely run both branches
    """
    with _evaluation_mode(adapter):
        # a batch of initial conditions, initialise once
        z0 = adapter.initialize(initial_conditions)
        z_base = adapter.clone_state(z0)
        p0_base = adapter.observe_probabilities(z_base)
        delta = delta.to(device=p0_base.device, dtype=p0_base.dtype)
        if delta.ndim == 1:
            delta = delta.expand(p0_base.shape[0], -1)
        
        # construct shocked branch
        intervention = adapter.apply_probability_impulse(z0, source_player, delta, config)
        z_shocked = intervention.z_shocked
        batch_size = p0_base.shape[0]
        failure_horizons = [None] * batch_size
        failure_branches = [None] * batch_size
        p_base_trajectory = []
        p_shocked_trajectory = []
        z_base_trajectory = []
        z_shocked_trajectory = []

        # run both branches for every horizon (steps)
        for h in range(config.horizon + 1):
            p_base = adapter.observe_probabilities(z_base)
            p_shocked = adapter.observe_probabilities(z_shocked)
            # total trajectory states
            p_base_trajectory.append(p_base)
            p_shocked_trajectory.append(p_shocked)
            # total latent states
            z_base_trajectory.append(adapter.state_tensor(z_base))
            z_shocked_trajectory.append(adapter.state_tensor(z_shocked))
            # check validity of both branches
            base_valid = _branch_validity(adapter, z_base, p_base, config)
            shocked_valid = _branch_validity(adapter, z_shocked, p_shocked, config)
            
            _record_failures(
                h, base_valid, shocked_valid,
                failure_horizons, failure_branches,
            )
            if h < config.horizon:
                z_base = adapter.step(z_base)
                z_shocked = adapter.step(z_shocked)

    full_horizon_valid = torch.tensor(
        [value is None for value in failure_horizons],
        dtype=torch.bool,
        device=intervention.feasible.device,
    )
    return PairedRollout(
        p_base=torch.stack(p_base_trajectory, dim=1),
        p_shocked=torch.stack(p_shocked_trajectory, dim=1),
        z_base=torch.stack(z_base_trajectory, dim=1),
        z_shocked=torch.stack(z_shocked_trajectory, dim=1),
        intervention=intervention,
        full_horizon_valid=full_horizon_valid,
        failure_horizons=failure_horizons,
        failure_branches=failure_branches,
    )
