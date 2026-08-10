"""Simplex checks and minimum-norm modified AL-RNN interventions."""

import torch

from .geometry import (euclidean_norm, player_probability, safe_cosine, simplex_feasible)
from .types import InterventionBatch


def impulse_delta(p_reference, alpha, sign, direction):
    """
    Construct target impulse delta for one player
    """
    direction = torch.as_tensor(direction, device=p_reference.device, dtype=p_reference.dtype)
    return (sign * alpha * direction).expand(p_reference.shape[0], -1)


def supports_all_impulses(p_source, alphas, signs, direction, atol=1e-6):
    """
    Check rows feasible for every configured sign at the largest alpha
    """
    valid = torch.ones(p_source.shape[0], dtype=torch.bool, device=p_source.device)
    
    for sign in signs:
        delta = impulse_delta(p_source, max(alphas), sign, direction)
        valid &= simplex_feasible(p_source + delta, atol) # check if all impulses are feasible
    
    return valid


def _quality_metrics(delta, delta_hat, config):
    """
    compute main metrics for impulse quality
    """
    delta_norm = euclidean_norm(delta).clamp_min(config.cosine_eps)
    ratio = euclidean_norm(delta_hat) / delta_norm
    cosine = safe_cosine(delta_hat, delta, config.cosine_eps)
    vector_error = euclidean_norm(delta_hat - delta) / delta_norm

    return ratio, cosine, vector_error


def _validate_interventions(feasible, target_same, cosine, vector_error, config):
    """Return a compact validity mask and one useful reason per row."""
    quality_ok = (
        torch.isfinite(cosine)
        & torch.isfinite(vector_error)
        & (cosine >= config.input_cosine_min) # decide impulse direction
        & (vector_error <= config.input_relative_error_max) # decide impulse magnitude
    )
    valid = feasible & target_same & quality_ok
    
    reasons = []
    for index in range(len(feasible)):
        if not bool(feasible[index]):
            reason = "infeasible_impulse"
        elif not bool(target_same[index]):
            reason = "target_modified_at_h0"
        elif not bool(quality_ok[index]):
            reason = "intervention_mismatch"
        else:
            reason = None
        reasons.append(reason)

    return valid, reasons



def apply_modified_intervention(model, z0, source_player, delta, config):
    """
    Apply a batched impulse to source observable coordinates
    """
    # get probability at h=0
    p0 = model.linear_score(z0).reshape(-1, 2, 3)
    p0_source = player_probability(p0, source_player)

    # construct desired probability p^* = p0 + delta and check feasibility
    # delta is impulse in probability space
    p_target = p0_source + delta
    feasible = simplex_feasible(p_target, config.probability_atol)

    # find the source observable latent coordinates
    # player 1: z[..., 0:N1], player 2: z[..., M1:M1+N2]
    source_start = 0 if source_player == 1 else model.M1
    N_source = model.N1 if source_player == 1 else model.N2
    V_source = model.V1 if source_player == 1 else model.V2
    source_observable = slice(source_start, source_start + N_source)

    # Softmax is invariant to adding the same scalar to all logits
    # C = I - (1/3)11^T removes this common-logit direction, leaving
    # only the two-dimensional subspace that determines RPS probabilities.
    C = torch.eye(3, device=z0.device, dtype=z0.dtype)
    C -= torch.ones_like(C) / 3.0
    
    # V_obs maps source observable latent coordinates to logits.
    # Therefore C @ V_obs maps a latent perturbation delta_z to its
    # probability-relevant centered-logit change.
    C_V_source = C @ V_source[:, :N_source]
    
    # For p = softmax(l), C log(p) = C l.  Thus the centered-logit
    # change required to move from p0 to p_star is
    # C @ [log(p_star) - log(p0)]
    tiny = torch.finfo(z0.dtype).tiny
    log_change = (
        p_target.clamp_min(tiny).log()
        - p0_source.clamp_min(tiny).log()
    )

    centered_log_change = log_change @ C.T
    
    # Solve (C @ V_obs) delta_z = C [log(p_star) - log(p0)].
    # The Moore-Penrose pseudoinverse gives the minimum-L2-norm latent
    # perturbation when multiple solutions are available.
    delta_z = centered_log_change @ torch.linalg.pinv(C_V_source).T


    z_shocked = z0.clone()
    # z_shocked = z0 + delta_z if feasible else z0
    z_shocked[..., source_observable] += torch.where(feasible.unsqueeze(-1), delta_z, torch.zeros_like(delta_z))
    # get p_shocked = softmax(V @ z_shocked + b)
    p_shocked = model.linear_score(z_shocked).reshape(-1, 2, 3)
    # delta_hat = p_shocked_source(0) - p0_source
    delta_hat = (
        player_probability(p_shocked, source_player)
        - p0_source
    )

    # compute actual matrics for impulse quality
    ratio, cosine, vector_error = _quality_metrics(delta, delta_hat, config)

    # find target player
    target_player = 2 if source_player == 1 else 1
    # it is not allowed to change the target player at t=0
    target_same = torch.isclose(
        player_probability(p_shocked, target_player),
        player_probability(p0, target_player),
        atol=max(config.probability_atol, 10.0 * torch.finfo(z0.dtype).eps),
        rtol=0.0,
    ).all(dim=-1)

    valid, reasons = _validate_interventions(feasible, target_same, cosine, vector_error, config)
    
    return InterventionBatch(z_shocked, feasible, valid, reasons, delta_hat, ratio, cosine)
