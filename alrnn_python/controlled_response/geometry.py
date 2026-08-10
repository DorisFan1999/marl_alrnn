"""Shared tensor geometry and RPS player indexing."""

import torch


def player_probability(values, player):
    """Return one player's three probabilities from [..., 2, 3] or [..., 6]."""
    if player not in (1, 2):
        raise ValueError("player must be 1 or 2")
    if values.shape[-2:] == (2, 3):
        return values[..., player - 1, :]
    if values.shape[-1] == 6:
        start = 3 * (player - 1)
        return values[..., start:start + 3]
    raise ValueError("probabilities must end in [2, 3] or [6]")


def as_player_probabilities(values):
    """Represent joint RPS probabilities with trailing shape [2, 3]."""
    if values.shape[-2:] == (2, 3):
        return values
    if values.shape[-1] == 6:
        return values.reshape(*values.shape[:-1], 2, 3)
    raise ValueError("joint probabilities must end in [2, 3] or [6]")


def euclidean_norm(values):
    """Compute Euclidean norm over the final dimension."""
    return torch.linalg.vector_norm(values, dim=-1)


def rms(values):
    """
    Compute root-mean-square of all supplied scalar values.
    not batched
    """
    return torch.sqrt(torch.mean(values.square()))


def safe_cosine(first, second, eps):
    """Return cosine similarity, using NaN when either norm is too small."""
    first_norm = euclidean_norm(first)
    second_norm = euclidean_norm(second)
    denominator = first_norm * second_norm
    cosine = (first * second).sum(dim=-1) / denominator.clamp_min(eps)
    nan = torch.full_like(cosine, torch.nan)
    return torch.where((first_norm < eps) | (second_norm < eps), nan, cosine)


def probability_validity(values, atol):
    """
    Return a per-batch validity mask for joint RPS probabilities
    for 2 players, values shape [..., 2, 3] or [..., 6]
    """
    values = as_player_probabilities(values) # ensure trailing shape [..., 2, 3]
    finite = torch.isfinite(values).all(dim=(-1, -2)) # prob. should be finite
    bounded = ((values >= -atol) & (values <= 1.0 + atol)).all(dim=(-1, -2)) # prob. should be in [0, 1]
    sums = values.sum(dim=-1) # prob. should sum to 1 per player

    normalized = torch.isclose(sums, torch.ones_like(sums), atol=atol, rtol=0.0).all(dim=-1)

    return finite & bounded & normalized


def simplex_feasible(probability, atol):
    """
    Return a per-row mask for valid three-action probability vectors
    for one player, probability shape [..., 3]
    """
    if probability.shape[-1] != 3:
        raise ValueError("probability must have three components")
    
    finite = torch.isfinite(probability).all(dim=-1)
    bounded = ((probability >= -atol) & (probability <= 1.0 + atol)).all(dim=-1)
    
    normalized = torch.isclose(probability.sum(dim=-1), torch.ones_like(probability[..., 0]), atol=atol, rtol=0.0)
    
    return finite & bounded & normalized
