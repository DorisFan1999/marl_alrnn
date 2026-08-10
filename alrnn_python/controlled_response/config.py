"""Configuration objects for controlled-response experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated settings shared by all controlled-response runs."""

    horizon: int = 300
    alphas: tuple[float, ...] = (0.005, 0.01, 0.02)
    direction: tuple[float, float, float] = (-1.0, 1.0, 0.0)
    signs: tuple[int, ...] = (1, -1)
    source_players: tuple[int, ...] = (1, 2)
    input_cosine_min: float = 0.999
    input_relative_error_max: float = 1e-3
    cosine_eps: float = 1e-10
    probability_atol: float = 1e-6

    def __post_init__(self):
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not self.alphas or any(alpha <= 0 for alpha in self.alphas):
            raise ValueError("alphas must contain positive values")
        if len(self.direction) != 3:
            raise ValueError("direction must contain three components")
        if abs(sum(self.direction)) > self.probability_atol:
            raise ValueError("direction components must sum to zero")
        if sum(value * value for value in self.direction) <= 0:
            raise ValueError("direction must be nonzero")
        if not self.signs or any(sign not in (-1, 1) for sign in self.signs):
            raise ValueError("signs must contain only -1 or 1")
        if not self.source_players or any(
            player not in (1, 2) for player in self.source_players
        ):
            raise ValueError("source_players must contain only 1 or 2")

    def delta(self, alpha, sign):
        """
        construct impulse vector
        delta = sign * alpha * direction
        """
        if alpha <= 0 or sign not in (-1, 1):
            raise ValueError("alpha must be positive and sign must be -1 or 1")
        return tuple(sign * alpha * value for value in self.direction)
