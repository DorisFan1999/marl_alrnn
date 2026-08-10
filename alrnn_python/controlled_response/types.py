"""Notebook-friendly result objects for controlled-response experiments."""

from dataclasses import dataclass, field

import torch


@dataclass
class InterventionDiagnostics:
    """Quality and feasibility diagnostics for one applied impulse."""

    feasible: bool # simplex_feasible
    valid: bool # feasible and target_same and quality_ok
    invalid_reason: str | None
    delta_hat: torch.Tensor # delta_hat = p_shocked_source(0) - p_base_source(0)
    input_norm_ratio: float # ||delta_hat|| / ||delta||
    input_cosine: float # cosine(delta_hat, delta)


@dataclass
class InterventionBatch:
    """Internal batched intervention output."""

    z_shocked: torch.Tensor
    feasible: torch.Tensor
    valid: torch.Tensor
    reasons: list[str | None]
    delta_hat: torch.Tensor
    input_norm_ratio: torch.Tensor
    input_cosine: torch.Tensor


@dataclass
class PairedRollout:
    """Batched baseline and shocked trajectories including h=0."""

    p_base: torch.Tensor
    p_shocked: torch.Tensor
    z_base: torch.Tensor
    z_shocked: torch.Tensor
    intervention: InterventionBatch
    full_horizon_valid: torch.Tensor
    failure_horizons: list[int | None]
    failure_branches: list[str | None]


@dataclass
class ResponseMetrics:
    """Complete response curve and its scalar trajectory summaries."""

    response: torch.Tensor
    norm: torch.Tensor
    cosine: torch.Tensor | None
    peak: float
    rms: float
    terminal: float
    time_to_peak: int


@dataclass
class ExperimentRun:
    """One trajectory under one source, sign, and impulse magnitude."""

    regime: str
    model_seed: int | None
    trajectory_id: int
    source_player: int
    target_player: int
    alpha: float
    sign: int
    direction: tuple[float, float, float]
    horizon: int
    diagnostics: InterventionDiagnostics
    full_horizon_valid: bool
    failure_horizon: int | None
    failure_branch: str | None
    p_base: torch.Tensor
    p_shocked: torch.Tensor
    own: ResponseMetrics | None
    cross: ResponseMetrics | None

    @property
    def valid(self):
        """Whether this run is eligible for full-horizon response analysis."""
        return self.diagnostics.valid and self.full_horizon_valid

    @property
    def invalid_reason(self):
        """Return intervention or rollout failure reason."""
        if self.diagnostics.invalid_reason is not None:
            return self.diagnostics.invalid_reason
        if not self.full_horizon_valid:
            return "invalid_rollout"
        return None

    def summary_record(self):
        """Return scalar fields only, suitable for a notebook table."""
        record = {
            "regime": self.regime,
            "model_seed": self.model_seed,
            "trajectory_id": self.trajectory_id,
            "source_player": self.source_player,
            "target_player": self.target_player,
            "alpha": self.alpha,
            "sign": self.sign,
            "direction": self.direction,
            "horizon": self.horizon,
            "feasible": self.diagnostics.feasible,
            "intervention_valid": self.diagnostics.valid,
            "invalid_reason": self.invalid_reason,
            "full_horizon_valid": self.full_horizon_valid,
            "failure_horizon": self.failure_horizon,
            "failure_branch": self.failure_branch,
            "input_norm_ratio": self.diagnostics.input_norm_ratio,
            "input_cosine": self.diagnostics.input_cosine,
        }
        for prefix, metrics in (("own", self.own), ("cross", self.cross)):
            for name in ("peak", "rms", "terminal", "time_to_peak"):
                record[f"{prefix}_{name}"] = (
                    getattr(metrics, name) if metrics is not None else float("nan")
                )
        return record


@dataclass
class ExperimentResult:
    """Collection of runs with lightweight notebook summary helpers."""

    runs: list[ExperimentRun] = field(default_factory=list)

    def valid_runs(self):
        """Return runs eligible for main response analysis."""
        return [run for run in self.runs if run.valid]

    def invalid_runs(self):
        """Return infeasible, low-quality, or incomplete runs."""
        return [run for run in self.runs if not run.valid]

    def summary_records(self):
        """Return scalar records without copying trajectory tensors."""
        return [run.summary_record() for run in self.runs]

    def summary_dataframe(self):
        """Return a pandas DataFrame when pandas is available."""
        try:
            import pandas as pd

        except ImportError as error:
            raise ImportError("pandas is required for summary_dataframe()") from error

        return pd.DataFrame(self.summary_records())
