"""Public notebook API for controlled initial perturbation experiments."""

from .adapters import ModifiedALRNNAdapter
from .config import ExperimentConfig
from .experiment import run_controlled_response_experiment
from .intervention import supports_all_impulses
from .metrics import scaling_records
from .plotting import (
    plot_own_cosine,
    plot_response_norms,
)
from .seed_analysis import (
    run_seed_response_analysis,
    summarize_across_seeds,
)
from .trajectory_plotting import plot_paired_probability_trajectories
from .types import ExperimentResult, ExperimentRun

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRun",
    "ModifiedALRNNAdapter",
    "plot_own_cosine",
    "plot_paired_probability_trajectories",
    "plot_response_norms",
    "run_controlled_response_experiment",
    "run_seed_response_analysis",
    "scaling_records",
    "supports_all_impulses",
    "summarize_across_seeds",
]
