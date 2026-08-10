"""Adapter for controlled-response experiments with Modified AL-RNN."""

from dataclasses import dataclass

import torch

from training.modified_alrnn_model import initialize_latent_state

from .geometry import as_player_probabilities
from .intervention import apply_modified_intervention


@dataclass
class ModifiedALRNNAdapter:
    """Expose a loaded Modified_AL_RNN through the experiment protocol."""

    model: object

    def initialize(self, x0):
        # Return a batch of initial latent states for the model, given initial conditions
        parameter = next(self.model.parameters())
        x0 = torch.as_tensor(x0, device=parameter.device, dtype=parameter.dtype)
        
        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)
        return initialize_latent_state(self.model, x0)

    def clone_state(self, z):
        return z.clone()

    def observe_probabilities(self, z):
        # Return the player probabilities for a given state tensor
        return as_player_probabilities(self.model.linear_score(z))

    def apply_probability_impulse(self, z0, source_player, delta, config):
        return apply_modified_intervention(self.model, z0, source_player, delta, config)

    def step(self, z):
        # free rollout one step
        return self.model(z)

    def state_tensor(self, z):
        return z
