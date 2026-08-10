from __future__ import annotations

import copy
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from metrics import power_spectrum_error, state_space_divergence_binning


class Modified_AL_RNN(nn.Module):
    """Modified autonomous latent recurrent network for two interacting players.

    The first M1-P1 and M2-P2 latent variables remain linear. The last P1
    and P2 latent variables use ReLU for players 1 and 2, respectively. N1
    and N2 are centered input dimensions; each player has a 3D RPS output.
    """

    def __init__(self, M1: int, P1: int, N1: int, M2: int, P2: int, N2: int, use_W12: bool = True, use_W21: bool = True):
        super().__init__()
        """
        if N1 <= 0:
            raise ValueError("N1 must be greater than zero")
        if N2 <= 0:
            raise ValueError("N2 must be greater than zero")
        if M1 < N1:
            raise ValueError("M1 must be greater than or equal to N1")
        if M2 < N2:
            raise ValueError("M2 must be greater than or equal to N2")
        if not 0 <= P1 <= M1:
            raise ValueError("P1 must satisfy 0 <= P1 <= M1")
        if not 0 <= P2 <= M2:
            raise ValueError("P2 must satisfy 0 <= P2 <= M2")
        """

        # initialise model dimensions
        self.M1 = M1
        self.P1 = P1
        self.N1 = N1

        self.M2 = M2
        self.P2 = P2
        self.N2 = N2

        self.M = self.M1 + self.M2
        self.P = self.P1 + self.P2
        self.N = self.N1 + self.N2
        
        # each player produces rock, paper, and scissors probabilities
        self.N1_out = 3
        self.N2_out = 3
        self.N_out = self.N1_out + self.N2_out

        self.use_W12 = use_W12
        self.use_W21 = use_W21

        # initialise model parameters A, W, h, B
        self.A1, self.W11, self.h1 = self.initialize_AWh_random(self.M1)
        self.A2, self.W22, self.h2 = self.initialize_AWh_random(self.M2)

        # W12: player 2 -> player 1; W21: player 1 -> player 2
        self.W12 = nn.Parameter(torch.randn(self.M1, self.M2) * 0.01)
        self.W21 = nn.Parameter(torch.randn(self.M2, self.M1) * 0.01)

        # z1 = x1 B1; z2 = x2 B2
        self.B1 = self.init_uniform((self.N1, self.M1))
        self.B2 = self.init_uniform((self.N2, self.M2))

        # linear score heads produce three unrestricted logits per player
        self.V1 = self.init_uniform((self.N1_out, self.M1))
        self.V2 = self.init_uniform((self.N2_out, self.M2))
        self.c1 = nn.Parameter(torch.zeros(self.N1_out))
        self.c2 = nn.Parameter(torch.zeros(self.N2_out))


    def forward(self, z):
        """z_t -> z_{t+1}"""
        # split latent state by player
        z1 = torch.clone(z[:, :self.M1])
        z2 = torch.clone(z[:, self.M1:])

        z1_unactivated = torch.clone(z1)
        z2_unactivated = torch.clone(z2)

        # Apply relu on the last P1 and P2 units
        if self.P1 > 0:
            z1[:, -self.P1:] = F.relu(z1[:, -self.P1:])
        if self.P2 > 0:
            z2[:, -self.P2:] = F.relu(z2[:, -self.P2:])

        # compute both next states from the same current states
        z1_next = (self.A1 * z1_unactivated + z1 @ self.W11.t() + self.h1)
        
        if self.use_W12:
            z1_next = z1_next + z2 @ self.W12.t()

        z2_next = (self.A2 * z2_unactivated + z2 @ self.W22.t() + self.h2)

        if self.use_W21:
            z2_next = z2_next + z1 @ self.W21.t()

        return torch.cat((z1_next, z2_next), dim=-1)


    def linear_score(self, z):
        """Map latent states to two complete 3D action probabilities."""
        z1 = z[..., :self.M1]
        z2 = z[..., self.M1:]

        u1 = z1 @ self.V1.t() + self.c1
        u2 = z2 @ self.V2.t() + self.c2

        # Normalize each player's logits separately
        u1 = F.softmax(u1, dim=-1)
        u2 = F.softmax(u2, dim=-1)

        return torch.cat((u1, u2), dim=-1)


    def initialize_AWh_random(self, M):
        # Randomly initialise A, W, h
        A = nn.Parameter(torch.diagonal(self.normalized_positive_definite(M),0,))
        W = nn.Parameter(torch.randn(M, M) * 0.01)
        h = nn.Parameter(torch.zeros(M))
        return A, W, h


    def normalized_positive_definite(self, M):
        R = np.random.randn(M, M).astype(np.float32)
        K = np.matmul(R.T, R) / M + np.eye(M)
        lambda_max = np.max(np.abs(np.linalg.eigvals(K)))
        return torch.tensor(K / lambda_max).float()

    def init_uniform(self, shape):
        tensor = torch.empty(*shape)
        r = 1 / math.sqrt(shape[0])
        torch.nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)


def initialize_latent_state(model, x):
    x1 = x[:, :model.N1]
    x2 = x[:, model.N1:]

    z1 = x1 @ model.B1
    z2 = x2 @ model.B2

    z1[:, :model.N1] = x1
    z2[:, :model.N2] = x2

    return torch.cat((z1, z2), dim=-1)


@torch.no_grad()
def predict_free_sequence(model, x, T):
    """Free-run the model from an initial readout state."""
    batch_size, n_readout = x.size()
    # Initialise first latent state
    z = initialize_latent_state(model, x)

    states = torch.empty(size=(T, batch_size, model.M), device=x.device)
    for t in range(T):
        z = model(z)
        states[t] = z
    return states.permute(1, 0, 2)


def nonredundant_player_probabilities(
    probabilities,
    n1_output,
    n2_output,
):
    """Drop one sum-constrained probability from each player's policy."""
    expected_dim = n1_output + n2_output
    if probabilities.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected probability dimension {expected_dim}, "
            f"found {probabilities.shape[-1]}"
        )
    if n1_output < 2 or n2_output < 2:
        raise ValueError("Each player must have at least two output probabilities")

    player1 = probabilities[..., :n1_output - 1]
    player2 = probabilities[..., n1_output:expected_dim - 1]
    return torch.cat((player1, player2), dim=-1)


def observed_latent_state(model, latent):
    """Return the observed coordinates from both players' latent states."""
    player1 = latent[..., :model.N1]
    player2 = latent[..., model.M1:model.M1 + model.N2]
    return torch.cat((player1, player2), dim=-1)


"""
Training Routine
"""
def teacher_force(z, x, alpha, model):
    """
    Inject each player's observation into its first latent variables.
    """
    x1 = x[:, :model.N1]
    x2 = x[:, model.N1:]

    z[:, :model.N1] = (alpha * x1 + (1 - alpha) * z[:, :model.N1])

    start2 = model.M1
    end2 = model.M1 + model.N2

    z[:, start2:end2] = (alpha * x2 + (1 - alpha) * z[:, start2:end2])
    return z


def predict_sequence_using_gtf(model, x, alpha, n_interleave):
    """Predict a batch of windows with generalized teacher forcing."""
    # Permute input to shape (sequence_length, batch_size, feature_dim)
    x_time_major = x.permute(1, 0, 2)
    T, batch_size, n_readout = x_time_major.size()

    states = torch.empty(size=(T, batch_size, model.M), device=x.device)

    # initialise latent state z_0 using x_0 and B1, B2
    z = initialize_latent_state(model, x_time_major[0])

    # Execute once every n_interleave steps
    for t in range(T):
        if (t % n_interleave == 0) and (t > 0):
            z = teacher_force(z, x_time_major[t], alpha=alpha, model=model)
        # forward prediction
        z = model(z)
        # keep latent state z
        states[t] = z
    return states.permute(1, 0, 2)


def train_sh(
    model,
    dataset,
    optimizer,
    scheduler,
    loss_fn,
    num_epochs,
    alpha,
    n_interleave,
    batches_per_epoch=50,
    ssi=25,
    use_best_model=True,
    checkpoint_interval=25,
    checkpoint_callback=None,
    progress_label=None,
    state_space_bins=30,
    state_loss_weight=1.0,
    probability_loss_weight=1.0,
):
    """Train with observed-state MSE plus probability-output MSE."""
    model.train()
    best_model = copy.deepcopy(model)
    losses, state_losses, probability_losses, klx, dh = [], [], [], [], []

    if progress_label is not None:
        print(f"[{progress_label}] started: 0/{num_epochs} epochs", file=sys.stderr, flush=True)

    with trange(num_epochs, desc="Training Progress", disable=progress_label is not None) as epochs:
        for epoch in epochs:
            epoch_losses = []
            epoch_state_losses = []
            epoch_probability_losses = []

            for _ in range(batches_per_epoch):
                optimizer.zero_grad()
                # input a batch of trajectory
                x, state_target, probability_target, _ = dataset.sample_batch_with_state_target()
                z_hat = predict_sequence_using_gtf(model, x, alpha, n_interleave)
                state_hat = observed_latent_state(model, z_hat)
                probability_hat = model.linear_score(z_hat)
                if state_target.shape[-1] != model.N:
                    raise ValueError(
                        f"Expected normalized state targets with dimension {model.N}, "
                        f"found {state_target.shape[-1]}"
                    )
                if probability_target.shape[-1] != model.N_out:
                    raise ValueError(
                        f"Expected full probability targets with dimension {model.N_out}, "
                        f"found {probability_target.shape[-1]}"
                    )
                state_loss = loss_fn(state_hat, state_target)
                probability_loss = loss_fn(probability_hat, probability_target)
                loss = (state_loss_weight * state_loss+ probability_loss_weight * probability_loss)
                loss.backward()
                # update parameters
                optimizer.step()
                epoch_losses.append(loss.item())
                epoch_state_losses.append(state_loss.item())
                epoch_probability_losses.append(probability_loss.item())

            # Adjust learning rate based on the scheduler
            scheduler.step()

            average_loss = sum(epoch_losses) / len(epoch_losses)
            if progress_label is None:
                epochs.set_postfix(loss=average_loss)
            losses.append(average_loss)
            state_losses.append(sum(epoch_state_losses) / len(epoch_state_losses))
            probability_losses.append(
                sum(epoch_probability_losses) / len(epoch_probability_losses)
            )

            completed_epochs = epoch + 1
            if (completed_epochs % ssi == 0) or (completed_epochs == num_epochs):
                with torch.no_grad():
                    reference = dataset.reference_sequence(trajectory_idx=0, max_length=10000).clone().detach()
                    reference_target = dataset.reference_target_sequence(trajectory_idx=0, max_length=10000).clone().detach()[1:]
                    z_test = predict_free_sequence(model, reference[0:1, :], len(reference_target))
                    generated = model.linear_score(z_test[0])
                    # Match original AL-RNN checkpoint selection in the
                    # nonredundant state dimension: one probability per
                    # player is fixed by the sum-to-one constraint.
                    generated_state = nonredundant_player_probabilities(
                        generated,
                        model.N1_out,
                        model.N2_out,
                    )
                    true_cloud = nonredundant_player_probabilities(
                        dataset.Y.clone().detach().reshape(-1, model.N_out),
                        model.N1_out,
                        model.N2_out,
                    )
                    klx.append(
                        state_space_divergence_binning(
                            generated_state,
                            true_cloud,
                            n_bins=state_space_bins,
                        )
                    )
                    # Calculate DH
                    dh.append(power_spectrum_error(generated, reference_target))

                    if torch.argmin(torch.tensor(klx)) + 1 == len(torch.tensor(klx)):
                        best_model = copy.deepcopy(model)

            checkpoint_due = (completed_epochs % checkpoint_interval == 0) or (completed_epochs == num_epochs)
            if checkpoint_due and checkpoint_callback is not None:
                checkpoint_callback(
                    completed_epochs,
                    model,
                    best_model,
                    losses,
                    state_losses,
                    probability_losses,
                    klx,
                    dh,
                )

            if checkpoint_due and progress_label is not None:
                print(
                    f"[{progress_label}] epoch {completed_epochs}/{num_epochs}, "
                    f"loss={average_loss:.6g}, checkpoint saved",
                    file=sys.stderr,
                    flush=True,
                )

    #If true the best model (according to Dstsp) during training is returned, else the one from the last epoch
    if use_best_model:
        model.load_state_dict(best_model.state_dict())

    # Keep the first three entries compatible with existing callers.
    return [losses, klx, dh, state_losses, probability_losses]
