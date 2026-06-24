from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from metrics import power_spectrum_error, state_space_divergence_binning


class AL_RNN(nn.Module):
    """Autonomous latent recurrent network with linear readout units."""

    def __init__(self, M: int, P: int, N: int):
        super().__init__()

        # initialise model dimensions
        self.M = M
        self.P = P
        self.N = N

        # initialise mode parameters A, W, h, B
        self.A, self.W, self.h = self.initialize_AWh_random()
        # z = xB
        self.B = self.init_uniform((self.N, self.M))

    def forward(self, z):
        """z_t -> z_{t+1}"""
        z_unactivated = torch.clone(z)

        # Apply relu on the last P units
        z[:, -self.P:] = F.relu(z[:, -self.P:])

        # compute forward function
        return self.A * z_unactivated + z @ self.W.t() + self.h


    def initialize_AWh_random(self):
        # Randomly initialise A, W, h
        A = nn.Parameter(torch.diagonal(self.normalized_positive_definite(self.M), 0))
        W = nn.Parameter(torch.randn(self.M, self.M) * 0.01)
        h = nn.Parameter(torch.zeros(self.M))
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


@torch.no_grad()
def predict_free_sequence(model, x, T):
    """Free-run the model from an initial readout state."""
    batch_size, n_readout = x.size()
    # Initialise first latent state
    z = x @ model.B 
    # Force the first N hidden variables to equal the actual observations
    z[:, :n_readout] = x

    states = torch.empty(size=(T, batch_size, model.M), device=x.device)
    for t in range(T):
        z = model(z)
        states[t] = z
    return states.permute(1, 0, 2)


"""
Training Routine
"""
def teacher_force(z, x, alpha, n_readout):
    """
    z_{t, 1:N} <- alpha * x_t + (1 - alpha) * z_{t, 1:N}
    """
    z[:, :n_readout] = alpha * x + (1 - alpha) * z[:, :n_readout]
    return z


def predict_sequence_using_gtf(model, x, alpha, n_interleave):
    """Predict a batch of windows with generalized teacher forcing."""
    # Permute input to shape (sequence_length, batch_size, feature_dim)
    x_time_major = x.permute(1, 0, 2) 
    T, batch_size, n_readout = x_time_major.size()

    states = torch.empty(size=(T, batch_size, model.M), device=x.device)

    # initialise latent state z_0 using x_0 * B
    z = x_time_major[0] @ model.B 
    # set first n dimension of z_0 = x_0
    z = teacher_force(z, x_time_major[0], alpha=1.0, n_readout=n_readout)

    # Execute once every n_interleave steps
    for t in range(T):
        if (t % n_interleave == 0) and (t > 0):
            z = teacher_force(z, x_time_major[t], alpha=alpha, n_readout=n_readout)
        # forward prediction
        z = model(z)
        # keep latent state z
        states[t] = z
    return states.permute(1, 0, 2)


def train_sh(model, dataset, optimizer, scheduler, loss_fn, num_epochs, alpha, n_interleave, batches_per_epoch=50, ssi=25, use_best_model=True):
    """Train AL-RNN on trajectory windows and keep the best free-run checkpoint."""
    model.train()
    best_model = copy.deepcopy(model)
    losses, klx, dh = [], [], []

    with trange(num_epochs, desc="Training Progress") as epochs:
        for epoch in epochs:
            epoch_losses = []

            for _ in range(batches_per_epoch):
                optimizer.zero_grad()
                # input a batch of trajectory
                x, y, _ = dataset.sample_batch()
                z_hat = predict_sequence_using_gtf(model, x, alpha, n_interleave)
                # cumpute loss of ground truth y and first N dim of z_hat
                loss = loss_fn(z_hat[:, :, :model.N], y)
                loss.backward()
                # update parameters
                optimizer.step()
                epoch_losses.append(loss.item())

            # Adjust learning rate based on the scheduler
            scheduler.step() 

            average_loss = sum(epoch_losses) / len(epoch_losses)
            epochs.set_postfix(loss=average_loss)
            losses.append(average_loss)

            if epoch % ssi == 0:
                with torch.no_grad():
                    reference = dataset.reference_sequence(trajectory_idx=0, max_length=10000).clone().detach()
                    z_test = predict_free_sequence(model, reference[0:1, :], len(reference)) # Predict sequence using teacher forcing
                    generated = z_test[0, :, :model.N]
                    # Calculate Dstsp
                    true_cloud = dataset.X.clone().detach().reshape(-1, model.N)
                    klx.append(state_space_divergence_binning(generated, true_cloud))
                    # Calculate DH
                    dh.append(power_spectrum_error(generated, reference)) 

                    if torch.argmin(torch.tensor(klx)) + 1 == len(torch.tensor(klx)):
                        best_model = copy.deepcopy(model)

    #If true the best model (according to Dstsp) during training is returned, else the one from the last epoch
    if use_best_model:  
        model.load_state_dict(best_model.state_dict())

    return [losses, klx, dh]
