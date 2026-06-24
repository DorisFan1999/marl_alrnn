import torch
from random import randint


class TimeSeriesDataset:
    """
    Dataset for multiple independent trajectories.

    Expected data shape:
        (n_trajectories, n_steps, state_dim)
    """

    def __init__(self, data, external_inputs=None, sequence_length=200, batch_size=16):
        self.X = torch.tensor(data, dtype=torch.float32)

        if self.X.ndim != 3:
            raise ValueError("data must have dimension=3")

        self.n_trajectories = self.X.shape[0]
        self.n_steps = self.X.shape[1]
        self.state_dim = self.X.shape[2]

        self.sequence_length = sequence_length
        self.batch_size = batch_size

        if self.sequence_length >= self.n_steps:
            raise ValueError("sequence_length must be smaller than n_steps")

        if external_inputs is not None:
            self.S = torch.tensor(external_inputs, dtype=torch.float32)
            if self.S.shape[:2] != self.X.shape[:2]:
                raise ValueError("external_inputs must match data on trajectory and time axes")
        else:
            self.S = None


    def __len__(self):
        windows_per_trajectory = self.n_steps - self.sequence_length
        return self.n_trajectories * windows_per_trajectory

    def _decode_index(self, idx):
        windows_per_trajectory = self.n_steps - self.sequence_length
        trajectory_idx = idx // windows_per_trajectory
        time_idx = idx % windows_per_trajectory
        return trajectory_idx, time_idx

    def __getitem__(self, idx):
        trajectory_idx, t = self._decode_index(idx)

        x = self.X[trajectory_idx, t:t + self.sequence_length, :]
        y = self.X[trajectory_idx, t + 1:t + self.sequence_length + 1, :]

        if self.S is None:
            return x, y, None

        s = self.S[trajectory_idx, t:t + self.sequence_length, :]
        return x, y, s

    def sample_batch(self):
        X, Y, S = [], [], []

        for _ in range(self.batch_size):
            idx = randint(0, len(self) - 1)
            x, y, s = self[idx]
            X.append(x)
            Y.append(y)
            S.append(s)

        X = torch.stack(X)
        Y = torch.stack(Y)

        if S[0] is None:
            return X, Y, None

        return X, Y, torch.stack(S)

    def reference_sequence(self, trajectory_idx=0, max_length=None):
        """
        Return one full trajectory for plotting or diagnostics.
        """
        if not 0 <= trajectory_idx < self.n_trajectories:
            raise IndexError("trajectory_idx is out of range")

        seq = self.X[trajectory_idx]

        if max_length is not None:
            seq = seq[:max_length]

        return seq
