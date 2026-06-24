"""Fixed point and local stability checks for AL-RNN."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from alrnn_eval_utils import normalization_arrays, one_step_readout


def true_fixed_point_raw(metadata, state_dim):
    # get nash equilibrium
    for key in ("true_fixed_point_raw", "mixed_nash_observation", "equilibrium", "nash"):
        if key in metadata:
            return np.asarray(metadata[key], dtype=np.float32)
    return np.zeros(state_dim, dtype=np.float32)


def true_fixed_point_norm(metadata, state_dim):
    # Normalise Nash equilibrium
    mean, std = normalization_arrays(metadata, state_dim)
    return (true_fixed_point_raw(metadata, state_dim) - mean) / std


def default_fixed_point_starts(ctx):
    # Generate a set of initial points for fixed-point optimization
    metadata = ctx["data"]["metadata"]
    starts = [
        np.zeros(ctx["N"], dtype=np.float32), # Normalised zero
        ctx["data"]["train_norm"].reshape(-1, ctx["N"]).mean(axis=0), # The mean of all training data points
        true_fixed_point_norm(metadata, ctx["N"]), # Real nornalised fixed point
    ]
    starts.extend(ctx["data"]["test_norm"][: min(20, len(ctx["data"]["test_norm"])), 0, :]) # 20 trajectories from test dataset
    return np.asarray(starts, dtype=np.float32)


def find_fixed_point(ctx, starts=None, steps=2500, lr=5e-2):
    metadata = ctx["data"]["metadata"]
    device = next(ctx["model"].parameters()).device
    mean, std = normalization_arrays(metadata, ctx["N"]) # back to raw coordinate

    if starts is None:
        starts = default_fixed_point_starts(ctx)
    starts = np.asarray(starts, dtype=np.float32)

    best_x = None
    best_loss = float("inf")

    for start in starts:
        x = torch.tensor(start, dtype=torch.float32, device=device).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=lr) # optimise x

        for _ in range(steps):
            optimizer.zero_grad()
            # compute fixed point residual
            residual = one_step_readout(ctx["model"], x)[0] - x 
            loss = torch.mean(residual ** 2)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            residual = one_step_readout(ctx["model"], x)[0] - x
            final_loss = float(torch.mean(residual ** 2).cpu()) # residual mse

        if final_loss < best_loss:
            best_loss = final_loss
            best_x = x.detach().cpu().numpy()

    fixed_raw = best_x * std + mean
    true_raw = true_fixed_point_raw(metadata, ctx["N"])

    return {
        "fixed_point_norm": best_x,
        "reconstruct_fixed_point": fixed_raw,
        "true_fixed_point": true_raw,
        "distance": float(np.linalg.norm(fixed_raw - true_raw)),
        "residual_mse": best_loss,
    }


def jacobian_function(ctx, x_star_norm):
    # compute jacobian around fixed point
    device = next(ctx["model"].parameters()).device
    x = torch.tensor(x_star_norm, dtype=torch.float32, device=device, requires_grad=True)

    def f(x_in):
        return one_step_readout(ctx["model"], x_in)[0] # f(x_t) = x_{t+1}

    jacobian = torch.autograd.functional.jacobian(f, x) # compute jacobian
    return jacobian.detach().cpu().numpy()


def eigenvalue_diagnostics(eigenvalue, tol=1e-2):
    eigenvalues = np.asarray(eigenvalue)
    mod = np.abs(eigenvalues)

    max_mod = float(mod.max())

    if max_mod < 1.0 - tol:
        stability = "locally stable (sink)"
    elif max_mod > 1.0 + tol:
        stability = "locally unstable (source)"
    else:
        stability = "approximately neutral"

    return {
        "eigenvalues": eigenvalues,
        "max_mod": max_mod,
        "stability": stability,
    }


def plot_fixed_point_stability(fixed_point, eig):
    eigvals = np.asarray(eig["eigenvalues"])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].plot(fixed_point["true_fixed_point"], "o-", label="true")
    axes[0].plot(fixed_point["reconstruct_fixed_point"], "x--", label="AL-RNN")
    axes[0].set_title("Fixed point")
    axes[0].legend()

    theta = np.linspace(0, 2 * np.pi, 256)
    axes[1].plot(np.cos(theta), np.sin(theta), linestyle="--", color="0.6")
    axes[1].scatter(eigvals.real, eigvals.imag)
    axes[1].axhline(0.0, color="black", linewidth=0.5)
    axes[1].axvline(0.0, color="black", linewidth=0.5)
    axes[1].set_title("Discrete eigenvalues")
    axes[1].axis("equal")

    fig.suptitle("Fixed point and local stability")
    fig.tight_layout()
    plt.show()


def fixed_point_stability_analysis(ctx, starts=None, steps=2500, lr=5e-2, tol=1e-2, plot=True):
    fixed_point = find_fixed_point(ctx, starts=starts, steps=steps, lr=lr)
    jacobian = jacobian_function(ctx, fixed_point["fixed_point_norm"])
    eigvals = np.linalg.eigvals(jacobian)
    eig = eigenvalue_diagnostics(eigvals, tol=tol)

    results = {
        "fixed_point_norm": fixed_point["fixed_point_norm"],
        "reconstruct_fixed_point": fixed_point["reconstruct_fixed_point"],
        "true_fixed_point": fixed_point["true_fixed_point"],
        "fixed_point_distance": fixed_point["distance"],
        "fixed_point_residual_mse": fixed_point["residual_mse"],
        "jacobian": jacobian,
        "eigenvalue_diagnostics": eig,
        "max_eigenvalue_mod": eig["max_mod"],
        "stability": eig["stability"],
    }

    if plot:
        plot_fixed_point_stability(fixed_point, eig)

    return results
