"""Polar-coordinate geometry checks for RPS AL-RNN rollouts.

The radial coordinate is measured around the mixed Nash equilibrium:

    r_t = ||q_t - q*||.

For RPS datasets in the current repo, test_raw and pred_raw are
pre-standardization 4D centered observations:

    (p1_rock - 1/3, p1_paper - 1/3, p2_rock - 1/3, p2_paper - 1/3).

The radius is computed in this same centered coordinate system. For the current
RPS metadata, mixed_nash_observation is the Nash point in the model/evaluation
observation coordinate, so it is used directly and is usually the origin.
Radial trend is measured on the generated trajectory by fitting

    log r_t ~= beta * t + c

where beta is the main diagnostic: negative for inward dynamics, near zero for
center dynamics, and positive for outward dynamics. The generated trajectory's
radius range is also reported to diagnose drift away from the Nash neighborhood.
"""

from __future__ import annotations

import numpy as np

from alrnn_eval_utils import rollout_on_test


def add_initial(ctx, rollout):
    indices = rollout["trajectory_indices"]
    initial = ctx["data"]["test_raw"][indices, :1, :]
    pred = np.concatenate([initial, rollout["pred_raw"]], axis=1)
    true = ctx["data"]["test_raw"][indices, : rollout["steps"] + 1, :]
    return pred, true


def nash(ctx):
    return np.asarray(
        ctx["data"]["metadata"].get("mixed_nash_observation", np.zeros(ctx["N"])),
        dtype=float,
    ) 


def radius(states, nash):
    # radius is the distance from the Nash point in the observation space
    return np.linalg.norm(states - nash, axis=-1)


def log_slope(radius, radius_eps):
    """
    We fit log r_t ~= beta * t + c for each rollout, and return the slope beta

    beta \approx \sum_i (t_i - t_mean) * (log r_i - log r_mean) / \sum_i (t_i - t_mean)^2
    """
    log_radius = np.log(np.maximum(radius, radius_eps))
    slopes = []

    for row in log_radius:
        y = row
        time_point = np.arange(len(row), dtype=float)
        time_point = time_point - time_point.mean()
        denominator = np.sum(time_point ** 2)
        slopes.append(float(np.sum(time_point * (y - y.mean())) / denominator))

    return np.asarray(slopes, dtype=float)


def omega(states, cols):
    coords = states[..., cols] # for each player
    velocities = []

    for xy in coords:
        # transform to polar coordinates and compute angular velocity
        theta = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
        # from start to end, the average angular velocity 
        velocities.append(float((theta[-1] - theta[0]) / (len(theta) - 1)))

    return np.asarray(velocities, dtype=float)


def rotation_sign(states, cols):
    """
    For each player, we compute the sign of the rotation at each step of the trajectory

    v_t = (p_rock_t - 1/3, p_paper_t - 1/3) = (x_t, y_t)

    we use the cross product to determine the sign of the rotation:
    v_t times v_{delta_t} = v_t cdot (v_{t+1} - v_t) = 

    If sign is positive, then the rotation is counter-clockwise, if negative, then clockwise. If zero, then no rotation.
    """
    coords = states[..., cols]
    # each step of displacement
    # v_{t+1} - v_t
    step = coords[:, 1:, :] - coords[:, :-1, :] # delta v_t
    # cross product of v_t and delta v_t
    # x_t * delta y_t - y_t * delta x_t
    cross = coords[:, :-1, 0] * step[:, :, 1] - coords[:, :-1, 1] * step[:, :, 0]

    # return to sequence of local rotation direction signs for each time step
    return np.sign(cross)


def step_accuracy(pred_sign, true_sign):
    return float(np.mean(pred_sign == true_sign))


def sign_sum(signs):
    return float(np.sum(signs))


def mean(x):
    return float(np.mean(x))


def std(x):
    return float(np.std(x))


def min_value(x):
    return float(np.min(x))


def max_value(x):
    return float(np.max(x))


def polar_geometry_analysis(
    ctx,
    radius_eps=1e-8,
    include_rotation=True,
):
    test = ctx["data"]["test_norm"]
    indices = list(range(len(test)))
    steps = test.shape[1] - 1
    rollout = rollout_on_test(ctx, trajectory_indices=indices, steps=steps)

    pred, true = add_initial(ctx, rollout)
    generated = rollout["pred_raw"]
    nash_point = nash(ctx)
    pred_radius = radius(pred, nash_point)
    generated_radius = radius(generated, nash_point)
    pred_log_radius_slope = log_slope(pred_radius, radius_eps)

    result = {
        "n_test_rollouts": len(indices),
        "steps": steps,
        "initial_radius_mean": mean(pred_radius[:, 0]),
        "final_radius_mean": mean(pred_radius[:, -1]),
        "generated_radius_min": min_value(generated_radius),
        "generated_radius_max": max_value(generated_radius),
        "pred_log_radius_slope_mean": mean(pred_log_radius_slope),
        "pred_log_radius_slope_std": std(pred_log_radius_slope),
        "rollout": rollout,
    }

    if include_rotation:
        pred_centered_for_rotation = pred - nash_point
        true_centered_for_rotation = true - nash_point

        pred_omega_p1 = omega(pred_centered_for_rotation, [0, 1])
        pred_omega_p2 = omega(pred_centered_for_rotation, [2, 3])
        
        true_omega_p1 = omega(true_centered_for_rotation, [0, 1])
        true_omega_p2 = omega(true_centered_for_rotation, [2, 3])

        pred_sign_p1 = rotation_sign(pred_centered_for_rotation, [0, 1])
        pred_sign_p2 = rotation_sign(pred_centered_for_rotation, [2, 3])
        true_sign_p1 = rotation_sign(true_centered_for_rotation, [0, 1])
        true_sign_p2 = rotation_sign(true_centered_for_rotation, [2, 3])

        rotation_correct_p1 = step_accuracy(pred_sign_p1, true_sign_p1)
        rotation_correct_p2 = step_accuracy(pred_sign_p2, true_sign_p2)
        rotation_correct_joint = float(
            np.mean(
                (pred_sign_p1 == true_sign_p1)
                & (pred_sign_p2 == true_sign_p2)
            )
        )

        result.update(
            {
                "pred_angular_velocity_p1_mean": mean(pred_omega_p1),
                "pred_angular_velocity_p2_mean": mean(pred_omega_p2),
                "true_angular_velocity_p1_mean": mean(true_omega_p1),
                "true_angular_velocity_p2_mean": mean(true_omega_p2),
                #"pred_sign_p1": pred_sign_p1,
                #"pred_sign_p2": pred_sign_p2,
                #"true_sign_p1": true_sign_p1,
                #"true_sign_p2": true_sign_p2,
                "pred_rotation_sign_sum_p1": sign_sum(pred_sign_p1),
                "pred_rotation_sign_sum_p2": sign_sum(pred_sign_p2),
                "true_rotation_sign_sum_p1": sign_sum(true_sign_p1),
                "true_rotation_sign_sum_p2": sign_sum(true_sign_p2),
                "step_rotation_accuracy_p1": rotation_correct_p1,
                "step_rotation_accuracy_p2": rotation_correct_p2,
                "step_rotation_accuracy_joint": rotation_correct_joint,
                "rotation_accuracy_p1": rotation_correct_p1,
                "rotation_accuracy_p2": rotation_correct_p2,
                "rotation_accuracy_joint": rotation_correct_joint,
            }
        )

    return result


rps_polar_geometry_analysis = polar_geometry_analysis
