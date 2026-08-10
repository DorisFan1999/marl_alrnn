"""RPS polar-coordinate checks for modified AL-RNN probability rollouts."""

from __future__ import annotations

import numpy as np

from modified_alrnn_eval_utils import (
    probability_summary,
    probability_to_centered_rps,
    rollout_on_test
)
from rps_polar_geometry import (
    log_slope,
    max_value,
    mean,
    min_value,
    omega,
    radius,
    rotation_sign,
    sign_sum,
    std,
    step_accuracy
)


def centered_rps_coordinates(probability, N1_output, N2_output):
    probability = np.asarray(probability)
    return probability_to_centered_rps(probability)


def add_initial(ctx, rollout):
    indices = rollout["trajectory_indices"]
    test_probability = ctx["data"]["test_probability"]
    initial = test_probability[indices, :1, :]
    pred_probability = np.concatenate([initial, rollout["pred_probability"]],axis=1)
    true_probability = test_probability[indices, :rollout["steps"] + 1, : ,]
    return pred_probability, true_probability


def polar_geometry_analysis(ctx, radius_eps=1e-8, include_rotation=True):
    test = ctx["data"]["test_probability"]
    target_validity = probability_summary(test, ctx["N1_output"], ctx["N2_output"])
    if not target_validity["valid"]:
        raise ValueError(
            "RPS polar geometry requires dataset observations to contain two "
            "three-action probability distributions"
        )

    indices = list(range(len(test)))
    steps = test.shape[1] - 1
    rollout = rollout_on_test(ctx, trajectory_indices=indices, steps=steps)
    pred_probability, true_probability = add_initial(ctx, rollout)
    pred = centered_rps_coordinates(pred_probability, ctx["N1_output"], ctx["N2_output"])
    true = centered_rps_coordinates(true_probability, ctx["N1_output"], ctx["N2_output"])
    generated = pred[:, 1:, :]
    nash_point = np.zeros(4, dtype=float)
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
        pred_omega_p1 = omega(pred, [0, 1])
        pred_omega_p2 = omega(pred, [2, 3])
        true_omega_p1 = omega(true, [0, 1])
        true_omega_p2 = omega(true, [2, 3])

        pred_sign_p1 = rotation_sign(pred, [0, 1])
        pred_sign_p2 = rotation_sign(pred, [2, 3])
        true_sign_p1 = rotation_sign(true, [0, 1])
        true_sign_p2 = rotation_sign(true, [2, 3])

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
