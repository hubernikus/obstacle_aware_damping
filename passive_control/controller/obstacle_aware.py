import time
import warnings
from typing import Optional
from abc import ABC, abstractmethod

from attrs import define, field

import numpy as np

# passive_control.of lukas
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.utils import get_orthogonal_basis

import passive_control.magic_numbers_and_enums as mn
from passive_control.agent import Agent


@define
class ObstacleAwarePassiveController(Controller):
    dynamic_avoider: ModulationAvoider

    lambda_dynamics: float = 100.0
    lambda_obstacle: float = 20.0
    lambda_remaining: float = mn.MALBDA_MAX

    critical_gamma: float = 3.0
    dimension: int = 2

    _normals_to_obstacles: np.ndarray = np.zeros(0)
    _distances_to_obstacles: np.ndarray = np.zeros(0)
    _gamma_of_obstacles: np.ndarray = np.zeros(0)

    def __attrs_post_init__(self):
        print("(TODO: remove this comment) Doing post init.")
        if (
            lambda_DS > mn.LAMBDA_MAX
            or lambda_perp > mn.LAMBDA_MAX
            or lambda_obs > mn.LAMBDA_MAX
        ):
            raise ValueError(f"lambda must be smaller than {mn.LAMBDA_MAX}")

    def compute_control_force(
        self, agent: Agent, desired_velocity: np.ndarray
    ) -> np.ndarray:
        """Returns control-force (without gravity vector)"""
        self.update_normal_list(agent.position)
        self.damping_matrix = self.compute_damping(desired_velocity)

        control_force = -self.damping_matrix @ (desired_velocity - agent.velocity)
        return control_force

    @property
    def obstacle_environment(self):
        return self.dynamic_avoider.obstacle_environment

    def update_normal_list(self, position: np.ndarray) -> None:
        self.normals_to_obstacles = np.zeros(
            (len(self.obstacle_environment, self.dimension))
        )
        self.distances_to_obstacles = np.zeros(len(self.obstacle_environment))
        self.gamma_of_obstacles = np.zeros(len(self.obstacle_environment))

        for obs in self.obstacle_environment:
            # gather the parameters wrt obstacle i
            normal = obs.get_normal_direction(
                position, in_obstacle_frame=False
            ).reshape(self.dimension, 1)

            self.normals_to_obstacles[:, ii] = np.append(
                self.obs_normals_list, normal, axis=1
            )

            self.gamma_of_obstacles[ii] = obs.get_gamma(
                position, in_obstacle_frame=False
            )
            self.distances_to_obstacles[ii] = self.gamma_of_obstacles[ii] - 1

    def compute_averaged_normal(self, normals, gammas) -> np.ndarray:
        weights = gammas - 1

        ind_negative = weights < 0
        if np.any(ind_negative):
            weights = ind_negative / np.sum(ind_negative)
        else:
            weights = 1 / weights
            weights = weights / np.sum(weights)

        averagd_normal = np.sum(
            normals * np.tile(weights, (normals.shape[0], 1)), axis=1
        )
        breakpoint()
        return averagd_normal

    def compute_danger_weight(self, min_gamma, averaged_normal) -> float:
        weight = max(self.gamma_critical - min_gamma / (self.gamma_critical - 1))
        # weight = weight ** (1 / np.linalg.norm(averaged_normal))
        return weight

    def compute_damping(self, desired_velocity):
        averaged_normal = self.compute_averaged_normal(
            self.obs_normals_list, self.obs_dist_list
        )
        weight = self.compute_danger_weight(
            np.min(self.gamma_of_obstacles), averaged_normal
        )

        damping_matrix_dynamics = self._compute_dynamics_damping(desired_velocity)

        if weight < 0:
            return damping_matrix_dynamics

        damping_matrix_obstacle = self._compute_obstacle_damping(
            averaged_normal, desired_velocity
        )

        return weight * damping_matrix_obstacle + (1 - weight) * damping_matrix_dynamics

    def _compute_dynamics_damping(
        self,
        desired_velocity: np.ndarray,
        averaged_normal: np.ndarray,
        min_gamma: np.ndarray,
    ) -> np.ndarray:
        weight = min(
            1,
            averaged_normal @ averaged_normal
            + (self.gamma_critical - min_gamma) / (self.gamma_critical - 1.0),
        )
        other_lambda = (
            weight * self.lambda_obstacle + (1 - weight) * self.lambda_remaining
        )

        damping_matrix = np.diag(
            np.hstack(
                (self.lambda_dynamics, other_lambda * np.ones(self.dimension - 1))
            )
        )

        basis_matrix = get_orthogonal_basis(desired_velocity)
        return basis_matrix @ damping_matrix @ basis_matrix.T

    def _compute_obstacle_damping(
        self, averaged_normal: np.ndarray, desired_velocity: np.ndarray
    ) -> np.ndarray:
        basis1 = averaged_normal / np.linalg.norm(averaged_normal)

        dotproduct = np.dot(basis1, desired_velocity)
        if not dotproduct:  # Zero value
            return get_orthogonal_basis(basis1)

        basis2 = desired_velocity - basis1 * dotproduct
        basis2 = np.linalg.norm(basis2)

        if self.dimension == 2:
            basis_matrix = np.vstack((basis1, basis2)).T
        elif self.dimension == 3:
            basis_matrix = np.vstack((basis1, basis2, np.cross(basis1, basis2))).T
        else:
            raise NotImplementedError(f"Not defined for dimensions >3.")

        weight = abs(dotproduct / np.linalg.norm(desired_velocity))

        damping_matrix = np.diag(
            np.hstack(
                (
                    self.lambda_obstacle,
                    (1 - weight) * self.lambda_dynamics
                    + weight * self.lambda_remaining,
                    self.lambda_remaining * np.ones(self.dimension - 2),
                )
            )
        )
        return basis_matrix @ damping_matrix @ basis_matrix.T


def test_simple_obstacle():
    pass
