from typing import Optional

from attrs import define, field

import numpy as np

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.utils import get_orthogonal_basis

import passive_control.magic_numbers_and_enums as mn
from passive_control.agent import Agent
from passive_control.controller import Controller


# TODO: activate slots when adapting of function is over...
@define(slots=False)
class ObstacleAwarePassivController(Controller):
    environment: ObstacleContainer = field(factory=ObstacleContainer)

    lambda_dynamics: float = field(default=100.0, validator=Controller.is_damping_value)
    lambda_obstacle: float = field(default=20.0, validator=Controller.is_damping_value)
    lambda_remaining: float = field(
        default=mn.LAMBDA_MAX, validator=Controller.is_damping_value
    )

    gamma_critical: float = 3.0
    dimension: int = 2

    _normals_to_obstacles: np.ndarray = np.zeros(0)
    _distances_to_obstacles: np.ndarray = np.zeros(0)
    _gammas_of_obstacles: np.ndarray = np.zeros(0)

    def compute_control_force(
        self, position: np.ndarray, velocity: np.ndarray, desired_velocity: np.ndarray
    ) -> np.ndarray:
        """Returns control-force (without gravity vector)"""
        self.update_normal_list(position)
        self.damping_matrix = self.compute_damping(desired_velocity, velocity)
        control_force = self.damping_matrix @ (desired_velocity - velocity)
        return control_force

    def compute_control_force_for_agent(
        self, agent: Agent, desired_velocity: np.ndarray
    ) -> np.ndarray:
        """Returns control-force (without gravity vector)"""
        return self.compute_control_force(
            agent.position, agent.velocity, desired_velocity
        )

    def update_normal_list(self, position: np.ndarray) -> None:
        n_obstacles = len(self.environment)
        self._normals_to_obstacles = np.zeros((self.dimension, n_obstacles))
        self._distances_to_obstacles = np.zeros(n_obstacles)
        self._gammas_of_obstacles = np.zeros(n_obstacles)

        for ii, obs in enumerate(self.environment):
            # gather the parameters wrt obstacle i
            self._normals_to_obstacles[:, ii] = obs.get_normal_direction(
                position, in_obstacle_frame=False
            )

            self._gammas_of_obstacles[ii] = obs.get_gamma(
                position, in_obstacle_frame=False
            )

            self._distances_to_obstacles[ii] = self._gammas_of_obstacles[ii] - 1

    @staticmethod
    def compute_averaged_normal(normals, gammas) -> np.ndarray:
        weights = gammas - 1

        ind_negative = weights < 0
        if np.any(ind_negative):
            weights = ind_negative / np.sum(ind_negative)
        else:
            weights = 1 / weights
            weights = weights / np.sum(weights)

        averaged_normal = np.sum(
            normals * np.tile(weights, (normals.shape[0], 1)), axis=1
        )
        return averaged_normal

    def compute_danger_weight(self, min_gamma, averaged_normal) -> float:
        if min_gamma <= 1:
            return 1.0
        elif min_gamma >= self.gamma_critical:
            return 0.0
        else:
            return (self.gamma_critical - min_gamma) / (self.gamma_critical - 1)

    def compute_damping(
        self, desired_velocity: np.ndarray, current_velocity: np.ndarray
    ) -> np.ndarray:
        averaged_normal = self.compute_averaged_normal(
            self._normals_to_obstacles, self._gammas_of_obstacles
        )
        min_gamma = np.min(self._gammas_of_obstacles)
        weight = self.compute_danger_weight(min_gamma, averaged_normal)

        damping_matrix_dynamics = self._compute_dynamics_damping(
            desired_velocity, averaged_normal, min_gamma
        )
        if weight < 0:
            return damping_matrix_dynamics

        damping_matrix_obstacle = self._compute_obstacle_damping(
            averaged_normal, desired_velocity, current_velocity
        )

        return weight * damping_matrix_obstacle + (1 - weight) * damping_matrix_dynamics

    def _compute_dynamics_damping(
        self,
        desired_velocity: np.ndarray,
        averaged_normal: np.ndarray,
        min_gamma: np.ndarray,
        power_weight: int | float = 1 / 2,
    ) -> np.ndarray:
        basis_matrix = get_orthogonal_basis(desired_velocity)

        # The weight is designed to be:
        # - smaller when normal larger, i.e., | n | -> 1
        # - smaller when normal & DS are perpendicular, i.e., | dotprod | -> 0
        # => perp_vector is small if normal is small or  normal & DS are aligned
        # - smaller when gamma is higher

        perp_vector = (basis_matrix[:, 0] @ averaged_normal) * basis_matrix[:, 0]
        perp_vector = perp_vector - averaged_normal

        weight = min(
            1,
            (perp_vector.T @ perp_vector) ** power_weight
            + ((self.gamma_critical - min_gamma) / (self.gamma_critical - 1.0))
            ** power_weight,
        )
        other_lambda = (
            weight * self.lambda_remaining + (1 - weight) * self.lambda_obstacle
        )

        damping_matrix = np.diag(
            np.hstack(
                (self.lambda_dynamics, other_lambda * np.ones(self.dimension - 1))
            )
        )

        return basis_matrix @ damping_matrix @ basis_matrix.T

    def _compute_obstacle_damping(
        self,
        averaged_normal: np.ndarray,
        desired_velocity: np.ndarray,
        current_velocity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        basis1 = averaged_normal / np.linalg.norm(averaged_normal)

        dotproduct = np.dot(basis1, desired_velocity)
        if not dotproduct:  # Zero value
            return get_orthogonal_basis(basis1)

        basis2 = desired_velocity - basis1 * dotproduct
        basis2 = basis2 / np.linalg.norm(basis2)

        if self.dimension == 2:
            basis_matrix = np.vstack((basis1, basis2))
        elif self.dimension == 3:
            basis_matrix = np.vstack((basis1, basis2, np.cross(basis1, basis2)))
        else:
            raise NotImplementedError(f"Not defined for dimensions > 3.")

        weight = abs(dotproduct / np.linalg.norm(desired_velocity))

        # Set desired matrix values
        if current_velocity is not None:
            delta_velocity = desired_velocity - current_velocity
            if basis1.T @ delta_velocity > 0:
                # When correcting away from obstacle, be stiff (!)
                lambda1 = self.lambda_obstacle
            else:
                lambda1 = self.lambda_remaining

        lambda2 = (1 - weight) * self.lambda_dynamics + weight * self.lambda_remaining
        damping_matrix = np.diag(
            np.hstack(
                (
                    lambda1,
                    lambda2,
                    self.lambda_remaining * np.ones(self.dimension - 2),
                )
            )
        )

        return basis_matrix @ damping_matrix @ basis_matrix.T
