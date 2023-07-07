import copy
from typing import Callable, Optional

from attrs import define, field

import numpy as np


def rk4_step(delta_time: float, functor: Callable, position: np.ndarray):
    """
    perform one time step of the robot with RK4 algorithm
    """
    if True:
        raise NotImplementedError("This should be fixed and improved")
    m1 = self.dt * self.xdot
    k1 = self.dt * self.func_dyn(self.x, self.xdot, t)  # (x, v, t)

    m2 = self.dt * (self.xdot + 0.5 * k1)
    k2 = self.dt * self.func_dyn(
        self.x + 0.5 * m1, self.xdot + 0.5 * k1, t + 0.5 * self.dt
    )

    m3 = self.dt * (self.xdot + 0.5 * k2)
    k3 = self.dt * self.func_dyn(
        self.x + 0.5 * m2, self.xdot + 0.5 * k2, t + 0.5 * self.dt
    )

    m4 = self.dt * (self.xdot + k3)
    k4 = self.dt * self.func_dyn(self.x + m3, self.xdot + k3, t + self.dt)

    # update of the state
    self.x += (m1 + 2 * m2 + 2 * m3 + m4) / 6
    self.xdot += (k1 + 2 * k2 + 2 * k3 + k4) / 6


def get_noise_vector(dimension, std_noise: float = 0.5):
    return np.random.normal(0, std_noise, dimension)


@define
class Agent:
    position: np.ndarray
    velocity: np.ndarray = field()
    # acceleration: np.ndarray = field()

    mass_matrix: np.ndarray = field()
    gravity_vector: np.ndarray = field()
    coriolis_matrix: np.ndarray = field()

    @velocity.default
    def _default_velocity(self):
        return np.zeros_like(self.position)

    @mass_matrix.default
    def _default_mass_matrix(self):
        return np.eye(self.position.shape[0])

    @gravity_vector.default
    def _default_gravity_vector(self):
        return np.zeros_like(self.position)

    @coriolis_matrix.default
    def _default_coriolis_matrix(self):
        return np.zeros((self.position.shape[0], self.position.shape[0]))

    def compute_acceleration(
        self, control_force: np.ndarray, external_force: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if external_force is not None:
            control_force += external_force

        acceleration = (
            control_force - self.gravity_vector - self.coriolis_matrix @ self.velocity
        ) @ np.linalg.pinv(self.mass_matrix)

        return acceleration

    def update_step(self, delta_time: float, control_force: float) -> None:
        acceleration = self.compute_acceleration(control_force)

        current_velocity = copy.deepcopy(self.velocity)
        self.velocity = self.velocity + acceleration * delta_time

        self.position = (
            self.position + 0.5 * (self.velocity + current_velocity) * delta_time
        )


def test_consruction():
    agent = Agent(position=np.zeros(2))
    assert np.allclose(agent.velocity, np.zeros(2))


if (__name__) == "__main__":
    test_consruction()

    print("Done testing.")
