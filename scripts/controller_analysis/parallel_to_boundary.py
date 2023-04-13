from dataclasses import dataclass, field
import numpy as np

import matplotlib.pyplot as plt

from vartools.states import Pose

from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import (
    plot_obstacles,
)
from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)

# import vartools

dimension = 2


class ParallelToBoundaryDS:
    dimension = 2

    def evaluate(self, position: np.ndarray) -> np.ndarray:
        return np.array([1.0, 0.0])


@dataclass
class SimpleController:
    s_obs: float = 10.0
    s_ds: float = 1.0

    D: np.ndarray = field(default_factory=lambda: np.zeros(dimension))

    def __post_init__(self):
        self.D = np.diag([self.s_ds, self.s_obs])

    def compute_force(self, velocity, desired_velocity):
        return self.D @ (desired_velocity - velocity)


@dataclass
class Agent:
    position: np.ndarray = field(default_factory=lambda: np.zeros(dimension))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(dimension))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(dimension))

    max_force: float = 100.0

    M: float = 1.0

    def apply_force(self, force):
        if (force_norm := np.linalg.norm(force)) > self.max_force:
            force = force / force_norm * self.max_force
        self.acceleration = force / self.M

    def euler_step(self, dt=1e-2):
        self.velocity = self.velocity + dt * self.acceleration
        self.position = self.position + dt * self.velocity


def main():
    x_lim = [-0.1, 2.0]
    y_lim = [-0.1, 1.1]

    dynamics = ParallelToBoundaryDS()

    agent: Agent = Agent()
    # Set initial velocity
    agent.velocity = dynamics.evaluate(agent.position)
    # Impact velocity
    agent.velocity = agent.velocity + np.array([0.0, 10.0])

    controller = SimpleController()

    fig, ax = plt.subplots(figsize=(5, 4))

    container = ObstacleContainer()
    container.append(
        Cuboid(axes_length=np.array([3, 0.4]), pose=Pose(position=np.array([1.0, 1.2])))
    )

    plot_obstacles(ax=ax, obstacle_container=container, x_lim=x_lim, y_lim=y_lim)
    plot_obstacle_dynamics(
        obstacle_container=container,
        dynamics=dynamics.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
    )

    # Plot baseline
    ax.plot(
        x_lim, np.zeros(2) * agent.position[1], color="black", alpha=0.5, linewidth=2.0
    )

    it_max = 500
    dt = 1e-2

    positions = np.zeros((dimension, it_max + 1))
    positions[:, 0] = agent.position

    for ii in range(it_max):
        force = controller.compute_force(
            agent.velocity, dynamics.evaluate(agent.position)
        )
        agent.apply_force(force)
        agent.euler_step(dt)

        positions[:, ii + 1] = agent.position

    plt.plot(dt * np.arange(it_max + 1), positions[0, :])
    # plt.ylim([-5, 5])


if (__name__) == "__main__":
    plt.ion()
    plt.close("all")
    main()
    # breakpoint()
