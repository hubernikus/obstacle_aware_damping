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
        return np.array([3.0, 0.0])


@dataclass
class SimpleController:
    s_obs: float = 10.0
    s_ds: float = 1.0

    # D: np.ndarray = field(default_factory=lambda: np.zeros(dimension))

    # def __post_init__(self):
    #     self._D = np.diag([self.s_ds, self.s_obs])

    @property
    def D(self) -> np.ndarray:
        return np.diag([self.s_ds, self.s_obs])

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


def main(save_figure=False):
    x_lim = [-0.35, 2.1]
    y_lim = [-0.25, 1.1]

    dynamics = ParallelToBoundaryDS()
    start_position = np.zeros(dimension)
    impact_velocity = np.array([0.0, 10.0])
    initial_velocity = dynamics.evaluate(start_position)

    agent: Agent = Agent(position=start_position)
    # Set initial velocity
    agent.velocity = initial_velocity + impact_velocity

    dt = 1e-2
    it_max = int((x_lim[1] - x_lim[0]) / (agent.velocity[0] * dt) + 1)

    controller = SimpleController()

    fig, ax = plt.subplots(figsize=(5, 4))
    # Plot velocity arrow
    arrow_scaling = 0.03
    ax.arrow(
        start_position[0],
        start_position[1],
        initial_velocity[0] * arrow_scaling,
        initial_velocity[1] * arrow_scaling,
        width=0.02,
        color="blue",
        zorder=3,
        # label="Initial velocity",
    )
    ax.arrow(
        start_position[0],
        start_position[1],
        impact_velocity[0] * arrow_scaling,
        impact_velocity[1] * arrow_scaling,
        width=0.02,
        # color="red",
        color="#740782ff",
        zorder=3,
        # label="Disturbance",
    )

    ax.plot(start_position[0], start_position[1], "o", color="black", zorder=10)
    ax.plot(start_position[0], 1.0, "o", color="black", zorder=10)

    plot_text = False
    if plot_text:
        ax.text(
            start_position[0] + initial_velocity[0] * arrow_scaling + 0.2,
            start_position[1] - 0.048,
            "Initial \nvelocity",
            backgroundcolor="white",
            size=9,
            color="blue",
            zorder=3,
        )

        ax.text(
            start_position[0] - 0.16,
            start_position[1] + impact_velocity[0] * arrow_scaling + 0.44,
            "Impact \nvelocity",
            backgroundcolor="white",
            size=9,
            color="red",
            zorder=2,
        )

    else:
        ax.text(
            start_position[0] + initial_velocity[0] * arrow_scaling + 0.15,
            start_position[1] - 0.1,
            r"$v^{0}$",
            backgroundcolor="white",
            size=12,
            color="blue",
            zorder=3,
        )

        ax.text(
            start_position[0] - 0.16,
            start_position[1] + impact_velocity[0] * arrow_scaling + 0.44,
            r"$v^{I}$",
            backgroundcolor="white",
            size=12,
            color="#740782ff",
            zorder=3,
        )

        ax.text(
            start_position[0] - 0.17,
            start_position[1] - 0.16,
            r"$p^0$",
            backgroundcolor="white",
            size=12,
            color="black",
            zorder=3,
        )

        ax.text(
            start_position[0] - 0.09,
            1.0 - 0.16,
            r"$\xi^b$",
            backgroundcolor="white",
            size=12,
            color="black",
            zorder=3,
        )

    container = ObstacleContainer()
    container.append(
        Cuboid(axes_length=np.array([3, 0.4]), pose=Pose(position=np.array([1.0, 1.2])))
    )

    plot_obstacles(
        ax=ax,
        obstacle_container=container,
        x_lim=x_lim,
        y_lim=y_lim,
        alpha_obstacle=1.0,
        zorder_obs=5.0,
    )
    plot_obstacle_dynamics(
        obstacle_container=container,
        dynamics=dynamics.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        n_grid=10,
        vectorfield_color="#7a7a7a7f",
    )

    # Plot baseline / Undisturbed motion
    ax.plot(
        x_lim,
        np.zeros(2) * agent.position[1],
        ":",
        color="black",
        alpha=0.5,
        linewidth=2.0,
        zorder=1,
        # label="Undisturbed",
    )

    # Plot uncontrolled, but disturbed
    undisturbed_position = agent.position + agent.velocity * dt * it_max
    ax.plot(
        [agent.position[0], undisturbed_position[0]],
        [agent.position[1], undisturbed_position[1]],
        ":",
        # label="Uncontrolled",
        color="red",
        alpha=0.6,
        linewidth=2.0,
        zorder=2,
    )

    positions = np.zeros((dimension, it_max + 1))
    positions[:, 0] = agent.position

    for ii in range(it_max):
        force = controller.compute_force(
            agent.velocity, dynamics.evaluate(agent.position)
        )
        agent.apply_force(force)
        agent.euler_step(dt)

        positions[:, ii + 1] = agent.position

    ax.plot(
        positions[0, :],
        positions[1, :],
        "k",
        # linestyle="dashdot",
        linewidth=2.0,
        label=r"$s^{obs} = 10$",
        color=colors[0],
    )

    # Very high damping - w/ max-force
    controller.s_obs = 20
    agent: Agent = Agent(position=start_position)
    # Set initial velocity
    agent.velocity = dynamics.evaluate(agent.position)
    # Impact velocity
    agent.velocity = agent.velocity + impact_velocity
    # agent.max_force = 1e10
    agent.max_force = 1e10

    positions = np.zeros((dimension, it_max + 1))
    positions[:, 0] = agent.position

    for ii in range(it_max):
        force = controller.compute_force(
            agent.velocity, dynamics.evaluate(agent.position)
        )
        agent.apply_force(force)
        agent.euler_step(dt)

        positions[:, ii + 1] = agent.position

    ax.plot(
        positions[0, :],
        positions[1, :],
        linewidth=2,
        label="$s^{obs} = 20$",
        color=colors[1],
        linestyle="--",
    )

    # Very high damping - w/ max-force
    controller.s_obs = 40
    agent: Agent = Agent(position=start_position)
    # Set initial velocity
    agent.velocity = dynamics.evaluate(agent.position)
    # Impact velocity
    agent.velocity = agent.velocity + impact_velocity
    agent.max_force = 1e10

    positions = np.zeros((dimension, it_max + 1))
    positions[:, 0] = agent.position

    for ii in range(it_max):
        force = controller.compute_force(
            agent.velocity, dynamics.evaluate(agent.position)
        )
        agent.apply_force(force)
        agent.euler_step(dt)

        positions[:, ii + 1] = agent.position

    ax.plot(
        positions[0, :],
        positions[1, :],
        linewidth=2,
        label="$s^{obs} = 40$",
        color=colors[2],
        linestyle=(0, (3, 1, 1, 1, 1, 1)),
    )

    # Very high damping - w/ max-force
    controller.s_obs = 40
    agent: Agent = Agent(position=start_position)
    # Set initial velocity
    agent.velocity = dynamics.evaluate(agent.position)
    # Impact velocity
    agent.velocity = agent.velocity + impact_velocity
    agent.max_force = 100

    positions = np.zeros((dimension, it_max + 1))
    positions[:, 0] = agent.position

    for ii in range(it_max):
        force = controller.compute_force(
            agent.velocity, dynamics.evaluate(agent.position)
        )
        agent.apply_force(force)
        agent.euler_step(dt)

        positions[:, ii + 1] = agent.position

    ax.plot(
        positions[0, :],
        positions[1, :],
        linestyle="dashdot",
        linewidth=2,
        label=r"$s^{obs} = 40$" + "\n" + r"$\tau^{max} = 100$",
        color=colors[3],
    )

    ax.legend(loc="center right")

    if save_figure:
        figname = "parallel_avoidance_obstacle"
        plt.savefig("figures/" + figname + filetype, bbox_inches="tight")


if (__name__) == "__main__":
    plt.ion()
    plt.close("all")
    filetype = ".pdf"

    colors = ["#DB7660", "#DB608F", "#47A88D", "#638030"]

    main(save_figure=True)

    # breakpoint()
