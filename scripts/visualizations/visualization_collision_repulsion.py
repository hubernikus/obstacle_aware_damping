from dataclasses import dataclass

import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import patches

from vartools.animator import Animator
from vartools.states import Pose
from vartools.dynamical_systems import (
    LinearSystem,
    QuadraticAxisConvergence,
    DynamicalSystem,
)

from dynamic_obstacle_avoidance.obstacles import StarshapedFlower
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid

from dynamic_obstacle_avoidance.avoidance import ModulationAvoider

from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from passive_control.agent import Agent
from passive_control.controller import Controller
from passive_control.controller import PassiveDynamicsController
from passive_control.controller import ObstacleAwarePassivController


class ParallelToBoundaryDS:
    dimension = 2

    def evaluate(self, position: np.ndarray) -> np.ndarray:
        return np.array([3.0, 0.0])


class AnimationCollisionAvoidance(Animator):
    # def setup(self, n_traj: int =  4):
    def setup(
        self,
        environment,
        controllers,
        avoider,
        x_lim=[-16, 12],
        y_lim=[-10, 10],
        attractor=None,
        disturbances=[],
    ):
        # Kind-of HD
        self.fig, self.ax = plt.subplots(figsize=(19.20, 10.80))

        self.environment = environment
        self.controllers = controllers
        self.disturbances = disturbances

        self.disturbance_positions = np.zeros((2, 0))
        self.disturbance_vectors = np.zeros((2, 0))

        # start_y_positions = [-1.6, -0.2, 1.0]
        self.start_position = [x_lim[0], 0.0]
        self.n_controllers = len(self.controllers)

        self.n_grid = 15
        if attractor is None:
            self.attractor = np.array([8.0, 0])
        else:
            self.attractor = attractor
        self.position = np.array([-8, 0.1])  # Start position

        self.dimension = 2
        self.trajectories = []
        for tt in range(self.n_controllers):
            self.trajectories.append(np.zeros((self.dimension, self.it_max + 1)))
            self.trajectories[tt][:, 0] = self.start_position

        self.x_lim = x_lim
        self.y_lim = y_lim

        self.avoider = avoider

        self.color_obstacle = "#005fa8ff"
        # self.color_ds = "#bd0500ff"

        # Create agents and set start positions
        self.agents = []
        for tt in range(self.n_controllers):
            velocity = self.avoider.evaluate_normalized(self.trajectories[tt][:, 0])
            self.agents.append(
                Agent(position=self.trajectories[tt][:, 0], velocity=velocity)
            )

    def update_step(self, ii: int) -> None:
        if not ii % 10:
            print(f"Iteration {ii}")

        disturbance_force = 0
        for disturbance in self.disturbances:
            if disturbance.timestep != ii:
                continue

            disturbance_force = disturbance.force

            self.disturbance_positions = np.vstack(
                (self.disturbance_positions.T, self.agents[0].position)
            ).T
            self.disturbance_vectors = np.vstack(
                (self.disturbance_vectors.T, disturbance.force)
            ).T

        for tt, (name, controller) in enumerate(self.controllers.items()):
            # Update agent
            agent = self.agents[tt]
            desired_velocity = self.avoider.evaluate_normalized(agent.position)

            force = controller.compute_force(
                position=agent.position,
                velocity=agent.velocity,
                desired_velocity=desired_velocity,
            )

            force = force + disturbance_force

            agent.update_step(self.dt_simulation, control_force=force)
            self.trajectories[tt][:, ii + 1] = agent.position

        for obs in self.environment:
            obs.pose.position = (
                self.dt_simulation * obs.twist.linear + obs.pose.position
            )
            obs.pose.orientation = (
                self.dt_simulation * obs.twist.angular + obs.pose.orientation
            )

        # Get normal
        self.ax.clear()

        # disturbance_color = "#dd34c1"
        disturbance_color = "#D542E0"

        # trajectory_colors = ["#DB7660", "#DB608F", "#47A88D", "#638030"]
        trajectory_colors = ["#5F5CE0", "#E0C958", "#51E0A6", "#E06C55"]
        linestyle_list = ["-", "--", (0, (3, 1, 1, 1, 1, 1)), "dashdot"]

        vel_scaling = 0.1
        arrow_width = 0.03

        force_scaling = 0.0005

        for pp in range(self.disturbance_positions.shape[1]):
            self.ax.arrow(
                self.disturbance_positions[0, pp],
                self.disturbance_positions[1, pp],
                self.disturbance_vectors[0, pp] * force_scaling,
                self.disturbance_vectors[1, pp] * force_scaling,
                width=arrow_width,
                label="Disturbance",
                color=disturbance_color,
                zorder=2,
            )
            # self.ax.plot(
            #     self.disturbance_positions[0, pp],
            #     self.disturbance_positions[1, pp],
            #     "ko",
            #     linewidth=4.0,
            #     markersize=18.0,
            #     # color="#4b1142",
            # )

        legend_label = None
        for tt, (name, controller) in enumerate(self.controllers.items()):
            agent = self.agents[tt]

            self.ax.arrow(
                agent.position[0],
                agent.position[1],
                agent.velocity[0] * vel_scaling,
                agent.velocity[1] * vel_scaling,
                width=arrow_width,
                # color="blue",
                color=trajectory_colors[tt],
                label=name,
                length_includes_head=True,
                zorder=3.0,
            )

            trajectory = self.trajectories[tt]
            self.ax.plot(
                trajectory[0, 0],
                trajectory[1, 0],
                "ko",
                linewidth=4.0,
                markersize=16.0,
            )
            self.ax.plot(
                trajectory[0, : ii + 1],
                trajectory[1, : ii + 1],
                linestyle=linestyle_list[tt],
                alpha=0.8,
                # color=self.color_obstacle,
                color=trajectory_colors[tt],
                linewidth=4.0,
                label=legend_label,
                zorder=-1,
            )
            self.ax.plot(
                trajectory[0, ii],
                trajectory[1, ii],
                "o",
                # color=self.color_obstacle,
                color=trajectory_colors[tt],
                markersize=16,
                zorder=4,
            )
            legend_label = None

        # Plot backgroundg
        plot_obstacles(
            ax=self.ax,
            obstacle_container=self.environment,
            x_range=self.x_lim,
            y_range=self.y_lim,
            # noTicks=True,
            showLabel=False,
            alpha_obstacle=1.0,
            # linecolor="white",
        )

        # self.ax.scatter(
        #     self.avoider.initial_dynamics.attractor_position[0],
        #     self.avoider.initial_dynamics.attractor_position[1],
        #     marker="*",
        #     s=200,
        #     color="white",
        #     zorder=5,
        # )

        self.ax.legend(fontsize="20", loc="lower right")

        plot_vectorfield = True
        if plot_vectorfield:
            plot_obstacle_dynamics(
                obstacle_container=self.environment,
                collision_check_functor=lambda x: (
                    self.environment.get_minimum_gamma(x) <= 1
                ),
                dynamics=self.avoider.evaluate_normalized,
                # dynamics=self.avoider.evaluate,
                x_lim=self.x_lim,
                y_lim=self.y_lim,
                ax=self.ax,
                do_quiver=True,
                n_grid=self.n_grid,
                show_ticks=False,
                vectorfield_color="#808080",
                quiver_scale=30,
            )


def create_environment():
    obstacle_environment = ObstacleContainer()

    obstacle_environment.append(
        Cuboid(axes_length=np.array([3, 0.4]), pose=Pose(position=np.array([1.0, 1.2])))
    )

    # obstacle_environment.append(
    #     Ellipse(
    #         axes_length=np.array([2, 2.0]),
    #         center_position=np.array([, 3.0]),
    #         orientation=0.0,
    #         margin_absolut=0.0,
    #         # relative_reference_point=np.array([0.0, 0]),
    #         tail_effect=False,
    #     )
    # )

    return obstacle_environment


@dataclass
class Disturbance:
    timestep: int
    force: np.ndarray


def animation_collision_repulsion(save_animation=False):
    delta_time = 0.01

    # lambda_max = 1.0 / delta_time
    lambda_max = 0.5 / delta_time

    dimension = 2

    # start_position = np.array([-2.5, -1.0])
    # initial_dynamics = LinearSystem(
    #     attractor_position=np.array([1e6, 0]),
    #     maximum_velocity=1.0,
    #     distance_decrease=1.0,
    # )

    initial_dynamics = ParallelToBoundaryDS()

    environment = create_environment()

    avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=environment,
    )

    controllers = {}

    lambda_obs = 10
    lambda_DS = 1.0 * lambda_obs
    lambda_perp = 1.0 * lambda_obs
    controllers[r"$s^{obs} = 10$"] = ObstacleAwarePassivController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        lambda_obstacle=lambda_obs,
        dimension=dimension,
        environment=environment,
    )

    lambda_obs = 20
    lambda_DS = 1.0 * lambda_obs
    lambda_perp = 1.0 * lambda_obs
    controllers[r"$s^{obs} = 20$"] = ObstacleAwarePassivController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        lambda_obstacle=lambda_obs,
        dimension=dimension,
        environment=environment,
    )

    lambda_obs = 40
    lambda_DS = 1.0 * lambda_obs
    lambda_perp = 1.0 * lambda_obs
    controllers[r"$s^{obs} = 40$"] = ObstacleAwarePassivController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        lambda_obstacle=lambda_obs,
        dimension=dimension,
        environment=environment,
    )

    lambda_obs = 40
    lambda_DS = 1.0 * lambda_obs
    lambda_perp = 1.0 * lambda_obs
    controllers[
        "$s^{obs} = 40$" + "\n" + r"$\tau^{max} = 100$"
    ] = ObstacleAwarePassivController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        lambda_obstacle=lambda_obs,
        maximal_force=100,
        dimension=dimension,
        environment=environment,
    )

    impact_velocity = 9.0  # Velocity after impact
    impact_force = impact_velocity / delta_time
    disturbances = [Disturbance(15, impact_force * np.array([0, 1.0]))]

    animator = AnimationCollisionAvoidance(
        dt_simulation=delta_time,
        dt_sleep=0.001,
        it_max=110,
        animation_name="collision_repulsion",
        file_type=".gif",
    )
    animator.setup(
        environment=environment,
        controllers=controllers,
        avoider=avoider,
        disturbances=disturbances,
        x_lim=[-0.35, 2.1],
        y_lim=[-0.25, 1.1],
    )
    animator.run(save_animation=save_animation)


if (__name__) == "__main__":
    # def main():

    plt.style.use("dark_background")
    animation_collision_repulsion(save_animation=True)
