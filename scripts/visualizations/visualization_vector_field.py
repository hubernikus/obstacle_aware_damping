import math
import numpy as np

import matplotlib.pyplot as plt

from vartools.animator import Animator
from vartools.dynamical_systems import LinearSystem, QuadraticAxisConvergence

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


class AnimationPassiveController(Animator):
    # def setup(self, n_traj: int =  4):
    def setup(
        self,
        environment,
        controller,
        avoider,
        x_lim=[-16, 12],
        y_lim=[-10, 10],
        attractor=None,
        n_traj: int = 10,
    ):
        # self.fig, self.ax = plt.subplots(figsize=(12, 9 / 4 * 3))
        # Kind-of HD
        self.fig, self.ax = plt.subplots(figsize=(19.20, 10.80))

        self.environment = environment
        self.controller = controller

        self.n_traj = n_traj
        self.start_positions = np.vstack(
            (
                np.ones(self.n_traj) * x_lim[0],
                np.linspace(y_lim[0], y_lim[1], self.n_traj),
            )
        )

        self.n_grid = 15
        if attractor is None:
            self.attractor = np.array([8.0, 0])
        else:
            self.attractor = attractor
        self.position = np.array([-8, 0.1])  # Start position

        self.dimension = 2
        self.trajectories = []
        for tt in range(self.n_traj):
            self.trajectories.append(np.zeros((self.dimension, self.it_max + 1)))
            self.trajectories[tt][:, 0] = self.start_positions[:, tt]
        # for ii in range(self.n_traj):
        #     self.trajectory = np.zeros((self.dimension, self.it_max))

        self.x_lim = x_lim
        self.y_lim = y_lim

        # self.initial_dynamics = LinearSystem(self.attractor, maximum_velocity=1.0)
        self.avoider = avoider

        # self.avoider = RotationalAvoider(
        #     initial_dynamics=self.initial_dynamics,
        #     obstacle_environment=self.environment,
        #     convergence_system=LinearSystem(self.attractor),
        #     convergence_radius=math.pi * 0.5,
        # )

        # self.trajectory_color = "green"
        cm = plt.get_cmap("gist_rainbow")
        self.color_list = [cm(1.0 * cc / self.n_traj) for cc in range(self.n_traj)]

        # Create agents and set start positions
        self.agents = []
        for ii in range(self.n_traj):
            self.agents.append(
                Agent(position=self.start_positions[:, ii], velocity=np.zeros(2))
            )

    def update_step(self, ii: int) -> None:
        if not ii % 10:
            print(f"Iteration {ii}")

        for tt in range(self.n_traj):
            # pos = self.trajectories[tt][:, ii]

            # Update agent
            agent = self.agents[tt]

            velocity = self.avoider.evaluate_normalized(agent.position)
            force = self.controller.compute_force(
                position=agent.position,
                velocity=agent.velocity,
                desired_velocity=velocity,
            )

            agent.update_step(self.dt_simulation, control_force=force)

            # Add to list
            self.trajectories[tt][:, ii + 1] = agent.position

        for obs in self.environment:
            obs.pose.position = (
                self.dt_simulation * obs.twist.linear + obs.pose.position
            )
            obs.pose.orientation = (
                self.dt_simulation * obs.twist.angular + obs.pose.orientation
            )

        self.ax.clear()

        for tt in range(self.n_traj):
            trajectory = self.trajectories[tt]
            self.ax.plot(
                trajectory[0, 0],
                trajectory[1, 0],
                "ko",
                linewidth=2.0,
            )
            self.ax.plot(
                trajectory[0, :ii],
                trajectory[1, :ii],
                "--",
                color=self.color_list[tt],
                linewidth=2.0,
            )
            self.ax.plot(
                trajectory[0, ii],
                trajectory[1, ii],
                "o",
                color=self.color_list[tt],
                markersize=8,
            )

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

        self.ax.scatter(
            self.avoider.initial_dynamics.attractor_position[0],
            self.avoider.initial_dynamics.attractor_position[1],
            marker="*",
            s=200,
            color="white",
            zorder=5,
        )

        plot_vectorfield = True
        if plot_vectorfield:
            plot_obstacle_dynamics(
                obstacle_container=self.environment,
                collision_check_functor=lambda x: (
                    self.environment.get_minimum_gamma(x) <= 1
                ),
                dynamics=self.avoider.evaluate_normalized,
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
    margin_absolut = 0.0

    center = np.array([0.2, 0.6])
    axes_length = np.array([2.0, 0.7])
    delta_ref = 0.5 * (axes_length[0] - axes_length[1])
    obstacle_environment.append(
        Cuboid(
            axes_length=axes_length,
            center_position=center + np.array([-delta_ref, 0]),
            orientation=0 * np.pi / 180.0,
            margin_absolut=margin_absolut,
            relative_reference_point=np.array([delta_ref, 0]),
            tail_effect=False,
        )
    )

    obstacle_environment.append(
        Cuboid(
            axes_length=axes_length,
            center_position=center + np.array([0, -delta_ref]),
            orientation=90 * np.pi / 180.0,
            margin_absolut=margin_absolut,
            relative_reference_point=np.array([delta_ref, 0]),
            tail_effect=False,
        )
    )

    # center = np.array([0.9, 0.2])
    # # axes_length = np.array([1.3, 0.3])
    # # delta_ref = 0.5 * (axes_length[0] - axes_length[1])
    # obstacle_environment.append(
    #     Cuboid(
    #         axes_length=axes_length,
    #         center_position=center + np.array([delta_ref, 0]),
    #         orientation=180 * np.pi / 180.0,
    #         margin_absolut=margin_absolut,
    #         relative_reference_point=np.array([delta_ref, 0]),
    #         tail_effect=False,
    #     )
    # )

    # obstacle_environment.append(
    #     Cuboid(
    #         axes_length=axes_length,
    #         center_position=center + np.array([0, delta_ref]),
    #         orientation=-90 * np.pi / 180.0,
    #         margin_absolut=margin_absolut,
    #         relative_reference_point=np.array([delta_ref, 0]),
    #         tail_effect=False,
    #     )
    # )
    return obstacle_environment


def animation_avoidance_with_controller(save_animation=False):
    delta_time = 0.05

    lambda_max = 1.0 / delta_time

    lambda_DS = 0.2 * lambda_max
    lambda_perp = 0.05 * lambda_max
    lambda_obs = 1.0 * lambda_max

    dimension = 2

    start_position = np.array([-2.5, -1.0])
    attractor_position = np.array([2.5, 1.0])

    initial_dynamics = LinearSystem(
        attractor_position=attractor_position,
        maximum_velocity=1.0,
        distance_decrease=1.0,
    )
    start_velocity = initial_dynamics.evaluate(start_position)

    environment = create_environment()

    avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=environment,
    )

    min_distance_both = [None] * 2
    position_lists_both = [None] * 2

    # controller = PassiveDynamicsController(
    #     lambda_dynamics=lambda_DS, lambda_remaining=lambda_perp, dimension=2
    # )

    controller = ObstacleAwarePassivController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        lambda_obstacle=lambda_obs,
        dimension=dimension,
        environment=environment,
    )

    animator = AnimationPassiveController(
        dt_simulation=delta_time,
        dt_sleep=0.001,
        it_max=200,
        animation_name="corner_avoidance_edge",
        file_type=".gif",
    )
    animator.setup(
        environment=environment,
        controller=controller,
        avoider=avoider,
        x_lim=[-3, 3],
        y_lim=[-2.0, 2.0],
    )
    animator.run(save_animation=save_animation)


def animation_disturbance(save_animation=False):
    delta_time = 0.05

    lambda_max = 1.0 / delta_time

    lambda_DS = 0.2 * lambda_max
    lambda_perp = 0.05 * lambda_max
    lambda_obs = 1.0 * lambda_max

    dimension = 2

    start_position = np.array([-2.5, -1.0])
    attractor_position = np.array([2.5, 1.0])

    initial_dynamics = LinearSystem(
        attractor_position=attractor_position,
        maximum_velocity=1.0,
        distance_decrease=1.0,
    )
    start_velocity = initial_dynamics.evaluate(start_position)

    environment = create_environment()

    avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=environment,
    )

    controller = PassiveDynamicsController(
        lambda_dynamics=lambda_DS, lambda_remaining=lambda_perp, dimension=2
    )

    controller = ObstacleAwarePassivController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        lambda_obstacle=lambda_obs,
        dimension=dimension,
        environment=environment,
    )

    animator = AnimationPassiveController(
        dt_simulation=delta_time,
        dt_sleep=0.001,
        it_max=200,
        animation_name="corner_avoidance_edge",
        file_type=".gif",
    )
    animator.setup(
        environment=environment,
        controller=controller,
        avoider=avoider,
        x_lim=[-3, 3],
        y_lim=[-2.0, 2.0],
    )
    animator.run(save_animation=save_animation)


if (__name__) == "__main__":
    # def main():
    plt.style.use("dark_background")
    # animation_avoidance_with_controller(save_animation=True)
    animation_disturbance(save_animation=False)
