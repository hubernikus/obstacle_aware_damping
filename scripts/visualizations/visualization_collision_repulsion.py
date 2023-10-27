from dataclasses import dataclass

import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import patches

from vartools.animator import Animator
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


class ParallelSinusMagnitudeDynamics(DynamicalSystem):
    def __init__(self):
        self.vector_nominal = np.array([1.0, 0])
        self.center_position = np.array([0, 0.8])

        self.frequency = 2.0

        self.base_magnitude = 1.0
        self.amplitude = 0.3

    def evaluate(self, position):
        dotprod = self.vector_nominal @ (position - self.center_position)
        magnitude = (
            np.sin(dotprod * self.frequency) * self.amplitude + self.base_magnitude
        )

        return self.vector_nominal * magnitude


class AnimationPassiveControllerComparison(Animator):
    # def setup(self, n_traj: int =  4):
    def setup(
        self,
        environment,
        controller,
        avoider,
        x_lim=[-16, 12],
        y_lim=[-10, 10],
        attractor=None,
        disturbances=[],
        # n_traj: int = 10,
    ):
        # Kind-of HD
        self.fig, self.ax = plt.subplots(figsize=(19.20, 10.80))

        self.environment = environment
        self.controller = controller
        self.disturbances = disturbances

        self.disturbance_positions = np.zeros((2, 0))
        self.disturbance_vectors = np.zeros((2, 0))

        # self.n_traj = n_traj
        # self.start_positions = np.vstack(
        #     (
        #         np.ones(self.n_traj) * x_lim[0],
        #         np.linspace(y_lim[0], y_lim[1], self.n_traj),
        #     )
        # )

        # start_y_positions = [-1.6, -0.2, 1.0]
        start_y_positions = [0.5]
        self.n_traj = len(start_y_positions)
        self.start_positions = np.vstack(
            (
                np.ones(self.n_traj) * x_lim[0],
                start_y_positions
                # np.linspace(y_lim[0], y_lim[1], self.n_traj),
            )
        )

        self.n_grid = 15
        if attractor is None:
            self.attractor = np.array([8.0, 0])
        else:
            self.attractor = attractor
        self.position = np.array([-8, 0.1])  # Start position

        self.dimension = 2
        self.trajectories_obstacle = []
        for tt in range(self.n_traj):
            self.trajectories_obstacle.append(
                np.zeros((self.dimension, self.it_max + 1))
            )
            self.trajectories_obstacle[tt][:, 0] = self.start_positions[:, tt]

        self.x_lim = x_lim
        self.y_lim = y_lim

        self.avoider = avoider

        self.color_obstacle = "#005fa8ff"
        # self.color_ds = "#bd0500ff"

        # Create agents and set start positions
        self.agents_obstacle = []
        for ii in range(self.n_traj):
            velocity = self.avoider.evaluate_normalized(self.start_positions[:, ii])
            self.agents_obstacle.append(
                Agent(position=self.start_positions[:, ii], velocity=velocity)
            )

    def update_step(self, ii: int) -> None:
        if not ii % 10:
            print(f"Iteration {ii}")

        for tt in range(self.n_traj):
            # Update agent
            agent = self.agents_obstacle[tt]
            desired_velocity = self.avoider.evaluate_normalized(agent.position)

            force = self.controller.compute_force(
                position=agent.position,
                velocity=agent.velocity,
                desired_velocity=desired_velocity,
            )

            for disturbance in self.disturbances:
                if disturbance.trajectory_number != tt:
                    continue

                if disturbance.timestep != ii:
                    continue

                force = force + disturbance.force

                self.disturbance_positions = np.vstack(
                    (self.disturbance_positions.T, agent.position)
                ).T
                self.disturbance_vectors = np.vstack(
                    (self.disturbance_vectors.T, disturbance.force)
                ).T

            agent.update_step(self.dt_simulation, control_force=force)
            self.trajectories_obstacle[tt][:, ii + 1] = agent.position

        for obs in self.environment:
            obs.pose.position = (
                self.dt_simulation * obs.twist.linear + obs.pose.position
            )
            obs.pose.orientation = (
                self.dt_simulation * obs.twist.angular + obs.pose.orientation
            )

        # Get normal
        self.ax.clear()

        arrow_width = 0.03
        position = self.trajectories_obstacle[tt][:, ii]

        ctrl = self.controller

        # surface_point = self.environment[0].get_point_on_surface(
        #     position, in_obstacle_frame=False
        # )

        if isinstance(self.controller, PassiveDynamicsController):
            D_matrix = self.controller.compute_damping(desired_velocity=velocity)
        else:
            D_matrix = self.controller.compute_damping(
                current_velocity=agent.velocity, desired_velocity=desired_velocity
            )

        eig_vals, eig_vecs = np.linalg.eig(D_matrix)

        # force_sacling = 0.3
        # force_color = "yellow"
        # self.ax.arrow(
        #     position[0],
        #     position[1],
        #     force[0] * force_sacling,
        #     force[1] * force_sacling,
        #     label="Control force",
        #     color=force_color,
        #     width=arrow_width,
        #     zorder=3,
        # )

        # # Velocity
        vel_scaling = 0.8
        agent = self.agents_obstacle[0]

        self.ax.arrow(
            position[0],
            position[1],
            desired_velocity[0] * vel_scaling,
            desired_velocity[1] * vel_scaling,
            width=arrow_width,
            color="#17BFE0",
            label="Desired velocity",
            length_includes_head=True,
            zorder=3.0,
        )

        self.ax.arrow(
            position[0],
            position[1],
            agent.velocity[0] * vel_scaling,
            agent.velocity[1] * vel_scaling,
            width=arrow_width,
            color="blue",
            label="Real velocity",
            length_includes_head=True,
            zorder=3.0,
        )

        # Plot passivity circle
        axes_length = np.sqrt(eig_vals)
        # axes_length = eig_vals

        velocity_D_frame = np.linalg.pinv(eig_vecs) @ desired_velocity

        pp = 2
        ellipse_factor = (np.sum((velocity_D_frame / axes_length) ** pp)) ** (1.0 / pp)

        # factor = np.linalg.norm(velocity_D_frame) / np.linalg.norm(agent.velocity)
        first_vec = eig_vecs[:, 0]
        # first_vec = [1, 0.0]

        angle = np.arctan2(first_vec[1], first_vec[0])
        width = ellipse_factor * axes_length[0]
        height = ellipse_factor * axes_length[1]
        # breakpoint()

        passivity_ellise = patches.Ellipse(
            (position[0], position[1]),
            width * 2 * vel_scaling,
            height * 2 * vel_scaling,
            angle=angle * 180 / math.pi,
            # angle=0,
            linewidth=2,
            fill=True,
            zorder=2.0,
            # color="red",
            color="orange",
            alpha=0.3,
            label="Non-passive region",
        )
        self.ax.add_patch(passivity_ellise)

        local_vels = np.linalg.pinv(eig_vecs) @ agent.velocity
        # real_vel_factor = np.sum((local_vels / (axes_length * ellipse_factor)) ** pp)
        real_vel_factor = np.sum((local_vels / (axes_length * ellipse_factor)) ** pp)

        # local_vels = np.linalg.pinv(eig_vecs) @ desired_velocity
        # real_vel_factor = np.sum((local_vels / (axes_length * ellipse_factor)) ** pp)

        if real_vel_factor > 1:
            passivity_color = "green"
            label = "(passive)"
        else:
            passivity_color = "red"
            label = "(non-passive)"

        velocity_difference = desired_velocity - agent.velocity
        tmp_width = np.linalg.norm(velocity_difference) * 0.1
        if tmp_width > arrow_width:
            tmp_width = arrow_width

        self.ax.arrow(
            # position[0] + desired_velocity[0] * vel_scaling,
            # position[1] + desired_velocity[1] * vel_scaling,
            position[0] + agent.velocity[0] * vel_scaling,
            position[1] + agent.velocity[1] * vel_scaling,
            velocity_difference[0] * vel_scaling,
            velocity_difference[1] * vel_scaling,
            width=tmp_width,
            color=passivity_color,
            label=f"Velocity difference \n {label}",
            length_includes_head=True,
            zorder=3.0,
        )

        legend_label = None
        for tt in range(self.n_traj):
            trajectory = self.trajectories_obstacle[tt]
            self.ax.plot(
                trajectory[0, 0],
                trajectory[1, 0],
                "ko",
                linewidth=4.0,
                markersize=16.0,
            )
            self.ax.plot(
                trajectory[0, :ii],
                trajectory[1, :ii],
                "--",
                color=self.color_obstacle,
                linewidth=4.0,
                label=legend_label,
                zorder=-1,
            )
            self.ax.plot(
                trajectory[0, ii],
                trajectory[1, ii],
                "o",
                color=self.color_obstacle,
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

        if self.disturbance_positions.shape[1] > 0:
            self.ax.quiver(
                self.disturbance_positions[0],
                self.disturbance_positions[1],
                self.disturbance_vectors[0],
                self.disturbance_vectors[1],
                label="Disturbance",
                color="#ff09f6ff",
                # color
                zorder=2,
            )

        self.ax.legend(fontsize="20", loc="upper right")

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
        Cuboid(
            # axes_length=np.array([2, 2.0]),
            axes_length=np.array([1.3, 5.0]),
            center_position=np.array([-2.2, 0.0]),
            # orientation=30 * np.pi / 180,
            orientation=0 * np.pi / 180,
            margin_absolut=0.0,
            # relative_reference_point=np.array([0.0, 0]),
            tail_effect=False,
        )
    )

    obstacle_environment.append(
        Cuboid(
            # axes_length=np.array([2, 2.0]),
            axes_length=np.array([1.3, 5.0]),
            center_position=np.array([1.0, 4.0]),
            # orientation=30 * np.pi / 180,
            orientation=0 * np.pi / 180,
            margin_absolut=0.0,
            # relative_reference_point=np.array([0.0, 0]),
            tail_effect=False,
        )
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
    trajectory_number: int = 0


def animation_collision_repulsion(save_animation=False):
    delta_time = 0.05

    # lambda_max = 1.0 / delta_time
    lambda_max = 0.5 / delta_time

    lambda_DS = 0.5 * lambda_max
    lambda_perp = 0.1 * lambda_max
    lambda_obs = 1.0 * lambda_max

    dimension = 2

    # start_position = np.array([-2.5, -1.0])
    initial_dynamics = LinearSystem(
        attractor_position=np.array([1e6, 0]),
        maximum_velocity=1.0,
        distance_decrease=1.0,
    )

    environment = create_environment()

    avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=environment,
    )

    controller_obstacle = ObstacleAwarePassivController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        lambda_obstacle=lambda_obs,
        dimension=dimension,
        environment=environment,
    )

    controller_obstacle.tail_effect = True

    # disturbances = [
    #     Disturbance(60, 60 * np.array([0.7, -0.7]), 1),
    #     Disturbance(80, np.array([0.5, -60]), 0),
    #     Disturbance(35, np.array([0.5, 60]), 2),
    #     Disturbance(15, np.array([-60, 0.5]), 2),
    # ]

    animator = AnimationPassiveControllerComparison(
        dt_simulation=delta_time,
        dt_sleep=0.001,
        it_max=250,
        animation_name="passivity_analysis",
        file_type=".gif",
    )
    animator.setup(
        environment=environment,
        controller=controller_obstacle,
        avoider=avoider,
        x_lim=[-4, 3],
        y_lim=[0.1, 4.0],
    )
    animator.run(save_animation=save_animation)


if (__name__) == "__main__":
    # def main():
    plt.style.use("dark_background")
    animation_collision_repulsion(save_animation=True)
