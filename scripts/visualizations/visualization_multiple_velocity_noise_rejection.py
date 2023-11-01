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


class AnimationVelocityNoise(Animator):
    def setup(
        self,
        controller,
        avoider,
        environment=[],
        trajectory_color="blue",
        trajectory_label="Obstacle aware",
        x_lim=[-16, 12],
        y_lim=[-10, 10],
        attractor=None,
        start_position=None,
        disturbances=[],
        n_traj=10,
        velocity_noise=0.0,
        velocity_measurement_noise=0.0,
        position_noise=0.0,
    ):
        self.velocity_noise_std = velocity_noise
        self.velocity_measurement_noise_std = velocity_measurement_noise
        self.position_noise_std = position_noise

        # Kind-of HD
        self.fig, self.ax = plt.subplots(figsize=(19.20, 10.80))

        self.trajectory_color = trajectory_color
        self.legend_label = trajectory_label

        self.environment = environment
        self.controller = controller
        self.disturbances = disturbances

        self.disturbance_positions = np.zeros((2, 0))
        self.disturbance_vectors = np.zeros((2, 0))

        # start_y_positions = [-1.6, -0.2, 1.0]
        if start_position is None:
            self.start_position = np.array([0, 0.0])
        else:
            self.start_position = start_position

        self.n_traj = n_traj

        self.collided_list = []

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
            self.trajectories[tt][:, 0] = self.start_position

        self.x_lim = x_lim
        self.y_lim = y_lim

        self.avoider = avoider

        self.color_obstacle = "#005fa8ff"
        # self.color_ds = "#bd0500ff"

        # Create agents and set start positions
        self.agents = []
        for tt in range(self.n_traj):
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

        for tt in range(self.n_traj):
            if tt in self.collided_list:
                self.trajectories[tt][:, ii + 1] = self.trajectories[tt][:, ii]
                continue

            # Update agent
            agent = self.agents[tt]
            desired_velocity = self.avoider.evaluate_normalized(agent.position)

            noise = self.velocity_measurement_noise_std * np.random.randn(
                self.dimension
            )
            measurent = agent.velocity + noise

            force = self.controller.compute_force(
                position=agent.position,
                velocity=measurent,
                desired_velocity=desired_velocity,
            )

            force = force + disturbance_force

            # Add a noise
            position_noise = np.random.randn(self.dimension) * self.position_noise_std
            agent.position = agent.position + position_noise
            velocity_noise = np.random.randn(self.dimension) * self.velocity_noise_std
            agent.velocity = agent.velocity + velocity_noise

            agent.update_step(self.dt_simulation, control_force=force)
            self.trajectories[tt][:, ii + 1] = agent.position

            if not self.environment.is_collision_free(agent.position):
                self.collided_list.append(tt)

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
        arrow_width = 0.1

        force_scaling = 0.0008

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

        legend_label = self.legend_label
        # for tt, (name, controller) in enumerate(self.controllers.items()):
        for tt in range(self.n_traj):
            agent = self.agents[tt]

            # self.ax.arrow(
            #     agent.position[0],
            #     agent.position[1],
            #     agent.velocity[0] * vel_scaling,
            #     agent.velocity[1] * vel_scaling,
            #     width=arrow_width,
            #     # color="blue",
            #     color=trajectory_colors[tt],
            #     label=name,
            #     length_includes_head=True,
            #     zorder=3.0,
            # )

            trajectory = self.trajectories[tt]
            # self.ax.plot(
            #     trajectory[0, 0],
            #     trajectory[1, 0],
            #     "ko",
            #     linewidth=4.0,
            #     markersize=16.0,
            # )

            self.ax.plot(
                trajectory[0, : ii + 1],
                trajectory[1, : ii + 1],
                # linestyle=linestyle_list[tt],
                alpha=0.8,
                # color=self.color_obstacle,
                # color=trajectory_colors[tt],
                color=self.trajectory_color,
                linewidth=4.0,
                # label=legend_label,
                label=legend_label,
                zorder=0,
            )
            self.ax.plot(
                trajectory[0, ii],
                trajectory[1, ii],
                "o",
                # color=self.color_obstacle,
                color=self.trajectory_color,
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

        self.ax.scatter(
            self.avoider.initial_dynamics.attractor_position[0],
            self.avoider.initial_dynamics.attractor_position[1],
            marker="*",
            s=200,
            color="white",
            zorder=5,
        )

        self.ax.legend(fontsize="20", loc="lower right")

        plot_vectorfield = True
        if plot_vectorfield:
            plot_obstacle_dynamics(
                obstacle_container=self.environment,
                # collision_check_functor=lambda x: (
                #     self.environment.get_minimum_gamma(x) <= 1
                # ),
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
            axes_length=np.array([2, 10]),
            pose=Pose(position=np.array([2.0, 0])),
            distance_scaling=0.3,
        )
    )

    obstacle_environment.append(
        Cuboid(
            axes_length=np.array([2, 10]),
            pose=Pose(position=np.array([6.0, 10])),
            distance_scaling=0.3,
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


def create_environment_two_corners():
    obstacle_environment = ObstacleContainer()
    margin_absolut = 0.15

    center = np.array([-0.2, -0.9])
    axes_length = np.array([1.3, 0.3])
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

    center = np.array([0.2, 0.9])
    axes_length = np.array([1.3, 0.3])
    delta_ref = 0.5 * (axes_length[0] - axes_length[1])
    obstacle_environment.append(
        Cuboid(
            axes_length=axes_length,
            center_position=center + np.array([delta_ref, 0]),
            orientation=180 * np.pi / 180.0,
            margin_absolut=margin_absolut,
            relative_reference_point=np.array([delta_ref, 0]),
            tail_effect=False,
        )
    )

    obstacle_environment.append(
        Cuboid(
            axes_length=axes_length,
            center_position=center + np.array([0, delta_ref]),
            orientation=-90 * np.pi / 180.0,
            margin_absolut=margin_absolut,
            relative_reference_point=np.array([delta_ref, 0]),
            tail_effect=False,
        )
    )

    return obstacle_environment


@dataclass
class Disturbance:
    timestep: int
    force: np.ndarray


def animation_velocity_noise(save_animation=False):
    delta_time = 0.04
    dimension = 2

    x_lim = [-2.5, 10.5]
    y_lim = [-2.0, 12]

    initial_dynamics = LinearSystem(
        attractor_position=np.array([9, 9]),
        maximum_velocity=5.0,
        distance_decrease=1.0,
    )

    environment = create_environment()
    # environment = []

    mass = 1.0

    lambda_obs = 2.0 * mass / delta_time
    lambda_DS = 0.5 * lambda_obs
    lambda_perp = 0.1 * lambda_obs

    controller_obstacle = ObstacleAwarePassivController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        lambda_obstacle=lambda_obs,
        dimension=dimension,
        environment=environment,
    )

    controller_ds = PassiveDynamicsController(
        lambda_dynamics=lambda_DS, lambda_remaining=lambda_perp, dimension=2
    )

    avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=environment,
    )

    velocity_noise = 1.5

    disturbances = []

    animator = AnimationVelocityNoise(
        dt_simulation=delta_time,
        dt_sleep=0.01,
        it_max=200,
        animation_name="velocity_disturbance_obstacle",
        file_type=".gif",
    )
    animator.setup(
        environment=environment,
        # controller=controller_ds,
        controller=controller_obstacle,
        avoider=avoider,
        disturbances=disturbances,
        trajectory_color="#005fa8ff",
        trajectory_label="Obstacle aware",
        x_lim=x_lim,
        y_lim=y_lim,
        velocity_noise=velocity_noise,
    )

    animator.run(save_animation=save_animation)

    animator = AnimationVelocityNoise(
        dt_simulation=delta_time,
        dt_sleep=0.01,
        it_max=200,
        animation_name="velocity_disturbance_ds",
        file_type=".gif",
    )
    animator.setup(
        environment=environment,
        # controller=controller_ds,
        controller=controller_ds,
        avoider=avoider,
        disturbances=disturbances,
        trajectory_color="#bd0500ff",
        trajectory_label="Dynamics preserving",
        x_lim=x_lim,
        y_lim=y_lim,
        velocity_noise=velocity_noise,
    )
    animator.run(save_animation=save_animation)


def animation_position_noise(save_animation=False):
    delta_time = 0.03
    dimension = 2

    start_position = np.array([-1.0, -2.0])
    attractor_position = np.array([1.0, 2.5])

    initial_dynamics = LinearSystem(
        attractor_position=attractor_position,
        maximum_velocity=1.0,
        distance_decrease=1.0,
    )

    environment = create_environment_two_corners()
    # environment = []

    mass = 1.0

    lambda_obs = 2.0 * mass / delta_time
    lambda_DS = 0.6 * lambda_obs
    lambda_perp = 0.1 * lambda_obs

    controller_obstacle = ObstacleAwarePassivController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        lambda_obstacle=lambda_obs,
        dimension=dimension,
        environment=environment,
    )

    controller_ds = PassiveDynamicsController(
        lambda_dynamics=lambda_DS, lambda_remaining=lambda_perp, dimension=2
    )

    avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=environment,
    )

    position_noise = 0.01

    # x_lim = [-2.5, 2.5]
    # y_lim = [-2.5, 2.5]

    x_lim = [-2, 2]
    y_lim = [-3.0, 3.0]

    it_max = 300

    disturbances = []

    animator = AnimationVelocityNoise(
        dt_simulation=delta_time,
        dt_sleep=0.01,
        it_max=it_max,
        animation_name="position_noise_obstacle",
        file_type=".gif",
    )
    animator.setup(
        environment=environment,
        # controller=controller_ds,
        controller=controller_obstacle,
        avoider=avoider,
        disturbances=disturbances,
        trajectory_color="#005fa8ff",
        trajectory_label="Obstacle aware",
        x_lim=x_lim,
        y_lim=y_lim,
        start_position=start_position,
        position_noise=position_noise,
    )

    animator.run(save_animation=save_animation)

    animator = AnimationVelocityNoise(
        dt_simulation=delta_time,
        dt_sleep=0.01,
        it_max=it_max,
        animation_name="position_noise_ds",
        file_type=".gif",
    )
    animator.setup(
        environment=environment,
        # controller=controller_ds,
        controller=controller_ds,
        avoider=avoider,
        disturbances=disturbances,
        trajectory_color="#bd0500ff",
        trajectory_label="Dynamics preserving",
        x_lim=x_lim,
        y_lim=y_lim,
        start_position=start_position,
        position_noise=position_noise,
    )
    animator.run(save_animation=save_animation)


def animation_velocity_measurement_noise(save_animation=False):
    delta_time = 0.04
    dimension = 2

    initial_dynamics = LinearSystem(
        attractor_position=np.array([9, 9]),
        maximum_velocity=5.0,
        distance_decrease=1.0,
    )

    environment = create_environment()
    # environment = []

    mass = 1.0

    lambda_obs = 2.0 * mass / delta_time
    lambda_DS = 0.5 * lambda_obs
    lambda_perp = 0.1 * lambda_obs

    controller_obstacle = ObstacleAwarePassivController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        lambda_obstacle=lambda_obs,
        dimension=dimension,
        environment=environment,
    )

    controller_ds = PassiveDynamicsController(
        lambda_dynamics=lambda_DS, lambda_remaining=lambda_perp, dimension=2
    )

    avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=environment,
    )

    velocity_noise = 2.0

    disturbances = []

    animator = AnimationVelocityNoise(
        dt_simulation=delta_time,
        dt_sleep=0.01,
        it_max=200,
        animation_name="velocity_measurement_noise_obstacle",
        file_type=".gif",
    )
    animator.setup(
        environment=environment,
        # controller=controller_ds,
        controller=controller_obstacle,
        avoider=avoider,
        disturbances=disturbances,
        trajectory_color="#005fa8ff",
        trajectory_label="Obstacle aware",
        x_lim=[-2.5, 10.5],
        y_lim=[-1.0, 11],
        velocity_measurement_noise=velocity_noise,
    )

    animator.run(save_animation=save_animation)

    animator = AnimationVelocityNoise(
        dt_simulation=delta_time,
        dt_sleep=0.01,
        it_max=200,
        animation_name="velocity_measurement_noise_ds",
        file_type=".gif",
    )
    animator.setup(
        environment=environment,
        # controller=controller_ds,
        controller=controller_ds,
        avoider=avoider,
        disturbances=disturbances,
        trajectory_color="#bd0500ff",
        trajectory_label="Dynamics preserving",
        x_lim=[-2.5, 10.5],
        y_lim=[-1.0, 11],
        velocity_measurement_noise=velocity_noise,
    )
    animator.run(save_animation=save_animation)


if (__name__) == "__main__":
    np.random.seed(0)
    # def main():

    plt.style.use("dark_background")
    # animation_velocity_noise(save_animation=False)
    animation_position_noise(save_animation=False)
    # animation_velocity_measurement_noise(save_animation=True)
