from typing import Callable
from collections import namedtuple

from attrs import define, field

import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)

from passive_control.agent import Agent
from passive_control.controller import Controller
from passive_control.controller import PassiveDynamicsController
from passive_control.controller import ObstacleAwarePassivController

# from passive_control.controller import RegulationController, TrackingController
from passive_control.robot_visualizer import Disturbance
import passive_control.magic_numbers_and_enums as mn

# from passive_control.draw_obs_overwrite import plot_obstacles
from passive_control.visualization import plot_qolo


def integrate_agent_trajectory(
    agent: Agent,
    dynamics: Callable[[np.ndarray], np.ndarray],
    controller: Controller,
    it_max=200,
    delta_time=0.01,
    disturbance_list=[],
):
    dimension = 2

    # Assumption of static environment

    position_list = np.zeros((dimension, it_max + 1))
    position_list[:, 0] = agent.position
    velocity_list = np.zeros((dimension, it_max + 1))
    velocity_list[:, 0] = agent.velocity

    for ii in range(it_max):
        if not (ii + 1) % 10:
            print(f"Step {ii+1} / {it_max}")

        velocity = dynamics(agent.position)
        force = controller.compute_force(
            position=agent.position, velocity=agent.velocity, desired_velocity=velocity
        )

        # for disturbance in disturbance_list:
        #     if disturbance.step == ii:
        #         force += disturbance.direction
        #         break

        agent.update_step(delta_time, control_force=force)

        position_list[:, ii + 1] = agent.position
        velocity_list[:, ii + 1] = agent.velocity

    return position_list, velocity_list


def create_environment():
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Cuboid(
            axes_length=[0.6, 0.6],
            center_position=np.array([0.0, 0.0]),
            margin_absolut=0.15,
            tail_effect=False,
        )
    )

    return obstacle_environment


def create_environment_two_obstacles():
    obstacle_environment = ObstacleContainer()
    margin_absolut = 0.15
    obstacle_environment.append(
        Cuboid(
            axes_length=[0.6, 4.0],
            center_position=np.array([-1.0, 2.0]),
            margin_absolut=margin_absolut,
            tail_effect=False,
        )
    )

    obstacle_environment.append(
        Cuboid(
            axes_length=[0.5, 4.0],
            center_position=np.array([1.0, -2.0]),
            margin_absolut=margin_absolut,
            tail_effect=False,
        )
    )

    return obstacle_environment


def create_environment_two_corners():
    obstacle_environment = ObstacleContainer()
    margin_absolut = 0.15

    center = np.array([-0.9, -0.2])
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

    center = np.array([0.9, 0.2])
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


def compute_distances_and_positions(
    std_noise_ranges,
    controller,
    avoider,
    environment,
    n_epoch,
    start_position,
    it_max,
    delta_time,
    velocity_noise=True,
):
    dimension = start_position.shape[0]

    min_distances = np.zeros((len(std_noise_ranges), n_epoch))
    position_lists = []
    for it_noise, std_noise in enumerate(std_noise_ranges):
        # positions = []
        position_lists.append([])

        for it_epoch in range(n_epoch):
            agent = Agent(position=start_position, velocity=np.zeros(2))

            position_lists[-1].append(np.zeros((dimension, it_max + 1)))
            position_lists[-1][-1][:, 0] = agent.position

            distances = np.zeros(it_max)
            for tt in range(it_max):
                if not (tt + 1) % 50:
                    print(f"Step {tt+1} / {it_max}")

                velocity = avoider.evaluate(agent.position)
                force = controller.compute_force(
                    position=agent.position,
                    velocity=agent.velocity,
                    desired_velocity=velocity,
                )

                if velocity_noise:
                    # Add position velocity noise
                    agent.velocity += std_noise * np.random.randn(2)
                else:
                    agent.position += std_noise * np.random.randn(2)

                agent.update_step(delta_time, control_force=force)
                position_lists[-1][-1][:, tt + 1] = agent.position
                distances[tt] = get_minimum_obstacle_distance(
                    agent.position, environment=environment
                )

            min_distances[it_noise, it_epoch] = np.min(distances)
    #     return min_distances
    return min_distances, position_lists


def multi_epoch_noisy_position(
    n_epoch=10,
    std_noise_ranges=np.arange(11) * 0.05,
    it_max=100,
    delta_time=0.03,
    visualize=False,
    save_figure=False,
):
    lambda_max = 1.0 / delta_time

    lambda_DS = 0.8 * lambda_max
    lambda_perp = 0.1 * lambda_max
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

    environment = create_environment_two_corners()

    avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=environment,
    )

    min_distance_both = [None] * 2
    position_lists_both = [None] * 2
    controller = PassiveDynamicsController(
        lambda_dynamics=lambda_DS, lambda_remaining=lambda_perp, dimension=2
    )

    min_distance_both[0], position_lists_both[0] = compute_distances_and_positions(
        std_noise_ranges,
        controller,
        avoider,
        environment,
        n_epoch,
        start_position,
        it_max,
        delta_time,
        velocity_noise=False,
    )

    controller = ObstacleAwarePassivController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        lambda_obstacle=lambda_obs,
        dimension=dimension,
        environment=environment,
    )

    min_distance_both[1], position_lists_both[1] = compute_distances_and_positions(
        std_noise_ranges,
        controller,
        avoider,
        environment,
        n_epoch,
        start_position,
        it_max,
        delta_time,
        velocity_noise=False,
    )

    # def visualize_distances(min_distances, save_figure=False):
    x_lim = [-3, 3]
    y_lim = [-2.0, 2.0]

    kwargs_meanline = {"linestyle": "-", "alpha": 1.0}

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for positions_lists, key in zip(position_lists_both, ["dynamics", "obstacle"]):
        # Analyze non-noisy
        traj = positions_lists[0][0]
        ax.plot(
            traj[0, :],
            traj[1, :],
            "-",
            color=plot_setup[key].color,
            linewidth=2.0,
            label=plot_setup[key].label,
        )

    plot_obstacle_dynamics(
        obstacle_container=environment,
        dynamics=avoider.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=60,
        ax=ax,
        attractor_position=initial_dynamics.attractor_position,
        do_quiver=False,
        show_ticks=True,
        vectorfield_color="#7a7a7a7f",
    )

    plot_obstacles(
        ax=ax,
        obstacle_container=environment,
        x_lim=x_lim,
        y_lim=y_lim,
        showLabel=False,
        draw_reference=True,
    )
    plot_qolo(start_position, start_velocity, ax)
    ax.set_xlabel(r"$\xi_1$ [m]")
    ax.set_ylabel(r"$\xi_2$ [m]")
    ax.legend(loc="upper left")

    if save_figure:
        figname = "trajectory_position_noise"
        plt.savefig("figures/" + figname + figtype, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(4.5, 2.0))
    ax.set_xlim([std_noise_ranges[0], std_noise_ranges[-1]])

    for min_distances, key in zip(min_distance_both, ["dynamics", "obstacle"]):
        mean_values = np.mean(min_distances, axis=1)
        std_values = np.std(min_distances, axis=1)

        ax.fill_between(
            std_noise_ranges,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.3,
            color=plot_setup[key].color,
            zorder=0,
        )

        ax.plot(
            std_noise_ranges,
            mean_values,
            linewidth=2.0,
            color=plot_setup[key].color,
            label=plot_setup[key].label,
            **kwargs_meanline,
            zorder=0,
        )

    ax.plot(ax.get_xlim(), [0, 0], "--", color="black", linewidth=2.0)

    ax.set_xlabel("Standard deviation of position noise [m]")
    ax.set_ylabel("Closest distance [m]")
    ax.legend(loc="lower left")

    if save_figure:
        figname = "comparison_position_noise"
        plt.savefig("figures/" + figname + figtype, bbox_inches="tight")


def multi_epoch_noisy_velocity(
    n_epoch=10,
    std_noise_ranges=np.arange(11) * 0.05,
    it_max=100,
    delta_time=0.03,
    visualize=False,
    save_figure=False,
):
    lambda_max = 1.0 / delta_time

    lambda_DS = 0.8 * lambda_max
    lambda_perp = 0.1 * lambda_max
    lambda_obs = 1.0 * lambda_max

    dimension = 2

    start_position = np.array([-2.5, 1.0])
    attractor_position = np.array([2.5, -1.0])

    initial_dynamics = LinearSystem(
        attractor_position=attractor_position,
        maximum_velocity=1.0,
        distance_decrease=1.0,
    )
    start_velocity = initial_dynamics.evaluate(start_position)

    environment = create_environment_two_obstacles()

    avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=environment,
    )

    min_distance_both = [None] * 2
    position_lists_both = [None] * 2
    controller = PassiveDynamicsController(
        lambda_dynamics=lambda_DS, lambda_remaining=lambda_perp, dimension=2
    )

    min_distance_both[0], position_lists_both[0] = compute_distances_and_positions(
        std_noise_ranges,
        controller,
        avoider,
        environment,
        n_epoch,
        start_position,
        it_max,
        delta_time,
    )

    controller = ObstacleAwarePassivController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        lambda_obstacle=lambda_obs,
        dimension=dimension,
        environment=environment,
    )

    min_distance_both[1], position_lists_both[1] = compute_distances_and_positions(
        std_noise_ranges,
        controller,
        avoider,
        environment,
        n_epoch,
        start_position,
        it_max,
        delta_time,
    )

    # def visualize_distances(min_distances, save_figure=False):
    x_lim = [-3, 3]
    y_lim = [-2.0, 2.0]

    kwargs_meanline = {"linestyle": "-", "alpha": 1.0}

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for positions_lists, key in zip(position_lists_both, ["dynamics", "obstacle"]):
        traj = positions_lists[-1][-1]
        ax.plot(
            traj[0, :],
            traj[1, :],
            "-",
            color=plot_setup[key].color,
            linewidth=2.0,
            label=plot_setup[key].label,
        )

    plot_obstacle_dynamics(
        obstacle_container=environment,
        dynamics=avoider.evaluate,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=60,
        ax=ax,
        attractor_position=initial_dynamics.attractor_position,
        do_quiver=False,
        show_ticks=True,
        vectorfield_color="#7a7a7a7f",
    )

    plot_obstacles(
        ax=ax,
        obstacle_container=environment,
        x_lim=x_lim,
        y_lim=y_lim,
        showLabel=False,
    )
    plot_qolo(start_position, start_velocity, ax)
    ax.set_xlabel(r"$\xi_1$ [m]")
    ax.set_ylabel(r"$\xi_2$ [m]")
    ax.legend(loc="lower left")

    if save_figure:
        figname = "trajectory_velocity_noise"
        plt.savefig("figures/" + figname + figtype, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(4.5, 2.0))
    ax.set_xlim([std_noise_ranges[0], std_noise_ranges[-1]])

    for min_distances, key in zip(min_distance_both, ["dynamics", "obstacle"]):
        mean_values = np.mean(min_distances, axis=1)
        std_values = np.std(min_distances, axis=1)

        ax.fill_between(
            std_noise_ranges,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.3,
            color=plot_setup[key].color,
            zorder=0,
        )

        ax.plot(
            std_noise_ranges,
            mean_values,
            linewidth=2.0,
            color=plot_setup[key].color,
            label=plot_setup[key].label,
            **kwargs_meanline,
            zorder=0,
        )

    ax.plot(ax.get_xlim(), [0, 0], "--", color="black", linewidth=2.0)

    ax.set_xlabel("Standard deviation of velocity noise [m/s]")
    ax.set_ylabel("Closest distance [m]")
    ax.legend()

    if save_figure:
        figname = "comparison_velocity_noise"
        plt.savefig("figures/" + figname + figtype, bbox_inches="tight")


def get_minimum_obstacle_distance(
    position: np.ndarray, environment: ObstacleContainer
) -> float:
    # distances = np.zeros(position.shape[1])
    distances = np.zeros(len(environment))
    for ii, obs in enumerate(environment):
        distances[ii] = obs.get_distance_to_surface(position, in_obstacle_frame=False)
    return np.min(distances)


if (__name__) == "__main__":
    figtype = ".pdf"

    np.random.seed(0)  # Keep use seed to better track the evolution

    plt.close("all")
    plt.ion()

    from scripts.plot_setup import plot_setup

    # plot_multiple_avoider(save_figure=True)
    # analysis_point()
    # min_distances = multi_epoch_noisy_velocity(
    #     visualize=True,
    #     it_max=400,
    #     delta_time=0.02,
    #     n_epoch=10,
    #     std_noise_ranges=np.arange(11) * 0.1,
    #     # n_epoch=2,
    #     # std_noise_ranges=np.arange(3) * 0.1,
    #     save_figure=True,
    # )

    min_distances = multi_epoch_noisy_position(
        visualize=True,
        it_max=400,
        delta_time=0.02,
        n_epoch=10,
        std_noise_ranges=np.arange(11) * 3.0e-3,
        # n_epoch=2,
        # std_noise_ranges=np.arange(3) * 0.1,
        save_figure=True,
    )
