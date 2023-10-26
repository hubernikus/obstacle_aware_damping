from typing import Callable
from collections import namedtuple

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

        force = controller.compute_force(
            velocity=agent.velocity,
            desired_velocity=dynamics(agent.position),
            position=agent.position,
        )

        for disturbance in disturbance_list:
            if disturbance.step == ii:
                force += disturbance.direction
                # print(f"Disturbance at {ii}")
                # print("position: ", agent.position)
                # print("velocity: ", agent.velocity)
                break

        agent.update_step(delta_time, control_force=force)

        position_list[:, ii + 1] = agent.position
        velocity_list[:, ii + 1] = agent.velocity

    return position_list, velocity_list


def create_three_body_container():
    obstacle_environment = ObstacleContainer()
    # SETUP A
    obstacle_environment.append(
        Cuboid(
            axes_length=[0.6, 0.6],
            center_position=np.array([0.0, 0.0]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.15,
            # orientation=10 * pi / 180,
            # linear_velocity = np.array([0.0, 1.0]),
            tail_effect=False,
            # repulsion_coeff=1.4,
        )
    )
    obstacle_environment.append(
        Cuboid(
            axes_length=[0.5, 0.5],
            center_position=np.array([0.0, 1.5]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.15,
            # orientation=10 * pi / 180,
            # linear_velocity = np.array([0.0, 0.5]),
            tail_effect=False,
            # repulsion_coeff=1.4,
        )
    )
    obstacle_environment.append(
        Cuboid(
            axes_length=[0.3, 0.3],
            center_position=np.array([1.0, 0.2]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.15,
            # orientation=10 * pi / 180,
            # linear_velocity = np.array([0.0, 0.5]),
            tail_effect=False,
            # repulsion_coeff=1.4,
        )
    )

    return obstacle_environment


def plot_disturbances(
    disturbances: list[Disturbance],
    trajectory_positions: np.ndarray,
    ax,
    arrow_scaling=5e-4,
    arrow_width=0.08,
    do_label=False,
):
    zorder_arrow = 2
    for ii, disturbance in enumerate(disturbances):
        position = trajectory_positions[:, disturbance.step]
        direction = disturbance.direction

        # small trick to only label one disturance
        if ii == 0:
            ax.arrow(
                position[0],
                position[1],
                direction[0] * arrow_scaling,
                direction[1] * arrow_scaling,
                width=arrow_width,
                # color="r",
                # color="#C73200",
                # color="#767820",
                # color="#168741",
                color="#740782",
                label=("Disturbances" if do_label else None),
                zorder=zorder_arrow,
            )
        else:
            ax.arrow(
                position[0],
                position[1],
                direction[0] * arrow_scaling,
                direction[1] * arrow_scaling,
                width=arrow_width,
                color="r",
                zorder=zorder_arrow,
            )


def analysis_point():
    lambda_DS = 100.0
    lambda_perp = 20.0
    lambda_obs = mn.LAMBDA_MAX

    dimension = 2
    delta_time = 0.01

    x_lim = [-2.5, 3]
    y_lim = [-1.5, 3.0]

    position_init = np.array([-0.78891176, -0.74863131])
    velocity_init = np.array([1.49812948, -0.27046507])

    obstacle_environment = create_three_body_container()

    initial_dynamics = LinearSystem(
        attractor_position=np.array([2.5, 0.5]),
        maximum_velocity=2,
        distance_decrease=0.5,
    )

    avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_environment,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_obstacles(
        ax=ax,
        obstacle_container=obstacle_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        showLabel=False,
    )

    atractor = initial_dynamics.attractor_position
    ax.plot(atractor[0], atractor[1], "k*", markersize=12)

    ax.plot()

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    disturbance = Disturbance(100, 700 * np.array([1.0, -1.0]))
    # disturbance = Disturbance(100, 700 * np.array([1.0, 1.0]))

    agent = Agent(position=position_init, velocity=velocity_init)
    controller = ObstacleAwarePassivController(
        environment=obstacle_environment,
        lambda_dynamics=100,
        lambda_obstacle=200,
        lambda_remaining=lambda_perp,
    )
    ax.plot(agent.position[0], agent.position[1], "o", color="black")
    arrowidth = 0.1
    ax.arrow(
        agent.position[0],
        agent.position[1],
        agent.velocity[0],
        agent.velocity[1],
        color="green",
        label="Initial",
        width=arrowidth,
    )
    ax.arrow(
        agent.position[0],
        agent.position[1],
        disturbance.direction[0],
        disturbance.direction[1],
        color="red",
        label="Disturbance",
        width=arrowidth,
    )

    # Single step
    print("initial", agent.velocity)
    delta_time = 0.01

    # Disturbance only
    velocity = avoider.evaluate(agent.position)
    force = disturbance.direction
    agent.update_step(delta_time, control_force=force)

    # Activate the controller (!)
    velocity = avoider.evaluate(agent.position)
    force = controller.compute_control_force(agent, desired_velocity=velocity)
    agent.update_step(delta_time, control_force=force)

    ax.plot(agent.position[0], agent.position[1], "o", color="black")
    ax.arrow(
        agent.position[0],
        agent.position[1],
        agent.velocity[0],
        agent.velocity[1],
        color="blue",
        label="Final",
        width=arrowidth,
    )
    # print("final", agent.velocity)

    ax.legend()
    ax.set_aspect("equal")
    breakpoint()


def plot_passivity_comparison(
    ax,
    initial_dynamics,
    obstacle_environment,
    x_init=np.array([-2.2, 0.5]),
    xdot_init=np.array([0.0, 0.0]),
    delta_time=0.1,
    it_max=200,
    disturbances=[],
    do_label=False,
):
    dimension = 2

    trajectory_kwargs = {"linewidth": 3.0, "zorder": 3}

    dynamic_avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_environment,
    )

    # must not be > 200 (num error, patch dt smaller) -> 200 makes xdot varies too much
    lambda_DS = 100.0
    lambda_perp = 20.0
    lambda_obs = mn.LAMBDA_MAX

    print("lambda_DS", lambda_DS)
    print("lambda_perp", lambda_perp)
    print("lambda_obs", lambda_obs)

    print("delta_time", delta_time)
    # disturbances = [Disturbance(80, 700 * np.array([1.0, -1.0]))]
    # disturbances = []

    do_undisturbed = True
    if do_undisturbed:
        #### Undisturbed Controller ###
        agent = Agent(position=x_init, velocity=xdot_init)
        controller = PassiveDynamicsController(
            lambda_dynamics=lambda_DS, lambda_remaining=lambda_perp
        )
        positions, velocities = integrate_agent_trajectory(
            agent=agent,
            dynamics=dynamic_avoider.evaluate,
            controller=controller,
            it_max=it_max,
            delta_time=delta_time,
        )
        ax.plot(
            positions[0, :],
            positions[1, :],
            "--",
            color="gray",
            label=("Undisturbed" if do_label else None),
            **trajectory_kwargs,
        )

    ### Passive DS - disturbed ###
    do_passive = True
    if do_passive:
        agent = Agent(position=x_init, velocity=xdot_init)
        controller = PassiveDynamicsController(
            lambda_dynamics=lambda_DS, lambda_remaining=lambda_perp
        )
        positions, velocities = integrate_agent_trajectory(
            agent=agent,
            dynamics=dynamic_avoider.evaluate,
            controller=controller,
            disturbance_list=disturbances,
            it_max=it_max,
            delta_time=delta_time,
        )
        ax.plot(
            positions[0, :],
            positions[1, :],
            "--",
            # color="#135e08",
            color=plot_setup["dynamics"].color,
            label=(plot_setup["dynamics"].label if do_label else None),
            **trajectory_kwargs,
        )

    do_obstacle_aware = True
    if do_obstacle_aware:
        ### Obstacle Aware Dynamics ###
        agent = Agent(position=x_init, velocity=xdot_init)
        controller = ObstacleAwarePassivController(
            environment=obstacle_environment,
            lambda_dynamics=lambda_DS,
            lambda_obstacle=lambda_obs,
            lambda_remaining=lambda_perp,
        )
        positions, velocities = integrate_agent_trajectory(
            agent=agent,
            dynamics=dynamic_avoider.evaluate,
            controller=controller,
            disturbance_list=disturbances,
            it_max=it_max,
            delta_time=delta_time,
        )

        ax.plot(
            positions[0, :],
            positions[1, :],
            "--",
            # color="blue",
            # color="#1132F0",
            # label=("Obstacle aware" if do_label else None),
            color=plot_setup["obstacle"].color,
            label=(plot_setup["obstacle"].label if do_label else None),
            **trajectory_kwargs,
        )

    # Disturbance and start position
    plot_qolo(positions[:, 0], velocities[:, 0], ax=ax)
    plot_disturbances(disturbances, positions, ax=ax, do_label=do_label)
    plot_qolo(
        positions[:, disturbances[0].step],
        velocities[:, disturbances[0].step],
        ax=ax,
    )


SimpleState = namedtuple("SimpleState", "position velocity")


def plot_multiple_avoider(save_figure=False):
    x_lim = [-2.5, 3]
    y_lim = [-1.3, 2.1]

    obstacle_environment = create_three_body_container()

    initial_dynamics = LinearSystem(
        attractor_position=np.array([2.5, 0.5]),
        maximum_velocity=2,
        distance_decrease=0.5,
    )

    avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_environment,
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    states = [
        SimpleState(position=[-2.2, 0.5], velocity=[0.0, 0.0]),
        SimpleState(position=[-2.2, 1.8], velocity=[0.0, 0.0]),
        SimpleState(position=[-2.2, -0.8], velocity=[0.0, 0.0]),
    ]

    disturbances = [
        [Disturbance(100, 700 * np.array([1.0, -1.0]))],
        [Disturbance(100, 700 * np.array([1.0, 0.5]))],
        [Disturbance(100, 700 * np.array([1.0, -1.0]))],
    ]

    # Do label once only (!)
    do_label = True
    for ii, state in enumerate(states):
        plot_passivity_comparison(
            ax,
            initial_dynamics,
            obstacle_environment,
            x_init=np.array(state.position),
            xdot_init=np.array(state.velocity),
            disturbances=disturbances[ii],
            it_max=400,
            do_label=do_label,
        )
        do_label = False

    plot_obstacles(
        ax=ax,
        obstacle_container=obstacle_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        showLabel=False,
    )

    do_quiver = True
    if do_quiver:
        plot_obstacle_dynamics(
            obstacle_container=obstacle_environment,
            dynamics=avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=60,
            ax=ax,
            attractor_position=initial_dynamics.attractor_position,
            do_quiver=False,
            show_ticks=False,
            vectorfield_color="#7a7a7a7f",
        )

    attractor = initial_dynamics.attractor_position
    ax.plot(attractor[0], attractor[1], "k*", markersize=12)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    ax.set_aspect("equal")
    ax.legend()

    if save_figure:
        figname = "multi_obstacle_with_damping"
        fig.savefig("figures/" + figname + figtype, bbox_inches="tight", dpi=300)


if (__name__) == "__main__":
    figtype = ".pdf"

    plt.close("all")
    plt.ion()

    from scripts.plot_setup import plot_setup

    # plot_passivity_comparison()
    plot_multiple_avoider(save_figure=True)
    # analysis_point()
