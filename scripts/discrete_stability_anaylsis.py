import numpy as np

import matplotlib.pyplot as plt


from vartools.dynamical_systems import ConstantValue

from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)


from passive_control.agent import Agent
from passive_control.controller import Controller
from passive_control.controller import PassiveDynamicsController
from passive_control.controller import ObstacleAwarePassivController


def evaluate_ideal_eigenvalue():
    delta_time = 0.1
    AA = np.eye(2)
    AA[0, 1] = 10
    AA[1, 1] = 100

    print("Transition matrix A")
    print(AA)

    result = np.linalg.eig(AA)
    breakpoint()

    U, S, Vh = np.linalg.svd(AA)
    print("Singular values S:", S)
    print("Eigenvectors U")
    print(U)

    print("Test values")
    print(np.linalg.det(AA - np.eye(2) * S[0]))
    print(np.linalg.det(AA - np.eye(2) * S[1]))


def run_controller_evaluate_positions(
    controller,
    dynamics,
    delta_time,
    start_position=np.array([-3, 0]),
    start_delta_velocity=np.array([0, -1.0]),
    it_max=100,
):
    dimension = start_position.shape[0]

    start_velocity = start_delta_velocity + dynamics.evaluate(start_position)

    agent = Agent(position=start_position, velocity=start_velocity)

    positions = np.zeros((dimension, it_max + 1))
    positions[:, 0] = agent.position
    for tt in range(it_max):
        if not (tt + 1) % 50:
            print(f"Step {tt+1} / {it_max}")

        velocity = dynamics.evaluate(agent.position)
        force = controller.compute_force(
            position=agent.position,
            velocity=agent.velocity,
            desired_velocity=velocity,
        )

        agent.update_step(delta_time, control_force=force)
        positions[:, tt + 1] = agent.position

    return positions


def evaluate_discrete_controller_with_different_eigenvalues(
    visualize=False,
    save_figure=False,
):
    dimension = 2

    delta_time = 0.4
    frequency = 1.0 / delta_time

    base_dynamics = np.array([1.0, 0.0])
    dynamics = ConstantValue(base_dynamics)

    lambda_fractions = [0.5, 2.0, 2.1, 2.5]

    x_lim = [-3, 5]
    y_lim = [-1.0, 1.0]

    # fig, ax = plt.subplots(figsize=(4.5, 2.0))
    # fig, ax = plt.subplots(figsize=(6.0, 4.0))
    fig, axs = plt.subplots(len(lambda_fractions), 1, figsize=(5.0, 6.0))

    for ii, lambda_fraction in enumerate(lambda_fractions):
        lambda_max = frequency * lambda_fraction

        lambda_DS = 0.9 * lambda_max
        lambda_perp = 1.0 * lambda_max
        lambda_obs = 1.0 * lambda_max

        controller = PassiveDynamicsController(
            lambda_dynamics=lambda_DS, lambda_remaining=lambda_perp, dimension=dimension
        )

        positions = run_controller_evaluate_positions(controller, dynamics, delta_time)

        ax = axs[ii]
        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=dynamics.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=10,
            ax=ax,
            # attractor_position=initial_dynamics.attractor_position,
            do_quiver=True,
            show_ticks=True,
            vectorfield_color="#7a7a7a7f",
        )

        ax.plot(
            positions[0, :],
            positions[1, :],
            marker=".",
            label=f"{lambda_fraction:.1f} f",
            linewidth=2.0,
            markersize=8.0,
        )

        if ii < len(lambda_fractions) - 1:
            ax.tick_params(
                which="both",
                bottom=False,
                top=False,
                left=True,
                right=False,
                labelbottom=False,
                labelleft=True,
            )
        else:
            ax.set_xlabel(r"$\xi_1$ [m]")

        ax.set_ylabel(r"$\xi_2$ [m]")
        ax.legend(loc="upper right")


if (__name__) == "__main__":
    figtype = ".pdf"

    plt.close("all")
    plt.ion()

    evaluate_discrete_controller_with_different_eigenvalues(visualize=True)
