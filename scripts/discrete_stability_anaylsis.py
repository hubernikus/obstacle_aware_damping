import numpy as np

import matplotlib.pyplot as plt


from vartools.dynamics import ConstantValue
from vartools.dynamics import LinearSystem

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
    start_position=np.array([0, 0]),
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

    start_delta_velocity = np.array([0.0, -1.0])
    base_dynamics = np.array([1.0, 0.0])
    dynamics = ConstantValue(base_dynamics)

    lambda_fractions = [0.5, 2.0, 2.1, 2.5]

    x_lim = [-1, 7]
    y_lim = [-1.0, 1.0]

    from matplotlib.cm import get_cmap

    # fig, ax = plt.subplots(figsize=(4.5, 2.0))
    # fig, ax = plt.subplots(figsize=(6.0, 4.0))
    # fig, axs = plt.subplots(len(lambda_fractions), 1, figsize=(5.0, 6.0))
    fig, axs = plt.subplots(len(lambda_fractions), 1, figsize=(4.5, 5.2))

    step_dx = delta_time * np.linalg.norm(base_dynamics)
    x_values = np.arange(x_lim[0], 0, step_dx)
    x_values = x_values - x_values[-1] - step_dx
    undisturbed_positions = np.vstack((x_values, np.zeros(x_values.shape)))

    for ii, lambda_fraction in enumerate(lambda_fractions):
        lambda_max = frequency * lambda_fraction

        lambda_DS = 0.9 * lambda_max
        lambda_perp = 1.0 * lambda_max
        lambda_obs = 1.0 * lambda_max

        controller = PassiveDynamicsController(
            lambda_dynamics=lambda_DS, lambda_remaining=lambda_perp, dimension=dimension
        )

        positions = run_controller_evaluate_positions(
            controller, dynamics, delta_time, start_delta_velocity=start_delta_velocity
        )

        # breakpoint()
        # Add start positions
        positions = np.hstack((undisturbed_positions, positions))

        ax = axs[ii]

        start_velocity = start_delta_velocity + base_dynamics
        start_position = [0, 0]

        plot_final_velocity = False
        if plot_final_velocity:
            ax.arrow(
                start_position[0],
                start_position[1],
                start_velocity[0] * 0.5,
                start_velocity[1] * 0.5,
                color="blue",
                # color="#740782ff",
                width=0.06,
            )

        ax.arrow(
            start_position[0],
            start_position[1],
            0,
            start_velocity[1] * 0.5,
            # color="blue",
            color="#740782ff",
            width=0.06,
        )

        ax.plot(start_position[0], start_position[1], "o", color="black", zorder=3)

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
            label=r"$s_i = $" + f"{lambda_fraction:.1f} " + r"$m / \Delta t$",
            linewidth=2.0,
            markersize=8.0,
            color=colors[ii],
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

    if save_figure:
        figname = "discrete_controller_parameters_comparison"
        plt.savefig("figures/" + figname + figtype, bbox_inches="tight")


def evaluate_discrete_controller_with_different_eigenvalues_stable(
    visualize=False,
    save_figure=False,
):
    dimension = 2

    delta_time = 0.2
    frequency = 1.0 / delta_time

    start_delta_velocity = np.array([0.0, 0.0])
    attractor_position = np.array([3.0, 0])
    # base_dynamics = np.array([1.0, 0.0])
    # dynamics = ConstantValue(base_dynamics)
    rot_angle = 30.0
    cos_ = np.cos(rot_angle * np.pi / 180)
    sin_ = np.sin(rot_angle * np.pi / 180)
    # A_matrix = np.array([[-1, 0.1], [-0.1, -1]])
    A_matrix = -np.array([[cos_, sin_], [-sin_, cos_]])

    dynamics = LinearSystem(
        attractor_position=attractor_position, A_matrix=A_matrix, maximum_velocity=5.0
    )
    start_position = np.array([0, 0.0])

    lambda_fractions = [0.5, 2.0, 2.1, 2.3]

    x_lim = [0, 4]
    y_lim = [-2.0, 2.0]

    from matplotlib.cm import get_cmap

    # fig, ax = plt.subplots(figsize=(4.5, 2.0))
    # fig, ax = plt.subplots(figsize=(6.0, 4.0))
    # fig, axs = plt.subplots(len(lambda_fractions), 1, figsize=(5.0, 6.0))
    n_row = 2
    n_col = 2
    fig, axs = plt.subplots(n_row, n_col, figsize=(4.5, 4.5))

    for ii, lambda_fraction in enumerate(lambda_fractions):
        lambda_max = frequency * lambda_fraction

        lambda_DS = 0.9 * lambda_max
        lambda_perp = 1.0 * lambda_max
        lambda_obs = 1.0 * lambda_max

        controller = PassiveDynamicsController(
            lambda_dynamics=lambda_DS, lambda_remaining=lambda_perp, dimension=dimension
        )

        positions = run_controller_evaluate_positions(
            controller, dynamics, delta_time, start_delta_velocity=start_delta_velocity
        )

        it_row = ii % n_row
        it_col = ii // n_col
        ax = axs[it_col, it_row]

        start_velocity = start_delta_velocity + dynamics.evaluate(start_position)
        start_position = [0, 0]
        # ax.arrow(
        #     start_position[0],
        #     start_position[1],
        #     start_velocity[0] * 0.5,
        #     start_velocity[1] * 0.5,
        #     color="blue",
        #     width=0.06,
        # )
        ax.plot(start_position[0], start_position[1], "o", color="black", zorder=3)
        ax.plot(
            attractor_position[0],
            attractor_position[1],
            "*",
            color="black",
            # markersize=1.0,
            zorder=4,
        )
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
            label=r"$s_i = $" + f"{lambda_fraction:.1f} " + r"$m / \Delta t$",
            linewidth=2.0,
            markersize=8.0,
            color=colors[ii],
        )

        if it_col + 1 < n_row:
            ax.set_xticks([])
        else:
            ax.set_xlabel(r"$\xi_1$ [m]")

        if it_row > 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel(r"$\xi_2$ [m]")

        ax.legend(loc="upper right")

    if save_figure:
        figname = "discrete_controller_parameters_comparison_stable"
        plt.savefig("figures/" + figname + figtype, bbox_inches="tight")


if (__name__) == "__main__":
    figtype = ".pdf"

    plt.close("all")
    plt.ion()

    colors = ["#DB7660", "#DB608F", "#47A88D", "#638030"]
    # colors = ["#B07146", "#963C87", "#47A88D", "#638030"

    # evaluate_discrete_controller_with_different_eigenvalues(
    #     visualize=True, save_figure=True
    # )

    evaluate_discrete_controller_with_different_eigenvalues_stable(
        visualize=True, save_figure=False
    )
