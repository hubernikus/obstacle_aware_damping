import math
import numpy as np
import matplotlib.pyplot as plt

from vartools.states import Pose
from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)

# This repository is might still get restructuring
from nonlinear_avoidance.avoidance import RotationalAvoider
from nonlinear_avoidance.rotation_container import RotationContainer


def two_obstacles_gamma(save_figure=False, n_resolution=10):
    x_lim = [-0.1, 7]
    y_lim = [-0.1, 4]
    # cmap_list = ["Blues_r", "Purples_r"]
    # obstacle_colors = ["#6666C1", "#905190"]

    cmap_list = ["Greens_r", "Purples_r"]
    obstacle_colors = ["#467446", "#724C72"]
    contourf_colors = ["#233A23", "#392639"]

    container = RotationContainer()
    container.append(
        Cuboid(
            pose=Pose(position=np.array([5.2, 2.2]), orientation=-10 / 180.0 * math.pi),
            axes_length=np.array([1.0, 2]),
        )
    )
    container.append(
        Ellipse(
            pose=Pose(position=np.array([2.0, 0.8])),
            axes_length=np.array([2, 1.0]),
        )
    )
    initial_dynamics = LinearSystem(
        attractor_position=np.array([6.5, 0.5]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    obstacle_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=container,
        convergence_system=initial_dynamics,
    )

    strings = [r"$\Gamma_1$", r"$\Gamma_2$"]
    fig, ax = plt.subplots(figsize=(5, 4))
    for oo, (obs, cmap, color) in enumerate(zip(container, cmap_list, obstacle_colors)):
        # gammas = np.zeros((n_grid, n_grid))

        dx = x_lim[1] - x_lim[0]
        dy = y_lim[1] - y_lim[0]
        pos_x_lim = [x_lim[0] - 0.5 * dx, x_lim[1] + 0.5 * dx]
        pos_y_lim = [y_lim[0] - 0.5 * dy, y_lim[1] + 0.5 * dy]

        pos_x_lim = x_lim
        pos_y_lim = y_lim

        nx = ny = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(pos_x_lim[0], pos_x_lim[1], nx),
            np.linspace(pos_y_lim[0], pos_y_lim[1], ny),
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        gammas = np.zeros((positions.shape[1]))

        for ii in range(positions.shape[1]):
            gammas[ii] = obs.get_gamma(positions[:, ii], in_global_frame=True)

        cont = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas.reshape(nx, ny),
            levels=np.arange(0.0, 8, 1.0),
            zorder=-2,
            cmap=cmap,
            alpha=0.5,
        )

        cont = ax.contour(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas.reshape(nx, ny),
            levels=np.arange(2.0, 9, 1.0),
            zorder=-2,
            cmap=cmap,
            alpha=1.0,
            label="Gamma",
        )

        def fmt(x):
            fmt_string = strings[oo] + f"= {x:.0f}"
            return fmt_string

        # ax.clabel(cont, inline=True, fontsize=12, colors="black")
        ax.clabel(
            cont,
            inline=True,
            fontsize=12,
            fmt=fmt,
            colors=contourf_colors[oo],
            zorder=0,
        )
        # ax.clabel(cont, fontsize=10)

        # circ_rad = (3 / gammas) ** 3
        # ax.scatter(positions[0, :], positions[1, :], s=circ_rad)

        plot_obstacles(
            [obs],
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            alpha_obstacle=1.0,
            obstacle_color=color,
        )

    # Streamplot
    n_grid = n_resolution
    nx = ny = n_grid
    x_vals, y_vals = np.meshgrid(
        np.linspace(pos_x_lim[0], pos_x_lim[1], nx),
        np.linspace(pos_y_lim[0], pos_y_lim[1], ny),
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities = np.zeros_like(positions)
    for pp in range(positions.shape[1]):
        if not container.is_collision_free(positions[:, pp]):
            continue
        velocities[:, pp] = obstacle_avoider.evaluate(positions[:, pp])

    # Plot dynamics
    color_dynamcis = "#5a5a5a"
    ax.streamplot(
        positions[0, :].reshape(n_grid, n_grid),
        positions[1, :].reshape(n_grid, n_grid),
        velocities[0, :].reshape(n_grid, n_grid),
        velocities[1, :].reshape(n_grid, n_grid),
        # color="black",
        color=color_dynamcis,
        # color="#414141",
        density=0.5,
        # color="red",
        # scale=50,
        zorder=-2,
    )

    ax.scatter(
        initial_dynamics.attractor_position[0],
        initial_dynamics.attractor_position[1],
        marker="*",
        s=200,
        # color="black",
        color=color_dynamcis,
        zorder=5,
    )

    show_ticks = False
    if not show_ticks:
        ax.tick_params(
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

    # Agent position
    position = np.array([2.5, 3.0])
    ax.scatter(
        position[0],
        position[1],
        marker=".",
        s=300,
        color="black",
        zorder=5,
    )

    arr_scale = 0.5
    normals = []
    gammas = []
    for oo, obs in enumerate(container):
        normals.append(obs.get_normal_direction(position, in_global_frame=True))
        gammas.append(obs.get_gamma(position, in_global_frame=True))
        ax.arrow(
            position[0],
            position[1],
            arr_scale * normals[-1][0],
            arr_scale * normals[-1][1],
            color=obstacle_colors[oo],
            # color=contourf_colors[oo],
            width=0.06,
            alpha=1.0,
        )

    # sum normals
    weights = 1 / np.array(gammas)
    weights = weights / np.sum(weights)
    normals = np.array(normals).T
    mean_normal = np.sum(np.tile(weights, (normals.shape[0], 1)) * normals, axis=1)

    ax.arrow(
        position[0],
        position[1],
        arr_scale * mean_normal[0],
        arr_scale * mean_normal[1],
        # color="red",
        color="#B20000",
        width=0.06,
        alpha=1.0,
        label="Averaged normal",
    )

    # ax.legend(loc="upper right")

    if save_figure:
        figname = "normal_and_gamma_field_visualization"
        plt.savefig("figures/" + figname + filetype, bbox_inches="tight")


if (__name__) == "__main__":
    plt.ion()
    plt.close("all")
    filetype = ".pdf"

    two_obstacles_gamma(save_figure=True, n_resolution=100)

    print("Done.")
