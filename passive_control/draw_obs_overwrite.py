#!/USSR/bin/python3
"""Obstacle Avoidance Algorithm script with vecotr field. """
# Author: Lukas Huber
# Date: 2018-02-15
# Email: lukas.huber@epfl.ch

import copy
import os
import warnings
from timeit import default_timer as timer

# Or use time.perf_counter()

import numpy as np
from numpy import pi

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
import matplotlib.image as mpimg

from scipy import ndimage

from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.avoidance import (
    obs_avoidance_interpolation_moving,
)
from dynamic_obstacle_avoidance.utils import obs_check_collision_2d

from dynamic_obstacle_avoidance.avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.avoidance.obs_dynamic_center_3d import (
    get_dynamic_center_obstacles,
)

from dynamic_obstacle_avoidance.avoidance import obs_avoidance_rk4

plt.ion()

import passive_control.magic_numbers_and_enums as mn


def plot_obstacles(
    obstacle_container,
    x_lim=None,
    y_lim=None,
    ax=None,
    pos_attractor=None,
    obstacle_color=None,
    show_obstacle_number=False,
    reference_point_number=False,
    drawVelArrow=True,
    noTicks=False,
    showLabel=False,
    draw_reference=False,
    draw_center=True,
    draw_wall_reference=False,
    border_linestyle="--",
    linecolor="black",
    linealpha=1,
    alpha_obstacle=0.8,
    velocity_arrow_factor=0.2,
    x_range=None,
    y_range=None,
    obs=None,
    absciss=0,  # added
    ordinate=1,  # added
):
    """Plot all obstacles & attractors"""
    if x_range is not None:
        # Depcreciated -> remove in the future
        x_lim = x_range

    if y_range is not None:
        # Depcreciated -> remove in the future
        y_lim = y_range

    if obs is not None:
        # Depreciated -> remove in the future
        obstacle_container = obs

    if ax is None:
        _, ax = plt.subplots()

    if pos_attractor is not None:
        ax.plot(
            pos_attractor[absciss],
            pos_attractor[ordinate],
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

    obs_polygon = []
    obs_polygon_sf = []

    if obstacle_color is None:
        obstacle_color = np.array([176, 124, 124]) / 255.0

    for n, obs in enumerate(obstacle_container):
        # Tiny bit outdated - newer obstacles wont have this

        if absciss == 0 and ordinate == 1:
            depth = 2
        elif absciss == 2 and ordinate == 1:
            depth = 0

        # added by thibaud to handle 3D: HUGLY PATCH - WORKS
        # we swap all coord since implemented function only handles xy
        # if mn.DIM == 3:
        #     obs.axes_length[[0,1,2]] = obs.axes_length[[absciss, ordinate, depth]]
        #     obs.axes_with_margin[[0,1,2]] = obs.axes_with_margin[[absciss, ordinate, depth]]
        #     obs.center_position[[0,1,2]] = obs.center_position[[absciss, ordinate, depth]]
        #     obs.global_reference_point[[0,1,2]] = obs.global_reference_point[[absciss, ordinate, depth]]
        #     obs.global_relative_reference_point[[0,1,2]] = obs.global_relative_reference_point[[absciss, ordinate, depth]]
        #     obs.linear_velocity[[0,1,2]] = obs.linear_velocity[[absciss, ordinate, depth]]
        #     obs.position[[0,1,2]] = obs.position[[absciss, ordinate, depth]]

        if hasattr(obs, "get_boundary_xy"):
            x_obs = np.array(obs.get_boundary_xy()).T

        else:
            # Outdated -> remove in the future
            obs.draw_obstacle()
            x_obs = obs.boundary_points_global_closed.T

        if hasattr(obs, "get_boundary_with_margin_xy"):
            x_obs_sf = np.array(obs.get_boundary_with_margin_xy()).T

        else:
            x_obs_sf = obs.boundary_points_margin_global_closed.T

        ax.plot(
            x_obs_sf[:, 0],
            x_obs_sf[:, 1],
            color=linecolor,
            linestyle=border_linestyle,
            alpha=linealpha,
            zorder=3,
        )

        if obs.is_boundary:
            if x_lim is None or y_lim is None:
                raise Exception(
                    "Outer boundary can only be defined with `x_lim` and `y_lim`."
                )
            outer_boundary = None
            if hasattr(obs, "global_outer_edge_points"):
                outer_boundary = obs.global_outer_edge_points

            if outer_boundary is None:
                outer_boundary = np.array(
                    [
                        [x_lim[0], x_lim[1], x_lim[1], x_lim[0]],
                        [y_lim[0], y_lim[0], y_lim[1], y_lim[1]],
                    ]
                )

            outer_boundary = outer_boundary.T
            boundary_polygon = plt.Polygon(
                outer_boundary, alpha=alpha_obstacle, zorder=-4
            )
            boundary_polygon.set_color(obstacle_color)
            ax.add_patch(boundary_polygon)

            obs_polygon.append(plt.Polygon(x_obs, alpha=1.0, zorder=-3))
            obs_polygon[n].set_color(np.array([1.0, 1.0, 1.0]))

        else:
            obs_polygon.append(plt.Polygon(x_obs, alpha=alpha_obstacle, zorder=0))

            # if obstacle_color is None:
            # obs_polygon[n].set_color(np.array([176,124,124])/255)
            # else:
            obs_polygon[n].set_color(obstacle_color)

        obs_polygon_sf.append(plt.Polygon(x_obs_sf, zorder=1, alpha=0.2))
        obs_polygon_sf[n].set_color([1, 1, 1])

        ax.add_patch(obs_polygon_sf[n])
        ax.add_patch(obs_polygon[n])

        if show_obstacle_number:
            ax.annotate(
                "{}".format(n + 1),
                xy=np.array(obs.center_position) + 0.16,
                textcoords="data",
                size=16,
                weight="bold",
            )

        # Automatic adaptation of center
        if draw_reference and not obs.is_boundary or draw_wall_reference:
            reference_point = obs.get_reference_point(in_global_frame=True)
            ax.plot(
                reference_point[0],
                reference_point[1],
                "k+",
                linewidth=12,
                markeredgewidth=2.4,
                markersize=8,
                zorder=3,
            )

        if (not obs.is_boundary or draw_wall_reference) and draw_center:
            ax.plot(
                obs.center_position[0],
                obs.center_position[1],
                "k.",
                zorder=3,
            )

        if reference_point_number:
            ax.annotate(
                "{}".format(n),
                xy=reference_point + 0.08,
                textcoords="data",
                size=16,
                weight="bold",
            )  #

        if (
            drawVelArrow
            and obs.linear_velocity is not None
            and np.linalg.norm(obs.linear_velocity) > 0
        ):
            # col=[0.5,0,0.9]
            col = [255 / 255.0, 51 / 255.0, 51 / 255.0]
            ax.arrow(
                obs.center_position[0],
                obs.center_position[1],
                obs.linear_velocity[0] * velocity_arrow_factor,
                obs.linear_velocity[1] * velocity_arrow_factor,
                # head_width=0.3, head_length=0.3, linewidth=10,
                head_width=0.1,
                head_length=0.1,
                linewidth=3,
                fc=col,
                ec=col,
                alpha=1,
                zorder=3,
            )

        # back to normal
        # added by thibaud to handle 3D:
        # if mn.DIM == 3:
        #     obs.axes_length[[absciss, ordinate, depth]] = obs.axes_length[[0,1,2]]
        #     obs.axes_with_margin[[absciss, ordinate, depth]] = obs.axes_with_margin[[0,1,2]]
        #     obs.center_position[[absciss, ordinate, depth]] = obs.center_position[[0,1,2]]
        #     obs.global_reference_point[[0,1,2]] = obs.global_reference_point[[absciss, ordinate, depth]]
        #     obs.global_relative_reference_point[[0,1,2]] = obs.global_relative_reference_point[[absciss, ordinate, depth]]
        #     obs.linear_velocity[[absciss, ordinate, depth]] = obs.linear_velocity[[0,1,2]]
        #     obs.position[[absciss, ordinate, depth]] = obs.position[[0,1,2]]

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if noTicks:
        ax.tick_params(
            axis="both",
            which="major",
            labelbottom=False,
            labelleft=False,
            bottom=False,
            top=False,
            left=False,
            right=False,
        )

    if showLabel:
        ax.set_xlabel(r"$\xi_1$")
        ax.set_ylabel(r"$\xi_2$")

    return ax
