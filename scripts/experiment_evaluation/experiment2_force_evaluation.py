from pathlib import Path
from typing import Optional

import numpy as np

from attrs import define, field

import pandas as pd
import matplotlib.pyplot as plt

from vartools.filter import filter_moving_average

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from passive_control.controller import Controller
from passive_control.controller import PassiveDynamicsController
from passive_control.controller import ObstacleAwarePassivController
import passive_control.magic_numbers_and_enums as mn

from passive_control.analysis import clean_all

# from passive_control.docker_helper import Simulated


def get_actual_distance(
    distance: float,
    margin_absolut: float = 0.12,
    distance_scaling: float = 10.0,
    boundary_power_factor: float = 1.0,
    center_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
    positions: Optional[np.ndarray] = None,
):
    """Computes the distance minus the margin.
    Use obstacle-values used in experiment, and invert gamma-computation.

    center_position: approximated center-position (real one not known)
    """
    distance = copy.deepcopy(distance)
    ind_negative = distance < 0

    # print(positions)

    if np.sum(ind_negative):
        # gamma = distance_center / (distance_center - distance_surface)
        # gamma = gamma**boundary_power_factor

        # => (distance_center - distance_surface) = distance_center / gamma
        # => (distance_center - distance_center / gamma) =  = distance_surface
        distance[ind_negative] = distance[ind_negative] + 1

        if positions is not None:
            distances_center = np.linalg.norm(
                positions[:, ind_negative]
                - np.tile(center_position, (np.sum(ind_negative), 1)).T,
                axis=0,
            )
        else:
            distances_center = 1

        distance[ind_negative] = distance[ind_negative] ** (1.0 / boundary_power_factor)
        distance[ind_negative] = (
            distances_center - distances_center / distance[ind_negative]
        )

    distance_scaled = distance / distance_scaling
    distance_without_margin = distance_scaled + margin_absolut

    return distance_without_margin


def plot_position_intervals_xyz(
    datahandler,
    ax,
    plot_indeces: list = [],
    plot_all=False,
):
    traj_label = datahandler.label
    dimension = 3

    positions = np.zeros((len(datahandler.dataframe["pos"]), dimension))
    velocities = np.zeros((len(datahandler.dataframe["vel"]), dimension))
    desired_vel = np.zeros((len(datahandler.dataframe["des_vel"]), dimension))

    for ii in range(len(datahandler.dataframe["pos"])):
        positions[ii, :] = datahandler.dataframe["pos"][ii]
        velocities[ii, :] = datahandler.dataframe["vel"][ii]
        desired_vel[ii, :] = datahandler.dataframe["des_vel"][ii]

    min_x = min(positions[:, 0])
    max_x = max(positions[:, 0])

    size_x = (positions[:, 0] - min_x) / (max_x - min_x)
    size_x = size_x**2 * 300 + 10
    # size_x = size_x * 400 + 10

    for ii, it_traj in enumerate(plot_indeces):
        # for ii, ax_label in enumerate(axes_names):
        interval = datahandler.intervals[it_traj]
        # breakpoint()
        ax.scatter(
            positions[interval, 1],
            positions[interval, 2],
            s=size_x[interval],
            # label=traj_label,
            color=datahandler.color,
            alpha=0.1,
            lw=0,
            zorder=0,
        )
        ax.plot(
            positions[interval, 1],
            positions[interval, 2],
            label=traj_label,
            color=datahandler.color,
            zorder=0,
            # alpha=0.1,
            # lw=0,
        )

        traj_label = None  # Only apply label once

    if plot_all:
        ax.scatter(
            positions[:, 1],
            positions[:, 2],
            s=size_x[:],
            # label=traj_label,
            color=datahandler.color,
            alpha=0.1,
            lw=0,
            zorder=0,
        )
        ax.plot(
            positions[:, 1],
            positions[:, 2],
            label=traj_label,
            color=datahandler.color,
            zorder=0,
            # alpha=0.1,
            # lw=0,
        )

    if plot_velocities:
        time_step_display = 140
        vel_scaling = 0.3
        arrow_kwargs = {
            "width": 0.01,
            "linewidth": 1.0,
            "edgecolor": "black",
            "zorder": 3,
        }

        for ii, it_traj in enumerate(plot_indeces):
            interval = datahandler.intervals[it_traj]
            for ii in range(100):
                if ii * time_step_display >= len(interval):
                    break

                it = interval[ii * time_step_display]

                ax.arrow(
                    positions[it, 1],
                    positions[it, 2],
                    velocities[it, 1] * vel_scaling,
                    velocities[it, 2] * vel_scaling,
                    facecolor=datahandler.color,
                    alpha=0.8,
                    **arrow_kwargs,
                )

                ax.arrow(
                    positions[it, 1],
                    positions[it, 2],
                    desired_vel[it, 1] * vel_scaling,
                    desired_vel[it, 2] * vel_scaling,
                    facecolor=datahandler.secondary_color,
                    linestyle="--",
                    **arrow_kwargs,
                )
                ax.plot(positions[it, 1], positions[it, 2], "ko", zorder=3)

    # ax.legend()

    # ax.set_xlabel("Step")
    # ax.set_ylabel("Position [m]")


def import_dataframe(
    filename,
    data_path=Path("/home/lukas/Code/obstacle_aware_damping") / "recordings",
    folder="new2_lukas",
):
    df_raw = pd.read_csv(data_path / folder / filename)
    return clean_all(df_raw)


def extract_sequences_intervals_experiment_two(
    dataframe,
    # z_range_start=[0.27, 0.295],
    y_start=0.395,
    # y_vel_max=0.25,
    y_vel_max=0.15,
    length_minimum=200,
    visualize=False,
    n_positive_velocities=5,
    end_cutoff=15,
    number_of_return_trajectories: Optional[int] = 10,
):
    dimension = 3
    positions = np.zeros((len(dataframe["pos"]), dimension))
    velocities = np.zeros((len(dataframe["vel"]), dimension))
    for ii in range(len(dataframe["pos"])):
        positions[ii, :] = dataframe["pos"][ii]
        velocities[ii, :] = dataframe["vel"][ii]

    if visualize:
        fig, axs = plt.subplots(3, 1)
        for ax in axs:
            ax.grid()

        axs[0].plot(positions[:, 0], label="Position x [m]")
        axs[2].plot(positions[:, 2], label="Position z [m]")
        ax = axs[1]

        # fig, ax = plt.subplots()
        ax.plot(positions[:, 1], label="Position y [m]")
        ax.plot(velocities[:, 1], label="Velocity y [m]")

        ax.plot([0, len(positions[:, 1])], [0, 0], ":", color="black")
        # ax.plot(ind_highs)
        # y_lim = ax.get_ylim()
        # for ii in range(interval_explicit.shape[0]):
        #     plt.plot(np.ones(2) * interval_explicit[ii], y_lim, color="black")
        # ax.set_ylim(y_lim)

    # In between y, close to x
    # it_start = np.logical_and(
    #     z_range_start[0] < positions[:, 2], positions[:, 2] < z_range_start[1]
    # )

    it_start = positions[:, 1] > y_start
    it_start_explicit = np.arange(positions.shape[0])[it_start]
    delta_start = it_start_explicit[1:] - it_start_explicit[:-1]
    # Always take last one
    it_start_explicit = it_start_explicit[np.hstack((delta_start > 5, True))]

    positions_filtered = filter_moving_average(positions[:, 1], 5)
    vel_from_pos = (positions_filtered - np.roll(positions_filtered, 1)) * 100

    # if visualize:
    if False:
        plt.close("all")
        fig, ax2 = plt.subplots()
        ax2.plot(vel_from_pos)
        ax2.set_ylim([-1.0, 1.0])

    it_end = vel_from_pos > y_vel_max
    it_end_explicit = np.arange(it_end.shape[0])[it_end]

    # Extract xx number of consecutive positive (y-direction-)velocities
    # it_positive = velocities[:, 1] >= 0
    # matr_positive = np.zeros((n_positive_velocities, it_positive.shape[0]))
    # for ii in range(matr_positive.shape[0]):
    #     matr_positive[ii, :] = np.roll(it_positive, (-1) * ii)
    # it_positive = np.sum(matr_positive, axis=0) == matr_positive.shape[0]
    # it_positive_explicit = np.arange(positions.shape[0])[it_positive]

    linekwargs = {"color": "gray"}

    it_end = 0
    intervals = []
    for ii, ii_start in enumerate(it_start_explicit):
        if visualize:
            ax.plot([it_end, ii_start], [0, 0], **linekwargs)
            ax.plot([ii_start, ii_start], [0, 1], **linekwargs)

        if ii_start >= it_end_explicit[-1]:
            break

        it_end_relative = np.searchsorted(it_end_explicit, ii_start)
        it_end = it_end_explicit[it_end_relative]

        # print("ii_start", ii_start)
        # print("it_end", it_end)

        if it_end - ii_start < length_minimum:
            # Short sequence indicates robot-outage
            continue

        if visualize:
            ax.plot([ii_start, it_end], [1, 1], **linekwargs)
            ax.plot([it_end, it_end], [1, 0], **linekwargs)

        interval = np.arange(ii_start, it_end - end_cutoff)

        # Cut some off at the end to avoid the recovery-peak
        intervals.append(interval)

    if number_of_return_trajectories is not None:
        if len(intervals) < number_of_return_trajectories:
            raise ValueError("Too few trajectories.")
        else:
            intervals = intervals[:number_of_return_trajectories]
    return intervals


def extract_sequences_intervals(
    dataframe, y_high=0.2, y_vel_max=-0.125, visualize=False
):
    dimension = 3
    positions = np.zeros((len(dataframe["pos"]), dimension))
    velocities = np.zeros((len(dataframe["vel"]), dimension))
    for ii in range(len(dataframe["pos"])):
        positions[ii, :] = dataframe["pos"][ii]
        velocities[ii, :] = dataframe["vel"][ii]

    ind_highs = positions[:, 1] > y_high
    ind_highs_explicit = np.arange(ind_highs.shape[0])[ind_highs]
    delta_step = ind_highs_explicit[1:] - ind_highs_explicit[:-1]

    # > 1, just in case we use 2
    interval_starts = delta_step > 10
    start_exlipcit = (
        np.arange(interval_starts.shape[0])[interval_starts] + 1
    )  # Take the right value
    interval_explicit = ind_highs_explicit[start_exlipcit]
    interval_explicit = np.hstack((0, interval_explicit, positions.shape[0] - 1))

    if visualize:
        fig, ax = plt.subplots()
        ax.plot(positions[:, 1])
        ax.plot(ind_highs)
        y_lim = ax.get_ylim()
        for ii in range(interval_explicit.shape[0]):
            plt.plot(np.ones(2) * interval_explicit[ii], y_lim, color="black")
        ax.set_ylim(y_lim)

    intervals = []
    for ii in range(interval_explicit.shape[0] - 1):
        interval = np.arange(interval_explicit[ii], interval_explicit[ii + 1])

        ind_min = np.argmin(positions[interval, 1])
        ind_max = np.argmax(positions[interval, 1])

        interval = np.arange(interval[ind_max], interval[ind_min])

        # Cut some off at the end to avoid the recovery-peak
        # limited = velocities[interval, 1] < y_vel_max
        # explicit_limited = interval[limited]
        # # intervals.append(np.arange(interval[ind_max], explicit_limited[-10]))
        # interval = np.arange(interval[ind_max], explicit_limited[-1])
        intervals.append(interval)

        if visualize:
            plt.plot([intervals[-1][0], intervals[-1][0]], y_lim, color="red")
            plt.plot([intervals[-1][-1], intervals[-1][-1]], y_lim, color="red")

    return intervals


def plot_position_intervals(dataframe, interval_list):
    fig, ax = plt.subplots(figsize=(5, 4))

    dimension = 3
    positions = np.zeros((len(dataframe["pos"]), dimension))
    for ii in range(len(dataframe["pos"])):
        positions[ii, :] = dataframe["pos"][ii]

    axes_names = ["x", "y", "z"]
    colors = ["red", "green", "blue"]
    for ii, ax_label in enumerate(axes_names):
        for interval in interval_list:
            ax.plot(positions[interval, ii], label=ax_label, color=colors[ii])
            ax_label = None

    # distances = np.zeros((len(dataframe["dists"])))
    # for ii in range(len(dataframe["dists"])):
    #     distances[ii] = dataframe["dists"][ii]
    # ax.plot(distances, "--", label="Minimum distance")

    ax.legend()
    # fig = df_damped.plot(y="pos", use_index=True, label="with damping")

    ax.set_xlabel("Step")
    ax.set_ylabel("Position [m]")

    return fig, ax


def plot_y_position_intervals(
    datahandler,
    ax=None,
    n_max=5,
):
    dimension = 3
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        # ax.set_xlabel("Time [s]")
        # ax.set_ylabel("Position y [m]")
        ax.set_xlabel("Position x [m]")
        ax.set_ylabel("Position y [m]")

        ax.set_ylim([0.05, 0.40])
        ax.set_xlim([0.5, -0.51])
        ax.set_aspect("equal", adjustable="box")
        ax.grid("on")

    else:
        fig = None

    positions = np.zeros((len(datahandler.dataframe["pos"]), dimension))
    for jj in range(len(datahandler.dataframe["pos"])):
        positions[jj, :] = datahandler.dataframe["pos"][jj]

    plot_all = True
    if plot_all:
        ax.plot(
            positions[:, 1],
            positions[:, 2],
            color=datahandler.color,
            marker=".",
            linestyle=":",
            linewidth=2.0,
        )
    else:
        for ii, interval in enumerate(datahandler.intervals):
            if ii >= n_max:
                # For visibility only plot partial
                break

            ax.plot(
                positions[interval, 1],
                positions[interval, 2],
                color=datahandler.color,
                marker=".",
                linestyle=datahandler.linestyles[ii],
                linewidth=2.0,
            )

    # plt.savefig("figures/" + figName + ".png", bbox_inches="tight")

    return fig, ax


def plot_velocities_all_directions(datahandler, ax=None, n_max=5):
    dimension = 3
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        # ax.set_xlabel("Time [s]")
        # ax.set_ylabel("Position y [m]")
        # ax.set_aspect("equal", adjustable="box")
        ax.grid("on")

    else:
        fig = None

    velocities = np.zeros((len(datahandler.dataframe["vel"]), dimension))
    for jj in range(len(datahandler.dataframe["vel"])):
        velocities[jj, :] = datahandler.dataframe["vel"][jj]

    axes_names = ["x", "y", "z"]
    colors = ["red", "green", "blue"]
    for ii, ax_label in enumerate(axes_names):
        for interval in datahandler.intervals:
            ax.plot(
                velocities[interval, ii], label=ax_label, color=colors[ii], marker="."
            )
            ax_label = None

    ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Position [m]")

    return fig, ax


def plot_distance_obstacle(
    datahandler,
    ax,
    n_max=5,
    plot_mean_value=False,
    plot_indeces: list = [],
    # traj_label=None,
):
    traj_label = copy.deepcopy(datahandler.label)

    dimension = 3
    positions = np.zeros((3, len(datahandler.dataframe["dists"])))
    distances = np.zeros((len(datahandler.dataframe["dists"])))
    for jj in range(len(datahandler.dataframe["dists"])):
        positions[:, jj] = datahandler.dataframe["pos"][jj]
        distances[jj] = datahandler.dataframe["dists"][jj]

    distances = get_actual_distance(
        distances,
        positions=positions,
        margin_absolut=main_obstacle.margin_absolut,
        distance_scaling=main_obstacle.distance_scaling,
        center_position=main_obstacle.center_position,
    )

    # distances_norm = np.linalg.norm(distances, axis=1)
    # breakpoint()
    for ii, it_traj in enumerate(plot_indeces):
        # print(it_traj)
        # for ii, interval in enumerate(datahandler.intervals):
        interval = datahandler.intervals[it_traj]
        ax.plot(
            datahandler.time_step * np.arange(interval.shape[0]),
            distances[interval],
            color=datahandler.color,
            linestyle=datahandler.linestyles[ii % len(datahandler.linestyles)],
            linewidth=2.0,
            label=traj_label,
        )
        traj_label = None

    values = distances
    if plot_mean_value:
        len_max = 0
        for ii, interval in enumerate(datahandler.intervals):
            len_max = max(interval.shape[0], len_max)

        ind_count = np.zeros(len_max)
        mean_value = np.zeros(len_max)
        for ii, interval in enumerate(datahandler.intervals):
            ind_count[: interval.shape[0]] = ind_count[: interval.shape[0]] + 1
            mean_value[: interval.shape[0]] = (
                mean_value[: interval.shape[0]] + values[interval]
            )
        mean_value = mean_value / ind_count

        values_matrix = np.ones((len(datahandler.intervals), len_max)) * mean_value
        for ii, interval in enumerate(datahandler.intervals):
            values_matrix[ii, : interval.shape[0]] = values[interval]

        std_value = np.std(values_matrix, axis=0)
        ax.fill_between(
            datahandler.time_step * np.arange(mean_value.shape[0]),
            mean_value - std_value,
            mean_value + std_value,
            alpha=0.3,
            color=datahandler.color,
            zorder=-1,
        )
        ax.plot(
            datahandler.time_step * np.arange(mean_value.shape[0]),
            mean_value,
            color=datahandler.color,
            label=traj_label,
            **kwargs_meanline,
        )

    min_distances = []
    for ii, interval in enumerate(datahandler.intervals):
        min_distances.append(min(values[interval]))

    # breakpoint()
    # plt.savefig("figures/" + figName + ".png", bbox_inches="tight")

    return min_distances


def import_second_experiment(n_traj_max=10):
    intervals = []

    folder = "recording_2022_07"
    dataframe = import_dataframe("recording_14-07-2023_15-28-39.csv", folder=folder)
    # intervals = extract_sequences_intervals(dataframe, visualize=False)
    intervals = extract_sequences_intervals_experiment_two(dataframe, visualize=True)
    # intervals = ]
    data_nodamping = DataStorer(
        dataframe,
        intervals,
        # color="red",
        color="#B0050B",
        secondary_color="#AB4346",
        label="Dynamics preserving",
        # it_nominal=2,
        # it_nominal=4,
        it_nominal=8,
    )

    dataframe = import_dataframe("recording_14-07-2023_15-32-02.csv", folder=folder)
    intervals = extract_sequences_intervals_experiment_two(dataframe, visualize=True)
    data_damped = DataStorer(
        dataframe,
        intervals,
        # color=(0, 0, 200),
        # secondary_color=(100, 100, 200),
        color="#084263",
        secondary_color="#4F778F",
        it_nominal=8,
        # it_nominal=1,
        label="Obstacle aware",  # "Dynamcis"
    )

    dataframe = import_dataframe("recording_14-07-2023_15-34-32.csv", folder=folder)
    intervals = extract_sequences_intervals_experiment_two(dataframe, visualize=True)
    data_no_interaction = DataStorer(
        dataframe,
        intervals,
        # secondary_color="gray",
        color="#696969",
        secondary_color="#808080",
        it_nominal=1,
        label="Undisturbed",
    )

    return (data_no_interaction, data_nodamping, data_damped)


@define(slots=False)
class DataStorer:
    dataframe: object
    intervals: object
    color: str | tuple[float | int]
    secondary_color: str | tuple[float | int]
    label: str

    linestyles: list = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))]
    time_step: float = 1.0 / 100

    it_nominal: Optional[int] = None


@define(slots=False)
class DummyObstacle:
    distance: float = 0.0
    normal: np.ndarray = field(factory=lambda: np.zeros(0))

    def get_gamma(self, *args, **kwargs):
        return self.distance + 1

    def get_normal_direction(self, *args, **kwargs):
        return self.normal


def plot_all_forces(datahandler, controller: Controller, ax, plot_indeces=[]):
    dimension = 3

    forces = np.zeros((len(datahandler.dataframe["des_vel"]), dimension))
    normals = np.zeros((len(datahandler.dataframe["des_vel"]), dimension))
    for jj in range(len(datahandler.dataframe["des_vel"])):
        velocity_desired = datahandler.dataframe["des_vel"][jj]
        velocity = datahandler.dataframe["vel"][jj]
        position = datahandler.dataframe["pos"][jj]

        if hasattr(controller, "environment"):
            # Recreate environment to compute the correct force
            controller.environment[0].normal = datahandler.dataframe["normals"][jj]
            controller.environment[0].distance = datahandler.dataframe["dists"][jj]

        forces[jj, :] = controller.compute_force(
            velocity=velocity,
            desired_velocity=velocity_desired,
            position=position,
        )

    forces_norm = np.linalg.norm(forces, axis=1)
    forces_norm = filter_moving_average(forces_norm)

    axes_names = ["Force x ", "Force y", "Force z"]
    for jj, it_traj in enumerate(plot_indeces):
        interval = datahandler.intervals[it_traj]
        for ii, ax_label in enumerate(axes_names):
            ax.plot(
                datahandler.time_step * np.arange(interval.shape[0]),
                forces[interval, ii],
                # label=ax_label,
                color=datahandler.color,
                linestyle=datahandler.linestyles[ii],
            )
            ax_label = None


def plot_forces(
    datahandler,
    ax,
    plot_indeces: list = [],
    plot_mean_force=False,
    traj_label=None,
):
    dimension = 3

    forces = np.zeros((len(datahandler.dataframe["des_vel"]), dimension))
    for jj in range(len(datahandler.dataframe["des_vel"])):
        velocity_desired = datahandler.dataframe["des_vel"][jj]
        velocity = datahandler.dataframe["vel"][jj]
        D = datahandler.dataframe["D"][jj]

        forces[jj, :] = D @ (velocity_desired - velocity)

    forces_norm = np.linalg.norm(forces, axis=1)
    # forces_norm = filter_moving_average(forces_norm, 3)

    max_forces = []
    for ii, it_traj in enumerate(plot_indeces):
        # for ii, interval in enumerate(datahandler.intervals):
        interval = datahandler.intervals[it_traj]

        # forces_norm = np.linalg.norm(forces[interval, :], axis=1)
        tot_forces = forces_norm[interval]
        # tot_forces = filter_moving_average(tot_forces, 5)
        ax.plot(
            datahandler.time_step * np.arange(interval.shape[0]),
            tot_forces,
            color=datahandler.color,
            linestyle=datahandler.linestyles[ii % len(datahandler.linestyles)],
            label=traj_label,
            zorder=2,
        )

        traj_label = None

    if plot_mean_force:
        # Use logarithmic value, add margin to avoid errors
        margin_force = 1e-9
        forces_norm = np.log(forces_norm + margin_force)

        len_max = 0
        for ii, interval in enumerate(datahandler.intervals):
            len_max = max(interval.shape[0], len_max)

        ind_count = np.zeros(len_max)
        mean_force = np.zeros(len_max)
        for ii, interval in enumerate(datahandler.intervals):
            ind_count[: interval.shape[0]] = ind_count[: interval.shape[0]] + 1
            mean_force[: interval.shape[0]] = (
                mean_force[: interval.shape[0]] + forces_norm[interval]
            )

        mean_force = mean_force / ind_count

        # Set default values when a trajectory is short
        force_values = np.ones((len(datahandler.intervals), len_max)) * mean_force
        for ii, interval in enumerate(datahandler.intervals):
            force_values[ii, : interval.shape[0]] = forces_norm[interval]

        std_force = np.std(force_values, axis=0)
        ax.fill_between(
            datahandler.time_step * np.arange(mean_force.shape[0]),
            np.exp(mean_force - std_force) + margin_force,
            np.exp(mean_force + std_force) + margin_force,
            alpha=0.3,
            color=datahandler.color,
            zorder=-1,
        )
        ax.plot(
            datahandler.time_step * np.arange(mean_force.shape[0]),
            np.exp(mean_force) + margin_force,
            ":",
            color=datahandler.color,
            alpha=0.8,
            label=traj_label,
            zorder=1,
        )

    # breakpoint()

    max_forces = []
    for ii, interval in enumerate(datahandler.intervals):
        max_forces.append(max(forces_norm[interval]))

    ax.legend()
    return max_forces


def plot_forces_all(
    data_no_interaction, data_nodamping, data_damped, save_figure=False
):
    dimension = 3

    lambda_DS = 100.0
    lambda_perp = 20.0
    lambda_obs = mn.LAMBDA_MAX

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force [N]")

    ax.set_xlim(x_lim_duration)
    # ax.set_ylim([-1.5, 5.0])

    x_vals = ax.get_xlim()

    # ax.plot(x_vals, [0, 0], ":", color="#8b0000", linewidth=2.0)
    ax.grid("on")

    # # No interaction
    max_forces = plot_forces(
        data_no_interaction,
        ax,
        plot_mean_force=True,
        # plot_indeces=[2],
        plot_indeces=[data_no_interaction.it_nominal],
        traj_label=data_no_interaction.label,
    )
    print(f"No-Interaction Force: {np.mean(max_forces)} \pm {np.std(max_forces)} ")

    # # No damping
    max_forces = plot_forces(
        data_nodamping,
        ax,
        plot_mean_force=True,
        # plot_indeces=np.arange(len(data_nodamping.intervals)),
        # plot_indeces=[3],
        plot_indeces=[data_nodamping.it_nominal],
        traj_label=data_nodamping.label,
    )
    print(f"No-Damping Force: {np.mean(max_forces)} \pm {np.std(max_forces)} ")

    # No ObstacleAwarePassivController
    max_forces = plot_forces(
        data_damped,
        ax,
        # plot_indeces=[8],
        plot_indeces=[data_damped.it_nominal],
        plot_mean_force=True,
        # plot_indeces=np.arange(len(data_damped.intervals)),
        traj_label=data_damped.label,
    )
    print(f"Obstacle-Aware Force: {np.mean(max_forces)} \pm {np.std(max_forces)} ")

    if save_figure:
        figname = "trajectory_comparison_force_magnitude"
        plt.savefig("figures/" + figname + figtype, bbox_inches="tight")


def plot_all_forces_comparison(
    data_no_interaction, data_nodamping, data_damped, save_figure=False
):
    dimension = 3

    lambda_DS = 100.0
    lambda_perp = 20.0
    lambda_obs = mn.LAMBDA_MAX

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force [N]")

    ax.set_xlim([0, 4.5])
    # ax.set_ylim([-1.5, 5.0])

    x_vals = ax.get_xlim()

    # ax.plot(x_vals, [0, 0], ":", color="#8b0000", linewidth=2.0)
    ax.grid("on")

    # # No interaction
    controller = PassiveDynamicsController(
        lambda_dynamics=lambda_DS, lambda_remaining=lambda_perp, dimension=3
    )
    max_forces = plot_all_forces(
        data_no_interaction,
        controller,
        ax=ax,
        plot_indeces=[0]
        # traj_label="No disturbance",
    )
    # print(f"No-Interaction Force: {np.mean(max_forces)} \pm {np.std(max_forces)} ")

    # # No damping
    controller = PassiveDynamicsController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        dimension=3,
    )
    max_forces = plot_all_forces(
        data_nodamping,
        controller,
        ax=ax,
        plot_indeces=[data_nodamping.it_nominal]
        # traj_label="No disturbance",
    )
    # print(f"No-Damping Force: {np.mean(max_forces)} \pm {np.std(max_forces)} ")

    # No ObstacleAwarePassivController
    controller = ObstacleAwarePassivController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        lambda_obstacle=lambda_obs,
        dimension=3,
        environment=ObstacleContainer([DummyObstacle()]),
    )
    max_forces = plot_all_forces(
        data_damped,
        controller,
        ax=ax,
        plot_indeces=[data_damped.it_nominal]
        # traj_label="No disturbance",
    )
    # print(f"Obstacle-Aware Force: {np.mean(max_forces)} \pm {np.std(max_forces)} ")


def plot_positions(data_no_interaction, data_nodamping, data_damped, save_figure=False):
    fig, ax = plot_y_position_intervals(data_no_interaction, n_max=1)
    plot_y_position_intervals(data_nodamping, ax=ax)
    plot_y_position_intervals(data_damped, ax=ax)

    if save_figure:
        figname = "robot_arm_trajectory_xy"
        plt.savefig("figures/" + figname + figtype, bbox_inches="tight")


def plot_positions_single_graph(
    data_no_interaction, data_nodamping, data_damped, save_figure=False
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("Position y [m]")
    ax.set_ylabel("Position z [m]")
    ax.set_aspect("equal", adjustable="box")

    # ax.set_xlim([0.455, -0.402])
    ax.set_ylim([-0.2, 0.4])
    ax.set_ylim([0.05, 0.36])

    x_vals = ax.get_xlim()

    # ax.plot(x_vals, [0, 0], ":", color="#8b0000", linewidth=2.0)
    ax.grid("on")

    plot_position_intervals_xyz(data_no_interaction, ax=ax, plot_indeces=[0])
    plot_position_intervals_xyz(
        data_nodamping, ax=ax, plot_indeces=[data_nodamping.it_nominal]
    )
    plot_position_intervals_xyz(
        data_damped, ax=ax, plot_indeces=[data_damped.it_nominal]
    )

    plt.legend(loc="lower left", ncol=3)

    if save_figure:
        figname = "robot_arm_trajectory_xyz"
        plt.savefig("figures/" + figname + figtype, bbox_inches="tight")


def plot_positions_single_graph(
    data_no_interaction, data_nodamping, data_damped, save_figure=False
):
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] * 2))
    ax.set_xlabel("Position y [m]")
    ax.set_ylabel("Position z [m]")
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim([-0.4, 0.40])
    ax.set_ylim([0.05, 0.45])
    # ax.set_ylim([0.05, 0.36])

    x_vals = ax.get_xlim()

    # ax.plot(x_vals, [0, 0], ":", color="#8b0000", linewidth=2.0)
    ax.grid("on")

    plot_position_intervals_xyz(
        data_no_interaction,
        ax=ax,
        plot_indeces=[data_no_interaction.it_nominal],
        # plot_all=True,
    )
    plot_position_intervals_xyz(
        data_nodamping,
        ax=ax,
        plot_indeces=[data_nodamping.it_nominal],
    )
    plot_position_intervals_xyz(
        data_damped,
        ax=ax,
        plot_indeces=[data_damped.it_nominal],
    )

    obstacle2d = Cuboid(
        center_position=main_obstacle.center_position[1:],
        margin_absolut=main_obstacle.margin_absolut,
        axes_length=main_obstacle.axes_length[1:],
    )

    plot_obstacles(
        obstacle_container=[obstacle2d],
        x_lim=ax.get_xlim(),
        y_lim=ax.get_ylim(),
        ax=ax,
    )

    # plt.legend(loc="lower left", ncol=3)
    plt.legend(loc="lower right")

    if save_figure:
        figname = "robot_arm_trajectory_xyz"
        plt.savefig("figures/" + figname + figtype, bbox_inches="tight")


def plot_velocities(
    data_no_interaction, data_nodamping, data_damped, save_figure=False
):
    fig, ax = plot_velocities_all_directions(data_no_interaction)
    # plot_y_velocity_intervals(data_nodamping, ax=ax)
    # plot_y_velocity_intervals(data_damped, ax=ax)

    if save_figure:
        figname = "robot_arm_trajectory_xy"
        plt.savefig("figures/" + figname + figtype, bbox_inches="tight")


def plot_closests_distance(
    data_no_interaction, data_nodamping, data_damped, save_figure=False
):
    dimension = 3

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance [m]")

    ax.set_xlim(x_lim_duration)
    ax.set_ylim([-0.2, 0.4])

    x_vals = ax.get_xlim()

    # ax.plot(x_vals, [0, 0], ":", color="#8b0000", linewidth=2.0)
    ax.plot(x_vals, [0, 0], ":", color="black", linewidth=2.0)
    ax.grid("on")

    min_distances_no_int = plot_distance_obstacle(
        data_no_interaction,
        ax=ax,
        plot_mean_value=True,
        plot_indeces=[data_no_interaction.it_nominal],
    )
    print(
        f"Distance [NoDist]: {np.mean(min_distances_no_int)} "
        + f" \pm {np.std(min_distances_no_int)}"
    )

    min_distances_ds = plot_distance_obstacle(
        data_nodamping,
        ax=ax,
        plot_indeces=[data_nodamping.it_nominal],
        # plot_indeces=np.arange(len(data_nodamping.intervals)),
        plot_mean_value=True
        # plot_indeces=np.arange(3),
    )

    print(
        f"Distance [PassiveDS]: {np.mean(min_distances_ds)} "
        + f" \pm {np.std(min_distances_ds)}"
    )

    min_distance_obs_aware = plot_distance_obstacle(
        data_damped,
        ax=ax,
        plot_indeces=[data_damped.it_nominal],
        plot_mean_value=True,
        # plot_indeces=np.arange(len(data_damped.intervals)),
        # plot_indeces=np.arange(3),
    )
    print(
        f"Distance [Aware]: {np.mean(min_distance_obs_aware)} "
        + f" \pm {np.std(min_distance_obs_aware)}"
    )

    ax.legend()

    if save_figure:
        figname = "robot_arm_trajectory_distance"
        plt.savefig("figures/" + figname + figtype, bbox_inches="tight")


def main():
    # df_damped.columns
    # Index(['D', 'vel', 'des_vel', 'pos', 'normals', 'dists'], dtype='object')
    # df_damped = import_dataframe("single_traverse_damped_1.csv")
    # df_no_damping = import_dataframe("single_traverse_nodamping_1.csv")

    # # Plot Positions
    # _, ax = plot_positions(df_damped)
    # ax.set_title("Obstacle aware")

    # intervals_damped = extract_sequences_intervals(df_damped, visualize=True)
    # plot_position_intervals(df_damped, intervals_damped)

    # intervals = extract_sequences_intervals(df_no_damping, visualize=True)
    # plot_position_intervals(df_no_damping, intervals)

    # plot_position_intervals(dataframe, intervals)

    # _, ax = plot_positions(df_no_damping)
    # ax.set_title("No damping")

    # _, ax = plot_positions(df_no_disturbance)
    # ax.set_title("No disturbance")

    # fig = df_damped.plot(y="pos", use_index=True, label="with damping")
    # df_no_raw.plot(y="pos", use_index=True, label="without damping")
    # plt.set_xlabel("Force")
    # df_clean_nothing.plot(y="Pos", use_index=True, label="no hit")
    # plt.title("Force")
    pass


if (__name__) == "__main__":
    figtype = ".pdf"
    figsize = (6.0, 2.2)

    x_lim_duration = [0, 3.5]

    kwargs_meanline = {"linestyle": ":", "alpha": 0.8}

    #
    # simulation_handler = Simulated()
    # obs_env = simulation_handler.create_env(
    #     obs_pos=np.array([0.4, 0, 0]),
    #     obs_axes_lenght=np.array([0.20] * 3),
    #     obs_vel=np.zeros(3),
    #     no_obs=False,
    # )

    # This obstacle is for represetnation purposes only.
    # The real obstacle did not have the exact same position for all trajectories
    main_obstacle = Cuboid(
        axes_length=np.array([0.20] * 3),
        margin_absolut=0.12,
        # axes_length=np.array([0.30] * 3),
        # margin_absolut=0.04,
        center_position=np.array([0.35, -0.00, 0.16]),
        distance_scaling=10.0,
    )

    # Only import data once... -> keep in local workspace after
    reimport_data = False
    if reimport_data or "data_no_interaction" not in locals():
        print("(Re-)Importing data-sequences.")
        (data_no_interaction, data_nodamping, data_damped) = import_second_experiment()

    plt.close("all")

    # plt.close("all")
    plt.ion()

    # main(data_no_interaction, data_nodamping, data_damped)

    plot_closests_distance(
        data_no_interaction, data_nodamping, data_damped, save_figure=True
    )
    plot_forces_all(data_no_interaction, data_nodamping, data_damped, save_figure=True)
    plot_positions_single_graph(
        data_no_interaction, data_nodamping, data_damped, save_figure=True
    )

    # plot_positions(data_no_interaction, data_nodamping, data_damped)
    # plot_velocities(data_no_interaction, data_nodamping, data_damped)

    # plot_all_forces_comparison(data_no_interaction, data_nodamping, data_damped)

    # plot_forces(data_no_interaction, data_nodamping, data_damped)
