import copy
from pathlib import Path
from typing import Optional

import numpy as np

from attrs import define, field

import pandas as pd
import matplotlib.pyplot as plt

from vartools.filter import filter_moving_average

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from passive_control.controller import Controller
from passive_control.controller import PassiveDynamicsController
from passive_control.controller import ObstacleAwarePassivController
import passive_control.magic_numbers_and_enums as mn


def clean_array(Serie: pd.Series):
    clean = Serie.copy()
    for i in range(Serie.shape[0]):
        yy = Serie[i]
        yy = yy.replace("[", " ")
        yy = yy.replace("]", " ")
        yy = yy.replace("\n", " ")
        yy = np.array(yy.split()).astype(float)
        clean[i] = yy
    return clean


def clean_matrix(Serie: pd.Series):
    clean = Serie.copy()
    for i in range(Serie.shape[0]):
        yy = Serie[i]
        yy = yy.replace("[", " ")
        yy = yy.replace("]", " ")
        yy = yy.replace("\n", " ")
        yy = np.array(yy.split()).astype(float)
        yy = yy.reshape(3, 3)
        clean[i] = yy
    return clean


def clean_float(Serie: pd.Series):
    clean = Serie.copy()
    for i in range(Serie.shape[0]):
        yy = Serie[i]
        yy = yy.replace("[", " ")
        yy = yy.replace("]", " ")
        yy = yy.replace("\n", " ")
        yy = float(yy)
        clean[i] = yy
    return clean


def clean_all(df):
    clean_D = clean_matrix(df["D"])
    # clean_eigen_values_obs = clean_matrix(df["eigen_values_obs"]).apply(np.diag)
    clean_pos = clean_array(df["pos"])
    clean_vel = clean_array(df["vel"])
    clean_des_vel = clean_array(df["des_vel"])
    clean_normal = clean_array(df["normals"])
    clean_dist = clean_float(df["dists"])
    df_clean = pd.concat(
        [
            clean_D,
            # clean_eigen_values_obs,
            clean_vel,
            clean_des_vel,
            clean_pos,
            clean_normal,
            clean_dist,
        ],
        axis=1,
    )
    return df_clean


def plot_position_intervals_xyz(
    datahandler,
    ax,
    plot_indeces: list = [],
):
    traj_label = datahandler.label
    dimension = 3
    positions = np.zeros((len(datahandler.dataframe["pos"]), dimension))
    for ii in range(len(datahandler.dataframe["pos"])):
        positions[ii, :] = datahandler.dataframe["pos"][ii]

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
    distances = np.zeros((len(datahandler.dataframe["dists"])))
    for jj in range(len(datahandler.dataframe["dists"])):
        distances[jj] = datahandler.dataframe["dists"][jj]

    # distances_norm = np.linalg.norm(distances, axis=1)
    # breakpoint()
    for ii, it_traj in enumerate(plot_indeces):
        # for ii, interval in enumerate(datahandler.intervals):
        interval = datahandler.intervals[ii]
        ax.plot(
            datahandler.time_step * np.arange(interval.shape[0]),
            distances[interval],
            color=datahandler.color,
            linestyle=datahandler.linestyles[ii],
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
        )

    min_distances = []
    for ii, interval in enumerate(datahandler.intervals):
        min_distances.append(min(values[interval]))

    # breakpoint()
    # plt.savefig("figures/" + figName + ".png", bbox_inches="tight")

    return min_distances


def import_dataframes():
    dataframe = import_dataframe("single_traverse_no_interaction.csv")
    intervals = extract_sequences_intervals(dataframe, visualize=False)
    data_no_interaction = DataStorer(
        dataframe, intervals, color="gray", label="Undisturbed"
    )

    dataframe = import_dataframe("single_traverse_nodamping_1.csv")
    intervals = extract_sequences_intervals(dataframe, visualize=False)
    data_nodamping = DataStorer(
        dataframe, intervals, color="blue", it_nominal=1, label="Dynamics"
    )

    dataframe = import_dataframe("single_traverse_damped_1.csv")
    intervals = extract_sequences_intervals(dataframe, visualize=False)
    data_damped = DataStorer(
        dataframe, intervals, color="green", it_nominal=1, label="Obstacle"
    )

    return (data_no_interaction, data_nodamping, data_damped)


@define(slots=False)
class DataStorer:
    dataframe: object
    intervals: object
    color: str
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


def plot_and_compute_force(
    datahandler,
    controller: Controller,
    ax,
    plot_indeces: list = [],
    plot_mean_force=False,
    traj_label=None,
):
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

    max_forces = []
    for ii, it_traj in enumerate(plot_indeces):
        # for ii, interval in enumerate(datahandler.intervals):
        interval = datahandler.intervals[it_traj]

        # forces_norm = np.linalg.norm(forces[interval, :], axis=1)
        ax.plot(
            datahandler.time_step * np.arange(interval.shape[0]),
            forces_norm[interval],
            color=datahandler.color,
            linestyle=datahandler.linestyles[ii],
            label=traj_label,
        )

        traj_label = None

    if plot_mean_force:
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

        force_values = np.ones((len(datahandler.intervals), len_max)) * mean_force
        for ii, interval in enumerate(datahandler.intervals):
            force_values[ii, : interval.shape[0]] = forces_norm[interval]

        std_force = np.std(force_values, axis=0)
        ax.fill_between(
            datahandler.time_step * np.arange(mean_force.shape[0]),
            mean_force - std_force,
            mean_force + std_force,
            alpha=0.3,
            color=datahandler.color,
            zorder=-1,
        )
        ax.plot(
            datahandler.time_step * np.arange(mean_force.shape[0]),
            mean_force,
            color=datahandler.color,
            label=traj_label,
        )

    max_forces = []
    for ii, interval in enumerate(datahandler.intervals):
        max_forces.append(max(forces_norm[interval]))

    ax.legend()
    return max_forces


def plot_forces(data_no_interaction, data_nodamping, data_damped, save_figure=False):
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
    max_forces = plot_and_compute_force(
        data_no_interaction,
        controller,
        ax,
        plot_mean_force=True,
        traj_label="No disturbance",
    )
    print(f"No-Interaction Force: {np.mean(max_forces)} \pm {np.std(max_forces)} ")

    # # No damping
    controller = PassiveDynamicsController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        dimension=3,
    )
    max_forces = plot_and_compute_force(
        data_nodamping,
        controller,
        ax,
        plot_indeces=[data_nodamping.it_nominal],
        traj_label="DS following",
    )
    print(f"No-Damping Force: {np.mean(max_forces)} \pm {np.std(max_forces)} ")

    # No ObstacleAwarePassivController
    controller = ObstacleAwarePassivController(
        lambda_dynamics=lambda_DS,
        lambda_remaining=lambda_perp,
        lambda_obstacle=lambda_obs,
        dimension=3,
        environment=ObstacleContainer([DummyObstacle()]),
    )
    max_forces = plot_and_compute_force(
        data_damped,
        controller,
        ax,
        plot_indeces=[data_damped.it_nominal],
        traj_label="Obstacle aware",
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

    ax.set_xlim([0.455, -0.402])
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

    ax.set_xlim([0, 7.2])
    ax.set_ylim([-1.5, 4.5])

    x_vals = ax.get_xlim()

    ax.plot(x_vals, [0, 0], ":", color="#8b0000", linewidth=2.0)
    ax.grid("on")

    min_distances_no_int = plot_distance_obstacle(
        data_no_interaction, ax=ax, plot_mean_value=True
    )
    print(
        f"Distance [NoDist]: {np.mean(min_distances_no_int)} "
        + f" \pm {np.std(min_distances_no_int)}"
    )
    min_distances_ds = plot_distance_obstacle(
        data_nodamping,
        ax=ax,
        plot_indeces=[data_nodamping.it_nominal]
        # plot_indeces=np.arange(3),
    )

    print(
        f"Distance [PassiveDS]: {np.mean(min_distances_ds)} "
        + f" \pm {np.std(min_distances_ds)}"
    )

    min_distance_obs_aware = plot_distance_obstacle(
        data_damped,
        ax=ax,
        plot_indeces=[data_damped.it_nominal]
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

    # Only import data once... -> keep in local workspace after
    reimport_data = True
    if reimport_data or "data_no_interaction" not in locals():
        print("Importing data-sequences.")
        (data_no_interaction, data_nodamping, data_damped) = import_dataframes()

    # plt.close("all")
    plt.ion()

    # main(data_no_interaction, data_nodamping, data_damped)

    # plot_closests_distance(data_no_interaction, data_nodamping, data_damped)
    # plot_forces(data_no_interaction, data_nodamping, data_damped, save_figure=True)

    plot_positions_single_graph(data_no_interaction, data_nodamping, data_damped)

    # plot_positions(data_no_interaction, data_nodamping, data_damped)
    # plot_velocities(data_no_interaction, data_nodamping, data_damped)

    # plot_all_forces_comparison(data_no_interaction, data_nodamping, data_damped)

    # plot_forces(data_no_interaction, data_nodamping, data_damped)
