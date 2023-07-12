from attrs import define

from pathlib import Path
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def plot_positions(dataframe):
    fig, ax = plt.subplots(figsize=(5, 4))

    dimension = 3
    positions = np.zeros((len(dataframe["pos"]), dimension))
    for ii in range(len(dataframe["pos"])):
        positions[ii, :] = dataframe["pos"][ii]

    axes_names = ["x", "y", "z"]
    for ii, ax_label in enumerate(axes_names):
        ax.plot(positions[:, ii], label=ax_label)

    # distances = np.zeros((len(dataframe["dists"])))
    # for ii in range(len(dataframe["dists"])):
    #     distances[ii] = dataframe["dists"][ii]
    # ax.plot(distances, "--", label="Minimum distance")

    ax.legend()
    # fig = df_damped.plot(y="pos", use_index=True, label="with damping")

    ax.set_xlabel("Step")
    ax.set_ylabel("Position [m]")

    return fig, ax


def import_dataframe(
    filename,
    data_path=Path("/home/lukas/Code/obstacle_aware_damping") / "recordings",
    folder="new2_lukas",
):
    df_raw = pd.read_csv(data_path / folder / filename)
    return clean_all(df_raw)


def extract_sequences_intervals(dataframe, y_high=0.2, visualize=False):
    dimension = 3
    positions = np.zeros((len(dataframe["pos"]), dimension))
    for ii in range(len(dataframe["pos"])):
        positions[ii, :] = dataframe["pos"][ii]

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

        intervals.append(np.arange(interval[ind_max], interval[ind_min]))

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


def plot_y_position_intervals(datahandler, ax=None, n_max=5):
    dimension = 3
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 2.8))
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

    for ii, interval in enumerate(datahandler.intervals):
        if ii >= n_max:
            # For visibility only plot partial
            break

        ax.plot(
            positions[interval, 1],
            positions[interval, 2],
            color=datahandler.color,
            linestyle=datahandler.linestyles[ii],
            linewidth=2.0,
        )

    # plt.savefig("figures/" + figName + ".png", bbox_inches="tight")

    return fig, ax


def plot_distance_obstacle(datahandler, ax=None, n_max=5):
    dimension = 3
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 2.8))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Distance [m]")

        ax.set_xlim([0, 7.2])
        ax.set_ylim([-1.5, 5.0])

        x_vals = ax.get_xlim()

        ax.plot(x_vals, [0, 0], ":", color="#8b0000", linewidth=2.0)
        ax.grid("on")

    else:
        fig = None

    distances = np.zeros((len(datahandler.dataframe["dists"]), dimension))
    for jj in range(len(datahandler.dataframe["dists"])):
        distances[jj, :] = datahandler.dataframe["dists"][jj]

    for ii, interval in enumerate(datahandler.intervals):
        if ii >= n_max:
            # For visibility only plot partial
            break

        ax.plot(
            datahandler.time_step * np.arange(interval.shape[0]),
            distances[interval],
            color=datahandler.color,
            linestyle=datahandler.linestyles[ii],
            linewidth=2.0,
        )

    # plt.savefig("figures/" + figName + ".png", bbox_inches="tight")

    return fig, ax


def import_dataframes():
    dataframe = import_dataframe("single_traverse_no_interaction.csv")
    intervals = extract_sequences_intervals(dataframe, visualize=False)
    data_no_interaction = DataStorer(dataframe, intervals, color="gray")

    dataframe = import_dataframe("single_traverse_nodamping_1.csv")
    intervals = extract_sequences_intervals(dataframe, visualize=False)
    data_nodamping = DataStorer(dataframe, intervals, color="blue")

    dataframe = import_dataframe("single_traverse_damped_1.csv")
    intervals = extract_sequences_intervals(dataframe, visualize=False)
    data_damped = DataStorer(dataframe, intervals, color="green")

    return (data_no_interaction, data_nodamping, data_damped)


@define(slots=False)
class DataStorer:
    dataframe: object
    intervals: object
    color: str

    linestyles: list = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))]
    time_step: float = 1.0 / 100


@define(slots=False)
class DummyObstacle:
    distance: float
    normal: np.ndarray

    def get_gamma(self, *args, **kwargs):
        return distance + 1

    def get_normal_direction(self, *args, **kwargs):
        return normals


def plot_and_compute_force(datahandler, controller: Controller, ax):
    dimension = 3

    forces = np.zeros((len(datahandler.dataframe["des_vel"]), dimension))
    normals = np.zeros((len(datahandler.dataframe["des_vel"]), dimension))
    for jj in range(len(datahandler.dataframe["des_vel"])):
        velocity_desired = datahandler.dataframe["des_vel"]
        velocity = datahandler.dataframe["vel"]
        position = datahandler.dataframe["pos"]

        if hasattr(controller, "environment"):
            # Recreate environment to compute the correct force
            controller.environment[0].normal = position = datahandler.dataframe[
                "normals"
            ]
            controller.environment[0].distance = position = datahandler.dataframe[
                "dists"
            ]

        forces[:, jj] = controller.compute_force(
            velocity=velocity,
            desired_velocity=velocity_desired,
            position=position,
        )

    axes_names = ["Force x ", "Force y", "Force z"]
    colors = ["red", "green", "blue"]
    for ii, ax_label in enumerate(axes_names):
        for interval in invertvals:
            ax.plot(forces[interval, ii], label=ax_label, color=colors[ii])
            ax_label = None


def main_force_evaluation(
    data_no_interaction, data_nodamping, data_damped, save_figure=False
):
    dimension = 3

    lambda_DS = 100.0
    lambda_perp = 20.0
    lambda_obs = mn.LAMBDA_MAX

    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance [m]")

    ax.set_xlim([0, 7.2])
    ax.set_ylim([-1.5, 5.0])

    x_vals = ax.get_xlim()

    ax.plot(x_vals, [0, 0], ":", color="#8b0000", linewidth=2.0)
    ax.grid("on")

    # No interaction
    controller = PassiveDynamicsController(
        lambda_dynamics=lambda_DS, lambda_remaining=lambda_perp
    )
    plot_and_compute_force(data_no_interaction, controller, ax)


def main(data_no_interaction, data_nodamping, data_damped, save_figure=False):
    fig, ax = plot_y_position_intervals(data_no_interaction, n_max=1)
    plot_y_position_intervals(data_nodamping, ax=ax)
    plot_y_position_intervals(data_damped, ax=ax)

    if save_figure:
        figname = "robot_arm_trajectory_xy"
        plt.savefig("figures/" + figname + figtype, bbox_inches="tight")

    fig, ax = plot_distance_obstacle(data_no_interaction, n_max=1)
    plot_distance_obstacle(data_nodamping, ax=ax)
    plot_distance_obstacle(data_damped, ax=ax)
    if save_figure:
        figname = "robot_arm_trajectory_distance"
        plt.savefig("figures/" + figname + figtype, bbox_inches="tight")

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


if (__name__) == "__main__":
    figtype = ".pdf"

    # Only import data once... -> keep in local workspace after
    reimport_data = False
    if reimport_data or "data_no_interaction" not in locals():
        print("Importing data-sequences.")
        (data_no_interaction, data_nodamping, data_damped) = import_dataframes()

    # plt.close("all")
    plt.ion()
    # main(data_no_interaction, data_nodamping, data_damped)
    main_force_evaluation(data_no_interaction, data_nodamping, data_damped)
