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

        velocity = dynamics(agent.position)
        force = controller.compute_control_force(agent, desired_velocity=velocity)

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


def create_environment():
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

    return obstacle_environment

class DataHandler:
    ranges: np.ndarray

    def __attrs_post_init__(self):
        self.measurements = [[]] * self.n_ranges.shape[0]
        
    def add_measurement(self, it_noise, data):
        self.measurements[it_nose].append(data)

    def get_mean(self, it_noise):
        pass

    def get_std(self, it_noise):
        pass
    

def multi_epoch_noisy_velocity(save_figure=False, n_epoch=10, noise_range=np.arrange(0, 1, 11)):
    x_lim = [-2.5, 3]
    y_lim = [-1.3, 2.1]

    obstacle_environment = create_three_body_container()

    initial_dynamics = LinearSystem(
        attractor_position=np.array([2.5, 0.5]),
        maximum_velocity=2,
        distance_decrease=0.5,
    )



if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    # plot_passivity_comparison()
    plot_multiple_avoider(save_figure=True)
    # analysis_point()
