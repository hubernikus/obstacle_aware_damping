import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

#from librairy of lukas : vartools
from vartools.dynamical_systems import LinearSystem

#from librairy of lukas : dynamic_obstacle_avoidance
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider

#from my librairies
from librairies.robot import Robot
from librairies.controller import RegulationController, TrackingController
from librairies.robot_animation import CotrolledRobotAnimation

from librairies.magic_numbers_and_enums import TypeOfDMatrix as TypeD

#TODO 
#add magic number for G = np.array([0.0, 0.0]), dim, dt_sim

def run_control_robot():
    dt_simulation = 0.01

    #initial condition
    x_init = np.array([0.3, 0.4])
    xdot_init = np.array([0.0, 0.0])

    #setup atractor 
    attractor_position = np.array([2.0, 0.0])

    #setup of obstacles
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Cuboid(
            axes_length=[0.4, 0.4],
            center_position=np.array([1.0, 0.25]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.1,
            # orientation=10 * pi / 180,
            #linear_velocity = np.array([0.0, 1.0]),
            tail_effect=False,
            # repulsion_coeff=1.4,
        )
    )
    # obstacle_environment.append(
    #     Cuboid(
    #         axes_length=[0.4, 0.4],
    #         center_position=np.array([0.7, -0.1]),
    #         # center_position=np.array([0.9, 0.25]),
    #         margin_absolut=0.1,
    #         # orientation=10 * pi / 180,
    #         #linear_velocity = np.array([0.0, 0.5]),
    #         tail_effect=False,
    #         # repulsion_coeff=1.4,
    #     )
    # )

    #setup of dynamical system
    initial_dynamics = LinearSystem(
        attractor_position = attractor_position,
        maximum_velocity=3,
        distance_decrease=0.3,
    )

    #setup of compliance matrix D, not used anymore
    D = 10*np.eye(2) #damping matrix
    #D[1,1] = 1        #less damped in y

    #setup of compliance values
    lambda_DS=100.0 #must not be > 200 (num error, patch dt smaller)
    lambda_obs=20.0

    ### ROBOT 1 : tau_c = 0, no command ###
    robot_not_controlled = Robot(
        x = x_init, 
        xdot = xdot_init, 
        dt = dt_simulation,
    ) 

    ### ROBOT 2 : tau_c regulates robot to origin ###
    #--> no more suported
    robot_regulated = Robot(
        x = x_init, 
        xdot = xdot_init, 
        dt = dt_simulation,
        controller = RegulationController(
            D=D,
        ),
    )

    ### ROBOT 3 : controlled via DS ###
    robot_tracked = Robot(
        x = x_init, 
        xdot = xdot_init, 
        dt = dt_simulation,
        controller = TrackingController(
            D=D,
            dynamic_avoider = ModulationAvoider(
                initial_dynamics=initial_dynamics,
                obstacle_environment=obstacle_environment,
            ),
            lambda_DS=lambda_DS,
            lambda_obs=lambda_obs,
            type_of_D_matrix = TypeD.DS_FOLLOWING, # TypeD.DS_FOLLOWING or TypeD.OBS_PASSIVITY
        ),
    )

    #setup of animator
    my_animation = CotrolledRobotAnimation(
        it_max = 300, #longer animation, default : 100
        dt_simulation = dt_simulation,
        dt_sleep=0.001,
    )

    my_animation.setup(
        #robot = robot_not_controlled,
        #robot = robot_regulated,
        robot = robot_tracked,
        obstacle_environment = obstacle_environment,
        x_lim = [-3, 3],
        y_lim = [-2.1, 2.1],
        draw_ideal_traj = True,
    )

    my_animation.run(save_animation=False)



if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    run_control_robot()
