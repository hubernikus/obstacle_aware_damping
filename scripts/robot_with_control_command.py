import numpy as np
import matplotlib.pyplot as plt

#from abc import ABC, abstractmethod

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
import librairies.magic_numbers_and_enums as mn

#just for plotting : global var, remoove when no bug
from librairies.robot_animation import s_list

def run_control_robot():
    dt_simulation = 0.01

    #initial condition
    x_init = np.array([-2.0, 0.3]) 
    xdot_init = np.array([0.0, 0.0])

    #setup atractor 
    attractor_position = np.array([2.0, 0.0])

    #setup of obstacles
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Cuboid(
            axes_length=[0.6, 0.6],
            center_position=np.array([0.0, 0.0]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.15,
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
    lambda_DS = 200.0 #must not be > 200 (num error, patch dt smaller)
    lambda_perp = 20.0
    lambda_obs_scaling = 20.0 #scaling factor
    if lambda_DS > mn.LAMBDA_MAX or lambda_perp > mn.LAMBDA_MAX or lambda_obs_scaling > mn.LAMBDA_MAX:
        raise ValueError(f"lambda must be smaller than {mn.LAMBDA_MAX}")


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
            dynamic_avoider = ModulationAvoider(
                initial_dynamics=initial_dynamics,
                obstacle_environment=obstacle_environment,
            ),
            lambda_DS=lambda_DS,
            lambda_perp=lambda_perp,
            lambda_obs_scaling = lambda_obs_scaling,
            type_of_D_matrix = TypeD.BOTH, # TypeD.DS_FOLLOWING or TypeD.OBS_PASSIVITY or TypeD.BOTH
        ),
    )

    #setup of animator
    my_animation = CotrolledRobotAnimation(
        it_max = 300, #longer animation, default : 100
        dt_simulation = dt_simulation,
        dt_sleep = dt_simulation,
    )

    my_animation.setup(
        robot = robot_tracked,
        obstacle_environment = obstacle_environment,
        x_lim = [-3, 3],
        y_lim = [-2.1, 2.1],
        draw_ideal_traj = True, 
        draw_qolo = True
    )

    my_animation.run(save_animation=False)


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    run_control_robot()

    #just for plotting s tank
    fig, ax = plt.subplots()
    x = np.linspace(0, len(s_list), len(s_list))
    plt.plot(x, s_list)
    plt.show()
    pass #add breakpoint here if want to plot s
