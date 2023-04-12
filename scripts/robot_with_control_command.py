import numpy as np
import matplotlib.pyplot as plt

# from librairy of lukas : vartools
from vartools.dynamical_systems import LinearSystem

# from librairy of lukas : dynamic_obstacle_avoidance
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider

# from my passive_control.
from passive_control.robot import Robot
from passive_control.controller import RegulationController, TrackingController
from passive_control.robot_animation import CotrolledRobotAnimation

from passive_control.magic_numbers_and_enums import TypeOfDMatrix as TypeD
from passive_control.magic_numbers_and_enums import Approach
import passive_control.magic_numbers_and_enums as mn

# just for plotting : global var, remoove when no bug
from passive_control.robot_animation import s_list


def run_control_robot():
    """
    Setup all the environment + robot + controller in 2D
    then run it
    """
    DIM = 2
    dt_simulation = (
        0.01  # attention bug when too small (bc plt takes too much time :( ))
    )

    # initial condition
    x_init = np.array([-2.2, 0.5])  # 0.3
    # x_init = np.array([0.1, 2.5])
    xdot_init = np.array([0.0, 0.0])

    # setup atractor
    attractor_position = np.array([2.0, 0.0])
    # attractor_position = np.array([0.0, -2.5])

    # setup of obstacles
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
    obstacle_environment.append(
        Cuboid(
            axes_length=[0.5, 0.5],
            center_position=np.array([0.0, 1.5]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.15,
            # orientation=10 * pi / 180,
            # linear_velocity = np.array([0.0, 0.5]),
            tail_effect=False,
            # repulsion_coeff=1.4,
        )
    )
    obstacle_environment.append(
        Cuboid(
            axes_length=[0.3, 0.3],
            center_position=np.array([1.0, 0.2]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.15,
            # orientation=10 * pi / 180,
            # linear_velocity = np.array([0.0, 0.5]),
            tail_effect=False,
            # repulsion_coeff=1.4,
        )
    )

    # SETUP B - used for pres
    # obstacle_environment.append(
    #     Cuboid(
    #         axes_length=[0.4, 1.8],
    #         center_position=np.array([-0.5, -0.5]),
    #         # center_position=np.array([0.9, 0.25]),
    #         margin_absolut=0.15,
    #         # orientation=10 * pi / 180,
    #         #linear_velocity = np.array([0.0, 1.0]),
    #         tail_effect=False,
    #         # repulsion_coeff=1.4,
    #     )
    # )

    # obstacle_environment.append(
    #     Cuboid(
    #         axes_length=[0.4, 1.8],
    #         center_position=np.array([0.8, 0.8]),
    #         # center_position=np.array([0.9, 0.25]),
    #         margin_absolut=0.15,
    #         # orientation=10 * pi / 180,
    #         #linear_velocity = np.array([0.0, 1.0]),
    #         tail_effect=False,
    #         # repulsion_coeff=1.4,
    #     )
    # )

    # SETUP C
    # obstacle_environment.append(
    #     Cuboid(
    #         axes_length=[0.6, 2.6],
    #         center_position=np.array([-0.3, 0.0]),
    #         # center_position=np.array([0.9, 0.25]),
    #         margin_absolut=0.15,
    #         # orientation=10 * pi / 180,
    #         #linear_velocity = np.array([0.0, 1.0]),
    #         tail_effect=False,
    #         # repulsion_coeff=1.4,
    #     )
    # )

    # SETUP D
    # obstacle_environment.append(
    #     Cuboid(
    #         axes_length=[1.6, 0.6],
    #         center_position=np.array([-0.3, 1.5]),
    #         # center_position=np.array([0.9, 0.25]),
    #         margin_absolut=0.15,
    #         # orientation=10 * pi / 180,
    #         #linear_velocity = np.array([0.0, 1.0]),
    #         tail_effect=False,
    #         # repulsion_coeff=1.4,
    #     )
    # )
    # obstacle_environment.append(
    #     Cuboid(
    #         axes_length=[3.0, 2.0],
    #         center_position=np.array([1.5, -0.8]),
    #         # center_position=np.array([0.9, 0.25]),
    #         margin_absolut=0.15,
    #         # orientation=10 * pi / 180,
    #         #linear_velocity = np.array([0.0, 1.0]),
    #         tail_effect=False,
    #         # repulsion_coeff=1.4,
    #     )
    # )
    # obstacle_environment.append(
    #     Cuboid(
    #         axes_length=[2.0, 2.0],
    #         center_position=np.array([-1.6, -0.8]),
    #         # center_position=np.array([0.9, 0.25]),
    #         margin_absolut=0.15,
    #         # orientation=10 * pi / 180,
    #         #linear_velocity = np.array([0.0, 1.0]),
    #         tail_effect=False,
    #         # repulsion_coeff=1.4,
    #     )
    # )

    # setup of dynamical system
    initial_dynamics = LinearSystem(
        attractor_position=attractor_position,
        maximum_velocity=3,
        distance_decrease=0.5,  # if too small, could lead to instable around atractor
    )

    # setup of compliance values
    lambda_DS = 100.0  # must not be > 200 (num error, patch dt smaller) -> 200 makes xdot varies too much
    # bc in tau_c compute -D@xdot becomes too big  + much more stable at atrat.
    lambda_perp = 20.0
    lambda_obs = mn.LAMBDA_MAX
    if (
        lambda_DS > mn.LAMBDA_MAX
        or lambda_perp > mn.LAMBDA_MAX
        or lambda_obs > mn.LAMBDA_MAX
    ):
        raise ValueError(f"lambda must be smaller than {mn.LAMBDA_MAX}")

    ### ROBOT 1 : tau_c regulates robot to origin ###
    # --> no more suported
    D = 10 * np.eye(2)  # damping matrix
    robot_regulated = Robot(
        x=x_init,
        xdot=xdot_init,
        dt=dt_simulation,
        controller=RegulationController(
            DIM=DIM,
            D=D,
        ),
    )

    ### ROBOT 2 : controlled via DS ###
    robot_tracked = Robot(
        DIM=DIM,
        x=x_init,
        xdot=xdot_init,
        dt=dt_simulation,
        noisy=False,
        controller=TrackingController(
            dynamic_avoider=ModulationAvoider(
                initial_dynamics=initial_dynamics,
                obstacle_environment=obstacle_environment,
            ),
            DIM=DIM,
            lambda_DS=lambda_DS,
            lambda_perp=lambda_perp,
            lambda_obs=lambda_obs,
            type_of_D_matrix=TypeD.BOTH,  # TypeD.DS_FOLLOWING or TypeD.OBS_PASSIVITY or TypeD.BOTH
            approach=Approach.WEIGHT_DS_OBS_MAT_V2,  # Aproach.ORTHO_BASIS or NON_ORTHO_BASIS or WEIGHT_DS_OBS_MAT
            with_E_storage=False,
        ),
    )

    # setup of animator
    my_animation = CotrolledRobotAnimation(
        it_max=300,  # longer animation
        dt_simulation=dt_simulation,
        dt_sleep=dt_simulation,
    )

    my_animation.setup(
        robot=robot_tracked,
        obstacle_environment=obstacle_environment,
        DIM=2,
        x_lim=[-2.5, 3],
        y_lim=[-1.3, 2.1],
        # x_lim = [-3,3],
        # y_lim = [-2.1,2.1],
        # x_lim = [-1.5,1.5], #[-1.5, 2.2], #-3
        # y_lim = [-3,3], #[-1.2, 2.1], #-2.1
        draw_ideal_traj=True,
        draw_qolo=True,
        rotate_qolo=True,
    )

    my_animation.run(save_animation=False)


def run_control_robot_3D():
    """
    Setup all the environment + robot + controller in 3D
    then run it
    """
    DIM = 3
    dt_simulation = (
        0.01  # attention bug when too small (bc plt takes too much time :( ))
    )

    # initial condition
    x_init = np.array([-2.0, 0.0, 0.0])  # np.array([-2.0, 0.3, 0.0])
    xdot_init = np.array([0.0, 0.0, 0.0])

    # setup atractor
    attractor_position = np.array([2.0, 0.0, 0.0])

    # setup of obstacles
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Cuboid(
            axes_length=[0.6, 0.6, 0.6],
            center_position=np.array([0.1, -0.1, 0.1]),  # z 0.1
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.15,
            # orientation=10 * pi / 180,
            linear_velocity=np.array([0.0, 0.0, 0.0]),  # necessary to specify in 3D
            tail_effect=False,
            # repulsion_coeff=1.4,
        )
    )
    obstacle_environment.append(
        Cuboid(
            axes_length=[0.5, 0.3, 1.0],
            center_position=np.array([1.0, -0.3, 1.0]),  # BUG x = z
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.15,
            # orientation=10 * pi / 180,
            linear_velocity=np.array(
                [0.0, 0.0, 0.0]
            ),  # necessary to specify in 3D (even when 0.0)
            tail_effect=False,
            # repulsion_coeff=1.4,
        )
    )

    # setup of dynamical system
    initial_dynamics = LinearSystem(
        attractor_position=attractor_position,
        dimension=3,
        maximum_velocity=3,
        distance_decrease=0.5,  # if too small, could lead to instable around atractor
    )

    # setup of compliance values
    lambda_DS = 100.0  # must not be > 200 (num error, patch dt smaller) -> 200 makes xdot varies too much
    # bc in tau_c compute -D@xdot becomes too big  + much more stable at atrat.
    lambda_perp = 20.0
    lambda_obs = mn.LAMBDA_MAX
    if (
        lambda_DS > mn.LAMBDA_MAX
        or lambda_perp > mn.LAMBDA_MAX
        or lambda_obs > mn.LAMBDA_MAX
    ):
        raise ValueError(f"lambda must be smaller than {mn.LAMBDA_MAX}")

    ### ROBOT 3 : controlled via DS ###
    robot_tracked = Robot(
        DIM=DIM,
        x=x_init,
        xdot=xdot_init,
        dt=dt_simulation,
        noisy=False,
        controller=TrackingController(
            dynamic_avoider=ModulationAvoider(
                initial_dynamics=initial_dynamics,
                obstacle_environment=obstacle_environment,
            ),
            DIM=DIM,
            lambda_DS=lambda_DS,
            lambda_perp=lambda_perp,
            lambda_obs=lambda_obs,
            type_of_D_matrix=TypeD.BOTH,  # TypeD.DS_FOLLOWING or TypeD.OBS_PASSIVITY or TypeD.BOTH
            approach=Approach.WEIGHT_DS_OBS_MAT_V2,  # Aproach.ORTHO_BASIS or NON_ORTHO_BASIS or WEIGHT_DS_OBS_MAT
            with_E_storage=False,
        ),
    )

    # setup of animator
    my_animation = CotrolledRobotAnimation(
        it_max=300,  # longer animation
        dt_simulation=dt_simulation,
        dt_sleep=dt_simulation,
    )

    my_animation.setup(
        robot=robot_tracked,
        obstacle_environment=obstacle_environment,
        DIM=3,
        x_lim=[-3, 3],
        y_lim=[-2.1, 2.1],
        draw_ideal_traj=False,
        draw_qolo=True,
        rotate_qolo=True,
    )

    my_animation.run(save_animation=False)


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    run_control_robot()
    # run_control_robot_3D()

    # just for plotting s tank, remoove when done, or implemment better
    plt.close("all")
    plt.ion()
    fig = plt.figure()
    x = np.linspace(0, len(s_list), len(s_list))
    plt.plot(x, s_list)
    plt.axhline(0, color="r")
    plt.axhline(mn.S_MAX, color="r")
    plt.xlabel("Iteration")
    plt.ylabel("Energy tank level")
    plt.title("Energy tank level at each iteration")
    # plt.show(block = True)
