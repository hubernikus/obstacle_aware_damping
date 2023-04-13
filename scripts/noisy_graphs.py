import numpy as np
import matplotlib.pyplot as plt

# from librairy of lukas : vartools
from vartools.dynamical_systems import LinearSystem

# from librairy of lukas : dynamic_obstacle_avoidance
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider

# from my librairies
from passive_control.robot import Robot
from passive_control.controller import RegulationController, TrackingController
from passive_control.robot_animation import CotrolledRobotAnimation

from passive_control.magic_numbers_and_enums import TypeOfDMatrix as TypeD
from passive_control.magic_numbers_and_enums import Approach
import passive_control.magic_numbers_and_enums as mn

# just for plotting : global var, remoove when no bug
from passive_control.robot_animation import s_list


def run_control_robot(noise_pos=0.0, noise_vel=0.0, is_obs_aw=True):

    mn.NOISE_STD_POS = noise_pos
    mn.NOISE_STD_VEL = noise_vel

    if is_obs_aw:
        type_of_contr = TypeD.BOTH
    else:
        type_of_contr = TypeD.DS_FOLLOWING

    dt_simulation = (
        0.01  # attention bug when too small (bc plt takes too much time :( ))
    )

    # initial condition
    x_init = np.array([-2.0, 0.3])  # 0.3
    xdot_init = np.array([0.0, 0.0])

    # setup atractor
    attractor_position = np.array([2.0, 0.0])

    # setup of obstacles
    obstacle_environment = ObstacleContainer()

    # obstacle_environment.append(
    #     Cuboid(
    #         axes_length=[0.6, 0.6],
    #         center_position=np.array([0.0, 0.0]),
    #         # center_position=np.array([0.9, 0.25]),
    #         margin_absolut=0.15,
    #         # orientation=10 * pi / 180,
    #         #linear_velocity = np.array([0.0, 1.0]),
    #         tail_effect=False,
    #         # repulsion_coeff=1.4,
    #     )
    # )

    obstacle_environment.append(
        Cuboid(
            axes_length=[0.4, 3.8],
            center_position=np.array([-0.5, -1.5]),
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
            axes_length=[0.4, 3.8],
            center_position=np.array([0.8, 1.8]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.15,
            # orientation=10 * pi / 180,
            # linear_velocity = np.array([0.0, 1.0]),
            tail_effect=False,
            # repulsion_coeff=1.4,
        )
    )

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

    ### ROBOT 3 : controlled via DS ###
    robot_tracked = Robot(
        x=x_init,
        xdot=xdot_init,
        dt=dt_simulation,
        noisy=True,
        controller=TrackingController(
            dynamic_avoider=ModulationAvoider(
                initial_dynamics=initial_dynamics,
                obstacle_environment=obstacle_environment,
            ),
            lambda_DS=lambda_DS,
            lambda_perp=lambda_perp,
            lambda_obs=lambda_obs,
            type_of_D_matrix=type_of_contr,  # TypeD.DS_FOLLOWING or TypeD.OBS_PASSIVITY or TypeD.BOTH
            approach=Approach.ORTHO_BASIS,
            with_E_storage=False,
        ),
    )

    # setup of animator
    my_animation = CotrolledRobotAnimation(
        it_max=200,  # longer animation
        dt_simulation=dt_simulation,
        dt_sleep=dt_simulation,
    )

    my_animation.setup(
        robot=robot_tracked,
        obstacle_environment=obstacle_environment,
        x_lim=[-3, 3],
        y_lim=[-2.1, 2.1],  # [-2.1, 2.1],
        draw_ideal_traj=True,
        draw_qolo=True,
        rotate_qolo=True,
    )

    my_animation.run(save_animation=False)

    return my_animation.get_d_min()


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    n = 8  # 8
    epochs = 8  # 8
    d_min_tab_obs_aw = np.zeros((n, epochs))
    d_min_tab_kronan = np.zeros((n, epochs))

    noise_level = np.linspace(0.0, 7.0, n)  # for velocity
    # noise_level = np.linspace(0.0,0.7,n) #for position

    # d_min_tab_obs_aw[0,0] = run_control_robot(noise_pos=0.0, noise_vel=4.0, is_obs_aw = True)

    for i, noise in enumerate(noise_level):
        print("noise :", noise)
        for e in range(epochs):
            print("epoch :", e, "noise std : ", noise)
            print("aware")
            dis_obs = run_control_robot(noise_pos=0.0, noise_vel=noise, is_obs_aw=True)
            # dis_obs = dis_obs if dis_obs>0 else 0
            d_min_tab_obs_aw[i, e] = dis_obs
            plt.close("all")
            print("kronan")
            dist_kro = run_control_robot(
                noise_pos=0.0, noise_vel=noise, is_obs_aw=False
            )
            # dist_kro = dist_kro if dist_kro>0 else 0
            d_min_tab_kronan[i, e] = dist_kro
            plt.close("all")
            if i == 0:
                d_min_tab_obs_aw[0, :] = d_min_tab_obs_aw[0, 0]
                d_min_tab_kronan[0, :] = d_min_tab_kronan[0, 0]
                break

    np.save("noisy_file\d_min_tab_obs_aw_temp.npy", d_min_tab_obs_aw)
    np.save("noisy_file\d_min_tab_kronan_temp.npy", d_min_tab_kronan)

    mean_obs = d_min_tab_obs_aw.mean(axis=1)
    std_obs = d_min_tab_obs_aw.std(axis=1)
    mean_kro = d_min_tab_kronan.mean(axis=1)
    std_kro = d_min_tab_kronan.std(axis=1)

    fig = plt.figure()
    # plt.errorbar(noise_level, mean, yerr=std)
    # obstacle aware plot
    plt.fill_between(
        noise_level, mean_obs + std_obs, mean_obs - std_obs, alpha=0.3, color="b"
    )
    plt.plot(noise_level, mean_obs, color="b", label="Obstacle aware passive control")
    # original kronander plot
    plt.fill_between(
        noise_level, mean_kro + std_kro, mean_kro - std_kro, alpha=0.3, color="r"
    )
    plt.plot(noise_level, mean_kro, color="r", label="Traditional passive control")
    plt.axhline(y=0.0, color="k", linestyle="-")
    # plt.plot(noise_level, np.zeros_like(noise_level), "k")
    # plt.title(f"Effect of velocity measurement noise over {epochs} epochs")
    plt.ylabel("Closest distance to obstacle during simulation [m]")
    plt.xlabel("Standard deviation of velocity measurement noise [m/s]")
    plt.legend(loc="upper right", prop={"size": 12})
    fig.show()
    plt.show()
    plt.savefig("vel_noise.png")

    # with clip ar 0
    d_min_tab_obs_aw[d_min_tab_obs_aw <= 0] = 0.0
    d_min_tab_kronan[d_min_tab_kronan <= 0] = 0.0
    mean_obs = d_min_tab_obs_aw.mean(axis=1)
    std_obs = d_min_tab_obs_aw.std(axis=1)
    mean_kro = d_min_tab_kronan.mean(axis=1)
    std_kro = d_min_tab_kronan.std(axis=1)

    fig2 = plt.figure()
    # plt.errorbar(noise_level, mean, yerr=std)
    # obstacle aware plot
    plt.fill_between(
        noise_level, mean_obs + std_obs, mean_obs - std_obs, alpha=0.3, color="b"
    )
    plt.plot(noise_level, mean_obs, color="b", label="Obstacle aware passive control")
    # original kronander plot
    plt.fill_between(
        noise_level, mean_kro + std_kro, mean_kro - std_kro, alpha=0.3, color="r"
    )
    plt.plot(noise_level, mean_kro, color="r", label="Traditional passive control")
    plt.axhline(y=0.0, color="k", linestyle="-")
    # plt.plot(noise_level, np.zeros_like(noise_level), "k")
    plt.title(f"Effect of velocity measurement noise over {epochs} epochs")
    plt.ylabel("Closest distance to obstacle during simulation [m]")
    plt.xlabel("Standard deviation of velocity measurement noise [m/s]")
    plt.legend(loc="upper right")
    fig2.show()
    plt.show()
    plt.savefig("vel_noise_cliped.png")

    plt.pause(100)
    pass

    # just for plotting s tank, remoove when done, or implemment better
    # fig, ax = plt.subplots()
    # fig = plt.figure()
    # x = np.linspace(0, len(s_list), len(s_list))
    # plt.plot(x, s_list)
    # fig.show()
    # plt.show()
    # pass #add breakpoint here if want to plot s
