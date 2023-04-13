import matplotlib.pyplot as plt

import numpy as np
from math import pi

# from librairy of lukas : dynamic_obstacle_avoidance
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

# from librairy of lukas : vartools
from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator


class DynamicalSystemAnimation(Animator):
    # class variable
    dim = 2

    def setup(
        self,
        initial_dynamics,
        obstacle_environment,
        start_position=np.array([-2.5, 0.5]),
        x_lim=[-1.5, 2],
        y_lim=[-0.5, 2.5],
    ):
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.obstacle_environment = obstacle_environment
        self.initial_dynamics = initial_dynamics

        self.dynamic_avoider = ModulationAvoider(
            initial_dynamics=self.initial_dynamics,
            obstacle_environment=self.obstacle_environment,
        )

        self.position_list = np.zeros((self.dim, self.it_max + 1))
        self.position_list[:, 0] = start_position

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def update_step(self, ii: int) -> None:
        print(f"iter : {ii}")

        # CALCULATION
        # ici diférent de lukas : moi ii et ii + 1 (lui ii -1 et ii)

        velocity = self.dynamic_avoider.evaluate(self.position_list[:, ii])
        self.position_list[:, ii + 1] = (
            velocity * self.dt_simulation + self.position_list[:, ii]
        )

        # Update obstacles
        self.obstacle_environment.do_velocity_step(delta_time=self.dt_simulation)

        # CLEARING
        self.ax.clear()

        # PLOTTING
        # past trajectory
        self.ax.plot(
            self.position_list[0, :ii], self.position_list[1, :ii], ":", color="#135e08"
        )
        # actual position
        self.ax.plot(
            self.position_list[0, ii],
            self.position_list[1, ii],
            "o",
            color="#135e08",
            markersize=12,
        )
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

        # atractor position
        self.ax.plot(
            self.initial_dynamics.attractor_position[0],
            self.initial_dynamics.attractor_position[1],
            "k*",
            markersize=8,
        )

        # obstacles positions
        plot_obstacles(
            ax=self.ax,
            obstacle_container=self.obstacle_environment,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            showLabel=False,
        )

        self.ax.grid(True)
        self.ax.set_aspect("equal", adjustable="box")

        pass


def run_obs_avoidance():
    # setup of environment
    obstacle_environment = ObstacleContainer()

    obstacle_environment.append(
        Cuboid(
            axes_length=[0.4, 1.3],
            center_position=np.array([0.0, 0.25]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.2,
            # orientation=10 * pi / 180,
            linear_velocity=np.array([-1.0, 0.0]),
            tail_effect=False,
            # repulsion_coeff=1.4,
        )
    )
    obstacle_environment.append(
        Cuboid(
            axes_length=[1.3, 0.4],
            center_position=np.array([-1.0, -0.5]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.2,
            # orientation=10 * pi / 180,
            linear_velocity=np.array([0.0, 1.0]),
            tail_effect=False,
            # repulsion_coeff=1.4,
        )
    )

    # setup of dynamics
    initial_dynamics = LinearSystem(
        attractor_position=np.array([2.0, 0.0]),
        maximum_velocity=2,
        distance_decrease=0.3,
    )

    # setup of animator
    my_animation = DynamicalSystemAnimation(
        dt_simulation=0.05,
        dt_sleep=0.01,
    )

    my_animation.setup(
        initial_dynamics,
        obstacle_environment,
        x_lim=[-3, 3],
        y_lim=[-2.1, 2.1],
    )

    my_animation.run(save_animation=False)


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    run_obs_avoidance()
