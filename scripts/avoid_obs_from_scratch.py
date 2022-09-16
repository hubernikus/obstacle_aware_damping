
import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
from math import pi

#from librairy of lukas : dynamic_obstacle_avoidance
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid

#from librairy of lukas : vartools
from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator


class DynamicalSystemAnimation(Animator):
    #class variable
    dim = 2

    def setup(
        self,
        initial_dynamics,
        obstacle_environment,
        start_position=np.array([1, 1]),
        x_lim=[-1.5, 2],
        y_lim=[-0.5, 2.5],
    ):
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.obstacle_environment = obstacle_environment
        self.initial_dynamics = initial_dynamics

        self.position_list = np.zeros((self.dim, self.it_max))
        self.position_list[:, 0] = start_position

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
    
    def update_step(self, ii: int) -> None:
        print(f"iter : {ii}")

        #CALCULATION
        #ici diférent de lukas : moi ii et ii + 1 (lui ii -1 et ii)
        velocity = self.initial_dynamics.evaluate(self.position_list[:,ii])
        self.position_list[:, ii + 1] = (
            velocity * self.dt_simulation + self.position_list[:, ii]
        )

        #CLEARING
        self.ax.clear()

        #PLOTTING
        #past trajectory
        self.ax.plot(
            self.position_list[0, :ii], self.position_list[1, :ii], ":", color="#135e08"
        )
        #actual position
        self.ax.plot(
            self.position_list[0, ii],
            self.position_list[1, ii],
            "o",
            color="#135e08",
            markersize=12,
        )
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

        #atractor position
        self.ax.plot(
            self.initial_dynamics.attractor_position[0],
            self.initial_dynamics.attractor_position[1],
            "k*",
            markersize=8,
        )

        #obstacles positions
        #...

        self.ax.grid(True)
        self.ax.set_aspect("equal", adjustable="box")

        pass


def run_obs_avoidance():
    #setup of environment
    obstacle_environment = ObstacleContainer()
    
    obstacle_environment.append(
        Cuboid(
            axes_length=[0.4, 1.3],
            center_position=np.array([2.2, 0.25]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.5,
            orientation=10 * pi / 180,
            tail_effect=False,
            # repulsion_coeff=1.4,
        )
    )
    

    #setup of dynamics
    initial_dynamics = LinearSystem(
        attractor_position=np.array([0.0, 0.0]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

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



    pass


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    run_obs_avoidance()

