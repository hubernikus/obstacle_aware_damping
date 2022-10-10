import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

#from librairy of lukas : vartools
from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

#from librairy of lukas : dynamic_obstacle_avoidance
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from dynamic_obstacle_avoidance.utils import get_orthogonal_basis

#from my librairies
from librairies.robot import Robot
from librairies.controller import RegulationController, TrackingController

#TODO 
#add magic number for G = np.array([0.0, 0.0]), dim...

class CotrolledRobotAnimation(Animator):
    #class variables
    dim = 2

    def setup(
        self,
        #start_position=np.array([-2.5, 0.5]),
        #start_velocity=np.array([0,0]),
        robot:Robot,
        obstacle_environment:ObstacleContainer,
        x_lim=[-1.5, 2],
        y_lim=[-0.5, 2.5],
        disturbance_magn = 200,
        draw_ideal_traj = False
    ):
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.robot = robot
        self.obstacle_environment = obstacle_environment

        self.position_list = np.zeros((self.dim, self.it_max + 1))
        self.position_list[:, 0] = robot.x.reshape((self.dim,))
        self.velocity_list = np.zeros((self.dim, self.it_max + 1))
        self.velocity_list[:, 0] = robot.xdot.reshape((self.dim,))

        self.disturbance_magn = disturbance_magn
        self.disturbance_list = np.empty((self.dim, 0))
        self.disturbance_pos_list = np.empty((self.dim, 0))
        self.new_disturbance = False

        self.position_list_ideal = np.zeros((self.dim, self.it_max + 1))
        self.position_list_ideal[:, 0] = robot.x.reshape((self.dim,))

        self.draw_ideal_traj = draw_ideal_traj

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def update_step(self, ii: int) -> None:
        print(f"iter : {ii + 1}") #actual is i + 1

        #CALCULATION
        self.robot.simulation_step()

        self.position_list[:, ii + 1] = self.robot.x
        self.velocity_list[:, ii + 1] = self.robot.xdot

        #without physic constrains - ideal
        if self.draw_ideal_traj and isinstance(self.robot.controller, TrackingController):
            velocity_ideal = self.robot.controller.dynamic_avoider.evaluate(self.position_list_ideal[:, ii])
            self.position_list_ideal[:, ii + 1] = (
                velocity_ideal * self.dt_simulation + self.position_list_ideal[:, ii]
            )
        
        # Update obstacles
        self.obstacle_environment.do_velocity_step(delta_time=self.dt_simulation)

        #record position of the new disturbance added with key pressed
        if self.new_disturbance:
            #bizarre line, recheck append
            self.disturbance_pos_list = np.append(self.disturbance_pos_list, 
                                                  self.position_list[:,ii].reshape((self.dim, 1)), axis = 1)
            self.new_disturbance = False

        for obs in self.obstacle_environment:
            #obs.get_normals() ##IMPLEMENT HERE##
            pass

        #CLEARING
        self.ax.clear()

        #PLOTTING
        #ideal trajectory - in black
        if self.draw_ideal_traj:
            self.ax.plot(
                self.position_list_ideal[0, :ii + 1], 
                self.position_list_ideal[1, :ii +1], 
                ":", color="#000000"
            )

        #past trajectory - in green
        self.ax.plot(
            self.position_list[0, :ii], self.position_list[1, :ii], ":", color="#135e08"
        )
        #actual position is i + 1 ?
        self.ax.plot(
            self.position_list[0, ii + 1],
            self.position_list[1, ii + 1],
            "o",
            color="#135e08",
            markersize=12,
        )
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

        #atractor position
        atractor = np.array([0.0, 0.0])
        if isinstance(self.robot.controller, TrackingController):
            atractor = self.robot.controller.dynamic_avoider.initial_dynamics.attractor_position
        self.ax.plot(
            atractor[0],
            atractor[1],
            "k*",
            markersize=8,
        )

        #obstacles positions
        plot_obstacles(
            ax=self.ax,
            obstacle_container=self.obstacle_environment,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            showLabel=False,
        )

        #disturbance drawing, bizzare lines, recheck, magic numbers ??
        for  disturbance, disturbance_pos in zip(self.disturbance_list.transpose(),
                                                 self.disturbance_pos_list.transpose()): #transpose ??
            self.ax.arrow(disturbance_pos[0], disturbance_pos[1],
                          disturbance[0]/10., disturbance[1]/10.0,
                          width=self.disturbance_magn/10000.0,
                          color= "r")


    #overwrite key detection of Animator
    #/!\ bug when 2 keys pressed at same time
    def on_press(self, event):
        if event.key.isspace():
            self.pause_toggle()

        elif event.key == "right":
            print("->")
            self.robot.tau_e = np.array([1.0, 0.0])*self.disturbance_magn
            self.disturbance_list = np.append(self.disturbance_list, (np.array([[1.0], [0.0]])), axis=1)
            self.new_disturbance = True
            #TEST 

        elif event.key == "left":
            print("<-")
            self.robot.tau_e = np.array([-1.0, 0.0])*self.disturbance_magn
            self.disturbance_list = np.append(self.disturbance_list, (np.array([[-1.0], [0.0]])), axis=1)
            self.new_disturbance = True

        elif event.key == "up":
            print("^\n|")
            self.robot.tau_e = np.array([0.0, 1.0])*self.disturbance_magn
            self.disturbance_list = np.append(self.disturbance_list, (np.array([[0.0], [1.0]])), axis=1)
            self.new_disturbance = True
    
        elif event.key == "down":
            print("|\nV")
            self.robot.tau_e = np.array([0.0, -1.0])*self.disturbance_magn
            self.disturbance_list = np.append(self.disturbance_list, (np.array([[0.0], [-1.0]])), axis=1)
            self.new_disturbance = True

        elif event.key == "d":
            self.step_forward()

        elif event.key == "a":
            self.step_back()


def run_control_robot():
    dt_simulation = 0.01

    #initial condition
    x_init = np.array([0.6, 0.3])
    xdot_init = np.array([0.0, 0.0])


    # #remove overwrite
    # x_init = np.array([-1.0, 2.0])
    # xdot_init = np.array([0.0, 0.0])

    #setup atractor if used
    attractor_position = np.array([2.0, 0.0])
    
    # #remove
    # attractor_position = np.array([0.0, 0.0])

    #setup of obstacles
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Cuboid(
            axes_length=[0.4, 0.4],
            center_position=np.array([1.0, 0.25]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.1,
            # orientation=10 * pi / 180,
            #linear_velocity = np.array([0.0, 0.5]),
            tail_effect=False,
            # repulsion_coeff=1.4,
        )
    )

    # obstacle_environment.append(
    #     Cuboid(
    #         axes_length=[0.4, 0.4],
    #         center_position=np.array([0.5, -0.25]),
    #         # center_position=np.array([0.9, 0.25]),
    #         margin_absolut=0.1,
    #         # orientation=10 * pi / 180,
    #         #linear_velocity = np.array([0.0, 0.5]),
    #         tail_effect=False,
    #         # repulsion_coeff=1.4,
    #     )
    # )

    ### ROBOT 1 : tau_c = 0, no command ###
    robot_not_controlled = Robot(
        x = x_init, 
        xdot = xdot_init, 
        dt = dt_simulation,
    ) 

    ### ROBOT 2 : tau_c regulates robot to origin ###
    D = 10*np.eye(2) #damping matrix
    #D[1,1] = 1        #less damped in y

    robot_regulated = Robot(
        x = x_init, 
        xdot = xdot_init, 
        dt = dt_simulation,
        controller = RegulationController(
            D=D,
        ),
    )

    ### ROBOT 3 : controlled via DS ###
    #setup of dynamics for robot 3
    initial_dynamics = LinearSystem(
        attractor_position = attractor_position,
        maximum_velocity=3,
        distance_decrease=0.3,
    )

    robot_tracked = Robot(
        x = x_init, 
        xdot = xdot_init, 
        dt = dt_simulation,
        controller = TrackingController(
            D=D, #observation : with D = 100, robot beter follows the dynamics f_des(x) than with D = 10
            dynamic_avoider = ModulationAvoider(
                initial_dynamics=initial_dynamics,
                obstacle_environment=obstacle_environment,
            ),
            lambda_DS=100,
            lambda_obs=20,
        ),
    )

    #setup of animator
    my_animation = CotrolledRobotAnimation(
        it_max = 200, #longer animation, default : 100
        dt_simulation=dt_simulation,
        dt_sleep=0.01,
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

    my_animation.fig.canvas.mpl_connect("button_press_event", lambda x: print("hey"))
    my_animation.run(save_animation=False)



if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    run_control_robot()
