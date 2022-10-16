import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import datetime
import os

#librairies of lukas
from vartools.animator import Animator

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles

#my librairies
from librairies.robot import Robot

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
        disturbance_scaling = 200,
        draw_ideal_traj = False,
    ):
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.robot = robot
        self.obstacle_environment = obstacle_environment

        self.position_list = np.zeros((self.dim, self.it_max + 1))
        self.position_list[:, 0] = robot.x.reshape((self.dim,))
        self.velocity_list = np.zeros((self.dim, self.it_max + 1))
        self.velocity_list[:, 0] = robot.xdot.reshape((self.dim,))

        self.disturbance_scaling = disturbance_scaling
        self.disturbance_list = np.empty((self.dim, 0))
        self.disturbance_pos_list = np.empty((self.dim, 0))
        self.new_disturbance = False
        self.x_press_disturbance = None
        self.y_press_disturbance = None

        self.position_list_ideal = np.zeros((self.dim, self.it_max + 1))
        self.position_list_ideal[:, 0] = robot.x.reshape((self.dim,))

        self.draw_ideal_traj = draw_ideal_traj

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def update_step(self, ii: int) -> None:
        print(f"iter : {ii + 1}") #because starting at 0
        
        ###################
        ### CALCULATION ###
        ###################

        # Update obstacles
        self.obstacle_environment.do_velocity_step(delta_time=self.dt_simulation)

        #record position of the new disturbance added with key pressed
        if self.new_disturbance:
            #bizarre line, recheck append
            self.disturbance_pos_list = np.append(self.disturbance_pos_list, 
                                                  self.position_list[:,ii].reshape((self.dim, 1)), axis = 1)
            self.new_disturbance = False
        
        #get the normal + distance of the obstacles for the D_matrix of the controller
        self.robot.controller.obs_normals_list = np.empty((self.dim, 0))
        self.robot.controller.obs_dist_list =np.empty(0)

        for obs in self.obstacle_environment:
            self.robot.controller.obs_normals_list = np.append(
                self.robot.controller.obs_normals_list,
                obs.get_normal_direction(
                    self.position_list[:, ii], in_obstacle_frame = False
                ).reshape(self.dim, 1),
                axis=1
            )
            self.robot.controller.obs_dist_list = np.append(
                self.robot.controller.obs_dist_list,
                obs.get_distance_to_surface(
                    self.position_list[:, ii], in_obstacle_frame = False
                )
            )
            #print(obs.get_distance_to_surface(self.position_list[:, ii], in_obstacle_frame = False))


        #updating the robot + record trajectory
        self.robot.simulation_step()
        
        self.position_list[:, ii + 1] = self.robot.x
        self.velocity_list[:, ii + 1] = self.robot.xdot

        #record trajectory without physical constrains - ideal
        if self.draw_ideal_traj:
            velocity_ideal = self.robot.controller.dynamic_avoider.evaluate(
                self.position_list_ideal[:, ii]
            )
            self.position_list_ideal[:, ii + 1] = (
                velocity_ideal * self.dt_simulation + self.position_list_ideal[:, ii]
            )

        #################
        #### CLEARING ###
        #################

        self.ax.clear()

        ################
        ### PLOTTING ###
        ################

        #ideal trajectory - in black
        if self.draw_ideal_traj:
            self.ax.plot(
                self.position_list_ideal[0, :ii], 
                self.position_list_ideal[1, :ii], 
                ":", color="#000000"
            )

        #past trajectory - in green
        if ii >= 1: #not first iter
            self.ax.plot(
                self.position_list[0, :(ii - 1)], self.position_list[1, :(ii - 1)], ":", color="#135e08"
            )
        #actual position is i, futur position is i + 1 (will be plot next cycle)
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
                          disturbance[0]/500.0, disturbance[1]/500.0,
                          width=self.disturbance_scaling/10000.0,
                          color= "r")


    #overwrite key detection of Animator -> not use anymore
    #/!\ bug when 2 keys pressed at same time
    def on_press_not_use(self, event):
        if True:
            raise NotImplementedError("Depreciated ---- remove")
        if event.key.isspace():
            self.pause_toggle()

        elif event.key == "right":
            print("->")
            self.robot.tau_e = np.array([1.0, 0.0])*self.disturbance_scaling
            self.disturbance_list = np.append(self.disturbance_list, (np.array([[1.0], [0.0]])), axis=1)
            self.new_disturbance = True
            #TEST 

        elif event.key == "left":
            print("<-")
            self.robot.tau_e = np.array([-1.0, 0.0])*self.disturbance_scaling
            self.disturbance_list = np.append(self.disturbance_list, (np.array([[-1.0], [0.0]])), axis=1)
            self.new_disturbance = True

        elif event.key == "up":
            print("^\n|")
            self.robot.tau_e = np.array([0.0, 1.0])*self.disturbance_scaling
            self.disturbance_list = np.append(self.disturbance_list, (np.array([[0.0], [1.0]])), axis=1)
            self.new_disturbance = True
    
        elif event.key == "down":
            print("|\nV")
            self.robot.tau_e = np.array([0.0, -1.0])*self.disturbance_scaling
            self.disturbance_list = np.append(self.disturbance_list, (np.array([[0.0], [-1.0]])), axis=1)
            self.new_disturbance = True

        elif event.key == "d":
            self.step_forward()

        elif event.key == "a":
            self.step_back()

    #overwrite run() to change "button_press_event"
    def run(self, save_animation: bool = False) -> None:
            """Runs the animation"""
            if self.fig is None:
                raise Exception("Member variable 'fig' is not defined.")

            # Initiate keyboard-actions
            self.fig.canvas.mpl_connect("button_press_event", self.record_click_coord) #modified
            self.fig.canvas.mpl_connect("button_release_event", self.add_disturbance)  #added
            self.fig.canvas.mpl_connect("key_press_event", self.on_press)

            if save_animation:
                if self.animation_name is None:
                    now = datetime.datetime.now()
                    animation_name = f"animation_{now:%Y-%m-%d_%H-%M-%S}" + self.file_type
                else:
                    # Set filetype
                    animation_name = self.animation_name + self.file_type

                print(f"Saving animation to: {animation_name}.")

                # breakpoint()
                anim = animation.FuncAnimation(
                    self.fig,
                    self.update_step,
                    frames=self.it_max,
                    interval=self.dt_sleep * 1000,  # Conversion [s] -> [ms]
                )

                # FFmpeg for
                writervideo = animation.FFMpegWriter(
                    fps=10,
                    # extra_args=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'],
                    # extra_args=['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2'],
                    # extra_args=['-vf', 'crop=1600:800']
                )

                anim.save(
                    os.path.join("figures", animation_name),
                    # metadata={"artist": "Lukas Huber"},
                    # We chose default 'pillow', beacuse 'ffmpeg' often gives errors
                    writer=writervideo,
                    # animation.PillowWriter()
                )
                print("Animation saving finished.")

            else:
                self.it_count = 0
                while self.it_max is None or self.it_count < self.it_max:
                    if not plt.fignum_exists(self.fig.number):
                        print("Stopped animation on closing of the figure.")
                        break

                    if self._animation_paused:
                        plt.pause(self.dt_sleep)
                        continue

                    self.update_step(self.it_count)

                    # Check convergence
                    if self.has_converged(self.it_count):
                        print(f"All trajectories converged at it={self.it_count}.")
                        break

                    # TODO: adapt dt_sleep based on
                    plt.pause(self.dt_sleep)

                    self.it_count += 1

    def record_click_coord(self, event):
        #print("press ", event.xdata, event.ydata)
        self.x_press_disturbance, self.y_press_disturbance = event.xdata, event.ydata

    def add_disturbance(self, event):
        #print("release : ", event.xdata, event.ydata)
        disturbance = np.array([event.xdata - self.x_press_disturbance,
                                event.ydata - self.y_press_disturbance])*self.disturbance_scaling
        if np.linalg.norm(disturbance) < 10.0:
            return
        #print("disturbance : ", disturbance)
        self.robot.tau_e = disturbance
        self.disturbance_list = np.append(self.disturbance_list, disturbance.reshape(self.dim,1), axis=1)
        self.new_disturbance = True
