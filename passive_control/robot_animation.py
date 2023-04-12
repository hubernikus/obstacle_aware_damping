import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation 
import datetime
import os
import matplotlib.image as mpimg
from scipy import ndimage

#passive_control.of lukas
from vartools.animator import Animator

from dynamic_obstacle_avoidance.containers import ObstacleContainer
#from dynamic_obstacle_avoidance.visualization import plot_obstacles #overwritten in draw_obs_overwrite

#my passive_control.
from passive_control.robot import Robot
import passive_control.magic_numbers_and_enums as mn
from passive_control.draw_obs_overwrite import plot_obstacles

#just for plotting : global
s_list = []

class CotrolledRobotAnimation(Animator):
    """
    Animator class that runs the main loop, simulates and vizualizes everything
    """

    def setup(
        self,
        robot:Robot,
        obstacle_environment:ObstacleContainer,
        DIM = 2,
        x_lim=[-1.5, 2],
        y_lim=[-0.5, 2.5],
        disturbance_scaling = 200.0,
        draw_ideal_traj = False,
        draw_qolo = False,
        rotate_qolo = False
    ):  
        self.DIM = DIM
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.robot = robot
        self.obstacle_environment = obstacle_environment

        self.position_list = np.zeros((self.DIM, self.it_max + 1))
        self.position_list[:, 0] = robot.x.reshape((self.DIM,))
        self.velocity_list = np.zeros((self.DIM, self.it_max + 1))
        self.velocity_list[:, 0] = robot.xdot.reshape((self.DIM,))

        self.disturbance_list = np.empty((self.DIM, 0))
        self.disturbance_pos_list = np.empty((self.DIM, 0))
        self.new_disturbance = False
        self.x_press_disturbance = None
        self.y_press_disturbance = None
        self.disturbance_scaling = disturbance_scaling

        self.position_list_ideal = np.zeros((self.DIM, self.it_max + 1))
        self.position_list_ideal[:, 0] = robot.x.reshape((self.DIM,))

        self.draw_ideal_traj = draw_ideal_traj

        self.draw_qolo = draw_qolo
        self.rotate_qolo = rotate_qolo

        self.qolo = mpimg.imread("./Qolo_T_CB_top_bumper_low_qual.png")
        self.qolo_length_x = mn.QOLO_LENGHT_X
        self.qolo_length_y = (1.0) * self.qolo.shape[0] / self.qolo.shape[1] * self.qolo_length_x

        if self.DIM == 2:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
        else:
            self.fig, self.ax = plt.subplots(1,2,figsize=(14, 7))

        self.d_min = 1000.0 #variable to record closest pos to obs, big init value

    def update_step(self, ii: int) -> None:
        """
        what has to be done during one iteration, is divided in 2 main phases :
        1. calculation : everethings that is related to the robot simulation,
                         recieves input disturbances
        2. clearing + drawing : vizualizes everything
        """
        print(f"iter : {ii + 1}") #because starting at 0
        
        ###################
        ### CALCULATION ###
        ###################

        # Update obstacles
        self.obstacle_environment.do_velocity_step(delta_time=self.dt_simulation)

        #add artificial disturbances : function to call to trigger automatic disturbances
        self.artificial_disturbances_2D(ii)

        #just for plotting the energy storage level s 
        s_list.append(self.robot.controller.s)
 
        #record position of the new disturbance added with mouse drag
        if self.new_disturbance:
            self.disturbance_pos_list = np.append(
                self.disturbance_pos_list, 
                self.position_list[:,ii].reshape((self.DIM, 1)), axis = 1
            )
            self.new_disturbance = False
        
        #get the normals + distances of the obstacles for the D_matrix of the controller
        self.robot.controller.obs_normals_list = np.empty((self.DIM, 0))
        self.robot.controller.obs_dist_list =np.empty(0)

        for obs in self.obstacle_environment:
            #gather the parameters wrt obstacle i
            normal = obs.get_normal_direction(
                    self.position_list[:, ii],
                    in_obstacle_frame = False,
                ).reshape(self.DIM, 1)
            self.robot.controller.obs_normals_list = np.append(
                self.robot.controller.obs_normals_list,
                normal,
                axis=1,
            )

            #get_gamma() works for more obstacles than get_distance_to_surface()
            d = obs.get_gamma(
                self.position_list[:, ii],
                in_obstacle_frame = False,                
            ) - 1
            self.robot.controller.obs_dist_list = np.append(
                self.robot.controller.obs_dist_list,
                d,
            )

            #to keep track of how close we we're during the whole simulation
            if d < self.d_min:
                self.d_min = d


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
        if self.DIM == 2:
            self.ax.clear()
        else:
            for ax in self.ax:
                ax.clear()

        ################
        ### PLOTTING ###
        ################

        if self.DIM == 2:
            self.plot_anim_2D(ii)
        else:
            self.plot_anim_3D(ii)
        
        

    def plot_anim_2D(self, ii):
        """
        Plot everything in 2D
        """
        #ax settings
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            right=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

        #ideal trajectory - in blue
        if self.draw_ideal_traj:
            self.ax.plot(
                self.position_list_ideal[0, :ii], 
                self.position_list_ideal[1, :ii], 
                ":", color="#0000FF",
                label= "Ideal trajectory"
            )

        #past trajectory - in green
        if ii >= 1: #not first iter
            self.ax.plot(
                self.position_list[0, :(ii - 1)], 
                self.position_list[1, :(ii - 1)], 
                ":", color="#135e08",
                label= "Real trajectory"
            )

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

        #disturbance drawing
        draw_scaling = 500.0
        for  i, (disturbance, disturbance_pos) in enumerate(zip(self.disturbance_list.transpose(),
                                                 self.disturbance_pos_list.transpose())):
            #small trick to only label one disturance
            if i == 0:
                self.ax.arrow(disturbance_pos[0], disturbance_pos[1],
                            disturbance[0]/draw_scaling, disturbance[1]/draw_scaling,
                            width=0.02,
                            color= "r",
                            label= "Disturbances",)
            else : 
                self.ax.arrow(disturbance_pos[0], disturbance_pos[1],
                            disturbance[0]/draw_scaling, disturbance[1]/draw_scaling,
                            width=0.02,
                            color= "r",)

        if self.draw_qolo:
            #carefull, rotating slows a lot the animation (-> reduce image resolution)
            if self.rotate_qolo:
                angle_rot = np.arctan2(self.velocity_list[1, ii], self.velocity_list[0, ii])
                qolo_rot = ndimage.rotate(
                    self.qolo, 
                    angle_rot * 180.0 / np.pi, cval=255
                )
                lenght_x_rotated = (
                    np.abs(np.cos(angle_rot)) * self.qolo_length_x + np.abs(np.sin(angle_rot)) \
                    * self.qolo_length_y
                )
                lenght_y_rotated = (
                    np.abs(np.sin(angle_rot)) * self.qolo_length_x + np.abs(np.cos(angle_rot)) \
                    * self.qolo_length_y
                )
            else:
                lenght_x_rotated = self.qolo_length_x
                lenght_y_rotated = self.qolo_length_y
                qolo_rot = self.qolo

            self.ax.imshow(
                (qolo_rot*255).astype('uint8'),
                extent = [
                    self.position_list[0,ii] - lenght_x_rotated/2,
                    self.position_list[0,ii] + lenght_x_rotated/2,
                    self.position_list[1,ii] - lenght_y_rotated/2,
                    self.position_list[1,ii] + lenght_y_rotated/2,
                ]
            )
        else:
            self.ax.plot(
                self.position_list[0, ii],
                self.position_list[1, ii],
                "o",
                color="#135e08",
                markersize=12,
            )
        
        plt.legend(loc = 1, prop={'size': 18}) #big : 18

    def plot_anim_3D(self, ii):
        """
        Plot everything in 3D, the screen it split, disturbance drawing works on both views
        """        
        plt.title("3D viewer (obstacle penetration only if both views penetrate")
        for i, ax in enumerate(self.ax):
            if i == 0: #plot xy plane
                absciss = 0 #x
                ordinate = 1 #y
                ax.set_title("View of XY plane - side view")
            else: #plot in zy plane
                absciss = 2 #z
                ordinate = 1 #y
                ax.set_title("View of ZY plane - front view")
            #ax settings
            ax.set_xlim(self.x_lim)
            ax.set_ylim(self.y_lim)
            ax.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                right=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )

            #ideal trajectory - in light blue
            if self.draw_ideal_traj:
                ax.plot(
                    self.position_list_ideal[absciss, :ii], 
                    self.position_list_ideal[ordinate, :ii], 
                    ":", color="#0000FF",
                    label= "Ideal trajectory"
                )

            #past trajectory - in green
            if ii >= 1: #not first iter
                ax.plot(
                    self.position_list[absciss, :(ii - 1)], 
                    self.position_list[ordinate, :(ii - 1)], 
                    ":", color="#135e08",
                    label= "Real trajectory"
                )

            #atractor position
            atractor = self.robot.controller.dynamic_avoider.initial_dynamics.attractor_position
            ax.plot(
                atractor[absciss],
                atractor[ordinate],
                "k*",
                markersize=8,
            )

            #obstacles positions
            plot_obstacles(
                ax=ax,
                obstacle_container=self.obstacle_environment,
                x_lim=self.x_lim,
                y_lim=self.y_lim,
                showLabel=False,
                absciss = absciss,
                ordinate = ordinate,
            )

            #disturbance drawing
            draw_scaling = 500.0
            for  i, (disturbance, disturbance_pos) in enumerate(zip(self.disturbance_list.transpose(),
                                                    self.disturbance_pos_list.transpose())):
                #small trick to only label one disturance
                if i == 0:
                    ax.arrow(disturbance_pos[absciss], disturbance_pos[ordinate],
                                disturbance[absciss]/draw_scaling, disturbance[ordinate]/draw_scaling,
                                width=0.02,
                                color= "r",
                                label= "Disturbances",)
                else : 
                    ax.arrow(disturbance_pos[absciss], disturbance_pos[ordinate],
                                disturbance[absciss]/draw_scaling, disturbance[ordinate]/draw_scaling,
                                width=0.02,
                                color= "r",)

            if self.draw_qolo:
                #carefull, rotating slows a lot the animation (-> lower image resolution)
                if self.rotate_qolo:
                    angle_rot = np.arctan2(self.velocity_list[ordinate, ii],
                                           self.velocity_list[absciss, ii]
                    )
                    qolo_rot = ndimage.rotate(
                        self.qolo, 
                        angle_rot * 180.0 / np.pi, cval=255
                    )
                    lenght_x_rotated = (
                        np.abs(np.cos(angle_rot)) * self.qolo_length_x + np.abs(np.sin(angle_rot)) \
                        * self.qolo_length_y
                    )
                    lenght_y_rotated = (
                        np.abs(np.sin(angle_rot)) * self.qolo_length_x + np.abs(np.cos(angle_rot)) \
                        * self.qolo_length_y
                    )
                else:
                    lenght_x_rotated = self.qolo_length_x
                    lenght_y_rotated = self.qolo_length_y
                    qolo_rot = self.qolo

                ax.imshow(
                    (qolo_rot*255).astype('uint8'),
                    extent = [
                        self.position_list[absciss,ii] - lenght_x_rotated/2,
                        self.position_list[absciss,ii] + lenght_x_rotated/2,
                        self.position_list[ordinate,ii] - lenght_y_rotated/2,
                        self.position_list[ordinate,ii] + lenght_y_rotated/2,
                    ]
                )
            else:
                ax.plot(
                    self.position_list[absciss, ii],
                    self.position_list[ordinate, ii],
                    "o",
                    color="#135e08",
                    markersize=12,
                )
            
        plt.legend()


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
    # only lines with #modified/added were changed
    def run(self, save_animation: bool = False) -> None:
            """Runs the animation"""
            if self.fig is None:
                raise Exception("Member variable 'fig' is not defined.")

            # Initiate keyboard-actions
            self.fig.canvas.mpl_connect("button_press_event", self.record_click_coord) #modified
            self.fig.canvas.mpl_connect("button_release_event", self.add_click_disturbance)  #added
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


    ### DISTURBANCES HANDLING ###

    def record_click_coord(self, event):
        """
        record if a click is perfom in ax (to draw a disturbance)
        """
        self.x_press_disturbance, self.y_press_disturbance = event.xdata, event.ydata
        #print(event.xdata, event.ydata)

    def add_click_disturbance(self, event):
        """
        record if a click is released in ax (to draw a disturbance)
        """
        if event.xdata is None:
            return
        
        if self.DIM == 2:
            disturbance = np.array(
                [event.xdata - self.x_press_disturbance,
                 event.ydata - self.y_press_disturbance]
            )*self.disturbance_scaling
        else:
            if event.inaxes is self.ax[0]: #dist in XY plane
                disturbance = np.array(
                    [event.xdata - self.x_press_disturbance,
                     event.ydata - self.y_press_disturbance,
                     0.0]
                )*self.disturbance_scaling
            else:
                disturbance = np.array( #dist in ZY plane
                    [0.0,
                     event.ydata - self.y_press_disturbance,
                     event.xdata - self.x_press_disturbance]
                )*self.disturbance_scaling

        
        if np.linalg.norm(disturbance) < 10.0:
            return
        self.robot.tau_e = disturbance
        self.disturbance_list = np.append(self.disturbance_list, disturbance.reshape(self.DIM,1), axis=1)
        self.new_disturbance = True

    def artificial_disturbances_2D(self, ii):
        # if ii == 20:
        #     disturbance = np.array([-1.,3.])*self.disturbance_scaling
        #     self.robot.tau_e = disturbance
        #     self.disturbance_list = np.append(self.disturbance_list, disturbance.reshape(self.DIM,1), axis=1)
        #     self.new_disturbance = True

        # if ii == 45:
        #     disturbance = np.array([-3., -2.])*self.disturbance_scaling
        #     self.robot.tau_e = disturbance
        #     self.disturbance_list = np.append(self.disturbance_list, disturbance.reshape(self.DIM,1), axis=1)
        #     self.new_disturbance = True
        # if ii == 35:
        #     disturbance = np.array([-3.,0.])*self.disturbance_scaling
        #     self.robot.tau_e = disturbance
        #     self.disturbance_list = np.append(self.disturbance_list, disturbance.reshape(self.DIM,1), axis=1)
        #     self.new_disturbance = True
        # if ii == 65:
        #     disturbance = np.array([2.,-4.])*self.disturbance_scaling
        #     self.robot.tau_e = disturbance
        #     self.disturbance_list = np.append(self.disturbance_list, disturbance.reshape(self.DIM,1), axis=1)
        #     self.new_disturbance = True
        # if ii == 50:
        #     disturbance = np.array([-1.0,-4.0])*self.disturbance_scaling
        #     self.robot.tau_e = disturbance
        #     self.disturbance_list = np.append(self.disturbance_list, disturbance.reshape(self.DIM,1), axis=1)
        #     self.new_disturbance = True
        # if ii == 110:
        #     disturbance = np.array([3.,1.])*self.disturbance_scaling
        #     self.robot.tau_e = disturbance
        #     self.disturbance_list = np.append(self.disturbance_list, disturbance.reshape(self.DIM,1), axis=1)
        #     self.new_disturbance = True
        # if ii == 140:
        #     disturbance = np.array([2.,1])*self.disturbance_scaling
        #     self.robot.tau_e = disturbance
        #     self.disturbance_list = np.append(self.disturbance_list, disturbance.reshape(self.DIM,1), axis=1)
        #     self.new_disturbance = True
        pass

    def get_d_min(self):
        return self.d_min