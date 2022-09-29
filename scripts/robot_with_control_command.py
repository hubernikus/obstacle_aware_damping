import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

#from librairy of lukas : vartools
from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

#TODO 
#add magic number for G = np.array([0.0, 0.0]), dim...

class Controller(ABC):
    """
    interface controller template
    """

    @abstractmethod
    def compute_tau_c():
        pass

class Robot:
    """
    Mass point robot 
    """
    #class variable
    dim = 2

    def __init__(
        self,
        M = np.eye(dim),
        C = np.zeros((dim,dim)), #10*np.eye(dim) with damping
        G = np.zeros(dim),

        tau_c = np.zeros(dim),  #control torque
        tau_e = np.zeros(dim),  #external disturbance torque

        controller:Controller = None,

        x = np.zeros(dim),      #curent position
        xdot = np.zeros(dim),   #current velocity

        dt = 0.01
    ):
        self.M = M
        self.C = C
        self.G = G

        self.controller = controller

        self.x = x
        self.xdot = xdot

        self.dt = dt

        self.tau_c = tau_c
        self.tau_e = tau_e

    def Func_dyn(self, pos, vel, time): 
        """
        Use by the Runge Kutta algorithme to evaluate the position&velocity at the next time step
        Func_dyn represents the right-hand side of the dynamic equation of the robot x'' = F(x,xdot,t)
        /!\ it is assumed that there is no coupling, i.e. M is diagonal
        """
        return np.divide((self.tau_c + self.tau_e - self.G - np.matmul(self.C,vel)),
                          np.diag(self.M))


    def simulation_step(self):
        """
        Performs one time step of the dynamics of the robot, update variables
        """
        
        #update tau_c
        if self.controller is not None:
            self.tau_c = self.controller.compute_tau_c(self.x, self.xdot)

        #update x and xdot
        self.rk4_step()

        #reset the disturbance, because taken account of it in rk4_step()
        self.tau_e = np.array([0.0, 0.0])


    def rk4_step(self):
        """
        perform one time step of the RK4 algorithme
        """
        t = 0 #time not used

        m1 = self.dt*self.xdot
        k1 = self.dt*self.Func_dyn(self.x, self.xdot, t)  #(x, v, t)

        m2 = self.dt*(self.xdot + 0.5*k1)
        k2 = self.dt*self.Func_dyn(self.x+0.5*m1, self.xdot+0.5*k1, t+0.5*self.dt)

        m3 = self.dt*(self.xdot + 0.5*k2)
        k3 = self.dt*self.Func_dyn(self.x+0.5*m2, self.xdot+0.5*k2, t+0.5*self.dt)

        m4 = self.dt*(self.xdot + k3)
        k4 = self.dt*self.Func_dyn(self.x+m3, self.xdot+k3, t+self.dt)

        #update of the state
        self.x += (m1 + 2*m2 + 2*m3 + m4)/6
        self.xdot += (k1 + 2*k2 + 2*k3 + k4)/6



class RegulationController(Controller):
    """
    in the form tau_c = G - D*x_dot - K*x , does regulation to 0
    """
    #class variables
    dim = 2

    def __init__(
        self,
        D = 10*np.eye(dim), 
        K = 100*np.eye(dim),
        G = np.zeros(dim),
    ):
        self.D = D
        self.K = K
        self.G = G

    def compute_tau_c(self, x, xdot):
        """
        return the torque control command of the regulation controller,
        """
        return self.G - np.matmul(self.D, xdot) - np.matmul(self.K, x)

class TrackingController(Controller):
    """
    in the form tau_c = G - D(xdot - f_desired(x))
    """
    #class variables
    dim = 2

    def __init__(
        self, 
        initial_dynamics:LinearSystem = None, 
        D = 10*np.eye(dim),
        G = np.zeros(dim),
    ):
        self.initial_dynamics = initial_dynamics
        self.D = D
        self.G = G

    def compute_tau_c(self, x, xdot):
        """
        return the torque control command of the DS-tracking controller,
        """
        x_des = self.initial_dynamics.evaluate(x)
        return self.G - np.matmul(self.D, (xdot - x_des))
    

class CotrolledRobotAnimation(Animator):
    #class variables
    dim = 2

    def setup(
        self,
        #start_position=np.array([-2.5, 0.5]),
        #start_velocity=np.array([0,0]),
        x_lim=[-1.5, 2],
        y_lim=[-0.5, 2.5],
        robot:Robot = None,
        disturbance_magn = 200,
    ):
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.robot = robot

        self.position_list = np.zeros((self.dim, self.it_max + 1))
        self.position_list[:, 0] = robot.x.reshape((self.dim,))
        self.velocity_list = np.zeros((self.dim, self.it_max + 1))
        self.velocity_list[:, 0] = robot.xdot.reshape((self.dim,))

        self.disturbance_magn = disturbance_magn
        self.disturbance_list = np.empty((self.dim, 0))
        self.disturbance_pos_list = np.empty((self.dim, 0))
        self.new_disturbance = False

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def update_step(self, ii: int) -> None:
        print(f"iter : {ii + 1}") #actual is i + 1

        #CALCULATION
        self.robot.simulation_step()

        self.position_list[:, ii + 1] = self.robot.x
        self.velocity_list[:, ii + 1] = self.robot.xdot

        #record position of the new disturbance added with key pressed
        if self.new_disturbance:
            #bizarre line, recheck append
            self.disturbance_pos_list = np.append(self.disturbance_pos_list, 
                                                  self.position_list[:,ii].reshape((self.dim, 1)), axis = 1)
            self.new_disturbance = False

        #CLEARING
        self.ax.clear()

        #PLOTTING
        #past trajectory
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
            atractor = self.robot.controller.initial_dynamics.attractor_position
        self.ax.plot(
            atractor[0],
            atractor[1],
            "k*",
            markersize=8,
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
    x_init = np.array([1.0, 0.5])
    xdot_init = np.array([5.0, 0.0])

    #other set of interesting initial conditions
    # x_init = np.array([2.0, 0.0])
    # xdot_init = np.array([0.0, 0.0])

    ### ROBOT 1 : tau_c = 0, no command ###
    robot_not_controlled = Robot(
        x = x_init, 
        xdot = xdot_init, 
        dt = dt_simulation,
    ) 

    ### ROBOT 2 : tau_c regulates robot to origin ###
    D = 10*np.eye(2) #damping matrix
    #D[1,1] = 1         #less damped in y

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
        attractor_position=np.array([2.0, 0.0]),
        maximum_velocity=2,
        distance_decrease=0.3,
    )

    robot_tracked = Robot(
        x = x_init, 
        xdot = xdot_init, 
        dt = dt_simulation,
        controller = TrackingController(
            D=D, #observation : with D = 100, robot beter follows the dynamics f_des(x) than with D = 10
            initial_dynamics = initial_dynamics,
        ),
    )

    #setup of animator
    my_animation = CotrolledRobotAnimation(
        it_max = 200, #longer animation, default : 100
        dt_simulation=dt_simulation,
        dt_sleep=0.01,
    )

    my_animation.setup(
        x_lim=[-3, 3],
        y_lim=[-2.1, 2.1],
        #robot = robot_not_controlled,
        #robot = robot_regulated,
        robot = robot_tracked
    )

    my_animation.run(save_animation=False)



if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    run_control_robot()
