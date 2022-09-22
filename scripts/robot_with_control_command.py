import numpy as np
import matplotlib.pyplot as plt

from vartools.animator import Animator

class Robot:
    #class variable
    dim = 2

    def __init__(
        self,
        M = np.eye(dim),
        C = np.zeros((dim,dim)), #10*np.eye(dim) with damping
        G = np.zeros((dim,1)),

        tau_c = np.zeros((dim,1)),  #control torque
        tau_e = np.zeros((dim,1)),  #external disturbance torque

        controller = None,

        x = np.zeros((dim,1)),      #curent position
        xdot = np.zeros((dim,1)),   #current velocity

        dt = 0.1
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

    def Func_dyn(self, pos, vel, time): #vel is xdot, pos is x
        #right-hand side of the dynamics of the robot x'' = F(x,xdot,t)
        #to simulatate spring : add "- 100*pos" after C*vel
        return np.divide((self.tau_c + self.tau_e - self.G - np.matmul(self.C,vel)),
                          np.diag(self.M).reshape((self.dim,1)))


    def step(self):
        
        #update tau_c
        if not (self.controller is None):
            self.tau_c = self.controller.get_tau_c(self.x, self.xdot)
        
        #update x and xdot
        self.rk4_step()
        pass

    def rk4_step(self):
        """
        perform one time step
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


class RegulationController:
    """
    in the form tau_c = G - (D*x_dot + K*x) , does regulation to 0
    """
    #class variables
    dim = 2

    def __init__(
        self,
        D = 10*np.eye(dim), 
        K = 100*np.eye(dim),

        G = np.zeros((dim,1)),
    ):
        self.D = D
        self.K = K
        self.G = G

    def get_tau_c(self, x, xdot):
        """
        return the torque control command
        """
        return self.G - ( np.matmul(self.D, xdot) + np.matmul(self.K, x) )



class CotrolledRobotAnimation(Animator):
    #class variables
    dim = 2

    def setup(
        self,
        #start_position=np.array([-2.5, 0.5]),
        #start_velocity=np.array([0,0]),
        x_lim=[-1.5, 2],
        y_lim=[-0.5, 2.5],
        robot = None,
    ):
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.robot = robot

        self.position_list = np.zeros((self.dim, self.it_max + 1))
        self.position_list[:, 0] = robot.x.reshape((2,))
        self.velocity_list = np.zeros((self.dim, self.it_max + 1))
        self.velocity_list[:, 0] = robot.xdot.reshape((2,))

        

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def update_step(self, ii: int) -> None:
        print(f"iter : {ii}")

        #CALCULATION
        #simple display example
        #self.position_list[:,ii] = np.array([0.1*ii, -0.5])

        self.robot.step()
        self.position_list[:,ii + 1] = self.robot.x.reshape((2,))
        self.velocity_list[:, ii + 1] = self.robot.xdot.reshape((2,))


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



def run_control_robot():
    dt_simulation = 0.01
    dt = dt_simulation #???

    #setup of robot

    robot_not_controlled = Robot(
        x = np.array([[1.],[0.5]]), 
        xdot = np.array([[1.],[0.]]), 
        dt = dt,
    ) #what is the simplest controller tau_c : 0 = no command

    robot_regulated = Robot(
        x = np.array([[1.],[0.5]]), 
        xdot = np.array([[1.],[0.]]), 
        dt = dt,
        controller = RegulationController(),
    )

    #setup of animator
    my_animation = CotrolledRobotAnimation(
        dt_simulation=dt_simulation,
        dt_sleep=0.01,
    )

    my_animation.setup(
        x_lim=[-3, 3],
        y_lim=[-2.1, 2.1],
        #robot = robot_not_controlled,
        robot = robot_regulated,
    )

    my_animation.run(save_animation=False)



if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    run_control_robot()
