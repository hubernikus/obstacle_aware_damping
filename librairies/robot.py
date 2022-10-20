import numpy as np

#my librairies
from librairies.controller import TrackingController
import librairies.magic_numbers_and_enums as mn

class Robot:
    """
    Mass point robot 
    """
    def __init__(
        self,
        M = np.eye(mn.DIM),
        C = np.zeros((mn.DIM,mn.DIM)), #10*np.eye(dim) with damping
        G = np.zeros(mn.DIM),

        tau_c = np.zeros(mn.DIM),  #control torque
        tau_e = np.zeros(mn.DIM),  #external disturbance torque

        controller:TrackingController = None,

        x = np.zeros(mn.DIM),      #curent position
        xdot = np.zeros(mn.DIM),   #current velocity

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
        Use by the Runge Kutta algorithm to evaluate the position&velocity at the next time step
        Func_dyn represents the right-hand side of the dynamic equation of the robot x'' = F(x,xdot,t)
        /!\ it is assumed that there is no coupling, i.e. M is diagonal
        """
        return np.divide((self.tau_c + self.tau_e - self.G - self.C@vel),
                          np.diag(self.M))


    def simulation_step(self):
        """
        Performs one time step of the dynamics of the robot, update variables
        """

        #update of D matrix to follow DS or passive to obs
        self.controller.update_D_matrix(self.x)

        #update tau_c
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
