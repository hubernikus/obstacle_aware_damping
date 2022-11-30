import numpy as np
from time import time

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
        #C = 10*np.eye(mn.DIM), # with damping 
        C = np.zeros((mn.DIM,mn.DIM)),

        tau_c = np.zeros(mn.DIM),  #control torque
        tau_e = np.zeros(mn.DIM),  #external disturbance torque

        controller:TrackingController = None,

        x = np.zeros(mn.DIM),      #curent position
        xdot = np.zeros(mn.DIM),   #current velocity

        dt = 0.01,

        noisy = False,
    ):
        self.M = M
        self.M_inv = np.linalg.inv(self.M)
        self.C = C

        self.controller = controller

        self.x = x
        self.xdot = xdot

        self.dt = dt

        self.tau_c = tau_c
        self.tau_e = tau_e

        self.noisy = noisy


    def simulation_step(self):
        """
        Performs one time step of the dynamics of the robot, update variables
        """
        ##############################################
        ## MEASUREMENTS, TORQUE COMMAND COMPUTATION ##
        ##############################################

        #udpate the energy tank - not used
        #self.controller.update_energy_tank(self.x, self.xdot, self.dt)

        #measurement of postition and velocity - with noise level
        x, xdot = self.measure_pos_vel()

        t_now = time()
        #update of D matrix to follow DS or passive to obs
        self.controller.update_D_matrix(x, xdot)

        #update tau_c
        self.tau_c = self.controller.compute_tau_c(x, xdot)
        t_diff = time() - t_now
        print(f"time to compute control command: {t_diff}")


        ###########################
        ## DYNAMICS OF THE ROBOT ##
        ###########################

        #update x and xdot - the real, bc its robot proper mecanic
        self.rk4_step()
        #self.euler_forward_step()

        ###########
        ## OTHER ##
        ###########

        #reset the disturbance, because taken account of it in rk4_step() 
        # i.e. disturbance are ponctual, only applied once to the system
        self.tau_e = np.zeros(mn.DIM)

    def func_dyn(self, pos, vel, time): 
        """
        Use by the Runge Kutta algorithm to evaluate the position&velocity at the next time step
        Func_dyn represents the right-hand side of the dynamic equation of the robot x'' = F(x,xdot,t)
        """
        return (self.tau_c + self.tau_e - mn.G - self.C@vel)@self.M_inv

    def rk4_step(self):
        """
        perform one time step of the RK4 algorithme
        """
        t = 0 #time not used

        m1 = self.dt*self.xdot
        k1 = self.dt*self.func_dyn(self.x, self.xdot, t)  #(x, v, t)

        m2 = self.dt*(self.xdot + 0.5*k1)
        k2 = self.dt*self.func_dyn(self.x+0.5*m1, self.xdot+0.5*k1, t+0.5*self.dt)

        m3 = self.dt*(self.xdot + 0.5*k2)
        k3 = self.dt*self.func_dyn(self.x+0.5*m2, self.xdot+0.5*k2, t+0.5*self.dt)

        m4 = self.dt*(self.xdot + k3)
        k4 = self.dt*self.func_dyn(self.x+m3, self.xdot+k3, t+self.dt)

        #update of the state
        self.x += (m1 + 2*m2 + 2*m3 + m4)/6
        self.xdot += (k1 + 2*k2 + 2*k3 + k4)/6

    def euler_forward_step(self):
        t = 0 #not used

        m1 = self.dt*self.xdot
        k1 = self.dt*self.func_dyn(self.x, self.xdot, t)

        #update of the state
        self.x += m1
        self.xdot += k1

    def measure_pos_vel(self):
        if self.noisy:
            x = self.x + mn.NOISE_MAGN_POS*np.random.normal(0,1,mn.DIM)
            xdot = self.xdot + mn.NOISE_MAGN_VEL*np.random.normal(0,1,mn.DIM)
        else:
            x = self.x
            xdot = self.xdot
        return x, xdot
