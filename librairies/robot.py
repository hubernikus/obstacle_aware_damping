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
        DIM = 2,
        M = None,
        C = None,
        G = None,

        controller:TrackingController = None,

        x = None,      #curent position
        xdot = None,   #current velocity

        dt = 0.01,

        noisy = False,
    ):
        self.DIM = DIM

        self.M = M
        if M is None:
            self.M = np.eye(self.DIM)
  
        self.M_inv = np.linalg.inv(self.M)
        
        self.C = C
        if C is None:
            self.C = np.zeros((self.DIM,self.DIM))
            #self.C = 10*np.eye(self.DIM), # with damping 

        self.G = G
        if G is None:
            self.G = np.zeros(self.DIM)

        if controller is None:
            raise ValueError("Specify a controller")
        self.controller = controller

        self.x = x
        if x is None:
            self.x = np.zeros(self.DIM)
        
        self.xdot = xdot
        if xdot is None:
            self.xdot = np.zeros(self.DIM)

        self.dt = dt

        self.tau_c = np.zeros(self.DIM)  #control torque
        self.tau_e = np.zeros(self.DIM)  #external disturbance torque            


        self.noisy = noisy


    def simulation_step(self):
        """
        Performs one time step of the dynamics of the robot, update variables
        """
        ##############################################
        ## MEASUREMENTS, TORQUE COMMAND COMPUTATION ##
        ##############################################

        #measurement of postition and velocity - with noise level
        x, xdot = self.measure_pos_vel()

        t_now = time()
        #update of D matrix to follow DS or passive to obs
        self.controller.update_D_matrix(x, xdot)

        #udpate the energy tank - not used
        self.controller.update_energy_tank(self.x, self.xdot, self.dt)

        #update tau_c
        self.tau_c = self.controller.compute_tau_c(x, xdot)
        t_diff = time() - t_now
        #print(f"time to compute control command: {t_diff}")


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
        self.tau_e = np.zeros(self.DIM)

    def func_dyn(self, pos, vel, time): 
        """
        Use by the Runge Kutta algorithm to evaluate the position&velocity at the next time step
        Func_dyn represents the right-hand side of the dynamic equation of the robot x'' = F(x,xdot,t)
        """
        return (self.tau_c + self.tau_e - self.G - self.C@vel)@self.M_inv

    def rk4_step(self):
        """
        perform one time step of the robot with RK4 algorithm
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
        """
        perform one time step of the robot with euler-forward algorithm
        """
        t = 0 #not used

        m1 = self.dt*self.xdot
        k1 = self.dt*self.func_dyn(self.x, self.xdot, t)

        #update of the state
        self.x += m1
        self.xdot += k1

    def measure_pos_vel(self):
        """
        adds noise to the simulation measurements (either on position, velocity or both)
        """
        if self.noisy:
            x = self.x + np.random.normal(0,mn.NOISE_STD_POS,self.DIM)
            xdot = self.xdot + np.random.normal(0, mn.NOISE_STD_VEL, self.DIM)
        else:
            x = self.x
            xdot = self.xdot
        return x, xdot
