import numpy as np
from abc import ABC, abstractmethod
import time

#librairies of lukas
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.utils import get_orthogonal_basis

#my librairies
from librairies.magic_numbers_and_enums import TypeOfDMatrix as TypeD
import librairies.magic_numbers_and_enums as mn


class Controller(ABC):
    """
    interface controller template
    """

    @abstractmethod
    def compute_tau_c():
        pass

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
        dynamic_avoider:ModulationAvoider,
        D = 10*np.eye(dim),
        G = np.zeros(dim),
        lambda_DS = 100.0, #to follow DS line
        lambda_perp = 20.0, #compliance perp to DS or obs
        lambda_obs_scaling = 20.0, #scaling factor prop to distance to obstacles, to be stiff
        type_of_D_matrix = TypeD.DS_FOLLOWING
        
    ):
        self.dynamic_avoider = dynamic_avoider
        self.D = D
        self.G = G

        self.lambda_DS = lambda_DS
        self.lambda_perp = lambda_perp
        self.lambda_obs_scaling = lambda_obs_scaling

        #self.lambda_mat = np.diag(np.array([self.lambda_DS, self.lambda_perp_DS]))

        self.obs_normals_list = np.empty((self.dim, 0))
        self.obs_dist_list = np.empty(0)

        self.type_of_D_matrix = type_of_D_matrix


    def compute_tau_c(self, x, xdot):
        """
        return the torque control command of the DS-tracking controller,
        """
        x_dot_des = self.dynamic_avoider.evaluate(x)
        tau_c = self.G - self.D@(xdot - x_dot_des)

        #physical constrains, value ?
        tau_c[tau_c > mn.MAX_TAU_C] = mn.MAX_TAU_C
        tau_c[tau_c < -mn.MAX_TAU_C] = -mn.MAX_TAU_C
        return tau_c

    def update_D_matrix(self, x):
        if self.type_of_D_matrix == TypeD.DS_FOLLOWING:
            D = self.update_D_matrix_wrt_DS(x)
        elif self.type_of_D_matrix == TypeD.OBS_PASSIVITY:
            D = self.update_D_matrix_wrt_obs()
        elif self.type_of_D_matrix == TypeD.BOTH:
            start = time.time()
            D = self.update_D_matrix_wrt_both(x)
            end = time.time()
            print(end - start)
        self.D = D
    
    def update_D_matrix_wrt_DS(self, x):
        x_dot_des = self.dynamic_avoider.evaluate(x)
        E = get_orthogonal_basis(x_dot_des)
        E_inv = np.linalg.inv(E)      #inv (and not transp.) bc not sure to be always orthonormal
        lambda_mat = np.array([[self.lambda_DS, 0.0], [0.0, self.lambda_perp]])
        return E@lambda_mat@E_inv

    def update_D_matrix_wrt_obs(self):
        if self.obs_dist_list.shape[0] > 1:
            raise NotImplementedError("passivity for obs only for 1 obs")
        if not self.obs_dist_list: #no obstacles
            return self.D
        for normal, dist in zip(self.obs_normals_list.T, self.obs_dist_list):
            #only implemented for 1 obs
            if dist <= 0: #if dist <0 where IN the obs
                return self.D

            E = get_orthogonal_basis(normal)
            #E[:,[0,1]] = E[:,[1,0]]
            E_inv = np.linalg.inv(E)

            lambda_mat = np.array([[self.lambda_obs_scaling/dist, 0.0], [0.0, self.lambda_perp]])
            #limit to avoid num error w/ rk4
            lambda_mat[lambda_mat > mn.LAMBDA_MAX] = mn.LAMBDA_MAX 
            

            return E@lambda_mat@E_inv

    #not used
    def update_D_matrix_wrt_both_trash(self, x):
        D_obs = self.update_D_matrix_wrt_obs()
        D_ds = self.update_D_matrix_wrt_DS(x)

        return D_ds*D_obs

    def update_D_matrix_wrt_both(self, x):
        if not self.obs_dist_list: #no obstacles
            return self.update_D_matrix_wrt_DS(self, x)
        
        #DS, attention si /0
        x_dot_des = self.dynamic_avoider.evaluate(x)
        e1_DS = x_dot_des/np.linalg.norm(x_dot_des)

        #obs
        if self.obs_dist_list.shape[0] > 1:
            raise NotImplementedError("passivity for obs only for 1 obs")
        for normal, dist in zip(self.obs_normals_list.T, self.obs_dist_list):
            e2_obs = normal
            if dist > 0:
                weight = max(0.0, 1.0 - dist/mn.DIST_CRIT)
            else:
                weight = 0

        #both
        perp_to_e1_DS = np.array([e1_DS[1], -e1_DS[0]]) #quel perp prendre ? 
        if np.dot(e2_obs, perp_to_e1_DS) < 0:
            perp_to_e1_DS = -perp_to_e1_DS

        e2_both = weight*e2_obs + (1-weight)*perp_to_e1_DS #prob avec le sens
        e2_both = e2_both/np.linalg.norm(e2_both)
            
        E = np.array([e1_DS, e2_both]).T #check if good
        E_inv = np.linalg.inv(E)

        lambda_comp = max(0,
         self.lambda_obs_scaling/dist - self.lambda_obs_scaling/mn.DIST_CRIT ) + self.lambda_perp
        lambda_comp = min(lambda_comp, mn.LAMBDA_MAX)
        #check if big enough
         

        lambda_mat = np.array([[self.lambda_DS, 0], [0, lambda_comp]])

        return E*lambda_mat*E_inv



