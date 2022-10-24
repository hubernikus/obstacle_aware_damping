import numpy as np
from abc import ABC, abstractmethod
import time

#librairies of lukas
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.utils import get_orthogonal_basis, warnings

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
    def __init__(
        self,
        D = 10*np.eye(mn.DIM), 
        K = 100*np.eye(mn.DIM),
        G = np.zeros(mn.DIM),
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
    def __init__(
        self,
        dynamic_avoider:ModulationAvoider,
        G = np.zeros(mn.DIM),
        lambda_DS = 100.0, #to follow DS line
        lambda_perp = 20.0, #compliance perp to DS or obs
        lambda_obs_scaling = 20.0, #scaling factor prop to distance to obstacles, to be stiff
        type_of_D_matrix = TypeD.DS_FOLLOWING
        
    ):
        self.dynamic_avoider = dynamic_avoider
        self.G = G

        self.lambda_DS = lambda_DS
        self.lambda_perp = lambda_perp
        self.lambda_obs_scaling = lambda_obs_scaling

        self.D = np.array([[self.lambda_DS, 0.0],[0.0, self.lambda_perp]])

        #self.lambda_mat = np.diag(np.array([self.lambda_DS, self.lambda_perp_DS]))

        self.obs_normals_list = np.empty((mn.DIM, 0))
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

    def update_D_matrix(self, x, xdot):
        if self.type_of_D_matrix == TypeD.DS_FOLLOWING:
            D = self.update_D_matrix_wrt_DS(x)
        elif self.type_of_D_matrix == TypeD.OBS_PASSIVITY:
            D = self.update_D_matrix_wrt_obs(xdot)
        elif self.type_of_D_matrix == TypeD.BOTH:
            D = self.update_D_matrix_wrt_both_ortho_basis(x, xdot)
        self.D = D

    
    def update_D_matrix_wrt_DS(self, x):
        x_dot_des = self.dynamic_avoider.evaluate(x)
        E = get_orthogonal_basis(x_dot_des)
        E_inv = np.linalg.inv(E)      #inv (and not transp.) bc not sure to be always orthonormal
        lambda_mat = np.array([[self.lambda_DS, 0.0], [0.0, self.lambda_perp]])
        return E@lambda_mat@E_inv

    def update_D_matrix_wrt_obs(self, xdot):
        if not self.obs_dist_list: #no obstacles
            return self.D

        if self.obs_dist_list.shape[0] > 1:
            raise NotImplementedError("passivity for obs only for 1 obs")
        
        for normal, dist in zip(self.obs_normals_list.T, self.obs_dist_list):
            #only implemented for 1 obs
            if dist <= 0: #if dist <0 where IN the obs
                return self.D
            weight = max(0.0, 1.0 - dist/mn.DIST_CRIT)
            e2 = normal

        e1 = np.array([e2[1], -e2[0]])
        E = np.array([e1, e2]).T
        #E[:,[0,1]] = E[:,[1,0]]
        E_inv = np.linalg.inv(E)

        lambda_2 = (1-weight)*self.lambda_perp + weight*mn.LAMBDA_MAX

        #if we go away from obs, we relax the stiffness
        if np.dot(xdot, e2) > 0:
            lambda_2 = self.lambda_perp

        lambda_mat = np.array([[self.lambda_perp, 0.0], [0.0, lambda_2]])
        
        #limit to avoid num error w/ rk4
        #lambda_mat[lambda_mat > mn.LAMBDA_MAX] = mn.LAMBDA_MAX 
        

        return E@lambda_mat@E_inv

    #not used -> retest
    def update_D_matrix_wrt_both_mat_mul(self, x, xdot):
        D_obs = self.update_D_matrix_wrt_obs(xdot)
        D_ds = self.update_D_matrix_wrt_DS(x)

        D_tot = D_ds@D_obs
        #D_tot[D_tot > 200.] = 200.

        return D_tot

    def update_D_matrix_wrt_both_ortho_basis(self, x, xdot):
        if not self.obs_dist_list: #no obstacles
            return self.update_D_matrix_wrt_DS(x)
        
        #DS, attention si /0
        x_dot_des = self.dynamic_avoider.evaluate(x)
        e1_DS = x_dot_des/np.linalg.norm(x_dot_des)

        perp_to_e1_DS = np.array([e1_DS[1], -e1_DS[0]])

        #obs
        if self.obs_dist_list.shape[0] > 1:
            raise NotImplementedError("passivity for obs only for 1 obs")
        for normal, dist in zip(self.obs_normals_list.T, self.obs_dist_list):
            e2_obs = normal
            if dist > 0: #we're in the obs
                weight = max(0.0, 1.0 - dist/mn.DIST_CRIT)
            else:
                weight = 0
        
        # voluntary construct in the other dir as ds
        perp_to_e2_obs = np.array([-e2_obs[1], e2_obs[0]])

        #both
        #finding closest perpendicular
        if np.dot(e2_obs, perp_to_e1_DS) < 0:
            perp_to_e1_DS = -perp_to_e1_DS

        if np.dot(e1_DS, perp_to_e2_obs) < 0:
            perp_to_e2_obs = -perp_to_e2_obs

        #weighting to still be about orthonormal ?? 
        e1_both = weight*perp_to_e2_obs + (1-weight)*e1_DS
        e1_both = e1_both/np.linalg.norm(e1_both)

        e2_both = weight*e2_obs + (1-weight)*perp_to_e1_DS #prob avec le sens
        e2_both = e2_both/np.linalg.norm(e2_both)

        print("dot", np.dot(e1_both, e2_both))

        #extreme case sould never happen
        if 1 - np.abs(np.dot(e1_both, e2_both)) < 1e-6:
            raise("What did trigger this, it shouldn't")
            
        E = np.array([e1_both, e2_both]).T
        E_inv = np.linalg.inv(E)

        #check if big enough
        lambda_1 = (1-weight)*self.lambda_DS #good ??
        lambda_2 = (1-weight)*self.lambda_perp + weight*mn.LAMBDA_MAX

        #if we go away from obs, we relax the stiffness
        if np.dot(xdot, e2_obs) > 0:
            lambda_2 = self.lambda_perp

        lambda_mat = np.array([[lambda_1, 0], [0, lambda_2]])
        ret = E@lambda_mat@E_inv
        return ret

    def update_D_matrix_wrt_both_not_orthogonal(self, x, xdot):
        if not self.obs_dist_list: #no obstacles
            return self.update_D_matrix_wrt_DS(x)
        
        #DS, attention si /0
        x_dot_des = self.dynamic_avoider.evaluate(x)
        e1_DS = x_dot_des/np.linalg.norm(x_dot_des)

        perp_to_e1_DS = np.array([e1_DS[1], -e1_DS[0]])

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
        #finding closest perpendicular
        if np.dot(e2_obs, perp_to_e1_DS) < 0:
            perp_to_e1_DS = -perp_to_e1_DS

        #e2 is a combinaison of compliance perp to DS and away from obs
        e2_both = weight*e2_obs + (1-weight)*perp_to_e1_DS

        #extreme case were we are on the boundary of obs + exactly at the saddle point of 
        # the obstacle avoidance ds modulation -> sould never realy happends
        if np.abs(np.dot(e1_DS, e2_both)) == 1:
            warnings.warn("Extreme case")
            return self.D

        #building basis matrix
        E = np.array([e1_DS, e2_both]).T #check if good
        E_inv = np.linalg.inv(E) 


        #HERE TEST THINGS
        #TEST 1
        # lambda_1 = (1-weight)*self.lambda_DS
        # lambda_2 = (1-weight)*self.lambda_perp + weight*mn.LAMBDA_MAX ## nothing better ?

        #TEST 2 - BETTER
        lambda_1 = self.lambda_DS
        lambda_2 = self.lambda_perp + weight*(mn.LAMBDA_MAX - self.lambda_perp)

        #if we go away from obs, we relax the stiffness
        if np.dot(xdot, e2_obs) > 0:
            lambda_2 = self.lambda_perp
        
        lambda_mat = np.array([[lambda_1, 0], [0, lambda_2]])
        D = E@lambda_mat@E_inv
        return D

