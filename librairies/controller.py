import numpy as np
from abc import ABC, abstractmethod

#librairies of lukas
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.utils import get_orthogonal_basis


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
        lambda_obs = 20.0, #suposed to be perp. to obs
        
    ):
        self.dynamic_avoider = dynamic_avoider
        self.D = D
        self.G = G
        self.lambda_mat = np.diag(np.array([lambda_DS, lambda_obs]))

        self.obs_normals_list = np.empty((self.dim, 0))
        self.obs_gamma_list = np.empty(0)


    def compute_tau_c(self, x, xdot):
        """
        return the torque control command of the DS-tracking controller,
        """
        x_dot_des = self.dynamic_avoider.evaluate(x)
        tau_c = self.G - self.D@(xdot - x_dot_des)
        return tau_c
    
    def update_D_matrix(self, x):
        #je viens de modif ca
        x_dot_des = self.dynamic_avoider.evaluate(x)
        E = get_orthogonal_basis(x_dot_des)
        E_inv = np.linalg.inv(E)      #inv (and not transp.) bc not sure to be always orthonormal
        self.D = E@self.lambda_mat@E_inv

    def update_D_matrix_wrt_obs(self):
        lambda_perp = 20
        lambda_obs_scaling = 20
        if self.obs_gamma_list.shape[0] > 1:
            raise NotImplementedError("passivity for obs only for 1 obs")
        for normal, gamma in zip(self.obs_normals_list.T, self.obs_gamma_list):
            #only implemented for 1 obs
            E = get_orthogonal_basis(normal)
            E_inv = np.linalg.inv(E)
            
            self.lambda_mat = np.array([[lambda_obs_scaling/(gamma-1), 0.0], [0.0, 20.0]])
            self.lambda_mat[self.lambda_mat > 200.0] = 200.0 #limit to avoid num error w/ rk4

        self.D = E@self.lambda_mat@E_inv

