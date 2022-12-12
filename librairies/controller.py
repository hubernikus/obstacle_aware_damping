import numpy as np
from abc import ABC, abstractmethod
import time
import warnings

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
    def __init__(
        self,
        D = 10*np.eye(mn.DIM), 
        K = 100*np.eye(mn.DIM),
    ):
        self.D = D
        self.K = K

    def compute_tau_c(self, x, xdot):
        """
        return the torque control command of the regulation controller,
        """
        return mn.G - np.matmul(self.D, xdot) - np.matmul(self.K, x)


class TrackingController(Controller):
    """
    in the form tau_c = G - D(xdot - f_desired(x)),
    Now also with energy tank system
    """
    def __init__(
        self,
        dynamic_avoider:ModulationAvoider,
        lambda_DS = 100.0, #to follow DS line
        lambda_perp = 20.0, #compliance perp to DS or obs
        lambda_obs = mn.LAMBDA_MAX,
        type_of_D_matrix = TypeD.BOTH,
        ortho_basis_approach = False,
        with_E_storage = False,
        
    ):
        self.dynamic_avoider = dynamic_avoider

        self.lambda_DS = lambda_DS
        self.lambda_perp = lambda_perp
        self.lambda_obs = lambda_obs

        if mn.DIM == 2:
            self.D = np.array([[self.lambda_DS, 0.0],
                               [0.0, self.lambda_perp]])
        elif mn.DIM == 3:
            self.D = np.array([[self.lambda_DS, 0.0, 0.0],
                               [0.0, self.lambda_perp, 0.0],
                               [0.0, 0.0, self.lambda_perp]])
        else:
            raise NotImplementedError("unknown dimension")

        self.obs_normals_list = np.empty((mn.DIM, 0))
        self.obs_dist_list = np.empty(0)

        self.type_of_D_matrix = type_of_D_matrix
        self.ortho_basis_approach = ortho_basis_approach

        #energy tank
        self.with_E_storage = with_E_storage
        self.s = mn.S_MAX/2
        self.z = 0 #initialized later

    def compute_tau_c(self, x, xdot):
        """
        return the torque control command of the DS-tracking controller,
        now also with energy storage
        """
    
        if not self.with_E_storage:
            x_dot_des = self.dynamic_avoider.evaluate(x)
            tau_c = mn.G - self.D@(xdot - x_dot_des)

            #physical constrains, value ? -> laisser ???
            tau_c[tau_c > mn.MAX_TAU_C] = mn.MAX_TAU_C
            tau_c[tau_c < -mn.MAX_TAU_C] = -mn.MAX_TAU_C

            return tau_c
        
        f_c, f_r = self.decomp_f(x)
        beta_r = self.get_beta_r()

        tau_c = mn.G - self.D@xdot + self.lambda_DS*f_c + beta_r*self.lambda_DS*f_r
        #limit tau_c ?? physical constrains

        return tau_c

    def update_D_matrix(self, x, xdot):
        # if np.linalg.norm(self.dynamic_avoider.evaluate(x)) < mn.EPS_CONVERGENCE:
        #     return

        if self.type_of_D_matrix == TypeD.DS_FOLLOWING:
            D = self.compute_D_matrix_wrt_DS(x)
        elif self.type_of_D_matrix == TypeD.OBS_PASSIVITY:
            D = self.compute_D_matrix_wrt_obs(xdot)
        elif self.type_of_D_matrix == TypeD.BOTH:
            if self.ortho_basis_approach:
                D = self.compute_D_matrix_wrt_both_ortho_basis(x, xdot)
            else: #probably better
                D = self.compute_D_matrix_wrt_both_not_orthogonal(x, xdot)
                
        #improve stability at atractor, not recompute always D???
        if np.linalg.norm(x - self.dynamic_avoider.initial_dynamics.attractor_position) < mn.EPS_CONVERGENCE:
            #D = np.array([[mn.LAMBDA_MAX, 0.0], [0.0, mn.LAMBDA_MAX]])
            #return
            pass #curently doing nothing

        #update the damping matrix
        self.D = D

    
    def compute_D_matrix_wrt_DS(self, x):
        #get desired velocity 
        x_dot_des = self.dynamic_avoider.evaluate(x)

        #construct the basis matrix, align to the DS direction 
        E = get_orthogonal_basis(x_dot_des)
        E_inv = np.linalg.inv(E)

        #contruct the matrix containing the damping coefficient along selective directions
        if mn.DIM == 2:
            lambda_mat = np.array([[self.lambda_DS, 0.0],
                                   [0.0, self.lambda_perp]])
        else:
            lambda_mat = np.array([[self.lambda_DS, 0.0, 0.0], 
                                   [0.0, self.lambda_perp, 0.0],
                                   [0.0, 0.0, self.lambda_perp]])

        #compose the damping matrix
        D = E@lambda_mat@E_inv
        return D

    def compute_D_matrix_wrt_obs(self, xdot):
        #if there is no obstacles, we just keep D as it is
        if not np.size(self.obs_dist_list):
            return self.D
        
        weight = 0
        e2 = np.zeros(mn.DIM)
        #get the normals and compute the weight of the obstacles
        for normal, dist in zip(self.obs_normals_list.T, self.obs_dist_list):
            #if dist <0 we're IN the obs
            if dist <= 0: 
                return self.D

            #weight is 1 at the boundary, 0 at a distance DIST_CRIT from the obstacle
            weight_i = max(0.0, 1.0 - dist/mn.DIST_CRIT)
            if weight_i > weight:
                weight = weight_i

            #e2 is a weighted linear combianation of all the normals
            e2 += normal/(dist + 1)

        e2_norm = np.linalg.norm(e2)
        if not e2_norm:
            #resultant normal not defined
            #what to do ?? -> return prev mat
            return self.D
        e2 = e2/e2_norm

        #construct the basis matrix, align with the normal of the obstacle
        
        if mn.DIM == 2:
            e1 = np.array([e2[1], -e2[0]])
            E = np.array([e1, e2]).T
            E_inv = np.linalg.inv(E)
        else :
            e1 = np.array([e2[1], -e2[0], 0.0]) # we have a degree of freedom -> 0.0
            if not any(e1): #e1 is [0,0,0]
                e1 = np.array([-e2[2], 0.0, e2[0]])
            e3 = np.cross(e1,e2)
            E = np.array([e1, e2, e3]).T
            E_inv = np.linalg.inv(E)

        #compute the damping coeficients along selective directions
        lambda_1 = self.lambda_perp
        lambda_2 = (1-weight)*self.lambda_perp + weight*self.lambda_obs
        lambda_3 = self.lambda_perp

        #if we go away from obs, we relax the stiffness
        if np.dot(xdot, e2) > 0:
            lambda_2 = self.lambda_perp

        #contruct the matrix containing the damping coefficient along selective directions
        if mn.DIM == 2:
            lambda_mat = np.array([[lambda_1, 0.0], [0.0, lambda_2]])
        else:
            lambda_mat = np.array([[lambda_1, 0.0, 0.0], 
                                   [0.0, lambda_2, 0.0],
                                   [0.0, 0.0, lambda_3]])
        
        #compose the damping matrix
        D = E@lambda_mat@E_inv
        return D

    #not used -> doesn't work
    def compute_D_matrix_wrt_both_mat_mul(self, x, xdot):
        D_obs = self.compute_D_matrix_wrt_obs(xdot)
        D_ds = self.compute_D_matrix_wrt_DS(x)

        D_tot = D_ds@D_obs
        #D_tot[D_tot > 200.] = 200.

        return D_tot

    def compute_D_matrix_wrt_both_ortho_basis(self, x, xdot):
        #if there is no obstacles, we want D to be stiff w.r.t. the DS
        if not np.size(self.obs_dist_list):
            return self.compute_D_matrix_wrt_DS(x)

        #get desired velocity - THAT takes long to compute
        x_dot_des = self.dynamic_avoider.evaluate(x)

        #if the desired velocity is too small, we risk numerical issue
        if np.linalg.norm(x_dot_des) < mn.EPSILON:
            #we just return the previous damping matrix
            return self.D
        
        #compute the vector align to the DS
        e1_DS = x_dot_des/np.linalg.norm(x_dot_des)

        #compute vector relative to obstacles
        weight = 0
        e2_obs = np.zeros(mn.DIM)
        #get the normals and compute the weight of the obstacles
        for normal, dist in zip(self.obs_normals_list.T, self.obs_dist_list):
            #if dist <0 we're IN the obs
            if dist <= 0: 
                return self.D

            #weight is 1 at the boundary, 0 at a distance DIST_CRIT from the obstacle
            #keep only biggest weight -> closer obstacle
            weight_i = max(0.0, 1.0 - dist/mn.DIST_CRIT)
            if weight_i > weight:
                weight = weight_i

            #e2_obs is a weighted linear combianation of all the normals
            e2_obs += normal/(dist + 1)

        e2_obs_norm = np.linalg.norm(e2_obs)
        if not e2_obs_norm:
            #resultant normal not defined
            #what to do ?? -> return prev mat
            return self.D
        e2_obs = e2_obs/e2_obs_norm

        # compute the basis of the DS : [e1_DS, e2_DS] or 3d
        if mn.DIM == 2:
            e2_DS = np.array([e1_DS[1], -e1_DS[0]])
            # construct the basis align with the normal of the obstacle
            # not that the normal to e2_obs is volontarly computed unlike the normal to e1_DS
            # this play a crucial role for the limit case
            e1_obs = np.array([-e2_obs[1], e2_obs[0]])
        else:
            e3 = np.cross(e1_DS, e2_obs)
            norm_e3 = np.linalg.norm(e3)
            if not norm_e3: #limit case if e1_DS//e2_obs -> DS aligned w/ normal
                warnings.warn("Limit case")
                return self.D #what to do ??
            else : 
                e3 = e3/norm_e3
            e2_DS = np.cross(e3, e1_DS)
            e1_obs = np.cross(e2_obs, e3)

        # we want both basis to be cointained in the same half-plane /space
        # we have this liberty since we always have 2 choice when choosing a perpendiular
        if np.dot(e2_obs, e2_DS) < 0:
            e2_DS = -e2_DS
        if np.dot(e1_DS, e1_obs) < 0:
            e1_obs = -e1_obs

        #we construct a new basis which is always othtonormal but "rotates" based of weight
        e1_both = weight*e1_obs + (1-weight)*e1_DS
        e1_both = e1_both/np.linalg.norm(e1_both)

        e2_both = weight*e2_obs + (1-weight)*e2_DS
        e2_both = e2_both/np.linalg.norm(e2_both)

        #extreme case sould never happen, e1, e2 are suposed to be a orthonormal basis
        if 1 - np.abs(np.dot(e1_both, e2_both)) < mn.EPSILON:
            raise("What did trigger this, it shouldn't")
            
        #construct the basis matrix
        if mn.DIM == 2:
            E = np.array([e1_DS, e2_both]).T
            E_inv = np.linalg.inv(E)
        else :
            E = np.array([e1_DS, e2_both, e3]).T
            E_inv = np.linalg.inv(E)

        #compute the damping coeficients along selective directions
        lambda_1 = self.lambda_DS 
        #lambda_1 = (1-weight)*self.lambda_DS #good ??
        lambda_2 = (1-weight)*self.lambda_perp + weight*self.lambda_obs
        lambda_3 = self.lambda_perp

        #if we go away from obs, we relax the stiffness
        if np.dot(xdot, e2_obs) > 0:
            lambda_2 = self.lambda_perp

        #contruct the matrix containing the damping coefficient along selective directions
        if mn.DIM == 2:
            lambda_mat = np.array([[lambda_1, 0.0], [0.0, lambda_2]])
        else:
            lambda_mat = np.array([[lambda_1, 0.0, 0.0], 
                                   [0.0, lambda_2, 0.0],
                                   [0.0, 0.0, lambda_3]])
        
        #compose the damping matrix
        D = E@lambda_mat@E_inv
        return D

    def compute_D_matrix_wrt_both_not_orthogonal(self, x, xdot):
        #if there is no obstacles, we want D to be stiff w.r.t. the DS
        if not np.size(self.obs_dist_list):
            return self.compute_D_matrix_wrt_DS(x)

        #get desired velocity
        x_dot_des = self.dynamic_avoider.evaluate(x)

        #if the desired velocity is too small, we risk numerical issue
        if np.linalg.norm(x_dot_des) < mn.EPSILON:
            #we just return the previous damping matrix
            return self.D
        
        #compute the vector align to the DS
        e1_DS = x_dot_des/np.linalg.norm(x_dot_des)
        
        #compute vector relative to obstacles
        weight = 0
        e2_obs = np.zeros(mn.DIM)
        #get the normals and compute the weight of the obstacles
        for normal, dist in zip(self.obs_normals_list.T, self.obs_dist_list):
            #if dist <0 we're IN the obs
            if dist <= 0: 
                return self.D

            #weight is 1 at the boundary, 0 at a distance DIST_CRIT from the obstacle
            #keep only biggest wight -> closer obstacle
            weight_i = max(0.0, 1.0 - dist/mn.DIST_CRIT)
            if weight_i > weight:
                weight = weight_i

            #e2_obs is a weighted linear combianation of all the normals
            e2_obs += normal/(dist + 1)

        e2_obs_norm = np.linalg.norm(e2_obs)
        if not e2_obs_norm:
            #resultant normal not defined
            #what to do ?? -> return prev mat
            return self.D
        e2_obs = e2_obs/e2_obs_norm


        # compute the basis of the DS : [e1_DS, e2_DS] or 3d
        if mn.DIM == 2:
            e2_DS = np.array([e1_DS[1], -e1_DS[0]])
        else:
            e3 = np.cross(e1_DS, e2_obs)
            if not np.any(e3): #limit case if e1_DS//e2_obs -> DS aligned w/ normal
                warnings.warn("Limit case")
                return self.D #what to do ??
            e2_DS = np.cross(e3, e1_DS)

        # we want both e2 to be cointained in the same half-plane/space
        # -> their weighted addition will not be collinear to e1_DS
        # we have this liberty since we always have 2 choice when choosing a perpendiular
        if np.dot(e2_obs, e2_DS) < 0:
            e2_DS = -e2_DS

        #e2 is a combinaison of compliance perp to DS and away from obs
        e2_both = weight*e2_obs + (1-weight)*e2_DS
        e2_both = e2_both/np.linalg.norm(e2_both)

        # extreme case were we are on the boundary of obs + exactly at the saddle point of 
        # the obstacle avoidance ds modulation -> sould never realy happends
        if np.abs(np.dot(e1_DS, e2_both)) == 1:
            warnings.warn("Extreme case")
            return self.D

        #construct the basis matrix
        if mn.DIM == 2:
            E = np.array([e1_DS, e2_both]).T
            E_inv = np.linalg.inv(E)
        else :
            E = np.array([e1_DS, e2_both, e3]).T
            E_inv = np.linalg.inv(E)

        #HERE TEST THINGS
        #TEST 1
        #lambda_1 = (1-weight)*self.lambda_DS
        # lambda_2 = (1-weight)*self.lambda_perp + weight*self.lambda_obs ## nothing better ?

        #TEST 2 - BETTER
        #compute the damping coeficients along selective directions
        lambda_1 = self.lambda_DS
        lambda_2 = (1-weight)*self.lambda_perp + weight*self.lambda_obs
        lambda_3 = self.lambda_perp

        #if we go away from obs, we relax the stiffness
        if np.dot(xdot, e2_obs) > 0:
            lambda_2 = self.lambda_perp
            pass
        
        #contruct the matrix containing the damping coefficient along selective directions
        if mn.DIM == 2:
            lambda_mat = np.array([[lambda_1, 0.0], [0.0, lambda_2]])
        else:
            lambda_mat = np.array([[lambda_1, 0.0, 0.0], 
                                   [0.0, lambda_2, 0.0],
                                   [0.0, 0.0, lambda_3]])
        
        #compose the damping matrix
        D = E@lambda_mat@E_inv
        return D

    #ENERGY TANK RELATED
    #all x, xdot must be from actual step and not future (i.e. must not be updated yet -> use temp var...)
    # D must not be updated yet
    def update_energy_tank(self, x, xdot, dt):

        if not self.with_E_storage:
            #avoid useless computation
            return

        _, f_r = self.decomp_f(x)
        self.z = xdot.T@f_r
        
        alpha = self.get_alpha()
        beta_s = self.get_beta_s()

        #using euler 1st order
        sdot = alpha*xdot.T@self.D@xdot - beta_s*self.lambda_DS*self.z
        self.s += dt*sdot
    
    def decomp_f(self, x):
        f_c = self.dynamic_avoider.initial_dynamics.evaluate(x) #unmodulated dynamic : conservative (if DS is lin.)
        f_r = self.dynamic_avoider.evaluate(x) - f_c
        return f_c, f_r

    def get_alpha(self):
        #return smooth_step_neg(mn.S_MAX-mn.DELTA_S, mn.S_MAX, self.s) #used in paper
        #return smooth_step_neg(0, mn.DELTA_S, self.s)

        #centered at S_MAX/2
        return smooth_step_neg(mn.S_MAX/2 - mn.DELTA_S, mn.S_MAX/2 + mn.DELTA_S, self.s)

    def get_beta_s(self):
        # ret = 1 - smooth_step(-mn.DELTA_Z, 0, self.z)*smooth_step_neg(mn.S_MAX, mn.S_MAX + mn.DELTA_S, self.s) \
        #         - smooth_step_neg(0, mn.DELTA_Z, self.z)*smooth_step(mn.S_MAX - mn.DELTA_S, mn.S_MAX, self.s)
        
        #modif at second H -> was a mistake
        ret = 1 - smooth_step(-mn.DELTA_Z, 0, self.z)*smooth_step_neg(0, mn.DELTA_S, self.s)\
                - smooth_step_neg(0, mn.DELTA_Z, self.z)*smooth_step(mn.S_MAX - mn.DELTA_S, mn.S_MAX, self.s)
        return ret

    def get_beta_r(self):
        # ret = (1 - smooth_step(-mn.DELTA_Z, 0, self.z)*smooth_step_neg(mn.S_MAX, mn.S_MAX + mn.DELTA_S, self.s)) \
        #       * (1 - (smooth_step(-mn.DELTA_Z, 0, self.z)* \
        #              smooth_step_neg(0, mn.DELTA_Z, self.z)*smooth_step(mn.S_MAX - mn.DELTA_S, mn.S_MAX, self.s)))
       
        #modif at second H -> was amistake
        ret = (1 - smooth_step(-mn.DELTA_Z, 0, self.z)*smooth_step_neg(0, mn.DELTA_S, self.s)) \
              * (1 - (smooth_step(-mn.DELTA_Z, 0, self.z)* \
                     smooth_step_neg(0, mn.DELTA_Z, self.z)* \
                     smooth_step(mn.S_MAX - mn.DELTA_S, mn.S_MAX, self.s)))
        return ret

### HELPER FUNTION ###

#helper smooth step functions
def smooth_step(a,b,x):
    if x < a: 
        return 0
    if x > b:
        return 1
    else:
        return 6*((x-a)/(b-a))**5 - 15*((x-a)/(b-a))**4 + 10*((x-a)/(b-a))**3

def smooth_step_neg(a,b,x):
    return 1 - smooth_step(a,b,x)