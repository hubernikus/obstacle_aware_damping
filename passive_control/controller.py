import numpy as np
from abc import ABC, abstractmethod
import time
import warnings

#passive_control.of lukas
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.utils import get_orthogonal_basis

#my passive_control.
from passive_control.magic_numbers_and_enums import TypeOfDMatrix as TypeD
from passive_control.magic_numbers_and_enums import Approach
import passive_control.magic_numbers_and_enums as mn


class Controller(ABC):
    """
    interface controller template
    """

    @abstractmethod
    def compute_tau_c():
        pass


class RegulationController(Controller):
    """
    not used anymore (but may work)
    in the form tau_c = G - D*x_dot - K*x , does regulation to 0
    """
    def __init__(
        self,
        DIM = 2,
        D = None, 
        K = None,
        G = None,
    ):
        self.DIM = DIM
        self.D = D
        if D is None:
            self.D = 10*np.eye(self.DIM)

        self.K = K
        if K is None:
            self.K = 100*np.eye(self.DIM)
        
        self.G = G
        if G is None:
            self.G = np.zeros(self.DIM)

    def compute_tau_c(self, x, xdot):
        """
        return the torque control command of the regulation controller,
        """
        return self.G - np.matmul(self.D, xdot) - np.matmul(self.K, x)


class TrackingController(Controller):
    """
    in the form tau_c = G - D(xdot - f_desired(x)),
    Now also with energy tank system
    """
    def __init__(
        self,
        dynamic_avoider:ModulationAvoider,
        DIM = 2,
        G = None,
        lambda_DS = 100.0, #to follow DS line
        lambda_perp = 20.0, #compliance perp to DS or obs
        lambda_obs = mn.LAMBDA_MAX,
        type_of_D_matrix = TypeD.BOTH,
        approach = Approach.ORTHO_BASIS,
        with_E_storage = False,
        
    ):
        self.DIM = DIM

        self.dynamic_avoider = dynamic_avoider

        self.G = G
        if G is None:
            self.G = np.zeros(self.DIM)

        self.lambda_DS = lambda_DS
        self.lambda_perp = lambda_perp
        self.lambda_obs = lambda_obs

        if self.DIM == 2:
            self.D = np.array([[self.lambda_DS, 0.0],
                               [0.0, self.lambda_perp]])
        elif self.DIM == 3:
            self.D = np.array([[self.lambda_DS, 0.0, 0.0],
                               [0.0, self.lambda_perp, 0.0],
                               [0.0, 0.0, self.lambda_perp]])
        else:
            raise NotImplementedError("unknown dimension")

        self.obs_normals_list = np.empty((self.DIM, 0))
        self.obs_dist_list = np.empty(0)

        self.type_of_D_matrix = type_of_D_matrix
        self.approach = approach

        #energy tank
        self.with_E_storage = with_E_storage
        self.s = mn.S_MAX/2
        self.z = 0 #initialized later

    def compute_tau_c(self, x, xdot):
        """
        return the torque control command of the DS-tracking controller,
        now also with energy storage
        """
    
        if not (self.with_E_storage and self.approach == Approach.ORTHO_BASIS and\
                self.type_of_D_matrix ==TypeD.BOTH):
            x_dot_des = self.dynamic_avoider.evaluate(x)
            tau_c = self.G - self.D@(xdot - x_dot_des)

            #physical constrains, value ?
            tau_c[tau_c > mn.MAX_TAU_C] = mn.MAX_TAU_C
            tau_c[tau_c < -mn.MAX_TAU_C] = -mn.MAX_TAU_C

            return tau_c
        
        #f_c, f_r = self.decomp_f(x)
        #beta_r = self.get_beta_r()

        beta_r_list = self.get_beta_r_list()
        #print(beta_r_list)

        #tau_c = self.G - self.D@xdot + self.lambda_DS*f_c + beta_r*self.lambda_DS*f_r
        tau_c = self.G - self.D@xdot + np.sum(self.f_decomp*beta_r_list*np.diag(self.lambda_mat), axis = 1)

        return tau_c

    def update_D_matrix(self, x, xdot):
        """
        General function that chose the way of updating the damping matix
        """
        # if np.linalg.norm(self.dynamic_avoider.evaluate(x)) < mn.EPS_CONVERGENCE:
        #     return

        if self.type_of_D_matrix == TypeD.DS_FOLLOWING:
            D = self.compute_D_matrix_wrt_DS(x)
        elif self.type_of_D_matrix == TypeD.OBS_PASSIVITY:
            D = self.compute_D_matrix_wrt_obs(xdot)
        elif self.type_of_D_matrix == TypeD.BOTH:
            #chose the way to control
            if self.approach == Approach.ORTHO_BASIS: #better
                D = self.compute_D_matrix_wrt_both_ortho_basis(x, xdot)
            elif self.approach == Approach.NON_ORTHO_BASIS:
                D = self.compute_D_matrix_wrt_both_not_orthogonal(x, xdot)
            elif self.approach == Approach.WEIGHT_DS_OBS_MAT:
                D = self.compute_D_matrix_wrt_both_mat_add(x, xdot)
            elif self.approach == Approach.WEIGHT_DS_OBS_MAT_V2:
                D = self.compute_D_matrix_wrt_both_mat_add_v2(x, xdot)
                
        #improve stability at atractor, not recompute always D???
        if np.linalg.norm(x - self.dynamic_avoider.initial_dynamics.attractor_position) < mn.EPS_CONVERGENCE:
            #D = np.array([[mn.LAMBDA_MAX, 0.0], [0.0, mn.LAMBDA_MAX]])
            #return
            pass #curently doing nothing

        #update the damping matrix
        self.D = D

    
    def compute_D_matrix_wrt_DS(self, x):
        """
        Computes the damping matrix, without considering obstacles.
        This implements the traditional passive control
        """
        #get desired velocity 
        x_dot_des = self.dynamic_avoider.evaluate(x)

        #construct the basis matrix, align to the DS direction 
        self.E = get_orthogonal_basis(x_dot_des)
        E_inv = np.linalg.inv(self.E)

        #contruct the matrix containing the damping coefficient along selective directions
        if self.DIM == 2:
            self.lambda_mat = np.array([[self.lambda_DS, 0.0],
                                   [0.0, self.lambda_perp]])
        else:
            self.lambda_mat = np.array([[self.lambda_DS, 0.0, 0.0], 
                                   [0.0, self.lambda_perp, 0.0],
                                   [0.0, 0.0, self.lambda_perp]])

        #compose the damping matrix
        D = self.E@self.lambda_mat@E_inv
        return D

    def compute_D_matrix_wrt_obs(self, xdot):
        """
        Compute the damping matri to be stiff against obstacle, 
        doesn't implement tracking stiffness
        """
        #if there is no obstacles, we just keep D as it is
        if not np.size(self.obs_dist_list):
            return self.D
        
        weight = 0
        div = 0
        e2 = np.zeros(self.DIM)
        #get the normals and compute the weight of the obstacles
        for normal, dist in zip(self.obs_normals_list.T, self.obs_dist_list):
            #if dist <0 we're IN the obs
            if dist <= 0: 
                return self.D

            #weight is 1 at the boundary, 0 at a distance DIST_CRIT from the obstacle
            weight_i = max(0.0, 1.0 - dist/mn.DIST_CRIT)
            if weight_i > weight:
                weight = weight_i

            #e2_obs is a weighted linear combianation of all the normals
            #e2_obs += normal/(dist + 1)
            e2 += normal/(dist + mn.EPSILON)
            div += 1/(dist + mn.EPSILON)

        e2 = e2/div

        e2_norm = np.linalg.norm(e2)
        if not e2_norm:
            #resultant normal not defined
            #-> return prev mat
            return self.D
        e2 = e2/e2_norm

        #construct the basis matrix, align with the normal of the obstacle
        
        if self.DIM == 2:
            e1 = np.array([e2[1], -e2[0]])
            self.E = np.array([e1, e2]).T
            E_inv = np.linalg.inv(self.E)
        else :
            e1 = np.array([e2[1], -e2[0], 0.0]) # we have a degree of freedom -> 0.0
            if not any(e1): #e1 is [0,0,0]
                e1 = np.array([-e2[2], 0.0, e2[0]])
            e3 = np.cross(e1,e2)
            self.E = np.array([e1, e2, e3]).T
            E_inv = np.linalg.inv(self.E)

        #compute the damping coeficients along selective directions
        lambda_1 = self.lambda_perp
        lambda_2 = (1-weight)*self.lambda_perp + weight*self.lambda_obs
        lambda_3 = self.lambda_perp

        #if we go away from obs, we relax the stiffness
        # if np.dot(xdot, e2) > 0:
        #     lambda_2 = self.lambda_perp
        DELTA_RELAX = 0.01 #mn.EPSILON
        res = np.dot(xdot, e2)
        lambda_2 = smooth_step(0, DELTA_RELAX, res)*self.lambda_perp +\
                   smooth_step_neg(0, DELTA_RELAX, res)*lambda_2

        #contruct the matrix containing the damping coefficient along selective directions
        if self.DIM == 2:
            self.lambda_mat = np.array([[lambda_1, 0.0], [0.0, lambda_2]])
        else:
            self.lambda_mat = np.array([[lambda_1, 0.0, 0.0], 
                                   [0.0, lambda_2, 0.0],
                                   [0.0, 0.0, lambda_3]])
        
        #compose the damping matrix
        D = self.E@self.lambda_mat@E_inv
        return D

    def compute_D_matrix_wrt_both_ortho_basis(self, x, xdot):
        """
        Method 1 :
        Compute the damping matrix with the new passive control. Will be stiff against 
        obstacles, while also beeing stiff along the direction of motion and
        compliant in the perpendicular directions.
        This function implements the first method, with D beeing a positive semi-definite matrix
        Works with many obstacles, and in 2/3 dimensions
            param : 
                x (np.array) : actual position
                xdot(np.array) : actual velocity
            return : 
                D (np.array) : the damping matrix
        """
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
        e2_obs = np.zeros(self.DIM)
        div = 0
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
            #e2_obs += normal/(dist + 1)
            e2_obs += normal/(dist + mn.EPSILON)
            div += 1/(dist + mn.EPSILON)

        e2_obs = e2_obs/div

        e2_obs_norm = np.linalg.norm(e2_obs)
        #if the normal gets too small, we reduce w->0 to just take the D_ds matrix
        delta_w_cont = 0.01
        weight = weight*smooth_step(0, delta_w_cont, e2_obs_norm)

        if e2_obs_norm:
            #resultant normal is defined
            e2_obs = e2_obs/e2_obs_norm

        # compute the basis of the DS : [e1_DS, e2_DS] or 3d
        if self.DIM == 2:
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
                #return self.D #what to do ??
                #its handle later as D->I
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
        if self.DIM == 2:
            self.E = np.array([e1_both, e2_both]).T
            E_inv = np.linalg.inv(self.E)
        else :
            self.E = np.array([e1_both, e2_both, e3]).T
            E_inv = np.linalg.inv(self.E)

        #compute the damping coeficients along selective directions
        lambda_1 = self.lambda_DS 
        #lambda_1 = (1-weight)*self.lambda_DS #good ??
        lambda_2 = (1-weight)*self.lambda_perp + weight*self.lambda_obs
        lambda_3 = self.lambda_perp
        
        #this is done such that at place where E is discutinous, D goes to identity, wich is still continouus
        y =  np.abs(np.dot(e1_DS, e2_obs))
        delta_E_cont = 0.01 #mn.EPSILON
        lambda_2 = lambda_1*smooth_step(1-delta_E_cont, 1, y) +\
                   lambda_2*smooth_step_neg(1-delta_E_cont, 1, y)
        lambda_3 = lambda_1*smooth_step(1-delta_E_cont, 1, y) +\
                   lambda_3*smooth_step_neg(1-delta_E_cont, 1, y)

        #if we go away from obs, we relax the stiffness
        # if np.dot(xdot, e2_obs) > 0:
        #     lambda_2 = self.lambda_perp
        DELTA_RELAX = 0.01 #mn.EPSILON
        res = np.dot(xdot, e2_obs)
        lambda_2 = smooth_step(0, DELTA_RELAX, res)*self.lambda_perp +\
                   smooth_step_neg(0, DELTA_RELAX, res)*lambda_2

        #contruct the matrix containing the damping coefficient along selective directions
        if self.DIM == 2:
            self.lambda_mat = np.array([[lambda_1, 0.0], [0.0, lambda_2]])
        else:
            self.lambda_mat = np.array([[lambda_1, 0.0, 0.0], 
                                   [0.0, lambda_2, 0.0],
                                   [0.0, 0.0, lambda_3]])
        
        #compose the damping matrix
        D = self.E@self.lambda_mat@E_inv
        return D

    def compute_D_matrix_wrt_both_not_orthogonal(self, x, xdot):
        """
        Method 2 :
        Compute the damping matrix with the new passive control. Will be stiff against 
        obstacles, while also beeing stiff along the direction of motion and
        compliant in the perpendicular directions.
        This function implements the second method, with e1 being always aligned with f = xdot_des
        Works with many obstacles, and in 2/3 dimensions
            param : 
                x (np.array) : actual position
                xdot(np.array) : actual velocity
            return : 
                D (np.array) : the damping matrix
        """
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
        e2_obs = np.zeros(self.DIM)
        div = 0
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
            #e2_obs += normal/(dist + 1)

            #maybe better
            e2_obs += normal/(dist + mn.EPSILON)
            div += 1/(dist + mn.EPSILON)

        e2_obs = e2_obs/div

        e2_obs_norm = np.linalg.norm(e2_obs)
        #if the normal gets too small, we reduce w->0 to just take the D_ds matrix
        delta_w_cont = 0.01
        weight = weight*smooth_step(0, delta_w_cont, e2_obs_norm)

        if e2_obs_norm:
            #resultant normal is defined
            e2_obs = e2_obs/e2_obs_norm


        # compute the basis of the DS : [e1_DS, e2_DS] or 3d
        if self.DIM == 2:
            e2_DS = np.array([e1_DS[1], -e1_DS[0]])
        else:
            e3 = np.cross(e1_DS, e2_obs)
            norm_e3 = np.linalg.norm(e3)
            if not norm_e3: #limit case if e1_DS//e2_obs -> DS aligned w/ normal
                warnings.warn("Limit case")
                #return self.D 
                #handle later as D->I
            else:
                e3 = e3/norm_e3
            e2_DS = np.cross(e3, e1_DS)

        # we want both e2 to be cointained in the same half-plane/space
        # -> their weighted addition will not be collinear to e1_DS
        # we have this liberty since we always have 2 choice when choosing a perpendiular
        if np.dot(e2_obs, e2_DS) < 0:
            e2_DS = -e2_DS

        #e2 is a combinaison of compliance perp to DS and away from obs
        e2_both = weight*e2_obs + (1-weight)*e2_DS
        e2_both = e2_both/np.linalg.norm(e2_both)

        #construct the basis matrix
        if self.DIM == 2:
            self.E = np.array([e1_DS, e2_both]).T
            E_inv = np.linalg.inv(self.E)
        else :
            self.E = np.array([e1_DS, e2_both, e3]).T
            E_inv = np.linalg.inv(self.E)

        #HERE TEST THINGS
        #TEST 1
        #lambda_1 = (1-weight)*self.lambda_DS
        # lambda_2 = (1-weight)*self.lambda_perp + weight*self.lambda_obs ## nothing better ?

        #TEST 2 - BETTER
        #compute the damping coeficients along selective directions
        lambda_1 = self.lambda_DS
        lambda_2 = (1-weight)*self.lambda_perp + weight*self.lambda_obs
        lambda_3 = self.lambda_perp

        #this is done such that at place where E is discutinous, D goes to identity, wich is still continouus
        # extreme case were we are on the boundary of obs + exactly at the saddle point of 
        # the obstacle avoidance ds modulation -> sould never realy happends
        y =  np.abs(np.dot(e1_DS, e2_obs))
        delta_E_cont = 0.01 #mn.EPSILON
        lambda_2 = lambda_1*smooth_step(1-delta_E_cont, 1, y) +\
                   lambda_2*smooth_step_neg(1-delta_E_cont, 1, y)
        lambda_3 = lambda_1*smooth_step(1-delta_E_cont, 1, y) +\
                   lambda_3*smooth_step_neg(1-delta_E_cont, 1, y)

        #if we go away from obs, we relax the stiffness
        # if np.dot(xdot, e2_obs) > 0:
        #     lambda_2 = self.lambda_perp
        DELTA_RELAX = 0.01 #mn.EPSILON
        res = np.dot(xdot, e2_obs)
        lambda_2 = smooth_step(0, DELTA_RELAX, res)*self.lambda_perp +\
                   smooth_step_neg(0, DELTA_RELAX, res)*lambda_2
        
        #contruct the matrix containing the damping coefficient along selective directions
        if self.DIM == 2:
            self.lambda_mat = np.array([[lambda_1, 0.0], [0.0, lambda_2]])
        else:
            self.lambda_mat = np.array([[lambda_1, 0.0, 0.0], 
                                   [0.0, lambda_2, 0.0],
                                   [0.0, 0.0, lambda_3]])
        
        #compose the damping matrix
        D = self.E@self.lambda_mat@E_inv
        return D

    def compute_D_matrix_wrt_both_mat_add(self, x, xdot):
        """
        Method 3 - prob wrong:
        Compute the damping matrix with the new passive control. Will be stiff against 
        obstacles, while also beeing stiff along the direction of motion and
        compliant in the perpendicular directions.
        This function implements the third method, with D beeing a positive semi-definite matrix
        but construct with a weighted comb of D_ds and D_obs.
        Works with many obstacles, and in 2/3 dimensions
            param : 
                x (np.array) : actual position
                xdot(np.array) : actual velocity
            return : 
                D (np.array) : the damping matrix
        """
        D_obs = self.compute_D_matrix_wrt_obs(xdot)
        D_ds = self.compute_D_matrix_wrt_DS(x)

        #if there is no obstacles, we want D to be stiff w.r.t. the DS
        if not np.size(self.obs_dist_list):
            return D_ds
    
        weight = 0
        div = 0
        e2 = np.zeros(self.DIM)
        #get the normals and compute the weight of the obstacles
        for normal, dist in zip(self.obs_normals_list.T, self.obs_dist_list):
            #if dist <0 we're IN the obs
            if dist <= 0: 
                return self.D

            #weight is 1 at the boundary, 0 at a distance DIST_CRIT from the obstacle
            #weight_i = max(0.0, 1.0 - dist/mn.DIST_CRIT)
            weight_i = smooth_step_neg(mn.DIST_CRIT * 0.5, mn.DIST_CRIT,  dist)
            if weight_i > weight:
                weight = weight_i

            #e2_obs is a weighted linear combianation of all the normals
            #e2_obs += normal/(dist + 1)
            e2 += normal/(dist + mn.EPSILON)
            div += 1/(dist + mn.EPSILON)

        e2 = e2/div

        e2_norm = np.linalg.norm(e2)
        #if the normal gets too small, we reduce w->0 to just take the D_ds matrix
        delta_w_cont = 0.01 
        weight = weight*smooth_step(0, delta_w_cont, e2_norm)
        if e2_norm:
            #resultant normal is defined
            e2 = e2/e2_norm
        
        D_tot = (1-weight)*D_ds + weight*D_obs
        return D_tot

    def compute_D_matrix_wrt_both_mat_add_v2(self, x, xdot):
        """
        Method 3 v2 : not compatible with storage
        Compute the damping matrix with the new passive control. Will be stiff against 
        obstacles, while also beeing stiff along the direction of motion and
        compliant in the perpendicular directions.
        This function implements the third method v2, with D beeing a positive semi-definite matrix
        but construct with a weighted comb of D_ds and D_obs.
        Works with many obstacles, and in 2/3 dimensions
            param : 
                x (np.array) : actual position
                xdot(np.array) : actual velocity
            return : 
                D (np.array) : the damping matrix
        """
        #############
        #### Dds ####
        #############
        #Ds matrix follow the ds, like in kronander paper
        # e1 : points in DS dir
        # e2, ... : arbitrary ortho basis

        #get desired velocity 
        x_dot_des = self.dynamic_avoider.evaluate(x)

        #if the desired velocity is too small, we risk numerical issue, we have converge (or sadle point)
        if np.linalg.norm(x_dot_des) < mn.EPSILON:
            #we just return the previous damping matrix
            return self.D
        
        #compute the vector align to the DS
        e1_DS = x_dot_des/np.linalg.norm(x_dot_des)

        #construct the basis matrix, align to the DS direction 
        E_DS = get_orthogonal_basis(e1_DS)
        E_DS_inv = np.linalg.inv(E_DS)

        #contruct the matrix containing the damping coefficient along selective directions
        if self.DIM == 2:
            lambda_mat_DS = np.array([[self.lambda_DS, 0.0],
                                   [0.0, self.lambda_perp]])
        else:
            lambda_mat_DS = np.array([[self.lambda_DS, 0.0, 0.0], 
                                   [0.0, self.lambda_perp, 0.0],
                                   [0.0, 0.0, self.lambda_perp]])

        #compose the damping matrix
        D_DS = E_DS@lambda_mat_DS@E_DS_inv

        #if there is no obstacles, we want D to be stiff w.r.t. the DS
        if not np.size(self.obs_dist_list):
            return D_DS

        ##############
        #### Dobs ####
        ##############
        #e1 : points in normal
        #e2 : proj of e1:ds that's ortho to e1_obs

        weight = 0
        div = 0
        e1_obs = np.zeros(self.DIM)
        #get the normals and compute the weight of the obstacles
        for normal, dist in zip(self.obs_normals_list.T, self.obs_dist_list):
            #if dist <0 we're IN the obs
            if dist <= 0: 
                return self.D

            #weight is 1 at the boundary, 0 at a distance DIST_CRIT from the obstacle
            weight_i = max(0.0, 1.0 - dist/mn.DIST_CRIT)
            if weight_i > weight:
                weight = weight_i

            #e1 is a weighted linear combination of all the normals
            e1_obs += normal/(dist + mn.EPSILON)
            div += 1/(dist + mn.EPSILON)

        e1_obs = e1_obs/div

        e1_obs_norm = np.linalg.norm(e1_obs)
        #if the normal gets too small, we reduce w->0 to just take the D_ds matrix
        delta_w_cont = 0.01 
        weight = weight*smooth_step(0, delta_w_cont, e1_obs_norm)
        if e1_obs_norm:
            #resultant normal is defined
            e1_obs = e1_obs/e1_obs_norm
        

        #basis construction depends on dimension
        if self.DIM == 3:
            #check if alignement
            if np.abs(np.dot(e1_DS, e1_obs)) == 1:
                #limit case : normal and DS are aligned
                #def e2 in limit case
                e2_obs = np.array([e1_obs[1], -e1_obs[0], 0])
                if not any(e2_obs):
                    e2_obs = np.array([e1_obs[2], 0, -e1_obs[0]])
                #def e3 in limit case
                e3_obs = np.cross(e1_obs, e2_obs)
            else:
                #base case
                #def e3
                e3_obs = np.cross(e1_obs, e1_DS)
                #def e2 : closest projection of e1_DS perp to e1_obs
                e2_obs = np.cross(e3_obs, e1_obs)

            #construct the basis matrix
            E_obs = np.array([e1_obs, e2_obs, e3_obs]).T
            E_obs_inv = np.linalg.inv(E_obs)
            lambda_mat_obs = np.array([[self.lambda_obs, 0.0, 0.0], 
                                   [0.0, self.lambda_DS, 0.0],
                                   [0.0, 0.0, self.lambda_perp]])

        elif self.DIM == 2:
            #def e2
            e2_obs = np.array([e1_obs[1], -e1_obs[0]])
            if np.dot(e2_obs, e1_DS) < 0:
                e2_obs = -e2_obs
            #construct the basis matrix
            E_obs = np.array([e1_obs, e2_obs]).T
            E_obs_inv = np.linalg.inv(E_obs)
            lambda_mat_obs = np.array([[self.lambda_obs, 0.0],
                                   [0.0, self.lambda_DS]])

        #if normal and DS begin to align, reduce lambda 2 to lambda_perp -> to keep continuity
        res = np.abs(np.dot(e1_DS, e1_obs))
        lambda_mat_obs[1,1] = self.lambda_perp*res + self.lambda_DS*(1-res)

        #if we go away from obs, we relax the stiffness, in obs directon
        DELTA_RELAX = 0.01
        res = np.dot(xdot, e1_obs)
        lambda_mat_obs[0,0] = smooth_step(0, DELTA_RELAX, res)*self.lambda_perp +\
                   smooth_step_neg(0, DELTA_RELAX, res)*self.lambda_obs

        #build D_obs           
        D_obs = E_obs@lambda_mat_obs@E_obs_inv


        #####################
        #### combine D's ####
        #####################

        D_tot = (1-weight)*D_DS + weight*D_obs
        return D_tot


    #ENERGY TANK RELATED
    #all x, xdot must be from actual step and not future (i.e. must not be updated yet)
    # D must be updated yet
    def update_energy_tank(self, x, xdot, dt):
        """
        Perform one time step (euler) to update the energy tank. Also updates the parameters 
        related to the energy storage (alpha, beta, ...)
        /!\ only designed for approach == Approach.ORTHO_BASIS
            param : 
                x (np.array) : actual position
                xdot(np.array) : actual velocity
        """

        if not self.with_E_storage:
            #avoid useless computation
            return

        #_, f_r = self.decomp_f(x)
        #self.z = xdot.T@f_r

        #get desired velocity
        f = self.dynamic_avoider.evaluate(x)
        self.f_decomp = (np.diag(self.E.T@f)@self.E.T).T #matrix with column = f1...fn : projection of f on each e1, ..., en
        self.z_list = self.f_decomp.T@xdot #need check
        
        alpha = self.get_alpha()
        #beta_s = self.get_beta_s()
        beta_s_list = self.get_beta_s_list()

        #using euler 1st order
        #sdot = alpha*xdot.T@self.D@xdot - beta_s*self.lambda_DS*self.z
        sdot = alpha*xdot.T@self.D@xdot - np.sum(beta_s_list*np.diag(self.lambda_mat)*self.z_list)#elem-wise
        self.s += dt*sdot
    
    def decomp_f(self, x):
        """
        Decompose f into a conservative + non-conservative part. Not used.
        """
        f_c = self.dynamic_avoider.initial_dynamics.evaluate(x) #unmodulated dynamic : conservative (if DS is lin.)
        f_r = self.dynamic_avoider.evaluate(x) - f_c
        return f_c, f_r

    def get_alpha(self):
        """
        computes alpha
            return : 
                alpha (int)
        """
        #return smooth_step_neg(mn.S_MAX/2 - mn.DELTA_S, mn.S_MAX/2 + mn.DELTA_S, self.s)
        return smooth_step_neg(mn.S_MAX-mn.DELTA_S, mn.S_MAX, self.s) #used in paper



    def get_beta_s(self):
        """
        computes beta_s, not used
            return : 
                beta_s (int)
        """
        # ret = 1 - smooth_step(-mn.DELTA_Z, 0, self.z)*smooth_step_neg(mn.S_MAX, mn.S_MAX + mn.DELTA_S, self.s) \
        #         - smooth_step_neg(0, mn.DELTA_Z, self.z)*smooth_step(mn.S_MAX - mn.DELTA_S, mn.S_MAX, self.s)
        
        #modif at second H -> was a mistake
        ret = 1 - smooth_step(-mn.DELTA_Z, 0, self.z)*smooth_step_neg(0, mn.DELTA_S, self.s)\
                - smooth_step_neg(0, mn.DELTA_Z, self.z)*smooth_step(mn.S_MAX - mn.DELTA_S, mn.S_MAX, self.s)
        return ret

    def get_beta_s_list(self):
        """
        computes all the beta_s, one for each dimension
            return : 
                beta_s_list (np.array)
        """
        #corec from paper, 2nd H
        ret = np.zeros(self.DIM)
        for i in range(self.DIM):
            ret[i] = 1 - smooth_step(-mn.DELTA_Z, 0, self.z_list[i])*smooth_step_neg(0, mn.DELTA_S, self.s)\
                       - smooth_step_neg(0, mn.DELTA_Z, self.z_list[i])*smooth_step(mn.S_MAX - mn.DELTA_S, mn.S_MAX, self.s)
        return ret

    def get_beta_r(self):
        """
        computes beta_R, not used
            return : 
                beta_s (int)
        """
        # ret = (1 - smooth_step(-mn.DELTA_Z, 0, self.z)*smooth_step_neg(mn.S_MAX, mn.S_MAX + mn.DELTA_S, self.s)) \
        #       * (1 - (smooth_step(-mn.DELTA_Z, 0, self.z)* \
        #              smooth_step_neg(0, mn.DELTA_Z, self.z)*smooth_step(mn.S_MAX - mn.DELTA_S, mn.S_MAX, self.s)))
       
        #modif at second H -> was amistake
        ret = (1 - smooth_step(-mn.DELTA_Z, 0, self.z)*smooth_step_neg(0, mn.DELTA_S, self.s)) \
              * (1 - (smooth_step(-mn.DELTA_Z, 0, self.z)* \
                     smooth_step_neg(0, mn.DELTA_Z, self.z)* \
                     smooth_step(mn.S_MAX - mn.DELTA_S, mn.S_MAX, self.s)))
        return ret

    def get_beta_r_list(self):
        """
        computes all the beta_R, one for each dimension
            return : 
                beta_R_list (np.array)
        """
        # ret = (1 - smooth_step(-mn.DELTA_Z, 0, self.z)*smooth_step_neg(mn.S_MAX, mn.S_MAX + mn.DELTA_S, self.s)) \
        #       * (1 - (smooth_step(-mn.DELTA_Z, 0, self.z)* \
        #              smooth_step_neg(0, mn.DELTA_Z, self.z)*smooth_step(mn.S_MAX - mn.DELTA_S, mn.S_MAX, self.s)))
       
        #modif at second H -> was amistake
        ret = np.zeros(self.DIM)
        for i in range(self.DIM):
            ret[i] = (1 - smooth_step(-mn.DELTA_Z, 0, self.z_list[i])*smooth_step_neg(0, mn.DELTA_S, self.s)) \
                * (1 - (smooth_step(-mn.DELTA_Z, 0, self.z_list[i])* \
                        smooth_step_neg(0, mn.DELTA_Z, self.z_list[i])* \
                        smooth_step(mn.S_MAX - mn.DELTA_S, mn.S_MAX, self.s)))
        return ret

### HELPER FUNTION ###

#helper smooth step functions
def smooth_step(a,b,x):
    """
    Implements a smooth step function
        param :
            a : low-limit
            b : high-limit 
            x : value where we want to evaluate the function
        return :
            H_a_b(x) (int) : the value of the smooth step function
    """
    if x < a: 
        return 0
    if x > b:
        return 1
    else:
        return 6*((x-a)/(b-a))**5 - 15*((x-a)/(b-a))**4 + 10*((x-a)/(b-a))**3

def smooth_step_neg(a,b,x):
    """
    Implements a negative smooth step function
        param :
            a : high-limit
            b : low-limit 
            x : value where we want to evaluate the function
        return :
            H_a_b(x) (int) : the value of the negative smooth step function
    """
    return 1 - smooth_step(a,b,x)