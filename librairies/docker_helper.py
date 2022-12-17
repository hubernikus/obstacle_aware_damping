import numpy as np
import warnings


#Lukas
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.avoidance.base_avoider import BaseAvoider
from dynamic_obstacle_avoidance.utils import get_orthogonal_basis

from vartools.dynamical_systems import LinearSystem

#new for obs viz
from franka_avoidance.optitrack_container import OptitrackContainer
from franka_avoidance.pybullet_handler import PybulletHandler


class Simulated():
    def __init__(
        self,
        lambda_DS = 100.0,
        lambda_perp = 20.0,
        lambda_obs = 200,
    ):
        self.lambda_DS = lambda_DS
        self.lambda_perp = lambda_perp
        self.lambda_obs = lambda_obs
        self.D = np.diag([self.lambda_DS, self.lambda_perp, self.lambda_obs])



    def create_env(self, obs_pos, obs_axes_lenght, obs_vel, no_obs):
        #old
        #self.obstacle_environment = ObstacleContainer()

        #new
        self.obstacle_environment = OptitrackContainer(use_optitrack=False)

        if no_obs:
            return
        for i, (pos, axes, vel) in enumerate(zip(obs_pos, obs_axes_lenght, obs_vel)):
            #old
            # self.obstacle_environment.append(
            #     Ellipse(
            #         axes_length = axes,
            #         center_position = pos,
            #         margin_absolut=0.15,
            #         # orientation=10 * pi / 180,
            #         linear_velocity = vel,
            #         tail_effect=False,
            #         # repulsion_coeff=1.4,
            #     )
            # )

            #new
            self.obstacle_environment.append(
                Ellipse(
                    center_position = pos,
                    axes_length = axes,
                    linear_velocity = vel,
                ),
                obstacle_id=i,
            )
         
        self.obstacle_environment.visualization_handler = PybulletHandler(self.obstacle_environment)
        return self.obstacle_environment
            
        

    def create_DS_copy(self, attractor_position, A_matrix, max_vel):
        self.initial_dynamics = LinearSystem(
            attractor_position = attractor_position,
            A_matrix = A_matrix,
            dimension=3,
            maximum_velocity=max_vel,
            distance_decrease=0.1, #if too small, could lead to instable around atractor 
        )

    def create_mod_avoider(self):
            self.dynamic_avoider = ModulationAvoider(
                initial_dynamics = self.initial_dynamics,
                obstacle_environment = self.obstacle_environment,
            )

    def compute_D_matrix_wrt_DS(self, x_dot_des):
        #print("compute D wrt DS only")
        #construct the basis matrix, align to the DS direction 
        E = get_orthogonal_basis(x_dot_des)
        E_inv = np.linalg.inv(E)


        #contruct the matrix containing the damping coefficient along selective directions
        lambda_mat = np.array([[self.lambda_DS, 0.0, 0.0], 
                                [0.0, self.lambda_perp, 0.0],
                                [0.0, 0.0, self.lambda_perp]])

        #compose the damping matrix
        D = E@lambda_mat@E_inv
        self.D = D
        return D

    def compute_D(self, x, xdot, x_dot_des):
        EPSILON = 1e-6
        DIM = 3
        DIST_CRIT = 1

        #if there is no obstacles, we want D to be stiff w.r.t. the DS
        if not len(self.obstacle_environment):
            #print("no obs")
            return self.compute_D_matrix_wrt_DS(x_dot_des)

        #print("compute D wrt obs")
        #KEEP ? if the desired velocity is too small, we risk numerical issue
        if np.linalg.norm(x_dot_des) < EPSILON:
            #we just return the previous damping matrix
            return self.D
        
        #compute the vector align to the DS
        e1_DS = x_dot_des/np.linalg.norm(x_dot_des)



        #get the normals and distance to the obstacles
        obs_normals_list = np.empty((DIM, 0))
        obs_dist_list = np.empty(0)
        for obs in self.obstacle_environment:
            #gather the parameters wrt obstacle i
            normal = obs.get_normal_direction(x, in_obstacle_frame = False).reshape(DIM, 1)
            obs_normals_list = np.append(obs_normals_list, normal, axis=1)

            d = obs.get_gamma(x, in_obstacle_frame = False) - 1
            obs_dist_list = np.append(obs_dist_list, d)


        #compute vector relative to obstacles
        weight = 0
        e2_obs = np.zeros(DIM)
        #compute the weight of the obstacles and resulting normal
        for normal, dist in zip(obs_normals_list.T, obs_dist_list):
            #if dist <0 we're IN the obs
            if dist <= 0: 
                return self.D

            #weight is 1 at the boundary, 0 at a distance DIST_CRIT from the obstacle
            #keep only biggest wight -> closer obstacle
            weight_i = max(0.0, 1.0 - dist/DIST_CRIT)
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
        if 1 - np.abs(np.dot(e1_both, e2_both)) < EPSILON:
            raise("What did trigger this, it shouldn't")
            
        #construct the basis matrix
        E = np.array([e1_DS, e2_both, e3]).T
        E_inv = np.linalg.inv(E)

        #compute the damping coeficients along selective directions
        lambda_1 = self.lambda_DS #(1-weight)*self.lambda_DS #good ??
        lambda_2 = (1-weight)*self.lambda_perp + weight*self.lambda_obs
        lambda_3 = self.lambda_perp

        #if we go away from obs, we relax the stiffness
        if np.dot(xdot, e2_obs) > 0:
            lambda_2 = self.lambda_perp

        #contruct the matrix containing the damping coefficient along selective directions
        lambda_mat = np.array([[lambda_1, 0.0, 0.0], 
                                [0.0, lambda_2, 0.0],
                                [0.0, 0.0, lambda_3]])
        
        #compose the damping matrix
        D = E@lambda_mat@E_inv
        self.D = D
        return D

