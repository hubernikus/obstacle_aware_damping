#!/usr/bin/env python3
import numpy as np

from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node

# LASA / Control-Libraries
import state_representation as sr
from controllers import create_cartesian_controller, CONTROLLER_TYPE
from dynamical_systems import create_cartesian_ds, DYNAMICAL_SYSTEM_TYPE
from network_interfaces.control_type import ControlType
from network_interfaces.zmq.network import CommandMessage

from vartools.linalg import get_orthogonal_basis

# Custom libraries
from franka_avoidance.robot_interface import RobotZmqInterface as RobotInterface

# Custom helper librairies
from librairies.docker_helper import Simulated

ANGLE_CONV = 'zyx'

class ObsstacleAwarePassiveCont(Node):
    def __init__(
        self,
        robot: RobotInterface,
        freq: float = 100.0,
        node_name="velocity_controller",
        is_simulation: bool = True,
    ) -> None:
        super().__init__(node_name)
        self.robot = robot
        self.rate = self.create_rate(freq)
        period = 1.0 / freq

        #ctrl not used
        self.ctrl = create_cartesian_controller(CONTROLLER_TYPE.COMPLIANT_TWIST)
        if is_simulation:
            print("Control mode: simulation")
            self.linear_principle_damping = 1.0
            self.linear_obstacle_damping = 1.2
            self.linear_orthogonal_damping = 0.4
            self.angular_stiffness = 0.5
            self.angular_damping = 0.5

        else:
            print("Control mode: real")
            self.linear_principle_damping = 50.0
            self.linear_obstacle_damping = 60.0
            self.linear_orthogonal_damping = 20.0
            self.angular_stiffness = 2.0
            self.angular_damping = 2.0

        self.ctrl.set_parameter_value(
            "linear_principle_damping",
            self.linear_principle_damping,
            sr.ParameterType.DOUBLE,
        )
        self.ctrl.set_parameter_value(
            "linear_orthogonal_damping",
            self.linear_orthogonal_damping,
            sr.ParameterType.DOUBLE,
        )
        self.ctrl.set_parameter_value(
            "angular_stiffness", self.angular_stiffness, sr.ParameterType.DOUBLE
        )
        self.ctrl.set_parameter_value(
            "angular_damping", self.angular_damping, sr.ParameterType.DOUBLE
        )

        ## unused
        self.ds = create_cartesian_ds(DYNAMICAL_SYSTEM_TYPE.POINT_ATTRACTOR)
        self.ds.set_parameter_value(
            "gain", [50.0, 50.0, 50.0, 20.0, 20.0, 20.0], sr.ParameterType.DOUBLE_ARRAY
        )

        self.clamp_linear = 0.25
        self.clamp_angular = 0.5

        # Get robot state to set up the target in the same frame
        while not (state := self.robot.get_state()) and rclpy.ok():
            print("Awaiting first state.")
        print("First state recieved.")

        #redifinned later
        self.attractor_position = np.array([0.6, -0.2, 0.5])
        self.attractor_quaternion = np.array([0.0, 1.0, 0.0, 0.0])

        ## unused
        target = sr.CartesianPose(
            state.ee_state.get_name(),
            self.attractor_position,
            self.attractor_quaternion,
            state.ee_state.get_reference_frame(),
        )
        ## unused
        self.ds.set_parameter_value(
            "attractor",
            target,
            sr.ParameterType.STATE,
            sr.StateType.CARTESIAN_POSE,
        )
        
        ############
        ############

        #create an instance where the DS is simulated

        self.sim = Simulated(
            lambda_DS=self.linear_principle_damping,
            lambda_perp=self.linear_orthogonal_damping,
            lambda_obs=self.linear_obstacle_damping,
        )

        # create obstacle env
        no_obs = True  # to disable obstacles
        obs_position = np.array([[0.0, 0.0, 0.0]])  # these are N x 3
        # for plot need to be all equal -> sphere, its the radius
        obs_axes_lenght = np.array([[0.20] * 3])
        obs_vel = np.array([[0.0] * 3])
        self.obstacle_env = self.sim.create_env(
            obs_position, obs_axes_lenght, obs_vel, no_obs
        )

        #create the DS
        self.A_lin = -np.diag([50.0, 50.0, 50.0])
        self.A_ang = -np.diag([20.0, 20.0, 20.0])
        self.attractor_A = np.array([0.4, 0.3, 0.4])
        self.attractor_B = np.array([0.4, -0.3, 0.4])
        self.attractor_position = self.attractor_A
        self.attrator_quaternion = np.array([0.0, 1.0, 0.0, 0.0])
        r = R.from_quat(self.attrator_quaternion)
        self.attractor_euler =  r.as_euler(ANGLE_CONV, degrees=False) #zyx #NOT SURE, what convention used ? 

        self.max_vel = 0.2          # ??? what param ? 
        self.sim.create_lin_DS(
            self.attractor_position,
            self.A_lin, 
            self.max_vel
        )
        self.sim.create_ang_DS(
            self.attractor_euler,
            self.A_ang,
            self.max_vel
        )

        # create modulation avoider to modulate final DS
        self.sim.create_mod_avoider()

        ############
        ############

        self.create_controller_dissipative()

        self.timer = self.create_timer(period, self.controller_callback)

    # def run(self):
    #     while rclpy.ok():
    #         self.controller_callback()

    def create_controller_dissipative(self) -> None:
        """Simple dissipative controller to obtain the desired state.
        The state is assumed to have zero force / torque,
        as this will be transferred further."""
        # initialize controller : Fcomannd = I*x_dd + D*x_d + K*x
        self.ctrl_dissipative = create_cartesian_controller(CONTROLLER_TYPE.IMPEDANCE)
        self.ctrl_dissipative.set_parameter_value(
            "stiffness", np.zeros((6, 6)), sr.ParameterType.MATRIX
        )
        self.ctrl_dissipative.set_parameter_value(
            "inertia", np.zeros((6, 6)), sr.ParameterType.MATRIX
        )

        D = np.diag(
            [
                self.linear_orthogonal_damping,
                self.linear_orthogonal_damping,
                self.linear_orthogonal_damping,
                self.angular_damping,
                self.angular_damping,
                self.angular_damping,
            ]
        )

        self.ctrl_dissipative.set_parameter_value("damping", D, sr.ParameterType.MATRIX)

    def update_dissipative_controller(
        self, 
        desired_twist: sr.CartesianTwist,
        ee_state : sr.CartesianState
    ) -> None:
        des_lin_vel = desired_twist.get_linear_velocity()
        if not (np.linalg.norm(des_lin_vel)):
            D = np.diag(
                [
                    self.linear_orthogonal_damping,
                    self.linear_orthogonal_damping,
                    self.linear_orthogonal_damping,
                    self.angular_damping,
                    self.angular_damping,
                    self.angular_damping,
                ]
            )
            self.ctrl_dissipative.set_parameter_value(
                "damping", D, sr.ParameterType.MATRIX
            )
            return
        
        ###########################
        ###########################
        #extract linear position and velocity from current state
        xyz = ee_state.get_pose()[:3]
        lin_vel = ee_state.get_linear_velocity()

        #construction of the damping matrix
        D = np.zeros((6, 6))

        ### to test, remoove 
        D_linear = np.diag(
            [
                self.linear_principle_damping,
                self.linear_orthogonal_damping,
                self.linear_orthogonal_damping,
            ]
        )
        D[:3, :3] = D_linear 
        ### end test

        #D[:3, :3] = self.sim.compute_D(xyz, lin_vel, des_lin_vel)
        D[3:, 3:] = np.eye(3) * self.angular_damping
        self.ctrl_dissipative.set_parameter_value("damping", D, sr.ParameterType.MATRIX)
        ###########################
        ###########################
        #previously 

        # D = np.zeros((6, 6))

        # # Angular damping
        # D[3:, 3:] = np.eye(3) * self.angular_damping

        # # Linear damping
        # D_linear = np.diag(
        #     [
        #         self.linear_principle_damping,
        #         self.linear_orthogonal_damping,
        #         self.linear_orthogonal_damping,
        #     ]
        # )
        # E = get_orthogonal_basis(des_lin_vel / lin_norm)
        # D[:3, :3] = E @ D_linear @ E.T
        # self.ctrl_dissipative.set_parameter_value("damping", D, sr.ParameterType.MATRIX)

    def controller_callback(self) -> None:
        command = CommandMessage()
        command.control_type = [ControlType.EFFORT.value]

        state = self.robot.get_state()
        if not state:
            return

        # !!!Current force / torque are eliminated in order to not 'overcompensate'
        state.ee_state.set_force(np.zeros(3))
        state.ee_state.set_torque(np.zeros(3))

        ######################
        ######################
        #computes desired twist
        pose = state.ee_state.get_pose()
        # print(dir(state.ee_state))
        # print(state.ee_state.get_orientation_coefficients())
        # print(state.ee_state.get_transformation_matrix())
        # breakpoint()
        r = R.from_quat(pose[3:]) #pose[3:7] is quaternion
        ang = r.as_euler(ANGLE_CONV, degrees=False) #zyx                     #what convention ???
        des_lin_vel = self.sim.dynamic_avoider.evaluate(
            pose[:3] #xyz
        )
        des_ang_vel = self.sim.initial_ang_dynamics.evaluate(
            ang #euler
        )

        ### NEED TO BE EULER ANGLES
        des_twist = np.concatenate((des_lin_vel, des_ang_vel))

        #traduction for cpp
        desired_twist = sr.CartesianTwist(
            state.ee_state.get_name(),
            des_twist,
            state.ee_state.get_reference_frame(),
        )   
        ######################
        ######################

        print("desired lin vel : ", desired_twist.get_linear_velocity())
        print("current lin vel : ", state.ee_state.get_linear_velocity())
        print("desired ang vel : ", desired_twist.get_angular_velocity())
        print("current ang vel : ", state.ee_state.get_angular_velocity())

        #desired_twist = sr.CartesianTwist(self.ds.evaluate(state.ee_state))
        desired_twist.clamp(self.clamp_linear, self.clamp_angular)

        # Update Damping-matrix based on desired velocity
        self.update_dissipative_controller(desired_twist, state.ee_state)

        cmnd_dissipative = self.ctrl_dissipative.compute_command(
            desired_twist, state.ee_state, state.jacobian
        )

        # command_torques = sr.JointTorques(cmnd_dissipative)

        command.joint_state = state.joint_state
        command.joint_state.set_torques(cmnd_dissipative.get_torques())
        self.robot.send_command(command)
        print("Command sent.")


if (__name__) == "__main__":
    print("[INFO] Starting Cartesian Damping controller  ...")
    rclpy.init()
    robot_interface = RobotInterface("*:1601", "*:1602")

    controller = ObsstacleAwarePassiveCont(
        robot=robot_interface, freq=100, is_simulation=False
    )

    try:
        rclpy.spin(controller)

    except KeyboardInterrupt:
        pass

    controller.destroy_node()

    rclpy.shutdown()