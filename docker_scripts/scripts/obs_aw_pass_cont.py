#!/usr/bin/env python3
from datetime import datetime

import numpy as np
import pandas as pd

import rclpy
from rclpy.node import Node

# LASA / Control-Libraries
import state_representation as sr
from controllers import create_cartesian_controller, CONTROLLER_TYPE
from dynamical_systems import create_cartesian_ds, DYNAMICAL_SYSTEM_TYPE
from network_interfaces.control_type import ControlType
from network_interfaces.zmq.network import CommandMessage

from vartools.linalg import get_orthogonal_basis

# Custom passive_control.
from franka_avoidance.robot_interface import RobotZmqInterface as RobotInterface
from franka_avoidance.velocity_publisher import VelocityPublisher

# Custom helper passive_control.
from passive_control.docker_helper import Simulated

ANGLE_CONV = "zyx"


class ObsstacleAwarePassiveCont(Node):
    def __init__(
        self,
        robot: RobotInterface,
        freq: float = 100.0,
        node_name="velocity_controller",
        is_simulation: bool = True,
        is_obstacle_aware: bool = True,
        data_handler=None,
    ) -> None:
        super().__init__(node_name)
        self.robot = robot
        self.rate = self.create_rate(freq)
        period = 1.0 / freq

        self.is_obstacle_aware = is_obstacle_aware

        self.data_handler = data_handler

        # ctrl not used
        self.ctrl = create_cartesian_controller(CONTROLLER_TYPE.COMPLIANT_TWIST)
        if is_simulation:
            print("Control mode: simulation")
            multiplier = 1.0  # increase for faster but less stable
            self.linear_principle_damping = 1.0 * multiplier
            self.linear_obstacle_damping = 1.2 * multiplier
            self.linear_orthogonal_damping = 0.4 * multiplier
            self.angular_stiffness = 0.5 * multiplier
            self.angular_damping = 0.5 * multiplier

        else:
            print("Control mode: real")
            multiplier = 8.0  # increase for faster but less stable
            self.linear_principle_damping = 8 * multiplier  # 50.0
            self.linear_obstacle_damping = 20 * multiplier  # 60.0
            self.linear_orthogonal_damping = 2 * multiplier  # 20.0
            self.angular_stiffness = 0.3 * multiplier  # 2
            self.angular_damping = 0.3 * multiplier  # 2

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

        ## only use for the angular DS (3 last dimensions)
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

        # redifinned later
        # self.attractor_A = np.array([0.4, -0.5, 0.25])
        # self.attractor_B = np.array([0.4, 0.5, 0.25])
        self.attractor_A = np.array([0.26, -0.53, 0.334])
        self.attractor_B = np.array([0.26, 0.53, 0.334])
        self.attractor_position = self.attractor_A
        self.attractor_quaternion = np.array([0.0, 1.0, 0.0, 0.0])

        target = sr.CartesianPose(
            state.ee_state.get_name(),
            self.attractor_position,
            self.attractor_quaternion,
            state.ee_state.get_reference_frame(),
        )

        self.ds.set_parameter_value(
            "attractor",
            target,
            sr.ParameterType.STATE,
            sr.StateType.CARTESIAN_POSE,
        )

        # create an instance where the DS for linear control is simulated
        self.sim = Simulated(
            lambda_DS=self.linear_principle_damping,
            lambda_perp=self.linear_orthogonal_damping,
            lambda_obs=self.linear_obstacle_damping,
        )

        # create obstacle env
        no_obs = False  # to disable obstacles
        obs_position = np.array([[0.4, 0.0, 0.3]])  # these are N x 3
        # for plot need to be all equal -> sphere, its the radius
        obs_axes_lenght = np.array([[0.20] * 3])
        obs_vel = np.array([[0.0] * 3])
        self.obstacle_env = self.sim.create_env(
            obs_position, obs_axes_lenght, obs_vel, no_obs
        )

        # create the linear DS (copy of the one in cpp)
        self.A_lin = -np.diag([50.0, 50.0, 50.0])
        self.max_vel = 0.2  # ??? what param ?
        self.sim.create_lin_DS(self.attractor_position, self.A_lin, self.max_vel)

        # create modulation avoider to modulate final DS
        self.sim.create_mod_avoider()

        # instanciate the controller
        self.create_controller_dissipative()

        robot_frame = "panda_link0"
        self.modulated_velocity_publisher = VelocityPublisher("modulated", robot_frame)
        # self.rotated_velocity_publisher = VelocityPublisher("rotated", robot_frame)

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
        self, desired_twist: sr.CartesianTwist, ee_state: sr.CartesianState
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

        # extract linear position and velocity from current state
        xyz = ee_state.get_pose()[:3]
        print("pos", xyz[0], xyz[1], xyz[2])
        lin_vel = ee_state.get_linear_velocity()

        # construction of the damping matrix
        D = np.zeros((6, 6))

        if self.is_obstacle_aware:
            # linear compunent are computed with new method
            # D[:3, :3] = self.sim.compute_D(xyz, lin_vel, des_lin_vel, self.data_handler)
            D[:3, :3] = self.sim.compute_D(xyz, lin_vel, des_lin_vel)
        else:
            # Linear damping
            D_linear = np.diag(
                [
                    self.linear_principle_damping,
                    self.linear_orthogonal_damping,
                    self.linear_orthogonal_damping,
                ]
            )
            E = get_orthogonal_basis(des_lin_vel / np.linalg.norm(des_lin_vel))
            D[:3, :3] = E @ D_linear @ E.T

        # angular component are trivial and isotropic
        D[3:, 3:] = np.eye(3) * self.angular_damping

        # update parameter
        self.ctrl_dissipative.set_parameter_value("damping", D, sr.ParameterType.MATRIX)

    def controller_callback(self) -> None:
        command = CommandMessage()
        command.control_type = [ControlType.EFFORT.value]

        state = self.robot.get_state()
        if not state:
            return

        # !!!Current force / torque are eliminated in order to not 'overcompensate'
        state.ee_state.set_force(np.zeros(3))
        state.ee_state.set_torque(np.zeros(3))

        # compute des_twist, only for angular (3 last dimension)
        desired_twist = sr.CartesianTwist(self.ds.evaluate(state.ee_state))

        # get the position of the robot
        xyz = state.ee_state.get_pose()[:3]

        # compute the real desired linear velocity (with the python DS)
        des_lin_vel = self.sim.dynamic_avoider.evaluate(xyz)

        self.modulated_velocity_publisher.publish(xyz, des_lin_vel)

        # overwritting of the desired linear velocity
        desired_twist.set_linear_velocity(des_lin_vel)

        # print("desired lin vel : ", desired_twist.get_linear_velocity())
        # print("current lin vel : ", state.ee_state.get_linear_velocity())
        # print("desired ang vel : ", desired_twist.get_angular_velocity())
        # print("current ang vel : ", state.ee_state.get_angular_velocity())

        desired_twist.clamp(self.clamp_linear, self.clamp_angular)

        # Update Damping-matrix based on desired velocity
        self.update_dissipative_controller(desired_twist, state.ee_state)

        if self.data_handler is not None:
            DIM = 3
            x = state.ee_state.get_position()

            D_matrix = self.ctrl_dissipative.get_parameter_value("damping")
            self.data_handler["D"].append(D_matrix[:3, :3])

            self.data_handler["pos"].append(state.ee_state.get_position())
            self.data_handler["vel"].append(state.ee_state.get_linear_velocity())
            self.data_handler["des_vel"].append(desired_twist.get_linear_velocity())

            # Compute minimal distance to obstacles
            # get the normals and distance to the obstacles
            obs_normals_list = np.empty((DIM, 0))
            obs_dist_list = np.empty(0)
            for obs in self.obstacle_env:
                # gather the parameters wrt obstacle i
                normal = obs.get_normal_direction(x, in_obstacle_frame=False).reshape(
                    DIM, 1
                )
                obs_normals_list = np.append(obs_normals_list, normal, axis=1)

                d = obs.get_gamma(x, in_obstacle_frame=False) - 1
                obs_dist_list = np.append(obs_dist_list, d)

            self.data_handler["normals"].append(obs_normals_list)
            self.data_handler["dists"].append(obs_dist_list)

        # Update data
        # desired_twist.set_linear_velocity(np.zeros(3))
        cmnd_dissipative = self.ctrl_dissipative.compute_command(
            desired_twist, state.ee_state, state.jacobian
        )

        # command_torques = sr.JointTorques(cmnd_dissipative)

        command.joint_state = state.joint_state
        command.joint_state.set_torques(cmnd_dissipative.get_torques())
        self.robot.send_command(command)
        print("Command sent.")
        print("heading to : ", self.attractor_position)

        # update obs
        if self.obstacle_env is not None:
            self.obstacle_env.update()

        ##############################s##############
        # feature to alternate between 2 atractors #
        ############################################

        EPS_PREC = 0.2  # 0.1 real val
        print("norm : ", np.linalg.norm(xyz - self.attractor_position))
        if np.linalg.norm(xyz - self.attractor_position) < EPS_PREC:
            if np.array_equal(self.attractor_position, self.attractor_A):
                print("Converged to A")
                self.attractor_position = self.attractor_B
            else:
                print("Converged to B")
                self.attractor_position = self.attractor_A

            # update cpp DS
            target = sr.CartesianPose(
                state.ee_state.get_name(),
                self.attractor_position,
                self.attractor_quaternion,
                state.ee_state.get_reference_frame(),
            )
            self.ds.set_parameter_value(
                "attractor",
                target,
                sr.ParameterType.STATE,
                sr.StateType.CARTESIAN_POSE,
            )

            # update python DS
            self.sim.create_lin_DS(self.attractor_position, self.A_lin, self.max_vel)
            self.sim.create_mod_avoider()
            # breakpoint()

    def destroy_node(self) -> None:
        df = pd.DataFrame.from_dict(self.data_handler, orient="index").transpose()
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        df.to_csv(f"recording_{dt_string}.csv")
        print("[INFO] Writing to file successful.")
        super().destroy_node()


if (__name__) == "__main__":
    print("[INFO] Starting Cartesian Damping controller  ...")
    rclpy.init()
    # robot_interface = RobotInterface("*:1601", "*:1602")
    robot_interface = RobotInterface.from_id(17)

    is_osbtacle_aware = True
    data_handler = {
        "D": [],
        "pos": [],
        "vel": [],
        "des_vel": [],
        "normals": [],
        "dists": [],
        "controller: [obstacle aware]": [is_osbtacle_aware],
    }

    controller = ObsstacleAwarePassiveCont(
        robot=robot_interface,
        freq=100,
        is_simulation=False,
        is_obstacle_aware=is_osbtacle_aware,
        # is_obstacle_aware=False,
        data_handler=data_handler,
    )

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass

    controller.destroy_node()

    rclpy.shutdown()
