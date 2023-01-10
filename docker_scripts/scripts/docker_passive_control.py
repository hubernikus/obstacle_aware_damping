#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node

import threading

import state_representation as sr
from controllers import create_cartesian_controller, CONTROLLER_TYPE
from dynamical_systems import create_cartesian_ds, DYNAMICAL_SYSTEM_TYPE
from network_interfaces.control_type import ControlType
from network_interfaces.zmq.network import CommandMessage

from scipy.spatial.transform import Rotation as R

# Custom libraries lukas
from franka_avoidance.robot_interface import RobotZmqInterface as RobotInterface
from dynamic_obstacle_avoidance.avoidance.base_avoider import BaseAvoider

# My custom librairies
from librairies.docker_helper import Simulated


class PassiveObsController(Node):
    def __init__(self, robot, freq: float = 100, node_name="passive_controller"):
        super().__init__(node_name)
        self.robot = robot
        self.rate = self.create_rate(freq)

        self.command = CommandMessage()
        self.command.control_type = [ControlType.EFFORT.value]

        # initialize controller : Fcomannd = I*x_dd + D*x_d + K*x
        self.ctrl = create_cartesian_controller(CONTROLLER_TYPE.IMPEDANCE)
        self.ctrl.set_parameter_value(
            "damping", np.eye(6), sr.ParameterType.MATRIX)
        self.ctrl.set_parameter_value(
            "stiffness", np.eye(6), sr.ParameterType.MATRIX)
        self.ctrl.set_parameter_value(
            "inertia", np.eye(6), sr.ParameterType.MATRIX)

        # usefull to print
        # print("methods : ", dir(self.ctrl))
        # print("PARAMS CTRL : ", self.ctrl.get_parameters())

        # sim is an instance containing the virtual copy of all the elements, used to compute D
        self.sim = Simulated(
            lambda_DS=10.0,  # 100
            lambda_perp=1.0,  # 20
            lambda_obs=10,  # 200
        )

        # create obstacle env
        obs_position = np.array([[0.0, 0.0, 0.0]])  # these are N x 3
        # for plot need to be all equal -> sphere, its the radius
        obs_axes_lenght = np.array([[0.20] * 3])
        obs_vel = np.array([[0.0] * 3])
        no_obs = False  # to disable obstacles
        self.obstacle_env = self.sim.create_env(
            obs_position, obs_axes_lenght, obs_vel, no_obs)

        # create the DS, NOT FORGET minus !!! (!= cpp )
        self.A_matrix = -np.diag([1.0, 1.0, 1.0])
        self.attractor_A = np.array([0.4, 0.3, 0.4])
        self.attractor_B = np.array([0.4, -0.3, 0.4])
        self.attractor_position = self.attractor_A
        self.attrator_quaternion = np.array([0.0, 1.0, 0.0, 0.0])
        self.max_vel = 0.2
        self.sim.create_DS_copy(self.attractor_position,
                                self.A_matrix, self.max_vel)

        # create modulation avoider to modulate final DS
        self.sim.create_mod_avoider()

        # to compare - remooove when debuged
        # self.ctrl_ref = create_cartesian_controller(CONTROLLER_TYPE.DISSIPATIVE)
        # self.ctrl_ref.set_parameter_value("eigenvalues", np.diag([20, 2, 2, 1, 1, 1]), sr.ParameterType.MATRIX)
        # self.ds = create_cartesian_ds(DYNAMICAL_SYSTEM_TYPE.POINT_ATTRACTOR)
        # self.ds.set_parameter_value(
        #      "gain", [1, 1, 1, 0.1, 0.1, 0.1], sr.ParameterType.DOUBLE_ARRAY #standart linear DS
        # )

    def run(self):
        target_set = False
        atr_a = True

        while rclpy.ok():
            state = self.robot.get_state()
            if not state:
                continue
            if False:
                pass
            # below, usefull when ds in in cpp
            # if not target_set:
            #     target = sr.CartesianPose(
            #         state.ee_state.get_name(),
            #         self.attractor_position,
            #         np.array([0.0, 1.0, 0.0, 0.0]), #quaternion coresponding to pointing down
            #         state.ee_state.get_reference_frame(),
            #     )
            #     self.ds.set_parameter_value(
            #         "attractor",
            #         target,
            #         sr.ParameterType.STATE,
            #         sr.StateType.CARTESIAN_POSE,
            #     )
            #     #print("### DS ###", self.ds.get_parameters())
            #     target_set = True
            else:
                ### CONTROL LOOP ###

                # 0. get the pose from the state
                x = state.ee_state.get_pose()

                # extract the position and velocity
                pos = x[0:3]
                # print(pos)
                pos_dot = state.ee_state.get_linear_velocity()

                # extract the angle
                # r = R.from_quat(x[3:7])
                # ang = r.as_euler('zyx', degrees=False)  # not sure which format

                # 1. compute desired velocity in cartesian coord
                pos_dot_des = self.sim.dynamic_avoider.evaluate(pos)

                # max_pos_des = max(abs(pos_dot_des))
                # if max_pos_des > 0.1:
                #     pos_dot_des = pos_dot_des/max_pos_des

                print("actual position : ", pos, " and ang : ",
                      x[3:7], " actual vel : ", pos_dot, " des_vel by avoider", pos_dot_des)

                # 2. compute the damping matrix and gains

                D = self.sim.compute_D(pos, pos_dot, pos_dot_des)
                KD_ang = np.diag([5.0] * 3)  # 1
                big_D = np.zeros((6, 6))
                big_D[0:3, 0:3] = D
                big_D[3:6, 3:6] = KD_ang
                print('D ', big_D)

                # 3. build the K matrix
                big_K = np.zeros((6, 6))
                KP_ang = np.diag([5.0] * 3)  # 5
                big_K[3:6, 3:6] = KP_ang
                print('K ', big_K)

                # 3. assign them to the robot
                self.ctrl.set_parameter_value(
                    "damping", big_D, sr.ParameterType.MATRIX)
                self.ctrl.set_parameter_value(
                    "stiffness", big_K, sr.ParameterType.MATRIX)

                # print("## PARAMS CTRL NEW ## : ", self.ctrl.get_parameters())

                # 4. construct des_ee_state
                des_ee_state = sr.CartesianState(
                    state.ee_state.get_name(),
                    state.ee_state.get_reference_frame(),
                )
                des_ee_state.set_pose(np.concatenate(
                    (np.zeros(3), self.attrator_quaternion)))  # vector 7x1
                des_ee_state.set_linear_velocity(pos_dot_des)
                des_ee_state.set_angular_velocity(np.zeros(3))

                # 5. compute the command with the impedance controller
                self.command_torques = sr.JointTorques(
                    self.ctrl.compute_command(
                        des_ee_state, state.ee_state, state.jacobian)
                )
                # print("\n com. torque :", self.command_torques.get_torques())

                # 6. send the ecommand to the robot
                self.command.joint_state = state.joint_state

                max_torque = max(abs(self.command_torques.get_torques()))
                if max_torque > 15:
                    print("torques were limited (>1)",
                          self.command_torques.get_torques())
                    self.command.joint_state.set_torques(
                        self.command_torques.get_torques()/max_torque)
                else:
                    self.command.joint_state.set_torques(
                        self.command_torques.get_torques())

                # self.command_torques.get_torques()
                # breakpoint()
                # print("torques : ", self.command_torques.get_torques())

                self.robot.send_command(self.command)

                # 7. draw obs
                if self.obstacle_env is not None:
                    # self.obstacle_env.update_obstacles()
                    #self.obstacle_env.update()
                    print('opti ok')

                # 8. switch atractor if converged
                EPS = 1e-1
                if atr_a:
                    # print("A")
                    pass
                else:
                    # print("B")
                    pass
                # print("current atractor position : ", self.attractor_position)
                if np.linalg.norm(pos - self.attractor_position) <= EPS:
                    # print("SWICH")
                    if atr_a:
                        self.attractor_position = self.attractor_B
                        atr_a = False
                    else:
                        self.attractor_position = self.attractor_A
                        atr_a = True

                    # update the DS
                    self.sim.create_DS_copy(
                        self.attractor_position, self.A_matrix, self.max_vel)

                    # update the modulation avoider to modulate final DS
                    self.sim.create_mod_avoider()

                # comparison with cpp#
                # ds_ret = self.ds.evaluate(state.ee_state)
                # sr.CartesianTwist
                # com_ref = sr.JointTorques(
                #     self.ctrl_ref.compute_command(ds_ret, state.ee_state, state.jacobian)
                # )
                # self.command.joint_state.set_torques(com_ref.get_torques())
                # command_torques has name 1->7 of joint + torque array 1x7
                # self.robot.send_command(self.command)
                ###

            self.rate.sleep()
        self.obstacle_env.shutdown()


if __name__ == "__main__":
    rclpy.init()
    # rospy.init_node("test", anonymous=True)
    # 16 is for the first franka robot
    robot_interface = RobotInterface("*:1601", "*:1602")

    # Spin in a separate thread
    controller = PassiveObsController(robot=robot_interface, freq=500)

    thread = threading.Thread(
        target=rclpy.spin, args=(controller,), daemon=True)
    thread.start()

    try:
        controller.run()

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    thread.join()
