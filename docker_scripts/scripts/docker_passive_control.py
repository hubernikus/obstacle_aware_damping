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

# Custom libraries lukas
from franka_avoidance.robot_interface import RobotZmqInterface as RobotInterface

# My custom librairies 
from librairies.docker_helper import Simulated

class PassiveController(Node):
    def __init__(self, robot, freq: float = 100, node_name="passive_controller"):
        super().__init__(node_name)
        self.robot = robot
        self.rate = self.create_rate(freq)

        self.command = CommandMessage()
        self.command.control_type = [ControlType.EFFORT.value]

        #create DS type xdot = -A(x - x_atr)
        self.ds = create_cartesian_ds(DYNAMICAL_SYSTEM_TYPE.POINT_ATTRACTOR)
        A_matrix = np.diag([50.0, 50.0, 50.0, 10.0, 10.0, 10.0])
        self.ds.set_parameter_value(
            "gain", np.diag(A_matrix), sr.ParameterType.DOUBLE_ARRAY #standart linear DS
        )
        print("## PARAMS DS ## : ", self.ds.get_parameters())

        #create a controller : is there a type t_c = G - D(xdot - f(x)) ??
        self.ctrl = create_cartesian_controller(CONTROLLER_TYPE.DISSIPATIVE)
        print("## PARAMS CTRL ## : ", self.ctrl.get_parameters())
        print(dir(self.ctrl))
        self.ctrl.set_parameter_value(
            "damping", [[10.0, 10.0, 10.0, 0.0, 0.0, 0.0],[10.0, 10.0, 10.0, 0.0, 0.0, 0.0],[10.0, 10.0, 10.0, 0.0, 0.0, 0.0],[10.0, 10.0, 10.0, 0.0, 0.0, 0.0],[10.0, 10.0, 10.0, 0.0, 0.0, 0.0],[10.0, 10.0, 10.0, 0.0, 0.0, 0.0]], sr.ParameterType.MATRIX
        )
        self.ctrl.set_parameter_value(
            "damping_eigenvalues", [10.0, 10.0, 10.0, 0.0, 0.0, 0.0], sr.ParameterType.DOUBLE_ARRAY
        )
        
        print("ITS NOT SET, WHY ??, is it private ?? ")
        # self.ctrl.set_parameter_value(
        #     "stiffness", 0.0, sr.ParameterType.DOUBLE
        # )
        # print("## PARAMS CTRL ## : ", self.ctrl.get_parameters())
        print("## PARAMS CTRL NEW ## : ", self.ctrl.get_parameters())

        #sim is an instance containing the virtual copy of all the elements
        self.sim = Simulated(
            lambda_DS = 100.0,
            lambda_perp = 20.0,
            lambda_obs = 200,
        )

        #create obstacle env
        obs_position = np.array([[0.0, 0.0, 0.0]]) #these are N x 3
        obs_axes_lenght = np.array([[0.6, 0.6, 0.6]])
        obs_vel = np.array([[0.0, 0.0, 0.0]])
        self.sim.create_env(obs_position, obs_axes_lenght, obs_vel)

        #create a copy of the DS that is understood by the mod. avoider
        #way to not have maximum_velocity=3 and distance_decrease=0.5 ??
        self.attractor_position = np.array([0.3, 0.4, 0.3, 0.0, np.pi, 0.0]) #angles to HARDCODED to match the one define after
        self.sim.create_DS_copy(self.attractor_position, A_matrix)

        #create modulation avoider to modulate final DS
        self.sim.create_mod_avoider()



    def run(self):
        target_set = False

        while rclpy.ok():
            state = self.robot.get_state()
            if not state:
                continue

            if not target_set:
                target = sr.CartesianPose(
                    state.ee_state.get_name(),
                    self.attractor_position[0:3],
                    np.array([0.0, 1.0, 0.0, 0.0]),
                    state.ee_state.get_reference_frame(),
                )
                self.ds.set_parameter_value(
                    "attractor",
                    target,
                    sr.ParameterType.STATE,
                    sr.StateType.CARTESIAN_POSE,
                )
                #print("### DS ###", self.ds.get_parameters())
                target_set = True
            else:
                #get current state
                x = state.ee_state.get_position()
                x_dot = state.ee_state.get_linear_velocity()

                # print("x : ", x)
                # print("xdot : ", x_dot)

                #get xdot_des of the MODULATED DS
                #x_dot_des = self.sim.dynamic_avoider.evaluate(x)
                #print(x_dot_des)
                #compute D

                #https://epfl-lasa.github.io/control-libraries/versions/main/_dissipative_8hpp_source.html
                #l163 : template<class S>
                # void Dissipative<S>::compute_damping(const S& desired_velocity) {
                # this->basis_ = this->compute_orthonormal_basis(desired_velocity);
                # auto diagonal_eigenvalues = this->damping_eigenvalues_->get_value().asDiagonal();
                # this->damping_->set_value(this->basis_ * diagonal_eigenvalues * this->basis_.transpose());
                # }




                #here control happen
                # print("\n state pos :", state.ee_state.get_position())
                # print("\n state vel :", state.ee_state.get_linear_velocity())
                # print("\n evaluate :", self.ds.evaluate(state.ee_state))
                #twist contains lin + ang speed evaluated by DS
                twist = sr.CartesianTwist(self.ds.evaluate(state.ee_state))
                # print("\n twist", twist)
                twist.clamp(0.25, 0.5)
                #command_torques has name 1->7 of joint + torque array 1x7
                self.command_torques = sr.JointTorques(
                    self.ctrl.compute_command(twist, state.ee_state, state.jacobian)
                )
                # print("\n com. torque :", self.command_torques)
                self.command.joint_state = state.joint_state
                self.command.joint_state.set_torques(self.command_torques.get_torques())

                self.robot.send_command(self.command)

            self.rate.sleep()


if __name__ == "__main__":
    rclpy.init()
    # rospy.init_node("test", anonymous=True)
    robot_interface = RobotInterface("*:1601", "*:1602") #16 is for the first franka robot

    # Spin in a separate thread
    controller = PassiveController(robot=robot_interface, freq=500)

    thread = threading.Thread(target=rclpy.spin, args=(controller,), daemon=True)
    thread.start()

    try:
        controller.run()

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    thread.join()