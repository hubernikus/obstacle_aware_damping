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

class PassiveController(Node):
    def __init__(self, robot, freq: float = 100, node_name="passive_controller"):
        super().__init__(node_name)
        self.robot = robot
        self.rate = self.create_rate(freq)

        self.command = CommandMessage()
        self.command.control_type = [ControlType.EFFORT.value]

        #create DS type xdot = -A(x - x_atr), now x is in 6d
        #no need, remoove when posible
        # self.ds = create_cartesian_ds(DYNAMICAL_SYSTEM_TYPE.POINT_ATTRACTOR)
        # A_matrix = np.diag([50.0, 50.0, 50.0, 10.0, 10.0, 10.0]) 
        # self.ds.set_parameter_value(
        #     "gain", np.diag(A_matrix), sr.ParameterType.DOUBLE_ARRAY #standart linear DS
        # )

        #create a controller : is there a type t_c = G - D(xdot - f(x)) ??
        self.ctrl = create_cartesian_controller(CONTROLLER_TYPE.IMPEDANCE)

        self.ctrl.set_parameter_value("damping", np.zeros((6,6)), sr.ParameterType.MATRIX)
        self.ctrl.set_parameter_value("stiffness", np.zeros((6,6)), sr.ParameterType.MATRIX)
        self.ctrl.set_parameter_value("inertia", np.zeros((6,6)), sr.ParameterType.MATRIX)

        #usefull to get methods
        print(dir(self.ctrl))

        # print("## PARAMS CTRL ## : ", self.ctrl.get_parameters())
        #print("## PARAMS CTRL NEW ## : ", self.ctrl.get_parameters())

        #sim is an instance containing the virtual copy of all the elements, used to compute D
        self.sim = Simulated(
            lambda_DS = 200.0, #100
            lambda_perp = 20.0, #20
            lambda_obs = 200, #200
        )

        #create obstacle env
        obs_position = np.array([[0.5, 0.0, 0.0]]) #these are N x 3
        obs_axes_lenght = np.array([[10., 0.3, 1.0]])
        obs_vel = np.array([[0.0, 0.0, 0.0]])
        
        #to test without obs
        no_obs = False
        self.sim.create_env(obs_position, obs_axes_lenght, obs_vel, no_obs)

        #create the DS that is understood by the mod. avoider !!!! NOT FORGET minus (!= cpp )
        self.A = -np.diag([1.0, 1.0, 1.0])
        self.attractor_A = np.array([0.5, 0.5, 0.3])
        self.attractor_B = np.array([0.5, -0.5, 0.3])
        self.attractor_position = self.attractor_A
        self.attrator_quaternion = np.array([0.0, 0.0, 0.0, 0.0])
        self.max_vel = 0.5
        self.sim.create_DS_copy(self.attractor_position, self.A, self.max_vel)

        #create modulation avoider to modulate final DS
        self.sim.create_mod_avoider()

        #to compare - remooove when debuged
        self.ctrl_ref = create_cartesian_controller(CONTROLLER_TYPE.DISSIPATIVE)
        self.ctrl_ref.set_parameter_value("eigenvalues", np.diag([20, 2, 2, 1, 1, 1]), sr.ParameterType.MATRIX)
        self.ds = create_cartesian_ds(DYNAMICAL_SYSTEM_TYPE.POINT_ATTRACTOR)
        self.ds.set_parameter_value(
             "gain", [1, 1, 1, 0.1, 0.1, 0.1], sr.ParameterType.DOUBLE_ARRAY #standart linear DS
        )




    def run(self):
        target_set = False
        atr_a = True

        while rclpy.ok():
            state = self.robot.get_state()
            if not state:
                continue
            #print("yo")
            #useless remoove when done
            if False:
                pass
            #below, comment when debug
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
                #get current state
                #x = state.ee_state.get_position()
                # x_dot = state.ee_state.get_linear_velocity()
                #print("x : ", x)
                # print("xdot : ", x_dot)

                #get xdot_des of the MODULATED DS
                #x = state.ee_state.get_pose()
                #print(x)
                #print(dir(state.ee_state))
                #print(dir(self.sim.obstacle_environment))
                # x_dot_des = self.sim.dynamic_avoider.evaluate(state.ee_state.get_position())
                # print(x_dot_des)
                #compute D

                #https://epfl-lasa.github.io/control-libraries/versions/main/_dissipative_8hpp_source.html
                #l163 : template<class S>
                # void Dissipative<S>::compute_damping(const S& desired_velocity) {
                # this->basis_ = this->compute_orthonormal_basis(desired_velocity);
                # auto diagonal_eigenvalues = this->damping_eigenvalues_->get_value().asDiagonal();
                # this->damping_->set_value(this->basis_ * diagonal_eigenvalues * this->basis_.transpose());
                # }

                #print(self.ctrl.get_parameter_value("damping"))
                #print(self.ctrl.get_parameter_value("damping_eigenvalues"))

                #here control happen
                # print("\n state pos :", state.ee_state.get_position())
                # print("\n state vel :", state.ee_state.get_linear_velocity())
                # print("\n evaluate :", self.ds.evaluate(state.ee_state))
                #twist contains lin + ang speed evaluated by DS


                #################

                #1. get the pose, vel from the state
                x = state.ee_state.get_pose()   
                #print(type(state.ee_state))
                pos = x[0:3]
                r = R.from_quat(x[3:7])
                #print(pos)
                ang = r.as_euler('zyx', degrees=False) #not sure which format

                pos_dot = state.ee_state.get_linear_velocity()
                #ang_dot = state.ee_state.get_angular_velocity()

                #print(ang_dot)
                #print(pos_dot)
                #print(x)

                #print(ang)
                #print("position", pos)

                #1. compute desired velocity in cartesian coord
                print("actual position : ", pos, " and ang : ", ang)
                print("actual vel : ", pos_dot)
                pos_dot_des = self.sim.dynamic_avoider.evaluate(pos)
                print("des_vel by avoider", pos_dot_des)

                #2. compute the damping matrix
                D = self.sim.compute_D(pos, pos_dot, pos_dot_des)
                #print("des_vel by avoider after ",pos_dot_des)
                KD_ang = np.diag([1] *3)
                big_D = np.zeros((6,6))
                big_D[0:3, 0:3] = D
                big_D[3:6, 3:6] = KD_ang
                #print(big_D)

                #3. build the K matrix
                big_K = np.zeros((6,6)) 
                KP_ang = np.diag([1000] *3)
                big_K[3:6, 3:6] = KP_ang

                #3. assign them to the robot
                self.ctrl.set_parameter_value("damping", big_D, sr.ParameterType.MATRIX)
                self.ctrl.set_parameter_value("stiffness", big_K, sr.ParameterType.MATRIX)


                #4. construct des_ee_state
                des_ee_state = sr.CartesianState(
                    state.ee_state.get_name(),
                    state.ee_state.get_reference_frame(),
                )
                des_ee_state.set_pose(np.concatenate((np.zeros(3), self.attrator_quaternion))) #matrix 7x1
                des_ee_state.set_linear_velocity(pos_dot_des)
                des_ee_state.set_angular_velocity(np.zeros(3))
                #des_ee_state.clamp(0.25, 0.5) #just added, need test
                #print("des vel", des_ee_state.get_linear_velocity())
                #print("real vel", state.ee_state.get_linear_velocity())
                #print("my des state : ", des_ee_state)
                #################

                #print("## PARAMS CTRL NEW ## : ", self.ctrl.get_parameters())
                
                #comparison ####
                #ds_ret = self.ds.evaluate(state.ee_state)
                #print(type(ds_ret)) #state_representation.CartesianState
                #print("cpp ds : ", ds_ret)
                # print("cpp ds twist : ", sr.CartesianTwist(ds_ret))
                #twist = sr.CartesianTwist(ds_ret)
                # sr.CartesianTwist
                # com_ref = sr.JointTorques(
                #     self.ctrl_ref.compute_command(ds_ret, state.ee_state, state.jacobian)
                # )
                #self.command.joint_state.set_torques(com_ref.get_torques())
                ####

                #print("\n com. torque ref", com_ref.get_torques())
                # print("\n twist", twist)
                #twist.clamp(0.25, 0.5)
                #command_torques has name 1->7 of joint + torque array 1x7

                #5. compute the command with the impedance controller
                self.command_torques = sr.JointTorques(
                    self.ctrl.compute_command(des_ee_state, state.ee_state, state.jacobian) #SWAPED !!! ??
                )
                #print("\n com. torque :", self.command_torques.get_torques())

                #6. send the ecommand to the robot
                self.command.joint_state = state.joint_state
                self.command.joint_state.set_torques(self.command_torques.get_torques())
                self.robot.send_command(self.command)

                #7. switch atractor if converge
                EPS = 1e-2
                print("curent atractor position : ", self.attractor_position)
                if np.linalg.norm(pos - self.attractor_position) <= EPS:
                    print("SWICH")
                    if atr_a:
                        self.attractor_position = self.attractor_B
                        atr_a = False
                    else:
                        self.attractor_position = self.attractor_A
                        atr_a = True

                    #update the DS
                    self.sim.create_DS_copy(self.attractor_position, self.A, self.max_vel)

                    #update the modulation avoider to modulate final DS
                    self.sim.create_mod_avoider()


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