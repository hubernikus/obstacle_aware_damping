import numpy as np
import matplotlib.pyplot as plt

from vartools.animator import Animator

class Robot:
    #class variable
    dim = 2
    def __init__(
        self,
        M = np.eye(dim),
        C = np.zeros((dim,dim)),
        G = np.zeros((dim,1)),
        controller = None,
        disturbance = None,
    ):
        self.M = M
        self.C = C
        self.G = G
        self.controller = controller
        self.disturbance = disturbance

class BasicController:
    """
    in the form tau_c = G - D*x_dot, does regulation to 0
    """
    #class variables
    dim = 2

    def __init__(
        self,
    ):
        pass

class CotrolledRobotAnimation(Animator):
    #class variables
    dim = 2

    def setup(
        self,
        start_position=np.array([-2.5, 0.5]),
        x_lim=[-1.5, 2],
        y_lim=[-0.5, 2.5],
    ):
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.position_list = np.zeros((self.dim, self.it_max + 1))
        self.position_list[:, 0] = start_position

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def update_step(self, ii: int) -> None:
        print(f"iter : {ii}")

        #CALCULATION
        self.position_list[:,ii] = np.array([0.1*ii, -0.5])


        #CLEARING
        self.ax.clear()

        #PLOTTING
        #past trajectory
        self.ax.plot(
            self.position_list[0, :ii], self.position_list[1, :ii], ":", color="#135e08"
        )
        #actual position
        self.ax.plot(
            self.position_list[0, ii],
            self.position_list[1, ii],
            "o",
            color="#135e08",
            markersize=12,
        )
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)


def run_control_robot():
    #setup of robot
    robot = Robot() #what is the simplest controller tau_c ??

    #setup of animator
    my_animation = CotrolledRobotAnimation(
        dt_simulation=0.05,
        dt_sleep=0.01,
    )

    my_animation.setup(
        x_lim=[-3, 3],
        y_lim=[-2.1, 2.1],
    )

    my_animation.run(save_animation=False)



if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    run_control_robot()
