import numpy as np
import matplotlib.pyplot as plt

from vartools.states import Pose

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles


def two_obstacles_gamma(visualize=False):
    x_lim = [-0.1, 5]
    y_lim = [-0.1, 4]
    container = ObstacleContainer()
    container.append(
        Cuboid(
            pose=Pose(position=np.array([0.5, 0.5])),
            axes_length=np.array([0.2, 0.2]),
        )
    )
    container.append(
        Ellipse(
            pose=Pose(position=np.array([0.5, 0.5])),
            axes_length=np.array([0.2, 0.2]),
        )
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_obstacles(container, ax=ax, x_lim=x_lim, y_lim=y_lim)




if __name__ == "__main__":
    two_obstacles_gamma(visualize=False)
