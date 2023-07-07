from typing import Optional

import numpy as np

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.utils import get_orthogonal_basis

from passive_control.agent import Agent
from passive_control.controller import ObstacleAwarePassivController


def test_simple_obstacle(visualize=False):
    from vartools.dynamical_systems import LinearSystem
    from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
    from dynamic_obstacle_avoidance.avoidance import ModulationAvoider

    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Cuboid(
            axes_length=[1.0, 1.0],
            center_position=np.array([0.0, 0.0]),
            margin_absolut=0.15,
            tail_effect=False,
        )
    )

    agent = Agent(position=np.array([-4, 1]))
    controller = ObstacleAwarePassivController(environment=obstacle_environment)

    initial_dynamics = LinearSystem(
        attractor_position=np.array([4.0, 0.0]),
        maximum_velocity=3,
        distance_decrease=0.5,
    )

    avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_environment,
    )

    if visualize:
        import matplotlib.pyplot as plt
        from passive_control.draw_obs_overwrite import plot_obstacles

        x_lim = [-5, 5]
        y_lim = [-4, 4]

        delta_time = 0.01
        it_max = 500

        positions = np.zeros((2, it_max + 1))
        positions[:, 0] = agent.position

        for ii in range(it_max):
            desired_velocity = avoider.evaluate(agent.position)
            control_force = controller.compute_control_force(agent, desired_velocity)
            agent.update_step(delta_time, control_force=control_force)

            if np.any(np.isnan(agent.position)):
                breakpoint()

            positions[:, ii + 1] = agent.position

        fig, ax = plt.subplots()
        ax.plot(positions[0, :], positions[1, :], "--", color="blue")
        ax.plot(positions[0, 0], positions[1, 0], "o", color="black")

        plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            showLabel=False,
        )

    agent.position = np.array([-4, 1])
    agent.velocity = np.zeros_like(agent.position)
    desired_velocity = avoider.evaluate(agent.position)
    control_force = controller.compute_control_force(agent, desired_velocity)
    assert control_force[0] > 0
    assert control_force[1] < 0


if (__name__) == "__main__":
    test_simple_obstacle(visualize=False)
