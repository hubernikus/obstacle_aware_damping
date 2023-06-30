import math
from dataclasses import dataclass

import numpy as np

import matplotlib.pyplot as plt


class PassiveController:
    def __init__(self):
        pass


@dataclass
class PointAgent:
    position: float
    # Default zero start velocity
    velocity: float = 0.0  # [m / s]
    acceleration: float = 0.0  # [m / s]

    # Set specific for this simulati
    mass: float = 10.0  # [kg]

    # Control Matrices
    stiffness: float = 0.0
    damping: float = 1.0  # [Ns / m]
    inertia: float = 0.0

    def evaluate_force(
        self,
        desired_position: float = 0.0,
        desired_velocity: float = 0.0,
        desired_acceleration: float = 0.0,
    ) -> float:
        force = (desired_position - self.position) * self.stiffness
        force += (desired_velocity - self.velocity) * self.damping
        force += (desired_acceleration - self.acceleration) * self.inertia
        return force

    def update_force_step(self, force: float, delta_time: float):
        # Simplest newton integral -> could be extended
        self.acceleration = force / self.mass
        self.velocity = self.velocity + self.acceleration * delta_time
        self.position = self.position + self.velocity * delta_time


def get_desired_position(time: float | np.ndarray) -> float | np.ndarray:
    # return 10 * np.sin(0.1 * time)
    return 0 * np.sin(0.1 * time)


def get_desired_stiffness(
    time: float | np.ndarray, k0: float = 0.1, mass: float = 10.0
) -> float | np.ndarray:
    # return k0 + 10 * np.sin(0.1 * time)
    return k0 + 0.9 * k0 * np.sin(math.sqrt(k0 / mass) * time)


def main(time_step: float = 0.01, time_max: float = 100):
    it_max = int(time_max / time_step)

    times = np.linspace(0, time_max, it_max + 1)
    desired_positions = get_desired_position(times)
    desired_stiffness = get_desired_stiffness(times)

    positions = np.zeros_like(desired_positions)
    positions[0] = 10

    agent = PointAgent(position=positions[0])
    for ii in range(it_max):
        agent.stiffness = desired_stiffness[ii]
        force = agent.evaluate_force(desired_position=desired_positions[ii])
        agent.update_force_step(force, delta_time=time_step)

        positions[ii + 1] = agent.position

    fig, ax = plt.subplots(figsize=(7, 5))
    # fig, axs = plt.subplots(2, 1, figsize=(7, 5))
    # ax = axs[0]
    ax.plot(times, positions, color="blue", label="Actual")
    ax.plot(times, desired_positions, "--", color="red", label="Reference")
    ax.plot(times, positions, color="blue", label="Actual")
    ax.set_ylim([-1e3, 1e3])
    ax.legend()
    ax.grid()

    # ax = axs[1]
    ax.plot(times, 20 * desired_stiffness, ":", color="green", label="Stiffness")
    ax.legend()
    ax.grid()

    print("Done")


if (__name__) == "__main__":
    plt.ion()
    plt.close("all")
    main()
