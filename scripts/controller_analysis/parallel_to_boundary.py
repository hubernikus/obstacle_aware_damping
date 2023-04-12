from dataclasses import dataclass, field
import numpy as np

import matplotlib.pyplot as plt

# import vartools

dimension = 1


class SimpleDS:
    def evaluate(self, position):
        return 0


@dataclass
class SimpleController:
    D: float = 100.0

    def compute_force(self, velocity, desired_velocity):
        return self.D * (desired_velocity - velocity)


@dataclass
class Agent:
    position: np.ndarray = field(default_factory=lambda: np.zeros(dimension))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(dimension))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(dimension))

    max_force: float = 10.0

    M: float = 1.0

    def apply_force(self, force):
        if (force_norm := np.linalg.norm(force)) > self.max_force:
            force = force / force_norm * self.max_force

        self.acceleration = force / self.M

    def euler_step(self, dt=1e-2):
        self.velocity = self.velocity + dt * self.acceleration
        self.position = self.position + dt * self.velocity


def main():
    agent: Agent = Agent()
    agent.velocity = np.array(5)
    controller = SimpleController()
    dynamics = SimpleDS()

    it_max = 500
    dt = 1e-2

    positions = np.zeros((dimension, it_max + 1))
    positions[:, 0] = agent.position

    for ii in range(it_max):
        force = controller.compute_force(
            agent.velocity, dynamics.evaluate(agent.position)
        )
        agent.apply_force(force)
        agent.euler_step(dt)

        positions[:, ii + 1] = agent.position

    plt.plot(dt * np.arange(it_max + 1), positions[0, :])
    # plt.ylim([-5, 5])



if (__name__) == "__main__":
    plt.ion()
    plt.close("all")
    main()
    # breakpoint()
