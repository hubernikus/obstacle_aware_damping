from typing import Callable

from attrs import define, field

from passive_control.agent import Agent
from passive_control.controller import Controller


@define
class AgentUpdater:
    agent: Agent
    dynamics: Callable[[np.ndarray], np.ndarray]
    controller: Controller

    def update(self, delta_time: float):
        velocity = self.dynamic(agent.position)
        force = self.controller.compute_control_force(agent, desired_velocity=velocity)
        self.agent.update_step(delta_time, control_force=force)
