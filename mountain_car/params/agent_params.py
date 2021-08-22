from dataclasses import dataclass


@dataclass()
class AgentParams:
    max_epsilon: float
    min_epsilon: float
    min_epsilon_step: int
    gamma: float
