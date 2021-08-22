import numpy as np

import torch
import torch.nn.functional

from .common import Agent
from ..params.agent_params import AgentParams


class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(2, 32, dtype=torch.float64)
        self.layer2 = torch.nn.Linear(32, 32, dtype=torch.float64)
        self.layer3 = torch.nn.Linear(32, 3, dtype=torch.float64)

    def forward(self, state):
        x = torch.tensor(state, dtype=torch.float64)
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class QLearningAgent(Agent):
    def __init__(self, params: AgentParams):
        self.params = params
        self.dqn = DQN()
        self.criterion = torch.nn.MSELoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.dqn.parameters())

    def choose_action(self, step, state):
        eps = self._calculate_epsilon(step)
        if np.random.rand() < eps:
            return np.random.choice([0, 1, 2])
        return self.dqn(state).argmax().item()

    def accept_feedback(self, step, prev_state, action, reward, new_state):
        target = reward + self.params.gamma * self.dqn(new_state).max().item()
        prediction = self.dqn(prev_state)[action]

        loss = self.criterion(prediction, torch.tensor(target).double())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _calculate_epsilon(self, step):
        min_eps = self.params.min_epsilon
        min_eps_step = self.params.min_epsilon_step
        if step >= min_eps_step:
            return min_eps
        max_eps = self.params.max_epsilon
        eps = max_eps - (max_eps - min_eps) * (min_eps_step - step) / min_eps_step
        return eps
