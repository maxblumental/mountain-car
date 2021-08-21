from typing import List

import gym
from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def accept_feedback(self, prev_state, new_state, reward):
        pass


class Hook(ABC):
    @abstractmethod
    def do_before_step(self, step, state):
        pass

    @abstractmethod
    def do_after_step(self, step, prev_state, action, new_state, reward):
        pass

    @abstractmethod
    def on_done(self, step, prev_state, action, new_state, reward):
        pass


def play_mountain_car(agent: Agent, hooks: List[Hook], steps_num: int = 100_000):
    env = gym.make("MountainCar-v0")

    state = env.reset()
    for step in range(steps_num):

        for hook in hooks:
            hook.do_before_step(step, state)

        action = agent.choose_action(state)

        new_state, reward, done, info = env.step(action)

        agent.accept_feedback(state, new_state, reward)

        for hook in hooks:
            hook.do_after_step(step, state, action, new_state, reward)

        if done:
            state = env.reset()

            for hook in hooks:
                hook.on_done(step, state, action, new_state, reward)
        else:
            state = new_state

    env.close()
