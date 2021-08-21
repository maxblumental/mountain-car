from typing import List

import gym

from mountain_car.agent import Agent
from mountain_car.hook import Hook
from mountain_car.params.play_params import PlayParams


def play_mountain_car(agent: Agent, hooks: List[Hook], params: PlayParams):
    env = gym.make("MountainCar-v0")

    state = env.reset()
    for step in range(params.steps_num):

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
