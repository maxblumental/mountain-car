import logging

import numpy as np

from .common import Hook
from ..params.logger_params import LoggerParams


class LoggerHook(Hook):
    def __init__(self, params: LoggerParams):
        self.params = params
        self.episode = 0
        self.total_reward = None
        self.max_xs = None
        self.current_max_x = None
        self.steps_num = None
        self.win_count = None
        self.reset_stats()

    def do_before_step(self, step, state):
        pass

    def do_after_step(self, step, prev_state, action, new_state, reward):
        self.steps_num += 1
        self.total_reward += reward
        self.current_max_x = max(self.current_max_x, new_state[0])

    def on_done(self, step, prev_state, action, new_state, reward):
        self.episode += 1

        if new_state[0] >= 0.5:
            self.win_count += 1

        period = self.params.period_in_episodes
        if self.episode % period == period - 1:
            self.log_stats()
            self.reset_stats()
        else:
            self.max_xs.append(self.current_max_x)
            self.current_max_x = -3

    def log_stats(self):
        period = self.params.period_in_episodes
        parts = [
            f"wins={self.win_count / period:.0%}",
            f"median_max_x={np.median(self.max_xs):.2f}",
            f"max_max_x={np.max(self.max_xs):.2f}",
            f"reward={self.total_reward / period:.1f}",
        ]
        logging.info("episode %d: %s", self.episode + 1, ", ".join(parts))

    def reset_stats(self):
        self.total_reward = 0
        self.steps_num = 0
        self.current_max_x = -3
        self.max_xs = []
        self.win_count = 0
