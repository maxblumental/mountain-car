from abc import ABC, abstractmethod


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
