from abc import abstractmethod, ABC


class Agent(ABC):
    @abstractmethod
    def choose_action(self, step, state):
        pass

    @abstractmethod
    def accept_feedback(self, step, prev_state, action, reward, new_state):
        pass
