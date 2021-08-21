from abc import abstractmethod, ABC


class Agent(ABC):
    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def accept_feedback(self, prev_state, new_state, reward):
        pass
