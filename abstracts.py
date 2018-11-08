from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Callable, Tuple

from game import Winner, Player

"""
Abstract interfaces for the different classes involved with MCTS
"""


class Policy(ABC):
    @abstractmethod
    def evaluate(self, player, visits, score, prior_propability):
        pass


class ABCMCNode(ABC):
    @abstractmethod
    def choose_best(self, policy, states):
        pass

    @abstractmethod
    def visit(self):
        pass

    @property
    @abstractmethod
    def visits(self):
        pass

    @property
    @abstractmethod
    def score(self):
        pass

    @abstractmethod
    def add_score(self, score):
        pass

    @abstractmethod
    def add_child(self, priority):
        pass

    @property
    @abstractmethod
    def expanded(self):
        pass

    @abstractmethod
    def expand(self):
        pass

    @abstractmethod
    def has_children(self):
        pass

    @abstractmethod
    def parent(self):
        pass


class ABCTreeSearch(ABC):
    @abstractmethod
    def search(self, state):
        pass


T = TypeVar('T')
State = Tuple[Player, T]


class Game(ABC, Generic[T]):
    @abstractmethod
    def initial_state(self) -> State:
        pass

    @abstractmethod
    def child_states(self, state: State) -> List[State]:
        pass

    @abstractmethod
    def describe_state_transition(self, state: State) -> Callable[[State], str]:
        pass

    @property
    @abstractmethod
    def max_child_states(self) -> int:
        pass
