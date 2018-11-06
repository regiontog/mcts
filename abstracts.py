from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Callable, Tuple

from game import Winner, Player


class Policy(ABC):
    @abstractmethod
    def evaluate(self, visits, score):
        pass


class ABCMCNode(ABC):
    @abstractmethod
    def choose_best(self, policy, states):
        pass

    @abstractmethod
    def visit(self):
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
    def winner(self, state: State) -> Winner:
        pass

    @property
    @abstractmethod
    def max_child_states(self) -> int:
        pass
