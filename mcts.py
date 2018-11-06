import numpy as np

from abc import abstractmethod
from random import choice

from abstracts import ABCMCNode, ABCTreeSearch, Game, Policy
from game import Winner, Player

# Data structures


class MCCommonNode(ABCMCNode):
    def __init__(self, max_child_states):
        self.mcs = max_child_states
        self.child_values = np.zeros([3, max_child_states], np.float32)
        self.children = []
        self.num_childs = 0
        self.exp = False

    def has_children(self):
        return self.num_childs > 0

    def add_child(self, priority):
        move_idx = len(self.children)
        self.child_values[2, move_idx] = priority

        self.num_childs += 1
        self.children.append(MCChildNode(
            self.mcs,
            self,
            move_idx,
        ))

    def choose_best(self, player, policy, states):
        move = minmax(player, policy.evaluate(player,
            self.child_values[0, :self.num_childs],
            self.child_values[1, :self.num_childs],
            self.child_values[2, :self.num_childs],
        ))

        return self.children[move], states[move]

    @property
    def expanded(self):
        return self.exp

    def expand(self):
        self.exp = True


class MCRootNode(MCCommonNode):
    def __init__(self, max_sub_states):
        super().__init__(max_sub_states)

    def visit(self):
        pass

    def add_score(self, score):
        pass

    def parent(self):
        return None


class MCChildNode(MCCommonNode):
    def __init__(self, max_sub_states: int, parent: ABCMCNode, move_idx: int):
        super().__init__(max_sub_states)

        self.mparent = parent
        self.move_idx = move_idx

    def visit(self):
        self.mparent.child_values[0, self.move_idx] += 1

    def add_score(self, score):
        self.mparent.child_values[1, self.move_idx] += score

    def parent(self):
        return self.mparent

# Policies


class TopScorePolicy(Policy):
    def evaluate(self, player, visits, score, priority):
        return score


class RandomPolicy(Policy):
    def evaluate(self, player, visits, score, priority):
        return np.random.random(getattr(visits, "shape", None))


class QUPolicy(Policy):
    @abstractmethod
    def U(self, visits, score, priority):
        pass

    @abstractmethod
    def Q(self, visits, score, priority):
        pass

    def evaluate(self, player, visits, score, priority):
        return self.Q(visits, score, priority) \
            + player.value * self.U(visits, score, priority)


class UCT1(QUPolicy):
    def __init__(self, c):
        self.c = c

    def U(self, visits, score, priority):
        return self.c * np.sqrt(np.log(visits) / (1 + visits))

    def Q(self, visits, score, priority):
        return score / (1 + visits)


# Algorithms

def minmax(player, array):
    mm = np.argmax if player is Player.Player1 else np.argmin
    return mm(array)


class MonteCarloTreeSearch(ABCTreeSearch):
    def __init__(self, game: Game, tree_policy, search_policy=TopScorePolicy()):
        self.game = game
        self.tree_policy = tree_policy
        self.search_policy = search_policy

    def search(self, state, iterations):
        root = MCRootNode(self.game.max_child_states)

        for _ in range(iterations):
            node, node_state = self.select(root, state)
            player, _ = node_state
            priorities, score = self.evaluate(node, node_state)
            self.expand(node, node_state, priorities)
            self.backpropagate(node, score * player.value)

        player, _ = state
        _, best_state = root.choose_best(player,
            self.search_policy, self.game.child_states(state))
        return best_state

    def select(self, root, state):
        current = root

        while current.expanded and current.has_children():
            player, _ = state
            current, state = current.choose_best(player,
                self.tree_policy, self.game.child_states(state))

        return current, state

    def expand(self, node, state, priorities):
        node.expand()

        for _, priority in zip(self.game.child_states(state), priorities):
            node.add_child(priority)

    @abstractmethod
    def evaluate(self, node, state):
        pass

    def backpropagate(self, node, score):
        current = node
        while current:
            current.visit()
            current.add_score(score)
            current = current.parent()


class RolloutMCTS(MonteCarloTreeSearch):
    def __init__(self, game, tree_policy, default_policy, **kwargs):
        super().__init__(game, tree_policy, **kwargs)

        self.default_policy = default_policy

    def evaluate(self, node, state):
        priorities = [1 for _ in range(self.game.max_child_states)]
        return priorities, self.rollout(state)

    def rollout(self, state):
        # TODO: multiple rollouts?

        player, _ = state
        player = Winner.from_player(player)

        while self.game.winner(state) is Winner.NA:
            cur_player, _ = state
            child_states = self.game.child_states(state)

            # Visits and scores for all nodes below here should be all zeroes
            zeros = np.zeros(len(child_states))
            move = minmax(cur_player, self.default_policy.evaluate(cur_player, zeros, zeros, zeros))

            state = child_states[move]

        return 1 if self.game.winner(state) is player else -1
