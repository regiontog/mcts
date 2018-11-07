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

    def add_child(self, prior_propability):
        move_idx = len(self.children)
        self.child_values[2, move_idx] = prior_propability

        self.num_childs += 1
        self.children.append(MCChildNode(
            self.mcs,
            self,
            move_idx,
        ))

    def choose_best(self, player, policy, states):
        move = minmax(player, policy.evaluate(player,
            self.child_values[0, :self.num_childs],
            self.visits,
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
    def __init__(self, max_sub_states, copy_from=None):
        super().__init__(max_sub_states)

        self.mvisits = 0
        self.mscore = 0

        if copy_from:
            self.mvisits = copy_from.visits
            self.mscore = copy_from.score
            self.exp = copy_from.exp
            self.children = copy_from.children
            self.num_childs = copy_from.num_childs
            self.child_values = np.array(copy_from.child_values)


    def visit(self):
        self.mvisits += 1

    @property
    def visits(self):
        return self.mvisits

    @property
    def score(self):
        return self.mscore

    def add_score(self, score):
        self.mscore += score

    def parent(self):
        return None


class MCChildNode(MCCommonNode):
    def __init__(self, max_sub_states: int, parent: ABCMCNode, move_idx: int):
        super().__init__(max_sub_states)

        self.mparent = parent
        self.move_idx = move_idx

    def visit(self):
        self.mparent.child_values[0, self.move_idx] += 1

    @property
    def visits(self):
        return self.mparent.child_values[0, self.move_idx]

    @property
    def score(self):
        return self.mparent.child_values[1, self.move_idx]

    def add_score(self, score):
        self.mparent.child_values[1, self.move_idx] += score

    def parent(self):
        return self.mparent

# Policies


class BestQualityPolicy(Policy):
    def evaluate(self, player, visits, parent_visits, score, prior_propability):
        return score/(1 + visits)

class MostVisitsPolicy(Policy):
    def evaluate(self, player, visits, parent_visits, score, prior_propability):
        return visits

class HighestScorePolicy(Policy):
    def evaluate(self, player, visits, parent_visits, score, prior_propability):
        return score


class RandomPolicy(Policy):
    def evaluate(self, player, visits, parent_visits, score, prior_propability):
        return np.random.random(getattr(visits, "shape", None))


class QUPolicy(Policy):
    @abstractmethod
    def U(self, visits, parent_visits, score, prior_propability):
        pass

    @abstractmethod
    def Q(self, visits, parent_visits, score, prior_propability):
        pass

    def evaluate(self, player, visits, parent_visits, score, prior_propability):
        return self.Q(visits, parent_visits, score, prior_propability) \
            + player.value * self.U(visits, parent_visits,
                                    score, prior_propability)


class UCB1(QUPolicy):
    def __init__(self, c):
        self.c = c

    def U(self, visits, parent_visits, score, prior_propability):
        return self.c * np.sqrt(np.log(parent_visits) / (1 + visits))

    def Q(self, visits, parent_visits, score, prior_propability):
        return score / (1 + visits)


# Algorithms

def minmax(player, array):
    mm = np.argmax if player is Player.Player1 else np.argmin
    return mm(array)


class MonteCarloTreeSearch(ABCTreeSearch):
    def __init__(self, game: Game, tree_policy, search_policy=HighestScorePolicy()):
        self.game = game
        self.tree_policy = tree_policy
        self.search_policy = search_policy

    def search(self, state, iterations, root=None):
        if root is None:
            root = MCRootNode(self.game.max_child_states)

        for _ in range(iterations):
            node, node_state = self.select(root, state)
            player, _ = node_state
            priorities, score = self.evaluate(node, node_state)
            self.expand(node, node_state, priorities)
            self.backpropagate(node, score * player.value)

        player, _ = state
        node, best_state = root.choose_best(
            player,
            self.search_policy,
            self.game.child_states(state)
        )

        # Create a new root without the reference to
        # parent so that python can free the memory thats unused
        standalone_new_root = MCRootNode(
            self.game.max_child_states,
            copy_from=node,
        )

        return standalone_new_root, best_state

    def select(self, root, state):
        current = root

        while current.expanded and current.has_children():
            player, _ = state
            current, state = current.choose_best(
                player,
                self.tree_policy,
                self.game.child_states(state)
            )

        return current, state

    def expand(self, node, state, priors):
        node.expand()

        for _, prior_propability in zip(self.game.child_states(state), priors):
            node.add_child(prior_propability)

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
            move = minmax(cur_player, self.default_policy.evaluate(
                cur_player, zeros, 0, zeros, zeros))

            state = child_states[move]

        return 1 if self.game.winner(state) is player else -1
