import numpy as np

from abc import abstractmethod
from random import choice

from abstracts import ABCMCNode, ABCTreeSearch, Game, Policy
from game import Winner, Player

# Data structures


class MCCommonNode(ABCMCNode):
    """
    Base class for MCRootNode and MCChildNode, implements all common behavior 

    Cannot be instanciated directly as some methods are unimplemented.
    """
    def __init__(self, max_child_states):
        self.mcs = max_child_states
        self.child_values = np.zeros([3, max_child_states], np.float32)
        self.children = []
        self.num_childs = 0
        self.exp = False

    def has_children(self):
        """
        Returns True if the self node has any children, will 
        return False node is not expanded or the expansion of 
        the node resulted in 0 children.
        """

        return self.num_childs > 0

    def add_child(self, prior_propability):
        """
        Add a single child to the self node and sets the prior 
        propability of the child.
        """
        move_idx = len(self.children)
        self.child_values[2, move_idx] = prior_propability

        self.num_childs += 1
        self.children.append(MCChildNode(
            self.mcs,
            self,
            move_idx,
        ))

    def choose_best(self, player, policy, states):
        """
        Returns the "best" child and the associated state
        according to the 'policy' passed from the perspective 
        of 'player'.
        """

        move = minmax(player, policy.evaluate(player,
            self.child_values[0, :self.num_childs],
            self.visits,
            self.child_values[1, :self.num_childs],
            self.child_values[2, :self.num_childs],
        ))

        return self.children[move], states[move]

    @property
    def expanded(self):
        """
        Whether or not the self node is expanded.
        """

        return self.exp

    def expand(self):
        """
        Expands the self node. No children are added by 
        this method, see 'add_child'.
        """

        self.exp = True


class MCRootNode(MCCommonNode):
    """
    Class for root nodes, is distinguished from 'MCChildNode' in that it has 
    no parent. As such it has to store it's own visits and score counts.
    """
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
        """
        Increments this nodes visit count.
        """

        self.mvisits += 1

    @property
    def visits(self):
        """
        Returns the visit count for this node.
        """

        return self.mvisits

    @property
    def score(self):
        """
        Returns the score of this node.
        """

        return self.mscore

    def add_score(self, score):
        """
        Adds to this node's score.
        """

        self.mscore += score

    def parent(self):
        """
        The root node has no parent, so this always returns None
        """

        return None


class MCChildNode(MCCommonNode):
    """
    Class for root nodes, is distinguished from 'MCRootNode' in that it has 
    no visits, scores and prior propabilities attribute as those are stored in it's 
    parent in an array for efficiency reasons.
    """
    def __init__(self, max_sub_states: int, parent: ABCMCNode, move_idx: int):
        super().__init__(max_sub_states)

        self.mparent = parent
        self.move_idx = move_idx

    def visit(self):
        """
        Increments this nodes visit count.
        """

        self.mparent.child_values[0, self.move_idx] += 1

    @property
    def visits(self):
        """
        Returns the visit count for this node.
        """

        return self.mparent.child_values[0, self.move_idx]

    @property
    def score(self):
        """
        Returns the score of this node.
        """

        return self.mparent.child_values[1, self.move_idx]

    def add_score(self, score):
        """
        Adds to this node's score.
        """

        self.mparent.child_values[1, self.move_idx] += score

    def parent(self):
        """
        Returns this child node's parent.
        """

        return self.mparent

# Policies


class BestQualityPolicy(Policy):
    """
    A simple policy that prefers high average scores.
    """

    def evaluate(self, player, visits, parent_visits, score, prior_propability):
        return score/(1 + visits)

class MostVisitsPolicy(Policy):
    """
    A simple policy that prefers nodes with many visits.
    """

    def evaluate(self, player, visits, parent_visits, score, prior_propability):
        return visits

class HighestScorePolicy(Policy):
    """
    A simple policy that prefers nodes with the highest score.
    """
    def evaluate(self, player, visits, parent_visits, score, prior_propability):
        return score


class RandomPolicy(Policy):
    """
    A policy that assigns a random value to the node(s).
    """

    def evaluate(self, player, visits, parent_visits, score, prior_propability):
        return np.random.random(getattr(visits, "shape", None))


class QUPolicy(Policy):
    """
    Abstract class for policies that awards bonuses to nodes unrelated 
    to it's quality. As such the bonus should be inverted depending on 
    the player perspective because of minmax. 
    """

    @abstractmethod
    def U(self, visits, parent_visits, score, prior_propability):
        """
        The bonus that may be inverted depending on the player.
        """
        pass

    @abstractmethod
    def Q(self, visits, parent_visits, score, prior_propability):
        """
        The quality of the node.
        """
        pass

    def evaluate(self, player, visits, parent_visits, score, prior_propability):
        return self.Q(visits, parent_visits, score, prior_propability) \
            + player.value * self.U(visits, parent_visits,
                                    score, prior_propability)


class UCB(QUPolicy):
    """
    Abstract class for policies that gives nodes an exploration bonus according to the Upper 
    Confidence Bounds approximation to the multi-armed bandit problem.
    """
    def __init__(self, exploration_bias):
        self.c = exploration_bias

    def U(self, visits, parent_visits, score, prior_propability):
        return self.c * np.sqrt(np.log(parent_visits) / (1 + visits))

class UCT(UCB):
    """
    A policy based on UCB that uses the average score as the quality of the node.
    """
    def __init__(self, exploration_bias=np.sqrt(2)):
        super().__init__(exploration_bias)

    def Q(self, visits, parent_visits, score, prior_propability):
        return score/(1 + visits)


# Algorithms

def minmax(player, array):
    """
    Returns the index of either the maximum or minimum 
    value in 'array' depending of 'player'.
    """

    mm = np.argmax if player is Player.Player1 else np.argmin
    return mm(array)


class MonteCarloTreeSearch(ABCTreeSearch):
    def __init__(self, game: Game, tree_policy, search_policy=HighestScorePolicy()):
        self.game = game
        self.tree_policy = tree_policy
        self.search_policy = search_policy

    def search(self, state, iterations, root=None):
        """
        Performs 'iterations' number of node evaluations in order to 
        make an informed decision about the best move to make in 'state'.

        Optionally reuses an existing tree.

        Returns the tuple (tree, state) of the estimated best child and accompanied state.
        """

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
        """
        Uses the tree policy to traverse the tree from 'root', 
        depth wise until a non-expanded node is found.

        Returns the leaf node (unexpanded).
        """

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
        """
        Expands the node and adds all child states of the leaf node to the tree.
        """

        node.expand()

        for _, prior_propability in zip(self.game.child_states(state), priors):
            node.add_child(prior_propability)

    @abstractmethod
    def evaluate(self, node, state):
        pass

    def backpropagate(self, node, score):
        """
        Writes results to the tree.

        Nodes that have been visited have their counts 
        incremented and the score is assigned.
        """

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
        """
        All child nodes get the same prior 
        propability as we don't know any better.

        Then a game is played out according to the default policy 
        until the game reaches it's end. The score is positive if 
        the rollout game ended in favor of the player who's turn it 
        was at 'state'.
        """

        priorities = [1 for _ in range(self.game.max_child_states)]
        return priorities, self.rollout(state)

    def rollout(self, state):
        player, _ = state
        player = Winner.from_player(player)

        while self.game.winner(state) is Winner.NA:
            cur_player, _ = state
            child_states = self.game.child_states(state)

            # Visits and scores for all nodes below leaf should be all zeroes
            zeros = np.zeros(len(child_states))
            move = minmax(cur_player, self.default_policy.evaluate(
                cur_player, zeros, 0, zeros, zeros))

            state = child_states[move]

        return 1 if self.game.winner(state) is player else -1
