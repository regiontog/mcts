from random import choice

from abstracts import Game
from game import Winner, Player, player_desc


class Nim(Game[int]):
    """
    A game of nim.
    """

    def __init__(self, n, k, starting_player=Player.Player1):
        self.N = n
        self.K = k

        if starting_player is None:
            self.P = choice((Player.Player1, Player.Player2))
        else:
            self.P = starting_player

    @property
    def max_child_states(self):
        return self.K

    def describe_state_transition(self, from_state):
        def inner(to):
            has_won = "" if to[1] != 0 else "\n{} wins.".format(
                player_desc[from_state[0]])

            return "{} selects {} stone(s). {} remaining.{}".format(
                player_desc[from_state[0]], from_state[1] -
                to[1], to[1], has_won
            )

        return inner

    def initial_state(self):
        return self.P, self.N

    def child_states(self, state):
        player, board = state
        next_player = player.swap()

        return [(next_player, board - n) for n in range(1, self.K + 1) if board - n >= 0]

    def winner(self, state):
        player, board = state

        if board == 0:
            return Winner.from_player(player.swap())

        return Winner.NA
