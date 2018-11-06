from random import choice

from game import Winner, Player
from abstracts import Game

from mcts import RolloutMCTS, UCT1, RandomPolicy


class Nim(Game[int]):
    def __init__(self, n, k, starting_player=Player.Player1):
        self.N = n
        self.K = k
        self.P = starting_player

    @property
    def max_child_states(self):
        return self.K

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


if __name__ == "__main__":
    game = Nim(10, 5, starting_player=Player.Player1)
    game_state = game.initial_state()

    mcts = RolloutMCTS(game, UCT1(1), RandomPolicy())

    while game.winner(game_state) is Winner.NA:
        print(game_state)
        game_state = mcts.search(game_state, 500)

    print(game_state)
    print(game.winner(game_state))
