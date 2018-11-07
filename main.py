from random import choice

from game import Winner, Player
from abstracts import Game

import multiprocessing

from mcts import RolloutMCTS, UCB1, RandomPolicy

player_desc = {
    Player.Player1: "Player 1",
    Player.Player2: "Player 2",
    None: "Random player",
}

winner_desc = {
    Winner.Player1: "player 1",
    Winner.Player2: "player 2",
}


class Nim(Game[int]):
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


if __name__ == "__main__":
    verbose = False
    do_sync = False or verbose

    # P = None
    P = Player.Player2
    W = Winner.Player1

    G = 100
    M = 10000
    N = 20
    K = 3

    def play_game(i):
        print("Game", i)

        game = Nim(N, K, starting_player=P)
        mcts = RolloutMCTS(game, UCB1(1), RandomPolicy())

        game_state = game.initial_state()
        tree = None

        while game.winner(game_state) is Winner.NA:
            # Tree search move for player 1
            if verbose:
                to = game.describe_state_transition(game_state)

            tree, game_state = mcts.search(game_state, M, root=tree)

            if verbose:
                print(to(game_state))

            # # Random move for player 2
            # p2_moves = game.child_states(game_state)
            # if p2_moves:
            #     tree, game_state = choice(list(zip(tree.children, p2_moves)))

        return 1 if game.winner(game_state) is W else 0

    if do_sync:
        wins = sum(play_game(i) for i in range(G))
    else:
        wins = sum(multiprocessing.Pool().imap_unordered(play_game, range(G)))

    print("{} has starting move and {} wins {} of {} games!".format(
        player_desc[P], winner_desc[W], wins, G))
