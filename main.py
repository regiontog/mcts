import multiprocessing
from random import choice


from game import Winner, Player, player_desc, winner_desc
from mcts import RolloutMCTS, UCT, RandomPolicy
from nim import Nim


if __name__ == "__main__":
    verbose = False
    do_sync = False or verbose

    # P = None
    P = Player.Player2
    W = Winner.Player1

    G = 100
    M = 1000
    N = 10
    K = 5

    def play_game(i):
        """
        Plays out one isolated game
        """

        print("Game", i)

        game = Nim(N, K, starting_player=P)
        mcts = RolloutMCTS(game, UCT(), RandomPolicy())

        game_state = game.initial_state()

        if verbose:
            to = game.describe_state_transition(game_state)

        tree = None

        while game.winner(game_state) is Winner.NA:
            # Tree search move for player 1
            tree, game_state = mcts.search(game_state, M, root=tree)

            if verbose:
                print(to(game_state))
                to = game.describe_state_transition(game_state)

            # # Random move for player 2
            # p2_moves = game.child_states(game_state)
            # if p2_moves:
            #     tree, game_state = choice(list(zip(tree.children, p2_moves)))

            # if verbose:
            #     print(to(game_state))
            #     to = game.describe_state_transition(game_state)

        return 1 if game.winner(game_state) is W else 0

    if do_sync:
        wins = sum(play_game(i) for i in range(G))
    else:
        wins = sum(multiprocessing.Pool().imap_unordered(play_game, range(G)))

    print("{} has starting move and {} wins {} of {} games!".format(
        player_desc[P], winner_desc[W], wins, G))
