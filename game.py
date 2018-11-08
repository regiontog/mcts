from enum import Enum, auto


class Player(Enum):
    """
    An algebraic datatype representing the players, the
    values associated with each player must have equal 
    magnitute but opposite sign. 
    """
    Player1 = 1
    Player2 = -1

    def swap(self):
        """
        Returns the other player
        """

        if self is Player.Player1:
            return Player.Player2
        else:
            return Player.Player1


class Winner(Enum):
    """
    An algebraic datatype representing each of the 
    different ending states a given gamestate could be in.
    Including ties and game not ended.
    """
    Player1 = auto()
    Player2 = auto()
    Tie = auto()
    NA = auto()

    @staticmethod
    def from_player(player):
        """
        Converts the Player datatype into the equivalent Winner datatype.
        """

        if player is Player.Player1:
            return Winner.Player1
        else:
            return Winner.Player2


# Lookup table mapping Players to their "names"
player_desc = {
    Player.Player1: "Player 1",
    Player.Player2: "Player 2",
    None: "Random player",
}

# Lookup table mapping Winners to their "names"
winner_desc = {
    Winner.Player1: "player 1",
    Winner.Player2: "player 2",
}
