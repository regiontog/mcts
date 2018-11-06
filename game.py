from enum import Enum, auto


class Player(Enum):
    Player1 = 1
    Player2 = -1

    def swap(self):
        if self is Player.Player1:
            return Player.Player2
        else:
            return Player.Player1


class Winner(Enum):
    Player1 = auto()
    Player2 = auto()
    Tie = auto()
    NA = auto()

    @staticmethod
    def from_player(player):
        if player is Player.Player1:
            return Winner.Player1
        else:
            return Winner.Player2
