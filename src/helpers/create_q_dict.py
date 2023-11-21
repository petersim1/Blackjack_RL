from typing import Optional, List
from copy import deepcopy

def init_q(moves_blacklist: List[str]=[], mode: Optional[str]=None) -> object:
    """
    Initialize the Q value object.
    I've gone back and forth about how to structure this.

    I think it's best to have the state represented as:
    (player_total, house_value, useable_ace)
    
    Exclude whether the player can double or split from the state.
    I'll just explicitly mask the Q values based on policy at the time,
    to tell our model whether these states are accessible or not.

    (A,A) is differentiated from (6,6) , for example, by the useable_ace var.

    Having a separate state for can_split such as:
    (player_total, house_value, useable_ace, can_split)
    will severely limit the number of occurrences in states where can_split=True,
    and we will not take advantage of the knowledge gained from states where
    player_total and house_value are the same, but with can_split = False.
    """

    moves = ["stay", "hit", "split", "double", "surrender"]
    moves = [m for m in moves if m not in moves_blacklist]
    Q = {}
    for p in range(4,22) :
        ace_arr = [False]
        if 11 < p < 21:
            ace_arr.append(True)
            
        for h in range(2,12) :
            for ace in ace_arr:
                Q[(p, h, ace)] = {m:0 for m in moves}

    if mode == "accepted":
        return init_accepted_q(Q)
    
    if mode == "house":
        return init_house_q(Q)

    return Q


def init_accepted_q(q):
    # taken from:
    # https://www.blackjackapprenticeship.com/blackjack-strategy-charts/

    # Uses a waterfall approach. We'll check valid_moves(), constrain the space
    # of values for a given key, then take the optimal move from the constrained space.
    accepted_q: dict = deepcopy(q)
    for (player, house, useable_ace), vals in accepted_q.items():
        if player <= 7:
            vals["split"] = int(house <= 7)
            vals["hit"] = 0.5
        if player == 8:
            vals["split"] = int(house in [5,6])
            vals["hit"] = 0.5
        if player == 9:
            vals["double"] = int(house in [3,4,5,6])
            vals["hit"] = 0.5
        if player == 10:
            vals["double"] = int(house <= 9)
            vals["hit"] = 0.5
        if player == 11:
            vals["double"] = 1
            vals["hit"] = 0.5
        if player == 12:
            vals["split"] = int(useable_ace or (house <= 6))
            vals["hit"] = 0.5 * int(house not in [4,5,6])
            vals["stay"] = 0.5 - vals["hit"]
        if player == 13:
            vals["double"] = int(useable_ace and (house in [5,6]))
            vals["hit"] = 0.5 * int(useable_ace or (house > 6))
            vals["stay"] = 0.25 * int(house <= 6)
        if player == 14:
            vals["split"] = int(house <= 7)
            vals["double"] = int(useable_ace and (house in [5,6]))
            vals["hit"] = 0.5 * int(useable_ace or (house > 6))
            vals["stay"] = 0.25 * int(house <= 6)
        if player == 15:
            vals["surrender"] = int((house == 10) and (not useable_ace))
            vals["double"] = int(useable_ace and (house in [4,5,6]))
            vals["hit"] = 0.5 * int(useable_ace or (house > 6))
            vals["stay"] = 0.25 * int(house <= 6)
        if player == 16:
            vals["surrender"] = 0.75 * int((house >= 9) and (not useable_ace))
            vals["split"] = 1
            vals["double"] = int(useable_ace and (house in [4,5,6]))
            vals["hit"] = 0.5 * int(useable_ace or (house > 6))
            vals["stay"] = 0.25 * int(house <= 6)
        if player == 17:
            vals["double"] = int(useable_ace and (house in [3,4,5,6]))
            vals["hit"] = 0.5 * int(useable_ace)
            vals["stay"] = 0.5 - vals["hit"]
        if player == 18:
            vals["split"] = int(house not in [7,10,11])
            vals["double"] = int(useable_ace and (house <= 6))
            vals["stay"] = 0.25
            vals["hit"] = 0.5 * int(useable_ace and (house >= 9))
        if player == 19:
            vals["double"] = int(useable_ace and (house == 6))
            vals["stay"] = 0.5
        if player == 20:
            vals["stay"] = 1
        if player == 21:
            vals["stay"] = 1

    return accepted_q


def init_house_q(q):
    house_q: dict = deepcopy(q)
    for (player, _, _), vals in house_q.items():
        if player >= 17:
            vals["stay"] = 1
        else :
            vals["hit"] = 1
    return house_q