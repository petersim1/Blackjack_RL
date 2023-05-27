from typing import Optional, List
from copy import deepcopy

def init_q(moves_blacklist: List[str]=[], mode: Optional[str]=None) -> object:
    """
    Initialize the Q value object.
    I've gone back and forth about how to structure this.
    I think it's best to tease out ability to split and ability to double as isolated states.

    Note: We'll skip states that are unreachable (ie having a total of 4 and not being able to split is impossible),
    this makes the visualizations simpler to create, without having to filter out invalid states downstream.
    """

    moves = ["stay", "hit", "split", "double", "surrender"]
    moves = [m for m in moves if m not in moves_blacklist]
    Q = {}
    for p in range(4,22) :
        ace_arr = [False]
        if 11 < p < 21:
            ace_arr.append(True)

        split_arr = [False]
        if not p % 2:
            split_arr.append(True)
            
        for h in range(2,12) :
            for ace in ace_arr:
                for split in split_arr:
                    if (p == 4) and (not split): continue
                    if (p == 12) and (not split) and ace: continue
                    if (p > 12) and (not p % 2) and split and ace: continue
                    Q[(p, h, ace, split)] = {m:0 for m in moves}

    if mode == "accepted":
        return init_accepted_q(Q)
    
    if mode == "house":
        return init_house_q(Q)

    return Q


def init_accepted_q(q):
    accepted_q = deepcopy(q)
    for (player, house, useable_ace, can_split), vals in accepted_q.items():
        if player <= 7:
            vals["split"] = int(can_split and (house <= 7))
            vals["hit"] = 0.5
        if player == 8:
            vals["split"] = int(can_split and (house in [5,6]))
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
            vals["split"] = int(can_split and (useable_ace or (house <= 6)))
            vals["hit"] = 0.5 * int(house not in [4,5,6])
            vals["stay"] = 0.5 - vals["hit"]
        if player == 13:
            vals["double"] = int(useable_ace and (house in [5,6]))
            vals["hit"] = 0.5 * int(useable_ace or (house > 6))
            vals["stay"] = 0.25 * int(house <= 6)
        if player == 14:
            vals["split"] = int(can_split and (house <= 7))
            vals["double"] = int(useable_ace and (house in [5,6]))
            vals["hit"] = 0.5 * int(useable_ace or (house > 6))
            vals["stay"] = 0.25 * int(house <= 6)
        if player == 15:
            vals["surrender"] = int(house == 10)
            vals["double"] = int(useable_ace and (house in [4,5,6]))
            vals["hit"] = 0.5 * int(useable_ace or (house > 6))
            vals["stay"] = 0.25 * int(house <= 6)
        if player == 16:
            vals["surrender"] = 0.75* int(house >= 9)
            vals["split"] = int(can_split)
            vals["double"] = int(useable_ace and (house in [4,5,6]))
            vals["hit"] = 0.5 * int(useable_ace or (house > 6))
            vals["stay"] = 0.25 * int(house <= 6)
        if player == 17:
            vals["double"] = int(useable_ace and (house in [3,4,5,6]))
            vals["hit"] = 0.5 * int(useable_ace)
            vals["stay"] = 0.5 - vals["hit"]
        if player == 18:
            vals["split"] = int(can_split and (house not in [7,10,11]))
            vals["double"] = int(useable_ace and (house <= 6))
            vals["stay"] = 0.25
            vals["hit"] = 0.5 * int(useable_ace and (house >= 9) and (not can_split))
        if player == 19:
            vals["double"] = int(useable_ace and (house == 6))
            vals["stay"] = 0.5
        if player == 20:
            vals["stay"] = 1
        if player == 21:
            vals["stay"] = 1

    return accepted_q


def init_house_q(q):
    house_q = deepcopy(q)
    for (player, _, _, _), vals in house_q.items():
        if player >= 17:
            vals["stay"] = 1
        else :
            vals["hit"] = 1
    return house_q