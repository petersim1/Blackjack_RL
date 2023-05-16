from typing import Optional, List
from copy import deepcopy

def init_q(moves_blacklist: List[str]=[], mode: Optional[str]=None) -> object:
    """
    Initialize the Q value object.
    Initially, I was isolating splittable vs. non-splittable,
    however, I really think this screws up the Q-learning process,
    as information can't be borrowed as easily.
    """

    moves = ["stay", "hit", "split", "double", "surrender"]
    moves = [m for m in moves if m not in moves_blacklist]
    Q = {}
    for p in range(4,22) :
        if 11 < p < 21:
            ace_arr = [False, True]
        else :
            ace_arr = [False]
            
        for h in range(2,12) :
            for ace in ace_arr:
                Q[(p, h, ace)] = {m:0 for m in moves}

    if mode == "accepted":
        return init_accepted_q(Q)
    
    if mode == "house":
        return init_house_q(Q)

    return Q


def init_accepted_q(q):
    accepted_q = deepcopy(q)
    for (player, house, useable_ace), vals in accepted_q.items():
        if player <= 7:
            if house <= 7:
                vals["split"] = 1
                vals["hit"] = 0.5
            else:
                vals["hit"] = 1
        if player == 8:
            if house in [5,6]:
                vals["split"] = 1
                vals["hit"] = 0.5
            else:
                vals["hit"] = 1
        if player == 9:
            if house in [3,4,5,6]:
                vals["double"] = 1
                vals["hit"] = 0.5
            else :
                vals["hit"] = 1
        if player == 10:
            if house <= 9:
                vals["double"] = 1
                vals["hit"] = 0.5
            else:
                vals["hit"] = 1
        if player == 11:
            vals["double"] = 1
            vals["hit"] = 0.5
        if player == 12:
            if useable_ace:
                vals["split"] = 1
            else:
                if house <= 6:
                    vals["split"] = 1
                else :
                    vals["hit"] = 1
                if house in [4,5,6]:
                    vals["stay"] = 0.5
                else:
                    vals["hit"] = 0.5
        if player == 13:
            if useable_ace:
                if house in [5,6]:
                    vals["double"] = 1
                    vals["hit"] = 0.5
                else:
                    vals["hit"] = 1
            else:
                if house <= 6:
                    vals["stay"] = 1
                else :
                    vals["hit"] = 1
        if player == 14:
            if useable_ace:
                if house in [5,6]:
                    vals["double"] = 1
                    vals["hit"] = 0.5
                else:
                    vals["hit"] = 1
            else:
                if house <= 7:
                    vals["split"] = 1
                else:
                    vals["hit"] = 1
                if house <= 6:
                    vals["stay"] = 0.5
                else :
                    vals["hit"] = 0.5
        if player == 15:
            if useable_ace:
                if house in [4,5,6]:
                    vals["double"] = 1
                    vals["hit"] = 0.5
                else:
                    vals["hit"] = 1
            else:
                if house == 10:
                    vals["surrender"] = 1
                if house <= 6:
                    vals["stay"] = 1
                else :
                    vals["hit"] = 0.5
        if player == 16:
            if useable_ace:
                if house in [4,5,6]:
                    vals["double"] = 1
                    vals["hit"] = 0.5
                else:
                    vals["hit"] = 1
            else:
                if house in [9,10,11]:
                    vals["surrender"] = 0.5
                vals["split"] = 1
                if house <= 6:
                    vals["stay"] = 0.25
                else :
                    vals["hit"] = 0.25
        if player == 17:
            if useable_ace:
                if house in [3,4,5,6]:
                    vals["double"] = 1
                    vals["hit"] = 0.5
                else:
                    vals["hit"] = 1
            else:
                vals["stay"] = 1
        if player == 18:
            if useable_ace:
                if house <= 6:
                    vals["double"] = 1
                    vals["stay"] = 0.5
                elif 6 < house < 9:
                    vals["stay"] = 1
                else:
                    vals["hit"] = 1
            else:
                if house not in [7,10,11]:
                    vals["split"] = 1
                vals["stay"] = 0.5
        if player == 19:
            if useable_ace:
                if house == 6:
                    vals["double"] = 1
                    vals["stay"] = 0.5
                else:
                    vals["stay"] = 1
            else:
                vals["stay"] = 1
        if player == 20:
            vals["stay"] = 1
        if player == 21:
            vals["stay"] = 1

    return accepted_q


def init_house_q(q):
    house_q = deepcopy(q)
    for (player, _, _), vals in house_q.items():
        if player >= 17:
            vals["stay"] = 1
        else :
            vals["hit"] = 1
    return house_q