from typing import Optional
from copy import deepcopy

def init_q(mode: Optional[str]=None) -> object:
    """Initialize the Q value object. Isolates splittable vs. non-splittable"""

    moves = ["stay", "hit", "split", "double", "surrender"]
    Q = {}
    for p in range(4,22) :
        if p > 4:
            if not p % 2:
                split_arr = [False, True]
            else :
                split_arr = [False]
        else :
            split_arr = [True]

        if 11 < p < 21:
            ace_arr = [False, True]
        else :
            ace_arr = [False]
            
        for h in range(2,12) :
            for ace in ace_arr:
                for split in split_arr:
                    if (p == 12) and (ace and not split): continue
                    if (p > 12) and (ace and split): continue
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
            if house <= 7:
                if can_split:
                    vals["split"] = 1
                else :
                    vals["hit"] = 1
            else:
                vals["hit"] = 1
        if player == 8:
            if house in [5,6]:
                if can_split:
                    vals["split"] = 1
                else:
                    vals["hit"] = 1
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
            if can_split:
                if useable_ace: #denotes pair of aces
                    vals["split"] = 1
                else:
                    if house <= 6:
                        vals["split"] = 1
                    else :
                        vals["hit"] = 1
            else:
                if house in [4,5,6]:
                    vals["stay"] = 1
                else :
                    vals["hit"] = 1
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
            elif can_split:
                if house <= 7:
                    vals["split"] = 1
                else:
                    vals["hit"] = 1
            else:
                if house <= 6:
                    vals["stay"] = 1
                else :
                    vals["hit"] = 1
        if player == 15:
            if useable_ace:
                if house in [4,5,6]:
                    vals["double"] = 1
                    vals["hit"] = 0.5
                else:
                    vals["hit"] = 1
            else:
                if house <= 6:
                    vals["stay"] = 1
                else :
                    if house == 10:
                        vals["surrender"] = 1
                        vals["hit"] = 0.5
                    else:
                        vals["hit"] = 1
        if player == 16:
            if useable_ace:
                if house in [4,5,6]:
                    vals["double"] = 1
                    vals["hit"] = 0.5
                else:
                    vals["hit"] = 1
            elif can_split:
                vals["split"] = 1
            else:
                if house <= 6:
                    vals["stay"] = 1
                else :
                    if house in [9,10,11]:
                        vals["surrender"] = 1
                        vals["hit"] = 0.5
                    else:
                        vals["hit"] = 1
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
            elif can_split:
                if house in [7,10,11]:
                    vals["stay"] = 1
                else:
                    vals["split"] = 1
            else:
                vals["stay"] = 1
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
    for (player, house, useable_ace, can_split), vals in house_q.items():
        if player >= 17:
            vals["stay"] = 1
        else :
            vals["hit"] = 1
    return house_q