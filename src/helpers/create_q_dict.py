from typing import Optional
from src.constants import card_map


def init_q(mode: Optional[str]=None) -> object:
    """Initialize the Q value object. Isolates splittable vs. non-splittable"""

    moves = ["stay", "hit", "split", "double", "surrender"]
    moves_no_split = [m for m in moves if m!="split"]

    Q = {"can_split": {}, "no_split": {}}
    
    for p in range(5,22) :
        for h in range(2,12) :
            if (21 > p > 11) :
                for a in [True,False] :
                    Q["no_split"][(p,h,a)] = {m:0 for m in moves_no_split}
            else :
                Q["no_split"][(p,h,False)] = {m:0 for m in moves_no_split}
    
    for c in card_map.values() :
        if c in ["J","Q","K"] :
            continue
        for h in range(2,12) :
            a = False if c!="A" else True
            Q["can_split"][(c,h,a)] = {m:0 for m in moves}

    if mode == "accepted":
        return init_accepted_q(Q)
    
    if mode == "house":
        return init_house_q(Q)

    return Q


def init_accepted_q(q):
    for split,s_pairs in q.items() :
        for s,vals in s_pairs.items() :
            p,h,a = s
            if split=="can_split" :
                if p == "A" :
                    p = 11
                if p <= 3 :
                    vals["hit"] = 0.5
                    if h <= 7 :
                        vals["split"] = 1
                if p == 4 :
                    vals["hit"] = 0.5
                    if h in [5,6] :
                        vals["split"] = 1
                if p == 5 :
                    vals["hit"] = 0.5
                    if h <= 9 :
                        vals["double"] = 1
                if p == 6 :
                    if h in [4,5,6] :
                        vals["stay"] = 0.5
                    else :
                        vals["hit"] = 0.5
                    if h <= 6 :
                        vals["split"] = 1
                if p == 7 :
                    if h <= 6 :
                        vals["stay"] = 0.5
                    else :
                        vals["hit"] = 0.5
                    if h <= 7 :
                        vals["split"] = 1
                if p == 8 :
                    if h <= 6 :
                        vals["stay"] = 0.5
                    else :
                        vals["hit"] = 0.5
                    vals["split"] = 1
                if p == 9 :
                    vals["stay"] = 0.5
                    if h in [2,3,4,5,6,8,9] :
                        vals["split"] = 1
                if p == 10 :
                    vals["stay"] = 0.5
                if p == 11 :
                    vals["split"] = 1
            else :
                if not a :
                    if p <= 8 :
                        vals["hit"] = 0.5
                    if p == 9 :
                        vals["hit"] = 0.5
                        if h in [3,4,5,6] :
                            vals["double"] = 1
                    if p == 10 :
                        vals["hit"] = 0.5
                        if h <= 9 :
                            vals["double"] = 1
                    if p == 11 :
                        vals["hit"] = 0.5
                        vals["double"] = 1
                    if p == 12 :
                        if 4 <= h <= 6 :
                            vals["stay"] = 0.5
                        else :
                            vals["hit"] = 0.5
                    if p in [13,14] :
                        if h <= 6 :
                            vals["stay"] = 0.5
                        else :
                            vals["hit"] = 0.5
                    if p == 15 :
                        if h <= 6 :
                            vals["stay"] = 0.5
                        else :
                            vals["hit"] = 0.5
                        if h == 10 :
                            vals["surrender"] = 1
                    if p == 16 :
                        if h <= 6 :
                            vals["stay"] = 0.5
                        else :
                            vals["hit"] = 0.5
                        if h >= 9 :
                            vals["surrender"] = 1
                    if p >= 17 :
                        vals["stay"] = 0.5
                else :
                    if p in [13,14] :
                        vals["hit"] = 0.5
                        if h in [5,6] :
                            vals["double"] = 1
                    if p in [15,16] :
                        vals["hit"] = 0.5
                        if h in [4,5,6] :
                            vals["double"] = 1
                    if p == 17 :
                        vals["hit"] = 0.5
                        if h in [3,4,5,6] :
                            vals["double"] = 1
                    if p == 18 :
                        if h <= 8 :
                            vals["stay"] = 0.5
                        else :
                            vals["hit"] = 0.5
                        if h <= 6 :
                            vals["double"] = 1
                    if p == 19 :
                        vals["stay"] = 0.5
                        if h == 6 :
                            vals["double"] = 1
                    if p == 20 :
                        vals["stay"] = 0.5 
    return q


def init_house_q(q):
    for split,s_pairs in q.items() :
        for s,vals in s_pairs.items() :
            p,h,a = s
            if split=="can_split" :
                if p == "A" :
                    p = 12
                else :
                    p = p*2
            if p >= 17 :
                vals["stay"] = 1
            else :
                vals["hit"] = 1
    return q