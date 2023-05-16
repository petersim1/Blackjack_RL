from typing import List, Dict
from pydantic import BaseModel

class StateActionPair(BaseModel):
    player_show: int
    house_show: int
    useable_ace: bool
    move: str

class RulesI(BaseModel):
    dealer_hit_soft17=True
    push_dealer22=False
    double_after_split=True
    hit_after_split_aces=False
    reduced_blackjack_payout=False
    allow_surrender=True

QMovesI = Dict[str, float]

ConditionalActionSpace = List[List[List[str]]]