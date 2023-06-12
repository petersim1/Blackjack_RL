from typing import List, Dict, Optional
from pydantic import BaseModel

class StateActionPair(BaseModel):
    player_show: int
    house_show: int
    useable_ace: bool
    can_split: bool
    move: str

class ReplayBuffer(BaseModel):
    obs: tuple
    move: str
    reward: float
    done: int
    obs_next: Optional[tuple]

class StateActionPairDeep(BaseModel):
    player_show: int
    house_show: int
    useable_ace: bool
    can_split: bool
    can_double: bool
    move: str

class StateActionPairDeepCount(BaseModel):
    player_show: int
    house_show: int
    useable_ace: bool
    can_split: bool
    can_double: bool
    count: float
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