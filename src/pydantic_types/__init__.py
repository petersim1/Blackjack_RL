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
    action_space: List[str]
    move: str
    reward: float
    done: int
    obs_next: Optional[tuple]
    action_space_next: Optional[List[str]]

class RulesI(BaseModel):
    dealer_hit_soft17=True
    push_dealer22=False
    double_after_split=True
    hit_after_split_aces=False
    reduced_blackjack_payout=False
    allow_surrender=True
    split_any_ten=True

QMovesI = Dict[str, float]

ConditionalActionSpace = List[List[List[str]]]