from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.modules.cards import Cards, Card
from src.pydantic_types import RulesI

@dataclass
class Player:
    """
    This module is to be used as an individual blackjack player.
    It is intended to be wrapped into the Game module, which controls the blackjack game entirely.

    In the Game module, in each round, this module is reset.

    We don't care for insurance bets, as they are side bets essentially, so we'll ignore completely.
    """
    base_wager: float
    wager: List[float]
    rules: RulesI
    i_hand: Optional[int] = 0
    cards: List[Cards] = field(default_factory=lambda : [[]])
    complete: List[bool] = field(default_factory=lambda : [False])
    surrendered: bool = False
    aces_split: bool = False

    def __post_init__(self):
        self.rules = RulesI(**self.rules)
        self.wager = [self.wager]

    @classmethod
    def from_state(cls, state):
        pass

    def _decorator(f):
        def inner(self, *args, **kwargs):
            self.i_hand = self.complete.index(False) if False in self.complete else -1
            return f(*args, **kwargs)
        return inner

    def _all_complete(self) -> bool:
        return all(self.complete)
    
    @_decorator
    def _deal_card(self, card: Card) -> None:
        """
        Updated slightly. Mark as complete if:
        - natural blackjack
        - bust
        - exactly 21 (including non-natural blackjack)
        This simplifies to marking as done if total >= 21
        """

        self.cards[self.i_hand].add_cards(card)

        self.complete[self.i_hand] = self.cards[self.i_hand].total >= 21

    @_decorator
    def _split(self, cards: List[Card]) -> None:
        """handles splitting of cards"""
        i_hand = self.i_hand
        self.aces_split = (self.cards[i_hand][0].card == "A") & (self.cards[i_hand][1].card == "A")

        card = self.cards[i_hand].remove_card(-1)
        self.cards.insert(i_hand + 1, Cards(card))
        self.wager.insert(i_hand + 1, self.base_wager)
        self.complete.insert(i_hand + 1, False)
        
        # Calling this fct will inherently create an updated "total" variable.
        self.cards[i_hand].add_cards(cards[0])
        self.cards[i_hand+1].add_cards(cards[1])

        if self.cards[i_hand].total == 21:
            self.complete[i_hand] = True
        if self.cards[i_hand + 1].total == 21:
            self.complete[i_hand + 1] = True

        if self.aces_split and (not self.rules.hit_after_split_aces) :
            if cards[0].card != "A":
                self.complete[i_hand] = True
            if cards[1].card != "A":
                self.complete[i_hand+1] = True

    @_decorator
    def get_value(self) -> Tuple[int, bool]:
        """ gets the card total and whether there's a useable ace for the current hand of cards for a player """
        return self.cards[self.i_hand].total, self.cards[self.i_hand].useable_ace
    
    @_decorator
    def get_valid_moves(self) -> List[str] :
        possible_moves = []
        i_hand = self.i_hand
        if i_hand < 0 :
            return possible_moves
        
        total, _ = self.get_value()
        
        n_hands = len(self.cards)
        n = len(self.cards[i_hand])
        
        can_hit = (not self.aces_split) | self.rules.hit_after_split_aces
        can_stay = (not self.aces_split) | self.rules.hit_after_split_aces
        can_surrender = (n==2) & (n_hands==1) & (self.rules.allow_surrender)
        can_split = (n==2) & (self.cards[i_hand][0] == self.cards[i_hand][1])
        can_double = (n==2) & (((n_hands > 1) & self.rules.double_after_split) | (n_hands == 1)) & can_hit
                
        if total < 21 :
            if can_stay: possible_moves.append("stay")
            if can_hit : possible_moves.append("hit")
            if can_split : possible_moves.append("split")
            if can_surrender : possible_moves.append("surrender")
            if can_double : possible_moves.append("double")
        if total == 21 :
            possible_moves.append("stay")
        
        return possible_moves
    
    
    def get_num_cards_draw(self, move: str) -> int :
        if move in ["hit","double"] : return 1
        if move == "split" : return 2
        return 0
    
    @_decorator
    def step(self, move: str, cards_give: List[Card]=[]) -> None:
        assert not self._all_complete() , "Player cannot move anymore!"
        assert len(cards_give) == self.get_num_cards_draw(move) , "Must provide proper # of cards!"
        
        if move == "hit":
            self._deal_card(cards_give) 
        if move == "stay":
            self.complete[self.i_hand] = True
        if move == "double":
            self.wager[self.i_hand] *= 2
            self._deal_card(cards_give)
            self.complete[self.i_hand] = True
        if move == "split":
            self._split(cards_give)
        if move == "surrender":
            self.surrendered = True
            self.complete[self.i_hand] = True

    def get_result(self, house_cards: Cards) -> Tuple[List[str], float] :

        blackjack_payout = 1.2 if self.rules.reduced_blackjack_payout else 1.5

        house_value = house_cards.total
        house_is_blackjack = (house_value == 21) & (len(house_cards.cards) == 2)
        
        text = []
        winnings = []
        if self.surrendered :
            return [["surrender"], [-self.base_wager / 2]]
        for i,cards in enumerate(self.cards) :
            val = cards.total
            # 21 after a split is not natural blackjack. It"s just 21, even on first two cards.
            is_blackjack = (val == 21) & (len(cards) == 2) & (len(self.cards) == 1)
            
            if val > 21 :
                text.append("bust")
                winnings.append(-self.wager[i])
            if val == 21 :
                if is_blackjack :
                    if house_is_blackjack :
                        text.append("push")
                        winnings.append(0)
                    else :
                        text.append("blackjack")
                        winnings.append(self.wager[i] * blackjack_payout)
                else :
                    if house_is_blackjack :
                        text.append("loss")
                        winnings.append(-self.wager[i])
                    else :
                        if house_value == 21 :
                            text.append("push")
                            winnings.append(0)
                        else :
                            if (self.rules.push_dealer22 and (house_value == 22)) : 
                                text.append("push")
                                winnings.append(0)
                            else :
                                text.append("win")
                                winnings.append(self.wager[i])
            if val < 21 :
                if house_value > 21 :
                    if (self.rules.push_dealer22 and (house_value == 22)) : 
                        text.append("push")
                        winnings.append(0)
                    else :
                        text.append("win")
                        winnings.append(self.wager[i])
                else :
                    if val > house_value :
                        text.append("win")
                        winnings.append(self.wager[i])
                    elif val < house_value :
                        text.append("loss")
                        winnings.append(-self.wager[i])
                    else :
                        text.append("push")
                        winnings.append(0)
    
        return text, winnings
    
    def is_move_valid(self, move: str) -> bool :
        return move in self.get_valid_moves()
        

    def get_cards(self) -> List[Cards] :
        return self.cards
        

    def is_done(self) -> bool:
        return self._all_complete()