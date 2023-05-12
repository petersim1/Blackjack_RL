from typing import List, Union, Optional, Tuple

from src.constants import card_values
from src.pydantic_types import RulesI


class Player:
    """
    This module is to be used as an individual blackjack player
    It is intended to be wrapped into the Game module, which controls the blackjack game.

    In the Game module, in each round, this module is reset.
    """
    
    def __init__(self, wager: float, rules: object={}) -> None :
        
        self.cards: List[List[Union[int,str]]] = [[]]
        
        self.base_wager = wager
        self.wager = [wager]
        self.rules = RulesI(**rules)

        self.complete = [False]
        self.insured = False
        self.surrendered = False
        self.aces_split = False

    
    def _get_cur_hand(self) -> Optional[int]:
        return self.complete.index(False) if False in self.complete else None
    

    def _all_complete(self) -> bool:
        return not self.complete.count(False)
    

    def _deal_card(self, card: Union[int, str]) -> None:
        i_hand = self._get_cur_hand()
        self.cards[i_hand].append(card)
        if self._get_value_cards(self.cards[i_hand])[0] >= 21 :
            self.complete[i_hand] = True
    

    def _split(self, cards: List[Union[int, str]]) -> None:
        i_hand = self._get_cur_hand()
        card = self.cards[i_hand].pop(-1)
        self.cards.insert(i_hand+1,[card])
        self.wager.insert(i_hand+1,self.base_wager)
        self.complete.insert(i_hand+1,False)
        
        self.cards[i_hand].append(cards[0])
        self.cards[i_hand+1].append(cards[1])
    

    def _get_value_cards(self, cards: List[Union[int, str]]) -> Tuple[int, bool] :
        useable_ace = False
        total = sum([card_values[card] for card in cards])
        if cards.count('A') :
            if total <= 11 :
                total += 10
                useable_ace = total < 21
        return total, useable_ace
    

    def get_value(self) -> Tuple[int, bool, bool, Optional[Union[int, str]]]:
        i_hand = self._get_cur_hand()
        n = len(self.cards[i_hand])
        can_split = (n==2) & (self.cards[i_hand][0] == self.cards[i_hand][1])
        useable_ace = False
        c1 = None
        if can_split :
            c1 = self.cards[i_hand][0]
            if c1 in ['J','Q','K'] :
                c1 = 10

        total, useable_ace = self._get_value_cards(self.cards[i_hand])

        return total, can_split, useable_ace, c1
        
         
    def get_valid_moves(self, house_show: Union[int,str]) -> List[str] :
        possible_moves = []
        i_hand = self._get_cur_hand()
        if i_hand is None :
            return possible_moves
        
        val, _, _, _ = self.get_value()
        
        n_hands = len(self.cards)
        n = len(self.cards[i_hand])
        
        can_hit = (not self.aces_split) | self.rules.hit_after_split_aces
        can_stay = (not self.aces_split) | self.rules.hit_after_split_aces
        can_split = (n==2) & (self.cards[i_hand][0] == self.cards[i_hand][1])
        can_insure = (house_show=='A') & (n==2) & (n_hands==1) & (not self.insured)
        can_surrender = (n==2) & (n_hands==1) & (self.rules.allow_late_surrender)
        can_double = (n==2) & (((n_hands > 1) & self.rules.double_after_split) | (n_hands == 1)) & can_hit
                
        if val < 21 :
            if can_stay: possible_moves.append("stay")
            if can_hit : possible_moves.append("hit")
            if can_split : possible_moves.append('split')
            if can_insure : possible_moves.append('insurance')
            if can_surrender : possible_moves.append('surrender')
            if can_double : possible_moves.append('double')
        if val == 21 :
            possible_moves.append("stay")
        
        return possible_moves
    
    
    def get_num_cards_draw(self, move: str) -> int :
        if move in ['hit','double'] :
            return 1
        if move == 'split' :
            return 2
        
        return 0
    
    
    def step(self, move: str, cards_give: List[Union[int, str]]=[]) -> None:
        assert not self._all_complete() , 'Player cannot move anymore!'
        assert len(cards_give) == self.get_num_cards_draw(move) , 'Must provide proper # of cards!'
        
        i_hand = self._get_cur_hand()
        if move == 'hit':
            self._deal_card(cards_give[0]) 
        if move == 'stay':
            self.complete[i_hand] = True
        if move == 'double':
            self.wager[i_hand] *= 2
            self._deal_card(cards_give[0])
            self.complete[i_hand] = True
        if move == 'insurance':
            self.insured = True
        if move == 'split':
            self.aces_split = (self.cards[i_hand][0] == "A") & (self.cards[i_hand][1] == "A")
            self._split(cards_give)
            if (not self.rules.hit_after_split_aces) and self.aces_split :
                if cards_give[0] != "A":
                    self.complete[i_hand] = True
                if cards_give[1] != "A":
                    self.complete[i_hand+1] = True
        if move == 'surrender':
            self.surrendered = True
            self.complete[i_hand] = True

            
    def get_result(self, house_value: int, house_cards: List[Union[int, str]]) -> Tuple[List[str], float] :
        house_is_blackjack = (house_value==21) & (len(house_cards)==2)
        blackjack_payout = 1.5 if not self.rules.reduced_blackjack_payout else 1.2
        
        text = []
        winnings = 0
        if self.insured :
            if house_is_blackjack : # insurance pays out 2:1
                winnings += self.base_wager 
            else :
                winnings -= self.base_wager/2
        if self.surrendered :
            return [['surrender'], winnings-self.base_wager/2]
        for i,cards in enumerate(self.cards) :
            val,_ = self._get_value_cards(cards)
            # 21 after a split is not natural blackjack. It's just 21, even on first two cards.
            is_blackjack = (val==21) & (len(cards)==2) & (len(self.cards) == 1)
            
            if val > 21 :
                text.append('bust')
                winnings -= self.wager[i]
            if val == 21 :
                if is_blackjack :
                    if house_is_blackjack :
                        text.append('push')
                    else :
                        text.append('blackjack')
                        winnings += self.wager[i]*blackjack_payout
                else :
                    if house_is_blackjack :
                        text.append('loss')
                        winnings -= self.wager[i]
                    else :
                        if house_value == 21 :
                            text.append('push')
                        else :
                            if (self.rules.push_dealer22 and (house_value == 22)) : 
                                text.append("push")
                            else :
                                text.append('win')
                                winnings += self.wager[i]

            if val < 21 :
                if house_value > 21 :
                    if (self.rules.push_dealer22 and (house_value == 22)) : 
                        text.append("push")
                    else :
                        text.append('win')
                        winnings += self.wager[i]
                else :
                    if val > house_value :
                        text.append('win')
                        winnings += self.wager[i]
                    elif val < house_value :
                        text.append('loss')
                        winnings -= self.wager[i]
                    else :
                        text.append('push')
        
        return text, winnings
        
            
    def is_move_valid(self, move: str, house_show: int) -> bool :
        return move in self.get_valid_moves(house_show)
        

    def get_cards(self) -> List[List[Union[int, str]]] :
        return self.cards
        

    def is_done(self) -> bool:
        return self._all_complete()