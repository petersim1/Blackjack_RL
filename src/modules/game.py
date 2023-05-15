import numpy as np
from typing import List, Union, Tuple
from src.constants import card_values, card_map
from src.modules.player import Player
from src.pydantic_types import RulesI


class Game :

    """
    This module controls the blackjack gameplay.
    It wraps the Player.py module, which represents each player and the house 

    Input :
        - player_module : class , uninitialized Player module from Player.py
        - shrink_deck : boolean , whether or not to remove selected cards from deck. If False, each card is drawn iid.
        - n_decks : number of decks to play with (default is 6, which is typical)
        - ratio_penetrate : ratio of cards that are playable (default is 2/3 of 6 decks). Only applicable when shrinkDeck == True.
        
        MUST call init_round() to start the round
    """
    
    def __init__(
            self,
            shrink_deck: bool=True,
            n_decks: int=6,
            ratio_penetrate: float=4/6,
            rules: object={}
        ) -> None:
        
        self.shrink_deck = shrink_deck # if False, will randomly select cards uniformly, and deck won't run out.
        self.n_decks = n_decks
        self.ratio_penetrate = ratio_penetrate
        self.n_rounds_played = 0
        self.reset_deck_after_round = False
        self.round_init = False
        self.rules = RulesI(**rules)
        
        self._init_deck()

            
    def _init_deck(self) -> None:
        self.cards = np.array([self.n_decks*4]*len(card_map))
        self.n_cards_played = 0 # In THIS deck.
        self.count = 0
        self.stop_card = int(self.n_decks*52*self.ratio_penetrate)
        

    def _init_players(self) -> None :
        self.players: List[type[Player]] = [Player(wager=wager, rules=self.rules.dict()) for wager in self.wagers]
        self.house: type[Player] = Player(0)
        

    def _update_count(self, card: Union[int, str]) -> None :        
        if 2 <= card_values[card] <=6 :
            self.count += 1
        if (card_values[card] >= 10) or (card_values[card] == 1) :
            self.count -= 1

            
    def _select_card(self, update_count: bool=True) -> Union[int, str]:
        ind = np.random.choice(list(card_map.keys()),p=self.cards/self.cards.sum())
        card = card_map[ind]
        if self.shrink_deck :
            self.cards[ind] -= 1
            self.n_cards_played += 1
        
        if (self.n_cards_played == self.stop_card) & self.shrink_deck :
            self.reset_deck_after_round = True
        
        if update_count :
            self._update_count(card)
        
        return card
        

    def init_round(self, wagers: List[float]) -> None :
        """Initializes the round. Resets the player, and will reshuffle as needed."""
        self.wagers = wagers
        
        self._init_players()
        
        self.n_rounds_played += 1
        self.round_init = True
        self.house_blackjack = False
        
        if self.reset_deck_after_round : 
            self._init_deck()
            self.reset_deck_after_round = False
        

    def reset_game(self) -> None:
        """Resets the entire game. ie game is over, reset module."""
        
        self._init_deck()
        self.players = []
        self.house = None
        self.n_rounds_played = 0
        self.round_init = False
        

    def deal_init(self) -> None:
        
        assert self.round_init , 'Must initialize round before dealing'

        for i in range(2) :
            for player in self.players :
                card = self._select_card()
                player._deal_card(card)
            card = self._select_card(update_count=(1-i)) # first card is shown, 2nd is hidden
            self.house._deal_card(card)
        
        house,_ = self.house._get_value_cards(self.house.cards[0])
        if house == 21 :
            self.house_blackjack = True # If house has blackjack, don't accept moves (except insurance + surrender)
        

    def get_house_show(self, show_value: bool=False) -> Union[int, str] :
        
        assert len(self.house.get_cards()[0]) , 'House has not been dealt yet'
        
        card = self.house.get_cards()[0][0]
        if show_value :
            return card_values[card] if card != 'A' else 11
        return card
             

    def step_house(self) -> None :
        
        house, ace = self.house._get_value_cards(self.house.cards[0])
        self._update_count(self.house.cards[0][-1]) # 2nd card is now displayed, so adjust count.
        
        while (house < 17) or ((house == 17) and ace and self.rules.dealer_hit_soft17) :
            card = self._select_card()
            self.house._deal_card(card)
            house, ace = self.house._get_value_cards(self.house.cards[0])
            

    def step_player(self, player: type[Player], move: str) -> None :
        n = player.get_num_cards_draw(move)
        cards = [self._select_card() for _ in range(n)]
        player.step(move, cards)
        
    
    def get_results(self) -> Tuple[List[List[str]], List[float]]:
        players = []
        winnings = []
        for player in self.players :
            text, win = player.get_result(self.house.cards[0])
            players.append(text)
            winnings.append(win)
                
        return players,winnings