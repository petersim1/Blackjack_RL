from dataclasses import dataclass, field
from typing import List, Union, Tuple

from src.modules.cards import Cards, Card
from src.modules.player import Player
from src.pydantic_types import RulesI


"""
This module controls the blackjack gameplay.
It wraps the Player module, which represents each player and the house,
and uses Cards for the shoe

Input :
    - player_module : class , uninitialized Player module from Player.py
    - shrink_deck : boolean , whether or not to remove selected cards from deck. If False, each card is drawn iid.
    - n_decks : number of decks to play with (default is 6, which is typical)
    - ratio_penetrate : ratio of cards that are playable (default is 2/3 of 6 decks). Only applicable when shrinkDeck == True.
    
MUST call init_round() to start the round

For the house, the first card is shown, the second card is hidden.
"""

@dataclass
class Game :
    rules: RulesI = field(default_factory=lambda : {})
    shrink_deck: bool = True # if False, will randomly select cards uniformly, and deck won't run out.
    n_decks: int = 6
    ratio_penetrate: float = 4 / 6
    n_rounds_played: int = field(init=False, default=0)
    reset_deck_after_round: bool = field(init=False, default=False)
    shoe: Cards = field(init=False)
    players: List[Player] = field(init=False)
    house: Player = field(init=False)
    count: int = field(init=False, default=0)
    true_count: float = field(init=False, default=0)


    def __post_init__(self):
        if not isinstance(self.rules, RulesI):
            self.rules = RulesI(**self.rules)
        self._init_deck()
                
    def _init_deck(self) -> None:
        self.shoe = Cards.init_from_deck(self.n_decks)
        self.count = 0
        self.true_count = 0   

    def _init_players(self) -> None :
        self.players = [Player(wager=wager, rules=self.rules) for wager in self.wagers]
        self.house = Player(wager=0)
        
            
    def _select_card(self) -> Card:
        card = self.shoe.select_card(deplete=self.shrink_deck)
        

        stop_card_met = len(self.shoe.cards) <= (int(self.n_decks * 52 * self.ratio_penetrate))

        if stop_card_met and self.shrink_deck:
            self.reset_deck_after_round = True
        
        return card
    
    def _decorator(f):
        def inner(self, *args, **kwargs):
            res = f(self, *args, **kwargs)
            self._update_count()
            return res
        return inner
    
    def _update_count(self) -> None:
        def count_impact(card: Card):
            if 2 <= card.value <= 6:
                return -1
            if (card.value == 1) or (card.value == 10):
                return 1
            return 0
        # count based off cards still in the shoe.
        count = sum(map(count_impact, self.shoe.cards))
        if not self.house_played:
            # need to fix for the hidden house card, which we don't know about until it's flipped.
            hidden_card = self.house.cards[0].cards[1]
            count += count_impact(hidden_card)

        self.true_count = count * 52 / len(self.shoe.cards)
        self.count = count

    def init_round(self, wagers: List[float]) -> None :
        """Initializes the round. Resets the player, and will reshuffle as needed."""
        self.wagers = wagers
        
        self._init_players()
        
        self.n_rounds_played += 1
        self.round_init = True
        self.house_blackjack = False
        self.house_played = False
        
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
        
    @_decorator
    def deal_init(self) -> None:
        
        assert self.round_init , 'Must initialize round before dealing'

        for _ in range(2) :
            for player in self.players :
                card = self._select_card()
                player._deal_card(card)
            card = self._select_card()
            self.house._deal_card(card)
        
        if self.house.cards[0].total == 21 :
            self.house_blackjack = True # If house has blackjack, don't accept moves (except insurance + surrender)
        
    def get_house_show(self) -> Card :
        
        assert len(self.house.cards[0].cards) , "House has not been dealt yet"

        return self.house.cards[0].cards[0]            


    @_decorator
    def step_house(self) -> None :

        house = self.house.cards[0].total
        useable_ace = self.house.cards[0].useable_ace
        
        while (house < 17) or ((house == 17) and useable_ace and self.rules.dealer_hit_soft17) :
            card = self._select_card()
            self.house._deal_card(card)
            house = self.house.cards[0].total
            useable_ace = self.house.cards[0].useable_ace
        self.house_played = True
            

    @_decorator
    def step_player(self, ind: int, move: str) -> None :
        n = Player.get_num_cards_draw(move)
        cards = [self._select_card() for _ in range(n)]
        self.players[ind].step(move, cards)
        
    
    def get_results(self) -> Tuple[List[List[str]], List[float]]:
        players = []
        winnings = []
        for player in self.players :
            text, win = player.get_result(self.house.cards[0])
            players.append(text)
            winnings.append(win)
                
        return players,winnings