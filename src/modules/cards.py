from dataclasses import dataclass, field
from typing import Union, List
from enum import Enum
from numpy.random import choice

class SuitEnum(Enum):
    hearts = "h"
    clubs = "c"
    diamonds = "d"
    spades = "s"

@dataclass
class Card:
    suit: SuitEnum
    card: Union[str, int]

    def __post_init__(self):
        self.suit = SuitEnum(self.suit)
        if isinstance(self.card, int):
            self.value = self.card
        else:
            self.value = 1 if self.card == "A" else 10

@dataclass
class Cards:
    cards: List[Card] = field(default_factory=list)
    requires_total: bool = True
    total: int = field(init=False)
    useable_ace: bool = field(init=False)

    @classmethod
    def init_from_deck(cls, n):
        # When initialized from deck, it assumes we don't
        # care about the total or useable_ace.
        cards = cls.generate_deck(n)
        return cls(cards, requires_total=False)

    @staticmethod
    def generate_deck(n) -> List[Card]:
        cards = []
        for suit in SuitEnum:
            for c in [2,3,4,5,6,7,8,9,10,"J","Q","K","A"]:
                cards.extend([Card(suit, c)] * n)
        return cards
    
    def _update_value(self) -> None:
        summed = 0
        aces = 0
        self.useable_ace = False
        for card in self.cards:
            summed += card.value
            aces += int(card.card == "A")
        if aces:
            if summed <= 11:
                self.useable_ace = True
                summed += 10
        self.total = summed
    
    def _decorator(f):
        def inner(self, *args, **kwargs):
            res = f(self, *args, **kwargs)
            if self.requires_total:
                self._update_value()
            return res
        return inner
    
    @_decorator
    def add_cards(self, cards: Union[Card, List[Card]]) -> None:
        if isinstance(cards, Card):
            self.cards.append(cards)
        else:
            self.cards.extend(cards)

    @_decorator
    def remove_card(self, ind: int) -> None:
        if ind >= len(self.cards):
            raise Exception("invalid index used")
        return self.cards.pop(ind)

    @_decorator
    def clear_cards(self):
        self.cards = []

    @_decorator
    def select_card(self, deplete: bool=True) -> Card:
        if not len(self.cards):
            raise Exception("no cards in the deck")
        ind = choice(len(self.cards))
        if deplete:
            return self.cards.pop(ind)
        return self.cards[ind]