# Blackjack_RL

This repository presents Reinforcement Learning strategies for blackjack.

It is ulimately split into 3 parts:
1. Q Learning
2. Deep Q Learning
3. Deep Q Learning with Card Count

In addition, there are modules to help simulate gameplay
- Game
- Player
- Cards
- Card

`Game` wraps the `Player` and `Cards` modules, and controls overall gameplay such as house moves, card deals, etc...

### Q Learning
Uses the SARSA algorithm to learn an optimal policy. It stores each state-action pair in memory. As this is a tractible solution without card count, it behaves quite well.

### Deep Q Learning
An adaption of Q Learning built within the Deep Learning framework (using pytorch). This still doesn't use card count. While it is a tractible solution without a neural network approximator, I include it to show the framework behind using this.

### Deep Q Learning with Card Count
Take the Deep Learning framework a bit further by incorporating card count. There are 2 main elements to card counting that I experiment with: running count, and true count. True count simply takes the running count and divides it by the number of decks remaining in the deck. This is likely a better metric, although more difficult to determine in practice, for learning the Q Network with count accounted for. Also, it'll help constrain the boundaries of possible values, by using true count. It's generally accepted that a higher true count is more favorable for a player.

To come....
Can we incorporate the Deep Q Learning with Card Count and a bankroll/betting strategy to optimize our rewards?

## Setup

`poetry install`, which will pull from the `poetry.lock` and `pyproject.toml` files to create a local env.

In your local Code Editor, specify the python interpreter path of your local poetry env.

## Module Overview

**Card:**
A simple class that represents suit (Enum), card, and a value (forcing Ace -> 1 for ease of use)

**Cards:**
A module that represents a list of *Card*, which is used for automatically calculating total card values (important for player + house, but not the shoe).
Can manipulate these lists with `add_card()`, `remove_card()`, `clear_cards()`, which are important for player + house 
Can also manipulate these lists with `select_card()`, which is importnt for the shoe.

**Player:**
Module used for players + the house. Used to dictate actions, get results, and to manage each hand for a given player (resulting from playing multiple hands at once, or from splits).
Heavily reliant on the *Cards* class to handle the manipulation of card lists and get current hand values.

**Game:**
Module for dictating overall gameplay. It wraps in *Player* classes for both players and the house, and it wraps in *Cards* class to manage the shoe.
This module manages the state of gameplay, handles initialization of rounds, hands, has memory for card count, and is aware of when to replenish the shoe.
