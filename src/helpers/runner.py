import asyncio
import numpy as np
from typing import List, Tuple

from src.pydantic_types import QMovesI
from src.modules.game import Game
from src.modules.player import Player

def select_action(state: QMovesI, policy: List[str], epsilon: float, method: str) -> str :
    """
    Get the best action according to a state, policy, epsilon value, and method.
    - Can use epsilon = -1 to serve as greedy.
    - Can use epsilon = 1 to serve as random.
    """
    assert method in ["epsilon", "thompson"], "invalid method selected"

    # masking of invalid states
    q_dict = {k:v for k,v in state.items() if k in policy}

    # softmax
    if method == "thompson" :
        exp = np.exp(np.array(list(q_dict.values())))
        p = exp / exp.sum()
        move = np.random.choice(list(q_dict.keys()), p=p)
        return move
    
    # epsilon-greedy
    if method == "epsilon" :
        n = np.random.rand()
        if n < epsilon :
            move = np.random.choice(policy)
        else :
            # possible that there are multiple "best" moves, sample from them.
            best_move = [k for k,v in q_dict.items() if v==max(list(q_dict.values()))]
            move = np.random.choice(best_move)

    return move

def play_round(
        blackjack: type[Game],
        q: object,
        wagers: List[float],
        verbose: bool=False
    ) -> Tuple[List[List[str]], List[float]]:

    """
    wagers denotes the wager per player

    returns:
        - players_text: (n_players x 1)
        - players_winnings: (n_players x 1)
    """

    blackjack.init_round(wagers)
    blackjack.deal_init()
    house_show = blackjack.get_house_show(show_value=True)

    player: type[Player]
    for player in blackjack.players :
        if verbose:
            print("Player Cards + Moves:")
        move = ""
        while not player.is_done() :
            player_show, useable_ace, can_split = player.get_value()
            policy = player.get_valid_moves()

            state = q[(player_show, house_show, useable_ace, can_split)]

            move = select_action(
                state=state,
                policy=policy,
                epsilon=-1,
                method="epsilon"
            )
            if verbose:
                print(player.cards, move)

            blackjack.step_player(player,move)

    if (verbose) & (move not in ["surrender", "stay"]):
        print(player.cards)

    blackjack.step_house()
    if verbose:
        print("\nHouse Cards")
        print(blackjack.house.cards)
        print("\nResult:")
    players_text, players_winnings = blackjack.get_results()

    return players_text, players_winnings

    
def play_n_rounds(
        blackjack: type[Game],
        q: object,
        wagers: List[float],
        n_rounds: int
    ) -> List[List[float]]:

    """
    wagers denotes the wager per round, per player. For this fct, I'll assume the wager is identical per round.
    ie, wagers should be [1 x n_players].

    returns:
        - rewards: (n_players x n_rounds)
    """
    rewards = [[] for _ in wagers]

    for i in range(n_rounds):
        _, players_winnings = play_round(
            blackjack=blackjack,
            q=q,
            wagers=wagers
        )

        for i,reward in enumerate(players_winnings):
            # reward is a list which represents the reward for each hand of a single player due to splitting.
            rewards[i].append(sum(reward))

    return rewards


async def play_n_games(
        q: object,
        wagers: List[float],
        n_rounds: int,
        n_games: int,
        game_hyperparams: object
    ) -> List[List[List[float]]]:

    """
    Runs N games of M rounds each, async. Game module is instantiated separately for each game, as they're independent.
    I'll assume equal wager per game, per player, per round.

    returns:
        - rewards: (n_games x n_players x n_rounds)
    """
    async def task():
        blackjack = Game(**game_hyperparams)
        return play_n_rounds(
            blackjack=blackjack,
            q=q,
            wagers=wagers,
            n_rounds=n_rounds
        )
    
    tasks = []
    for _ in range(n_games):
        tasks.append(asyncio.create_task(task()))

    return await asyncio.gather(*tasks)