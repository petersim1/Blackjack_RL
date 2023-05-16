import numpy as np
from typing import List

from src.helpers.runner import select_action
from src.modules.game import Game
from src.modules.player import Player

def cummulative_rewards_per_round(rewards_games: List[List[List[float]]]) -> np.ndarray:

    n_rounds = np.array(rewards_games).shape[-1]
    return np.cumsum(rewards_games, axis=2)[:,:,-1]/n_rounds


def mean_cum_rewards(rewards_games: List[List[List[float]]]) -> np.ndarray:

    cummed: np.ndarray = cummulative_rewards_per_round(rewards_games)
    return cummed.mean(axis=0)


def median_cum_rewards(rewards_games: List[List[List[float]]]) -> np.ndarray:

    cummed: np.ndarray = cummulative_rewards_per_round(rewards_games)
    return np.median(cummed, axis=0)


def compare_to_accepted(
        q: object,
        accepted_q: object,
        game_hyperparams: object,
        n_rounds: int
    ):
    """ Simulates game play, and determines how many moves were 'optimal' according to a baseline policy """

    blackjack = Game(**game_hyperparams)

    correct_moves = []

    for _ in range(n_rounds):

        blackjack.init_round([1])
        blackjack.deal_init()
        house_show = blackjack.get_house_show(show_value=True)

        for player in blackjack.players :
            player: type[Player]
            while not player.is_done() :
                player_show, useable_ace = player.get_value()
                policy = player.get_valid_moves()

                state = q[(player_show, house_show, useable_ace)]
                accepted_state = accepted_q[(player_show, house_show, useable_ace)]

                move = select_action(
                    state=state,
                    policy=policy,
                    epsilon=-1,
                    method="epsilon"
                )
                accepted_move = select_action(
                    state=accepted_state,
                    policy=policy,
                    epsilon=-1,
                    method="epsilon"
                )
                correct_moves.append(move == accepted_move)

                blackjack.step_player(player,move)
    
    return sum(correct_moves) / len(correct_moves)