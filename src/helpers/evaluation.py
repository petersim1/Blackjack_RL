import asyncio
import numpy as np
from typing import List

from src.helpers.runner import select_action, play_round
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
        q: dict,
        accepted_q: dict,
        game_hyperparams: object,
        n_rounds: int
    ):
    """ Simulates game play, and determines how many moves were 'optimal' according to a baseline policy """

    game = Game(**game_hyperparams)

    correct_moves = []

    for _ in range(n_rounds):

        game.init_round([1])
        game.deal_init()
        house_card_show = game.get_house_show()
        house_value = house_card_show.value if house_card_show.value > 1 else 11

        for i in range(len(game.players)) :
            player = game.players[i]
            while not player.is_done() :
                player_show, useable_ace = player.get_value()
                policy = player.get_valid_moves()

                state = q[(player_show, house_value, useable_ace)]
                accepted_state = accepted_q[(player_show, house_value, useable_ace)]

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

                game.step_player(i, move)
    
    return sum(correct_moves) / len(correct_moves)


async def play_until_bankroll(q: object, wager: float, bankroll: float, max_rounds: int, game_hyperparams: object):

    assert max_rounds <= 1_000, "use a valid max_rounds <= 1,000"

    game = Game(**game_hyperparams)

    n_rounds = 0
    n_rounds_profitable = 0
    bankroll_init = bankroll
    profits = 0

    while (bankroll_init > wager) and (n_rounds < max_rounds):

        _, rewards = play_round(
            game=game,
            q=q,
            wagers=[wager],
            verbose=False
        )
        bankroll_init += sum(rewards[0])
        profits += sum(rewards[0])
        n_rounds += 1
        n_rounds_profitable += int(bankroll_init > bankroll)

    return n_rounds, n_rounds_profitable, profits


async def play_games_bankroll(
        q: object,
        wager: float,
        bankroll: float,
        max_rounds: int,
        n_games: int,
        game_hyperparams: object
    ):

    tasks = []
    for _ in range(n_games):
        tasks.append(asyncio.create_task(play_until_bankroll(q, wager, bankroll, max_rounds, game_hyperparams)))

    res = await asyncio.gather(*tasks)

    rounds = [r[0] for r in res]
    profitable = [r[1] for r in res]
    profits = [r[2] for r in res]

    return rounds, profitable, profits


async def play_games_bankrolls(
        q: object,
        wager: float,
        max_rounds: int,
        n_games: int,
        bankrolls: List[float],
        game_hyperparams: object
    ):

    tasks = []
    for bankroll in bankrolls:
        tasks.append(asyncio.create_task(play_games_bankroll(q, wager, bankroll, max_rounds, n_games, game_hyperparams)))

    res = await asyncio.gather(*tasks)

    rounds = [r[0] for r in res]
    profitable = [r[1] for r in res]
    profits = [r[2] for r in res]


    return rounds, profitable, profits


def assess_static_outcomes(game: Game, q: object, n_rounds: int):

    results = {}
    busts = {}
    player_results = {"soft":{},"hard":{}}

    for _ in range(n_rounds) :

        game.init_round(wagers=[1])
        game.deal_init()
        player: Player = game.players[0] # only 1 player, so i"ll just extract that specific player module.
        house_card_show = game.get_house_show()
        house_value = house_card_show.value if house_card_show.value > 1 else 11
        
        r_p = player.get_value()
        soft = r_p[1]
        if r_p[0] not in player_results["soft"] :
            player_results["soft"][r_p[0]] = {"n": 0, "rewards": 0}
            player_results["hard"][r_p[0]] = {"n": 0, "rewards": 0}

        if house_value not in results :
            results[house_value] = {"n": 0, "bust": 0, "17": 0, "18": 0, "19": 0, "20": 0,"21": 0}
        results[house_value]["n"] += 1

        while not player.is_done() :

            player_show, useable_ace = player.get_value()

            policy = player.get_valid_moves()
            policy = [p for p in policy if p!="insurance"]

            move = select_action(q[(player_show, house_value, useable_ace)], policy, -1, "epsilon")

            game.step_player(0, move)


        game.step_house()
        
        total, _ = game.house.get_value()

        if total > 21 :
            results[house_value]["bust"] += 1
            if total not in busts :
                busts[total] = 0
            busts[total] += 1
        if total <= 21 :
            results[house_value][str(total)] += 1
            
        _, r_player = player.get_result(game.house.cards[0])
        player_results["soft" if soft else "hard"][r_p[0]]["n"] += 1
        player_results["soft" if soft else "hard"][r_p[0]]["rewards"] += sum(r_player)

    return results, busts, player_results