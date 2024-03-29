from __future__ import \
    annotations  # required for preventing the cyclical import of type annotations

import asyncio
from typing import TYPE_CHECKING, List

import numpy as np

from src.modules.game import Game

from .action import select_action

if TYPE_CHECKING:
    # if type_checking, import the modules for type hinting. Otherwise we get cyclical import errors. # noqa: E501
    from src.deep_learning.modules import Net


def play_round(
    blackjack: Game,
    model: Net,
    wagers: List[float],
    include_count: bool,
):
    blackjack.init_round(wagers)
    blackjack.deal_init()

    house_card_show = blackjack.get_house_show()
    house_value = house_card_show.value if house_card_show.value > 1 else 11

    for i, player in enumerate(blackjack.players):
        while not player.is_done():
            player_total, useable_ace = player.get_value()
            policy = player.get_valid_moves()

            if include_count:
                observation = (
                    player_total,
                    house_value,
                    2 * int(useable_ace) - 1,
                    # 2 * int("split" in policy) - 1,
                    # 2 * int("double" in policy) - 1,
                    blackjack.true_count
                )
            else:
                observation = (
                    player_total,
                    house_value,
                    2 * int(useable_ace) - 1,
                    # 2 * int("split" in policy) - 1,
                    # 2 * int("double" in policy) - 1
                )
            move = select_action(
                model=model,
                method="argmax",
                policy=policy,
                observation=observation,
            )

            blackjack.step_player(i, move)

    blackjack.step_house(only_reveal_card=True)
    while not blackjack.house_done():
        blackjack.step_house()

    _, players_winnings = blackjack.get_results()

    return players_winnings


async def play_rounds(
    blackjack: Game,
    model: Net,
    n_rounds: int,
    wagers: List[float],
    include_count: bool,
):
    rewards = [[] for _ in wagers]

    for i in range(n_rounds):
        players_rewards = play_round(
            blackjack=blackjack,
            model=model,
            wagers=wagers,
            include_count=include_count,
        )

        for i, reward in enumerate(players_rewards):
            # reward is a list which represents the reward for each hand of a single player due to splitting. # noqa: E501
            rewards[i].append(sum(reward))

    return rewards


async def play_games(
    model: Net,
    n_games: int,
    n_rounds: int,
    wagers: List[float],
    include_count: bool,
    game_hyperparams: object,
):
    # Will use the optimal move to carry out gameplay
    model.eval()
    tasks = []
    for _ in range(n_games):
        blackjack = Game(**game_hyperparams)
        tasks.append(
            asyncio.create_task(
                play_rounds(
                    blackjack=blackjack,
                    model=model,
                    n_rounds=n_rounds,
                    wagers=wagers,
                    include_count=include_count,
                )
            )
        )

    rewards = await asyncio.gather(*tasks)

    return np.array(rewards)
