from __future__ import (
    annotations,
)  # required for preventing the cyclical import of type annotations

import asyncio
from typing import TYPE_CHECKING, List

import numpy as np

from src.modules.game import Game

from .helpers import create_state_action

if TYPE_CHECKING:
    # if type_checking, import the modules for type hinting. Otherwise we get cyclical import errors. # noqa: E501
    from src.deep_learning.modules import Net
    from src.modules.player import Player


def play_round(
    blackjack: type[Game],
    model: type[Net],
    wagers: List[float],
    include_count: bool,
    method: str = "argmax",
):
    model.eval()

    blackjack.init_round(wagers)
    blackjack.deal_init()

    house_show = blackjack.get_house_show(show_value=True)

    count = blackjack.get_count()
    true_count = count * 52 / blackjack.cards.sum()

    for player in blackjack.players:
        player: type[Player]
        while not player.is_done():
            _, move, _ = create_state_action(
                player=player,
                house_show=house_show,
                include_count=include_count,
                true_count=true_count,
                model=model,
                method=method,
            )

            blackjack.step_player(player, move)

    blackjack.step_house()
    _, players_winnings = blackjack.get_results()

    return players_winnings


async def play_rounds(
    blackjack: type[Game],
    model: type[Net],
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
    model: type[Net],
    n_games: int,
    n_rounds: int,
    wagers: List[float],
    include_count: bool,
    game_hyperparams: object,
):
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
