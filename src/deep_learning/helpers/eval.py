from __future__ import annotations # required for preventing the cyclical import of type annotations
from typing import List, TYPE_CHECKING
import numpy as np
import torch
import asyncio

from src.modules.game import Game

if TYPE_CHECKING:
    # if type_checking, import the modules for type hinting. Otherwise we get cyclical import errors.
    from src.modules.player import Player
    from src.deep_learning.modules import Net
    
def get_observation(include_count: bool, **kwargs):

    if include_count:
        return (kwargs["player_total"], kwargs["house_show"], kwargs["useable_ace"], kwargs["can_split"], kwargs["can_double"], kwargs["count"])
    
    return (kwargs["player_total"], kwargs["house_show"], kwargs["useable_ace"], kwargs["can_split"], kwargs["can_double"])


def get_action(model: type[Net], method: str, observation: tuple) -> str:

    if method == "random":
        move = np.random.choice(model.moves)
        return move
    
    obs_t = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    _, _, action_ind = model.act(obs=obs_t, method=method)
    move = model.moves[action_ind[0][0].item()]

    return move

def create_state(player: type[Player], house_show: int, include_count: bool, true_count: float):

    player_total, useable_ace = player.get_value()

    policy = player.get_valid_moves()
    policy = [p for p in policy if p != "surrender"]

    can_split = "split" in policy
    can_double = "double" in policy

    observation = get_observation(
            include_count=include_count,
            player_total=player_total,
            house_show=house_show,
            useable_ace=2 * int(useable_ace) - 1,
            can_split=2 * int(can_split) - 1,
            can_double=2 * int(can_double) - 1,
            count=true_count
        )

    return observation


def create_state_action(
        player: type[Player],
        house_show: int,
        include_count: bool,
        true_count: float,
        model: type[Net],
        method: str="argmax"
):
    policy = player.get_valid_moves()
    policy = [p for p in policy if p != "surrender"]

    observation = create_state(
        player=player,
        house_show=house_show,
        include_count=include_count,
        true_count=true_count
    )

    move = get_action(
        model=model,
        method=method,
        observation=observation
    )

    is_valid_move = move in policy

    return is_valid_move, observation, move


def play_round(
        blackjack: type[Game],
        model: type[Net],
        wagers: List[float],
        include_count: bool,
        method: str="argmax"
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

            is_valid_move, _, move = create_state_action(
                player=player,
                house_show=house_show,
                include_count=include_count,
                true_count=true_count,
                model=model,
                method=method
            )

            if not is_valid_move:
                return [[-1.5]] # I can tinker with this value.

            blackjack.step_player(player, move)

    blackjack.step_house()
    _, players_winnings = blackjack.get_results()

    return players_winnings

async def play_rounds(blackjack: type[Game], model: type[Net], n_rounds: int, wagers: List[float], include_count: bool):
    rewards = [[] for _ in wagers]

    for i in range(n_rounds):
        players_rewards = play_round(
            blackjack=blackjack,
            model=model,
            wagers=wagers,
            include_count=include_count
        )

        for i,reward in enumerate(players_rewards):
            # reward is a list which represents the reward for each hand of a single player due to splitting.
            rewards[i].append(sum(reward))

    return rewards

async def play_games(
        model: type[Net],
        n_games: int,
        n_rounds: int,
        wagers: List[float],
        include_count: bool,
        game_hyperparams: object
):

    tasks = []
    for _ in range(n_games):
        blackjack = Game(**game_hyperparams)
        tasks.append(
            asyncio.create_task(
                play_rounds(blackjack=blackjack, model=model, n_rounds=n_rounds, wagers=wagers, include_count=include_count)
            ))
        
    rewards = await asyncio.gather(*tasks)

    return np.array(rewards)

__all__ = ["get_observation", "get_action", "create_state_action","play_round", "play_rounds", "play_games"]