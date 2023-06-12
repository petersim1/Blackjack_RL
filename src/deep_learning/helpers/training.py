from __future__ import annotations
import numpy as np
from collections import deque
import torch
from typing import List, TYPE_CHECKING
from src.pydantic_types import ReplayBuffer
import asyncio

from src.modules.game import Game

if TYPE_CHECKING:
    # if type_checking, import the modules for type hinting. Otherwise we get cyclical import errors.
    from src.modules.player import Player
    from src.deep_learning.modules import Net

def gather_buffer_obs(
        replay_buffer: deque[ReplayBuffer],
        batch_size: int,
        moves: List[str]
):
    transition_inds = np.random.choice(len(replay_buffer), batch_size, replace=True)

    obs_t = torch.tensor([replay_buffer[i].obs for i in transition_inds], dtype=torch.float32)
    moves_t = torch.tensor([moves.index(replay_buffer[i].move) for i in transition_inds], dtype=torch.int64).unsqueeze(-1)
    rewards_t = torch.tensor([replay_buffer[i].reward for i in transition_inds], dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.tensor([replay_buffer[i].done for i in transition_inds], dtype=torch.float32).unsqueeze(-1)
    obs_next_t = torch.tensor([replay_buffer[i].obs_next or (0,0,0,0,0) for i in transition_inds], dtype=torch.float32)

    return obs_t, moves_t, rewards_t, dones_t, obs_next_t


def gather_target_obs(
        target_net: type[Net],
        obs_next_t: torch.Tensor,
        rewards_t: torch.Tensor,
        dones_t: torch.Tensor,
        gamma: float,
):

    with torch.no_grad():
        _, target_q_argmax, _ = target_net.act(obs_next_t, method="argmax")
        targets_t = rewards_t + torch.nan_to_num(gamma * (1 - dones_t) * target_q_argmax, nan=0)

    return targets_t


def play_round(blackjack: type[Game], model: type[Net], wagers: List[float]):

    model.eval()
    
    blackjack.init_round(wagers)
    blackjack.deal_init()

    house_show = blackjack.get_house_show(show_value=True)

    for player in blackjack.players:
        player: type[Player]
        while not player.is_done():

            player_total, useable_ace = player.get_value()

            policy = player.get_valid_moves()
            policy = [p for p in policy if p != "surrender"]

            can_split = "split" in policy
            can_double = "double" in policy

            obs = (player_total, house_show, 2*int(useable_ace)-1, 2*int(can_split)-1, 2*int(can_double)-1)

            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            # _, _, action_ind = model.act(obs=obs_t, method="argmax", avail_actions=[policy])
            _, _, action_ind = model.act(obs=obs_t, method="argmax")
            move = model.moves[action_ind[0][0].item()]
            if move not in policy:
                return [[-1]] # I can tinker with this value.

            blackjack.step_player(player, move)

    blackjack.step_house()
    _, players_winnings = blackjack.get_results()

    return players_winnings


async def play_rounds(blackjack: type[Game], model: type[Net], n_rounds: int, wagers: List[float]):
    rewards = [[] for _ in wagers]

    for i in range(n_rounds):
        players_rewards = play_round(
            blackjack=blackjack,
            model=model,
            wagers=wagers
        )

        for i,reward in enumerate(players_rewards):
            # reward is a list which represents the reward for each hand of a single player due to splitting.
            rewards[i].append(sum(reward))

    return rewards


async def play_games(model: type[Net], n_games: int, n_rounds: int, wagers: List[float], game_hyperparams: object):

    tasks = []
    for _ in range(n_games):
        blackjack = Game(**game_hyperparams)
        tasks.append(
            asyncio.create_task(
            play_rounds(blackjack=blackjack, model=model, n_rounds=n_rounds, wagers=wagers)
            ))
        
    rewards = await asyncio.gather(*tasks)

    return np.array(rewards)