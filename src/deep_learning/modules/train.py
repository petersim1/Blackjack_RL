from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from src.deep_learning.modules.replay import ReplayBuffer
from src.deep_learning.utils.replay_buffer import (gather_buffer_obs,
                                                   gather_target_obs,
                                                   update_replay_buffer)
from src.deep_learning.utils.runner import play_games

if TYPE_CHECKING:
    # if type_checking, import the modules for type hinting. Otherwise we get cyclical import errors. # noqa: E501
    from src.deep_learning.modules import Net
    from src.modules.game import Game


class Trainer:
    def __init__(
            self, online_net, target_net, replay_size, include_count,
            implicit_masking=True
    ):
        self.online_net: Net = online_net
        self.target_net: Net = target_net

        self.implicit_masking = implicit_masking

        self.include_count = include_count

        self.replay_size = replay_size
        self.replay_buffer = ReplayBuffer(capacity=replay_size)

    def copy_online_to_target(self):
        self.target_net.load_state_dict(deepcopy(self.online_net.state_dict()))

    def update_buffer(self, blackjack: Game, method: str = "random"):
        with torch.no_grad():
            update_replay_buffer(
                blackjack=blackjack,
                buffer=self.replay_buffer,
                model=self.online_net,
                include_count=self.include_count,
                include_continuous_count=False,
                method=method,
                implicit_masking=self.implicit_masking,
            )

    def train_epoch(
            self, batch_size: int, gamma: float, loss_fct: nn.modules.loss._Loss,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.ExponentialLR):
        (
            obs_t,
            action_space,
            moves_t,
            rewards_t,
            dones_t,
            obs_next_t,
            action_space_next,
        ) = gather_buffer_obs(
            replay_buffer=self.replay_buffer,
            batch_size=batch_size,
            moves=self.online_net.moves,
        )

        targets_t = gather_target_obs(
            target_net=self.target_net,
            obs_next_t=obs_next_t,
            action_space_next=action_space_next,
            rewards_t=rewards_t,
            dones_t=dones_t,
            gamma=gamma,
        )

        self.online_net.train()

        q_values: torch.Tensor = self.online_net.forward(obs_t)
        action_q_values = q_values.gather(1, moves_t)
        # _, action_q_values, _ = self.online_net.act(
        #     obs=obs_t, avail_actions=action_space
        # )

        optimizer.zero_grad()
        loss: torch.Tensor = loss_fct(action_q_values, targets_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        return loss.item()

    async def eval(self, n_games: int, n_rounds: int, wagers, game_hyperparams):
        self.online_net.eval()

        r = await play_games(
            model=self.online_net,
            n_games=n_games,
            n_rounds=n_rounds,
            wagers=wagers,
            include_count=self.include_count,
            game_hyperparams=game_hyperparams,
            implicit_masking=self.implicit_masking
        )
        mean_reward = np.mean(r[:, 0, :])

        return mean_reward
