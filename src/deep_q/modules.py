from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.deep_q.helpers import play_games


class Net(nn.Module):
    def __init__(self, input_dim, hidden_layers=[]):
        super(Net, self).__init__()

        assert len(hidden_layers), "must have at least 1 hidden layer"

        self.moves = ["stay", "hit", "double", "split"]

        self.input_dim = input_dim
        self.output_dim = len(self.moves)
        self.hidden_layers = hidden_layers
        self.fc_input = nn.Linear(self.input_dim, self.hidden_layers[0])

        self.fc_hidden = []
        for i in range(len(self.hidden_layers) - 1):
            self.fc_hidden.append(
                nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1])
            )

        self.fc_output = nn.Linear(self.hidden_layers[-1], self.output_dim)

    def mask(self, valid_moves):
        def to_mask(moves):
            return [1 if (move in moves) else torch.nan for move in self.moves]

        return torch.tensor(list(map(to_mask, valid_moves)))

    def forward(self, data):
        x_t = F.relu(self.fc_input(data))
        for layer in self.fc_hidden:
            x_t = F.relu(layer(x_t))
        return self.fc_output(x_t)

    def act(self, obs, method="argmax", avail_actions=[]):
        """
        inputs:
        - obs: (batch_size, 5)
        - avail_actions: empty or (batch_size, n_i)

        returns:
        - q_avail_t: (batch_size, 5)
        - q_selection_t: (batch_size, 1)
        - actions_t: (batch_size, 1)
        """

        assert method in ["argmax", "softmax"], "must use a valid method"

        with torch.no_grad():
            q_values_t = self.forward(obs)

            q_avail_t = q_values_t

            if avail_actions:
                mask_t = self.mask(avail_actions)
                q_avail_t = torch.nan_to_num(q_avail_t * mask_t, nan=-torch.inf)

            if method == "argmax":
                actions_t = torch.argmax(q_avail_t, dim=1, keepdim=True).detach()
            else:
                action_p = F.softmax(q_avail_t, dim=1).detach().numpy()
                actions_t = torch.tensor(
                    list(
                        map(
                            lambda x: np.random.choice(x.shape[0], p=x),
                            action_p,
                        )
                    )
                ).unsqueeze(-1)

            q_selection_t = q_avail_t.gather(1, index=actions_t)

        return q_avail_t, q_selection_t, actions_t


class EarlyStop:
    def __init__(self, leniency):
        self.leniency = leniency
        self.counter = 0
        self.min_performance = -10
        self.stop = False

    def _update(self, performance: float):
        if performance < self.min_performance:
            self.counter += 1
        else:
            self.counter = 0

        self.min_performance = max(self.min_performance, performance)

        if self.counter == self.leniency:
            self.stop = True


class Trainer(EarlyStop):
    def __init__(
        self,
        online_net: type[Net],
        target_net: type[Net],
        loss_fct,
        optimizer_fct,
        optimizer_kwargs,
        use_early_stop: bool = True,
    ):
        EarlyStop.__init__(self, leniency=10)
        self.online_net = online_net
        self.target_net = target_net
        self.copy_online_to_target()

        self.loss = loss_fct()
        self.optimizer = optimizer_fct(self.online_net.parameters(), **optimizer_kwargs)
        self.use_early_stop = use_early_stop
        if use_early_stop:
            self.best_state = deepcopy(self.online_net.state_dict())

    def train_epoch(self, replay_buffer, batch_size, gamma):
        """
        This step doesn't actually require inputs,
        it simply samples from the replay buffer.
        """

        transition_inds = np.random.choice(len(replay_buffer), batch_size, replace=True)

        obs_t = torch.tensor(
            [replay_buffer[i][0] for i in transition_inds], dtype=torch.float32
        )
        # a_s = [replay_buffer[i][1] for i in transition_inds]
        moves_t = torch.tensor(
            [self.online_net.moves.index(replay_buffer[i][2]) for i in transition_inds],
            dtype=torch.int64,
        ).unsqueeze(-1)
        rewards_t = torch.tensor(
            [replay_buffer[i][3] for i in transition_inds], dtype=torch.float32
        ).unsqueeze(-1)
        dones_t = torch.tensor(
            [replay_buffer[i][4] for i in transition_inds], dtype=torch.float32
        ).unsqueeze(-1)
        obs_next_t = torch.tensor(
            [replay_buffer[i][5] or (0, 0, 0, 0, 0) for i in transition_inds],
            dtype=torch.float32,
        )
        a_s_next = [replay_buffer[i][6] or ["stay"] for i in transition_inds]

        self.online_net.train()

        # target_q_argmax is determined for all next states, even if it's a terminal state. # noqa: E501
        # For terminal states, results don't make sense, but it doesn't matter since dones_t will force us to ignore it anyways # noqa: E501

        with torch.no_grad():
            _, target_q_argmax, _ = self.target_net.act(
                obs_next_t, method="argmax", avail_actions=a_s_next
            )
            # _, target_q_argmax, _ = self.target_net.act(obs_next_t, method="argmax")
            targets = rewards_t + torch.nan_to_num(
                gamma * (1 - dones_t) * target_q_argmax, nan=0
            )

        q_values = self.online_net.forward(obs_t)
        action_q_values = q_values.gather(1, moves_t)

        loss = self.loss(action_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    async def eval(
        self,
        n_games: int,
        n_rounds: int,
        wagers: List[float],
        game_hyperparams: object,
    ):
        self.online_net.eval()

        r = await play_games(
            model=self.online_net,
            n_games=n_games,
            n_rounds=n_rounds,
            wagers=wagers,
            game_hyperparams=game_hyperparams,
        )
        mean_reward = np.mean(r[:, 0, :])

        if self.use_early_stop:
            self._update(mean_reward)
            if not self.counter:
                self.best_state = deepcopy(self.online_net.state_dict())

        return mean_reward

    def copy_online_to_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def get_best_state(self, backtrack: bool = False):
        if self.use_early_stop and backtrack:
            return self.best_state
        return self.online_net.state_dict()
