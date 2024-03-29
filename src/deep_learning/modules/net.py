from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim, hidden_layers=[]):
        super().__init__()

        assert len(hidden_layers), "must have at least 1 hidden layer"

        self.moves = ["stay", "hit", "double", "split", "surrender"]

        self.input_dim = input_dim
        self.output_dim = len(self.moves)
        self.hidden_layers = hidden_layers
        self.fc_input = nn.Linear(self.input_dim, self.hidden_layers[0])

        self.fc_hidden = nn.ModuleList()
        for i in range(len(self.hidden_layers) - 1):
            self.fc_hidden.append(
                nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1])
            )

        self.fc_output = nn.Linear(self.hidden_layers[-1], self.output_dim)

    def mask(self, valid_moves):
        def to_mask(moves):
            return [move not in moves for move in self.moves]

        return torch.tensor(list(map(to_mask, valid_moves)))

    def forward(self, data):
        x_t = F.relu(self.fc_input(data))
        for layer in self.fc_hidden:
            x_t = F.relu(layer(x_t))
        return self.fc_output(x_t)

    def act(
            self, obs, method="argmax", avail_actions=[]
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used for explicit masking

        inputs:
        - obs: (batch_size, input_dim)
        - avail_actions: empty or (batch_size, n_i)

        returns:
        - q_avail_t: (batch_size, len(self.moves))
        - q_selection_t: (batch_size, 1)
        - actions_t: (batch_size, 1)
        """

        assert method in ["argmax", "softmax"], "must use a valid method"
        with torch.no_grad():
            q_values_t = self.forward(obs)

            q_avail_t: torch.Tensor = q_values_t

            if avail_actions:
                mask_t = self.mask(avail_actions)
                q_avail_t = q_avail_t.masked_fill(mask_t, -torch.inf)

            if method == "argmax":
                actions_t = torch.argmax(q_avail_t, dim=1, keepdim=True)
            else:
                action_p = F.softmax(q_avail_t, dim=1)
                actions_t = torch.multinomial(action_p, num_samples=1)

            q_selection_t = q_avail_t.gather(1, index=actions_t)

        return q_avail_t, q_selection_t, actions_t
