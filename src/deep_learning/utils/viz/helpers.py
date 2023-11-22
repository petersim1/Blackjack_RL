from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    # if type_checking, import the modules for type hinting. Otherwise we get cyclical import errors. # noqa: E501
    from src.deep_learning.modules import Net


def gen_all_states(include_count: bool = False, true_count: float = 0):
    all_states = []

    for player in range(4, 22):
        for house in range(2, 12):
            for ace in [-1, 1]:
                if (player < 12) and (ace > 0):
                    continue
                if (player == 21) and (ace > 0):
                    continue
                for split in [-1, 1]:
                    if (player == 4) and (split < 0):
                        continue
                    if (player == 12) and (split < 0) and (ace > 0):
                        continue
                    if (player > 12) and (not player % 2) and (split > 0) and (ace > 0):
                        continue
                    if (player % 2) and (split > 0):
                        continue
                    for double in [-1, 1]:
                        if (split > 0) and (double < 0):
                            continue
                        if (player == 5) and (double < 0):
                            continue
                        if include_count:
                            all_states.append(
                                (player, house, ace, split, double, true_count)
                            )
                        else:
                            all_states.append((player, house, ace, split, double))

    return all_states


def gen_value_tensor(model: type[Net], data: List[Tuple], avail_actions: List[str]):
    value_det = np.zeros((21 + 1, 11 + 1))

    data_t = torch.tensor(data, dtype=torch.float32)
    actions = [avail_actions for _ in data]

    q_values_max: torch.Tensor
    with torch.no_grad():
        _, q_values_max, _ = model.act(data_t, method="argmax", avail_actions=actions)

    for i, q in enumerate(q_values_max):
        r = data[i][0]
        c = data[i][1]
        if data[i][3] > 0:
            r = int(r / 2)
            if r == 6:
                if data[i][2] > 0:
                    r = 11
        value_det[r, c] = q

    return value_det, q_values_max.min().item(), q_values_max.max().item()


def gen_action_tensor(model: type[Net], data: List[Tuple], avail_actions: List[str]):
    value_action = np.empty((21 + 1, 11 + 1), dtype="O")

    data_t = torch.tensor(data, dtype=torch.float32)
    actions = [avail_actions for _ in data]

    q_values: torch.Tensor
    with torch.no_grad():
        q_values, _, _ = model.act(data_t, method="argmax", avail_actions=actions)

    for i, q_value in enumerate(q_values):
        r = data[i][0]
        c = data[i][1]
        if data[i][3] > 0:
            r = int(r / 2)
            if r == 6:
                if data[i][2] > 0:
                    r = 11

        q_max_i = torch.argmax(q_value)
        max_a = model.moves[q_max_i][:2].title()

        if (model.moves[q_max_i] == "double") and (data[i][3] < 0):
            q_value[q_max_i] = -torch.inf
            q_max_i_2 = torch.argmax(q_value)
            max_a = max_a + "/" + model.moves[q_max_i_2][:2].title()

        value_action[r, c] = max_a

    return value_action


__all__ = ["gen_all_states", "gen_value_tensor", "gen_action_tensor"]
