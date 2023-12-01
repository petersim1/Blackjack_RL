from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    # if type_checking, import the modules for type hinting. Otherwise we get cyclical import errors. # noqa: E501
    from src.deep_learning.modules import Net


def gen_all_states():
    all_states = []
    for p in range(4, 21):
        for h in range(2, 12):
            if p >= 12:
                all_states.append((p, h, 1))
            all_states.append((p, h, -1))
    return all_states


def get_masked_output_space(
        model: Net, data: List[Tuple], action_space: List[List[str]]):

    data_t = torch.tensor(data, dtype=torch.float32)
    actions = [action_space]*len(data)
    q_masked, _, _ = model.act(data_t, method="argmax", avail_actions=actions)

    return q_masked


def fill_value_grid(model: Net, states: List[Tuple], hand_type: str):

    if hand_type == "hard":
        fill = np.full((16, 10), np.nan)
        policy = ["hit", "stay", "double", "surrender"]
    if hand_type == "soft":
        fill = np.full((8, 10), np.nan)
        policy = ["hit", "stay", "double", "surrender"]
    if hand_type == "split":
        fill = np.full((10, 10), np.nan)
        policy = ["hit", "stay", "double", "split", "surrender"]

    q_masked = get_masked_output_space(
        model=model,
        data=states,
        action_space=policy
    )

    for i, q in enumerate(q_masked):
        player, house, useable_ace = states[i]
        can_split = (not player % 2) and\
            ((useable_ace < 0) or (player == 12)) and\
            ("split" in policy)
        if can_split:
            r = 11 if (useable_ace > 0) else int(player / 2)
            player_ind = 11 - r
        else:
            player_ind = 20 - player
        fill[player_ind, house - 2] = torch.max(q)

    return fill


def fill_string_grid(model: Net, states: List[Tuple], hand_type: str):
    if hand_type == "hard":
        fill = np.empty((16, 10), dtype="O")
        policy = ["hit", "stay", "double", "surrender"]
    if hand_type == "soft":
        fill = np.empty((8, 10), dtype="O")
        policy = ["hit", "stay", "double", "surrender"]
    if hand_type == "split":
        fill = np.empty((10, 10), dtype="O")
        policy = ["hit", "stay", "double", "split", "surrender"]

    q_masked = get_masked_output_space(
        model=model,
        data=states,
        action_space=policy
    )

    for i, q in enumerate(q_masked):
        player, house, useable_ace = states[i]
        can_split = (not player % 2) and\
            ((useable_ace < 0) or (player == 12)) and\
            ("split" in policy)
        if can_split:
            r = 11 if (useable_ace > 0) else int(player / 2)
            player_ind = 11 - r
        else:
            player_ind = 20 - player

        max_ind1 = torch.argmax(q).item()
        max_ind2 = None
        max_val2 = ""
        if (model.moves[max_ind1] in ["double", "surrender"]) and (not can_split):
            _, _, next_best = model.act(
                obs=torch.tensor(states[i], dtype=torch.float32),
                avail_actions=[["stay", "hit"]]
            )
            max_ind2 = next_best[0].item()

        if model.moves[max_ind1] in ["surrender", "split"]:
            str_fill = model.moves[max_ind1][:2].title()
        else:
            str_fill = model.moves[max_ind1][:1].title()
        if max_ind2 is not None:
            if model.moves[max_ind2] in ["surrender", "split"]:
                max_val2 = model.moves[max_ind2][:2].title()
            else:
                max_val2 = model.moves[max_ind2][:1].title()
            str_fill += "/" + max_val2

        fill[player_ind, house - 2] = str_fill

    return fill


def generate_grid(model: Net, return_type: str = "string"):
    """
    generates the grids for:
    - hard totals (will include all actions except split)
    - soft totals (will include all actions except split)
    - ability to split (will include all actions)
    """
    assert return_type in ["string", "value"], "invalid return_type given."

    all_states = gen_all_states()
    hard_states = list(filter(lambda x : (x[2] < 0) and (x[0] > 4), all_states))
    soft_states = list(
        filter(
            lambda x : (x[2] > 0) and (
                not ((not x[0] % 2) and ((x[2] < 0) or (x[0] == 12)))
            ),
            all_states
        ))
    split_states = list(
        filter(
            lambda x : (not x[0] % 2) and ((x[2] < 0) or (x[0] == 12)),
            all_states
        ))

    if return_type == "value":
        hard_totals = fill_value_grid(
            model=model,
            states=hard_states,
            hand_type="hard"
        )
        soft_totals = fill_value_grid(
            model=model,
            states=soft_states,
            hand_type="soft"
        )
        split_totals = fill_value_grid(
            model=model,
            states=split_states,
            hand_type="split"
        )
    else:
        hard_totals = fill_string_grid(
            model=model,
            states=hard_states,
            hand_type="hard"
        )
        soft_totals = fill_string_grid(
            model=model,
            states=soft_states,
            hand_type="soft"
        )
        split_totals = fill_string_grid(
            model=model,
            states=split_states,
            hand_type="split"
        )

    return hard_totals, soft_totals, split_totals


def gen_value_tensor(model: type[Net], data: List[Tuple], avail_actions: List[str]):
    value_det = np.zeros((21, 11))

    data_t = torch.tensor(data, dtype=torch.float32)
    actions = [avail_actions for _ in data]

    q_values_max: torch.Tensor
    _, q_values_max, _ = model.act(data_t, method="argmax", avail_actions=actions)

    for i, q in enumerate(q_values_max):
        r = data[i][0]
        c = data[i][1]
        can_split = (not r % 2) and\
            ((data[i][2] < 0) or (r == 12)) and\
            ("split" in actions[i])
        if can_split:
            r = int(r / 2)
            if r == 6:
                if data[i][2] > 0:
                    r = 11
        value_det[r, c] = q

    return value_det


def gen_action_tensor(model: type[Net], data: List[Tuple], avail_actions: List[str]):
    value_action = np.empty((21 + 1, 11 + 1), dtype="O")

    data_t = torch.tensor(data, dtype=torch.float32)
    actions = [avail_actions for _ in data]

    q_values: torch.Tensor
    q_values, _, _ = model.act(data_t, method="argmax", avail_actions=actions)

    for i, q_value in enumerate(q_values):
        r = data[i][0]
        c = data[i][1]
        can_split = (not r % 2) and\
            ((data[i][2] < 0) or (r == 12)) and\
            ("split" in actions[i])
        if can_split:
            r = int(r / 2)
            if r == 6:
                if data[i][2] > 0:
                    r = 11

        q_max_i = torch.argmax(q_value)
        max_a = model.moves[q_max_i][:2].title()

        if (model.moves[q_max_i] == "double"):
            q_value[q_max_i] = -torch.inf
            q_max_i_2 = torch.argmax(q_value)
            max_a = max_a + "/" + model.moves[q_max_i_2][:2].title()

        value_action[r, c] = max_a

    return value_action


__all__ = ["gen_all_states", "gen_value_tensor", "gen_action_tensor", "generate_grid"]
