from __future__ import \
    annotations  # required for preventing the cyclical import of type annotations

from typing import TYPE_CHECKING, List

import numpy as np
import torch

if TYPE_CHECKING:
    # if type_checking, import the modules for type hinting. Otherwise we get cyclical import errors. # noqa: E501
    from src.deep_learning.modules import Net
    from src.modules.player import Player


def get_observation(include_count: bool, **kwargs):
    if include_count:
        return (
            kwargs["player_total"],
            kwargs["house_show"],
            kwargs["useable_ace"],
            kwargs["can_split"],
            kwargs["can_double"],
            kwargs["count"],
        )

    return (
        kwargs["player_total"],
        kwargs["house_show"],
        kwargs["useable_ace"],
        kwargs["can_split"],
        kwargs["can_double"],
    )


def get_action(
    model: Net, method: str, policy: List[str], observation: tuple
) -> str:
    if method == "random":
        move = np.random.choice(policy)
        return move
    with torch.no_grad():
        obs_t = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        _, _, action_ind = model.act(obs=obs_t, method=method, avail_actions=[policy])
        move = model.moves[action_ind[0][0].item()]

    return move


def create_state_action(
    player: Player,
    house_show: int,
    include_count: bool,
    true_count: float,
    model: Net,
    method: str = "argmax",
):
    player_total, useable_ace = player.get_value()

    policy = player.get_valid_moves()

    if not model.allow_surrender:
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
        count=true_count,
    )

    move = get_action(
        model=model, method=method, policy=policy, observation=observation
    )

    return observation, move, policy


__all__ = ["get_observation", "get_action", "create_state_action"]
