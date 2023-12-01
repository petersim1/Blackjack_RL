from __future__ import \
    annotations  # required for preventing the cyclical import of type annotations

from typing import TYPE_CHECKING, List

import numpy as np
import torch

if TYPE_CHECKING:
    # if type_checking, import the modules for type hinting. Otherwise we get cyclical import errors. # noqa: E501
    from src.deep_learning.modules import Net


def select_action(
    model: Net, method: str, policy: List[str], observation: tuple,
) -> str:
    if method == "random":
        return np.random.choice(policy)
    with torch.no_grad():
        obs_t = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        _, _, action_ind = model.act(
            obs=obs_t,
            method=method,
            avail_actions=[policy]
        )
        move = model.moves[action_ind[0][0].item()]
        return move


__all__ = ["select_action"]
