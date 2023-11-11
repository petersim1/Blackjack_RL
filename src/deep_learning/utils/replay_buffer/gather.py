from __future__ import annotations
import numpy as np
from collections import deque
import torch
from typing import List, TYPE_CHECKING
from src.pydantic_types import ReplayBuffer

if TYPE_CHECKING:
    # if type_checking, import the modules for type hinting. Otherwise we get cyclical import errors.
    from src.deep_learning.modules import Net

def gather_buffer_obs(
        replay_buffer: deque[ReplayBuffer],
        batch_size: int,
        moves: List[str]
):
    transition_inds = np.random.choice(len(replay_buffer), batch_size, replace=True)

    dim = len(replay_buffer[0].obs)
    filler = tuple([0]*dim)

    obs_t = torch.tensor([replay_buffer[i].obs for i in transition_inds], dtype=torch.float32)
    action_space = [replay_buffer[i].action_space or ["hit"] for i in transition_inds]
    moves_t = torch.tensor([moves.index(replay_buffer[i].move) for i in transition_inds], dtype=torch.int64).unsqueeze(-1)
    rewards_t = torch.tensor([replay_buffer[i].reward for i in transition_inds], dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.tensor([replay_buffer[i].done for i in transition_inds], dtype=torch.float32).unsqueeze(-1)
    obs_next_t = torch.tensor([replay_buffer[i].obs_next or filler for i in transition_inds], dtype=torch.float32)
    action_space_next = [replay_buffer[i].action_space_next or ["hit"] for i in transition_inds]

    return obs_t, action_space, moves_t, rewards_t, dones_t, obs_next_t, action_space_next


def gather_target_obs(
        target_net: type[Net],
        action_space_next: List[List[str]],
        obs_next_t: torch.Tensor,
        rewards_t: torch.Tensor,
        dones_t: torch.Tensor,
        gamma: float,
) -> torch.Tensor:

    with torch.no_grad():
        _, target_q_argmax, _ = target_net.act(obs_next_t, method="argmax", avail_actions=action_space_next)
        targets_t = rewards_t + torch.nan_to_num(gamma * (1 - dones_t) * target_q_argmax, nan=0)

    return targets_t

__all__ = ["gather_buffer_obs", "gather_target_obs"]