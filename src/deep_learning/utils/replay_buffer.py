from __future__ import \
    annotations  # required for preventing the cyclical import of type annotations

from typing import TYPE_CHECKING, List, Tuple

import torch

from src.modules.game import Game
from src.pydantic_types import ReplayBufferI

from .action import select_action

if TYPE_CHECKING:
    # if type_checking, import the modules for type hinting. Otherwise we get cyclical import errors. # noqa: E501
    from src.deep_learning.modules import Net, ReplayBuffer


def generate_state_action_pairs(
    blackjack: Game,
    player_ind: int,
    model: Net,
    include_count: bool,
    include_continuous_count: bool,
    method: str = "random",
    implicit_masking: bool = True,
) -> Tuple[List[List[Tuple]], bool]:
    house_card_show = blackjack.get_house_show()
    house_value = house_card_show.value if house_card_show.value > 1 else 11
    player = blackjack.players[player_ind]

    s_a = [[]]
    invalid_move = False

    true_count = blackjack.true_count

    while not player.is_done():
        player_total, useable_ace = player.get_value()
        policy = player.get_valid_moves()

        if include_count:
            if implicit_masking:
                observation = (
                    player_total,
                    house_value,
                    2 * int(useable_ace) - 1,
                    2 * int("split" in policy) - 1,
                    2 * int("double" in policy) - 1,
                    true_count
                )
            else:
                observation = (
                    player_total,
                    house_value,
                    2 * int(useable_ace) - 1,
                    true_count
                )
        else:
            if implicit_masking:
                observation = (
                    player_total,
                    house_value,
                    2 * int(useable_ace) - 1,
                    2 * int("split" in policy) - 1,
                    2 * int("double" in policy) - 1
                )
            else:
                observation = (
                    player_total,
                    house_value,
                    2 * int(useable_ace) - 1,
                )

        move = select_action(
            model=model, method=method, policy=policy, observation=observation
        )

        nHand = player.i_hand  # need this for isolating "split" moves.

        s_a_pair = (observation, policy, move)
        s_a[nHand].append(s_a_pair)

        if move not in policy:
            # will only ever happen if implicit_masking = True
            invalid_move = True
            break

        if move == "split":
            s_a.append(s_a[nHand].copy())

        blackjack.step_player(player_ind, move)

        if include_continuous_count:
            true_count = blackjack.true_count

    return s_a, invalid_move


def update_replay_buffer(
    blackjack: Game,
    buffer: ReplayBuffer,
    model: Net,
    include_count: bool,
    include_continuous_count: bool,
    method: str = "random",
    implicit_masking: bool = True,
):
    blackjack.init_round([1])
    blackjack.deal_init()

    s_a_pairs, invalid_state = generate_state_action_pairs(
        blackjack=blackjack,
        player_ind=0,
        model=model,
        include_count=include_count,
        include_continuous_count=include_continuous_count,
        method=method,
        implicit_masking=implicit_masking,
    )

    rewards = [-0.2]*len(s_a_pairs)

    if not invalid_state:
        blackjack.step_house(only_reveal_card=True)
        while not blackjack.house_done():
            blackjack.step_house()

        _, results = blackjack.get_results()

        rewards = results[0]

    for i, s_a_pair_hand in enumerate(s_a_pairs):
        for j, s_a_pair in enumerate(s_a_pair_hand):
            state_obs, action_space, move = s_a_pair
            # might want to further incentivize hitting
            # reward = 0.25*int(s_a_pair.move in ["hit", "split"])
            reward = 0
            done = 0

            if j == len(s_a_pair_hand) - 1:
                # I leaning towards taking mean reward, versus reward of each separate split hand # noqa: E501
                # This is because, there's no guarantee that sampling the replay buffer will pick up # noqa: E501
                # all observations.
                # reward = sum(rewards) / len(rewards)
                reward = rewards[i]
                state_obs_next = None
                action_space_next = None
                done = 1
            else:
                if move == "split":
                    reward = sum(rewards) / len(rewards)
                state_obs_next, action_space_next, _ = s_a_pair_hand[j + 1]

            buffer.push(
                ReplayBufferI(
                    obs=state_obs,
                    action_space=action_space,
                    move=move,
                    reward=reward,
                    done=done,
                    obs_next=state_obs_next,
                    action_space_next=action_space_next,
                )
            )


def gather_buffer_obs(replay_buffer: ReplayBuffer, batch_size: int, moves: List[str]):
    observations = replay_buffer.sample(batch_size=batch_size)

    dim = len(observations[0].obs)
    filler = tuple([0] * dim)

    (obs, action_space, moves_, rewards, dones, obs_next, action_space_next) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for observation in observations:
        obs.append(observation.obs)
        action_space.append(observation.action_space or ["hit"])
        moves_.append(moves.index(observation.move))
        rewards.append(observation.reward)
        dones.append(observation.done)
        obs_next.append(observation.obs_next or filler)
        action_space_next.append(observation.action_space_next or ["hit"])

    obs_t = torch.tensor(obs, dtype=torch.float32)
    moves_t = torch.tensor(moves_, dtype=torch.int64).unsqueeze(-1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)
    obs_next_t = torch.tensor(obs_next, dtype=torch.float32)

    return (
        obs_t,
        action_space,
        moves_t,
        rewards_t,
        dones_t,
        obs_next_t,
        action_space_next,
    )


def gather_target_obs(
    target_net: type[Net],
    action_space_next: List[List[str]],
    obs_next_t: torch.Tensor,
    rewards_t: torch.Tensor,
    dones_t: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    _, target_q_argmax, _ = target_net.act(
        obs_next_t, method="argmax", avail_actions=action_space_next
    )
    targets_t = rewards_t + torch.nan_to_num(
        gamma * (1 - dones_t) * target_q_argmax, nan=0
    )

    return targets_t


__all__ = [
    "generate_state_action_pairs",
    "update_replay_buffer",
    "gather_buffer_obs",
    "gather_target_obs"
]
