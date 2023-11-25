from __future__ import \
    annotations  # required for preventing the cyclical import of type annotations

from collections import deque
from typing import TYPE_CHECKING, List, Tuple

from src.modules.game import Game
from src.pydantic_types import ReplayBuffer

from ..play import create_state_action

if TYPE_CHECKING:
    # if type_checking, import the modules for type hinting. Otherwise we get cyclical import errors. # noqa: E501
    from src.deep_learning.modules import Net
    from src.modules.player import Player


def state_action_generator(
    blackjack: Game,
    player_ind: int,
    model: Net,
    include_count: bool,
    include_continuous_count: bool,
    method: str = "random",
) -> List[List[Tuple]]:
    house_card_show = blackjack.get_house_show()
    house_value = house_card_show.value if house_card_show.value > 1 else 11
    player: Player = blackjack.players[player_ind]

    s_a = [[]]

    true_count = blackjack.true_count

    while not player.is_done():
        observation, move, policy = create_state_action(
            player=player,
            house_show=house_value,
            include_count=include_count,
            true_count=true_count,
            model=model,
            method=method,
        )

        nHand = player.i_hand  # need this for isolating "split" moves.
        s_a_pair = (observation, move, policy)
        s_a[nHand].append(s_a_pair)

        if move == "split":
            s_a.append(s_a[nHand].copy())

        blackjack.step_player(player_ind, move)

        if include_continuous_count:
            true_count = blackjack.true_count

    return s_a


def buffer_collector(
    buffer: deque,
    s_a_pairs: List[Tuple[Tuple, str]],
    rewards: List[float],
):
    """
    Gathering of rewards with state-action pairs.
    The most challenging part is figuring out how to handle rewards for splitting.
    - treat as normal. Invidual reward per hand
    - for each split observation (2 are generated per split), give each the average total reward.
    # noqa: E501
    """
    for i, s_a_pair_hand in enumerate(s_a_pairs):
        for j, s_a_pair in enumerate(s_a_pair_hand):
            state_obs, move, action_space = s_a_pair
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
                state_obs_next = s_a_pair_hand[j + 1][0]
                action_space_next = s_a_pair_hand[j + 1][2]

            buffer.append(
                ReplayBuffer(
                    obs=state_obs,
                    action_space=action_space,
                    move=move,
                    reward=reward,
                    done=done,
                    obs_next=state_obs_next,
                    action_space_next=action_space_next,
                )
            )


def update_replay_buffer(
    blackjack: Game,
    buffer: deque,
    model: Net,
    include_count: bool,
    include_continuous_count: bool,
    method: str = "random",
):
    blackjack.init_round([1])
    blackjack.deal_init()

    if blackjack.house_blackjack:
        return

    s_a_pairs = state_action_generator(
        blackjack=blackjack,
        player_ind=0,
        model=model,
        include_count=include_count,
        include_continuous_count=include_continuous_count,
        method=method,
    )

    blackjack.step_house(only_reveal_card=True)
    while not blackjack.house_done():
        blackjack.step_house()

    _, results = blackjack.get_results()

    buffer_collector(
        buffer=buffer,
        s_a_pairs=s_a_pairs,
        rewards=results[0],
    )


__all__ = ["update_replay_buffer"]
