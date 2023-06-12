from __future__ import annotations # required for preventing the cyclical import of type annotations
from collections import deque
from typing import List, Tuple, TYPE_CHECKING
import numpy as np
import torch

from src.modules.game import Game
from src.pydantic_types import ReplayBuffer

if TYPE_CHECKING:
    # if type_checking, import the modules for type hinting. Otherwise we get cyclical import errors.
    from src.modules.player import Player
    from src.deep_learning.modules import Net


def get_observation(include_count: bool, **kwargs):

    if include_count:
        return (kwargs["player_total"], kwargs["house_show"], kwargs["useable_ace"], kwargs["can_split"], kwargs["can_double"], kwargs["count"])
    
    return (kwargs["player_total"], kwargs["house_show"], kwargs["useable_ace"], kwargs["can_split"], kwargs["can_double"])


def get_action(model: type[Net], method: str, observation: tuple) -> str:

    if method == "random":
        move = np.random.choice(model.moves)
        return move
    
    obs_t = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    _, _, action_ind = model.act(obs=obs_t, method=method)
    move = model.moves[action_ind[0][0].item()]

    return move


def state_action_generator(
        blackjack: type[Game],
        player: type[Player],
        model: type[Net],
        include_count: bool,
        include_continuous_count: bool,
        method: str="random",
) -> Tuple[bool, List[List[Tuple]]]:
    house_show = blackjack.get_house_show(show_value=True)

    count = blackjack.get_count()
    true_count = count * 52 / blackjack.cards.sum()

    s_a = [[]]

    while not player.is_done() :

        player_total, useable_ace = player.get_value()
        nHand = player._get_cur_hand() # need this for isolating "split" moves.

        policy = player.get_valid_moves()
        policy = [p for p in policy if p != "surrender"]

        can_split = "split" in policy
        can_double = "double" in policy

        # I figure that using [-1,1] could help the ReLU fct more than [0,1]

        observation = get_observation(
            include_count=include_count,
            player_total=player_total,
            house_show=house_show,
            useable_ace=2 * int(useable_ace) - 1,
            can_split=2 * int(can_split) - 1,
            can_double=2 * int(can_double) - 1,
            count=true_count
        )

        move = get_action(
            model=model,
            method=method,
            observation=observation
        )

        s_a_pair = (observation, move)
        
        s_a[nHand].append(s_a_pair)

        if move not in policy:
            # Can change the penalty as a hyperparameter of learning process.
            return True, s_a

        if move == "split" :
            s_a.append(s_a[nHand].copy())

        blackjack.step_player(player, move)

        if include_continuous_count:
            count = blackjack.get_count()
            true_count = count * 52 / blackjack.cards.sum()

    return False, s_a


def buffer_collector(
        buffer: deque,
        s_a_pairs: List[Tuple[Tuple, str]],
        rewards: List[float],
):
    for i,s_a_pair_hand in enumerate(s_a_pairs):
        for j,s_a_pair in enumerate(s_a_pair_hand):

            state_obs = s_a_pair[0]
            move = s_a_pair[1]
            # might want to further incentivize hitting
            # reward = 0.25*int(s_a_pair.move in ["hit", "split"])
            reward = 0
            done = 0

            if j == len(s_a_pair_hand) - 1:
                # reward = sum(reward_hands)
                reward = rewards[i]
                state_obs_new = None
                done = 1
            else:
                state_obs_new = s_a_pair_hand[j+1][0]

            buffer.append(ReplayBuffer(
                obs=state_obs,
                move=move,
                reward=reward,
                done=done,
                obs_next=state_obs_new
            ))


def update_replay_buffer(
        blackjack: type[Game],
        buffer: deque,
        model: type[Net],
        include_count: bool,
        include_continuous_count: bool,
        misstep_penalty: float=-1.5,
        method: str="random",
):
    blackjack.init_round([1])
    blackjack.deal_init()

    if blackjack.house_blackjack: return

    player: type[Player] = blackjack.players[0]

    misstep, s_a_pairs = state_action_generator(
        blackjack=blackjack,
        player=player,
        model=model,
        include_count=include_count,
        include_continuous_count=include_continuous_count,
        method=method,
    )

    # if we take an invalid move, we can't advance the blackjack module.
    # force the misstep penalty as the rewards. Otherwise, proceed with module.
    reward_hands = [misstep_penalty for _ in player.cards]
    if not misstep:
        blackjack.step_house()
        _, reward_hands = player.get_result(blackjack.house.cards[0])
    
    buffer_collector(
        buffer=buffer,
        s_a_pairs=s_a_pairs,
        rewards=reward_hands,
    )


__all__ = ["update_replay_buffer"]