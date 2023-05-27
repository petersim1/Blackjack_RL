from __future__ import annotations # required for preventing the cyclical import of type annotations
from collections import deque
from typing import List, TYPE_CHECKING
import numpy as np
import torch
import asyncio

from src.modules.game import Game
from src.pydantic_types import StateActionPairDeep

if TYPE_CHECKING:
    # if type_checking, import the modules for type hinting. Otherwise we get cyclical import errors.
    from src.modules.player import Player
    from src.deep_q.modules import Net

def update_replay_buffer(blackjack: type[Game], buffer: deque, model: type[Net], mode="random"):
    """ step to update the replay buffer """

    assert mode in ["random", "argmax", "softmax"]

    model.eval()

    blackjack.init_round([1])
    blackjack.deal_init()

    if blackjack.house_blackjack: return

    player = blackjack.players[0]
    player: type[Player]

    s_a = [[]]
    action_space = [[]]
    
    house_show = blackjack.get_house_show(show_value=True)

    if blackjack.house_blackjack: return

    while not player.is_done() :

        player_total, useable_ace = player.get_value()
        nHand = player._get_cur_hand() # need this for isolating "split" moves.

        policy = player.get_valid_moves()
        policy = [p for p in policy if p != "surrender"]

        action_space[nHand].append((policy))

        can_split = "split" in policy
        can_double = "double" in policy

        # I figure that using [-1,1] could help the ReLu fct more than [0,1]
        obs = (player_total, house_show, 2*int(useable_ace)-1, 2*int(can_split)-1, 2*int(can_double)-1)

        if mode == "random":
            # move = np.random.choice(policy) # completely random within valid action space
            move = np.random.choice(model.moves)
        elif mode == "argmax":
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            # _, _, action_ind = model.act(obs=obs_t, method="argmax", avail_actions=[policy])
            _, _, action_ind = model.act(obs=obs_t, method="argmax")
            move = model.moves[action_ind[0][0].item()]
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            # _, _, action_ind = model.act(obs=obs_t, method="softmax", avail_actions=[policy])
            _, _, action_ind = model.act(obs=obs_t, method="softmax")
            move = model.moves[action_ind[0][0].item()]
        
        if move not in policy:
            # Can change the penalty as a hyperparameter of learning process.
            buffer.append(
                (obs, policy, move, -1.5, 1, None, None)
            )
            return
        
        s_a_pair = StateActionPairDeep(
            player_show=player_total,
            house_show=house_show,
            useable_ace=useable_ace,
            can_split=can_split,
            can_double=can_double,
            move=move
        )
        s_a[nHand].append(s_a_pair)

        if move == "split" :
            s_a.append(s_a[nHand].copy())
            action_space.append(action_space[nHand].copy())

        blackjack.step_player(player, move)

    blackjack.step_house()

    _, reward_hands = player.get_result(blackjack.house.cards[0])

    s_a_pair: StateActionPairDeep
    look_forward: StateActionPairDeep
    for i,s_a_pair_hand in enumerate(s_a):
        for j,s_a_pair in enumerate(s_a_pair_hand):

            state_obs = (
                s_a_pair.player_show,
                s_a_pair.house_show,
                2*int(s_a_pair.useable_ace)-1,
                2*int(s_a_pair.can_split)-1,
                2*int(s_a_pair.can_double)-1
            )
            move = s_a_pair.move
            reward = 0
            done = 0
            a_s = action_space[i][j]

            if j == len(s_a_pair_hand) - 1:
                # reward = sum(reward_hands)
                reward = reward_hands[i]
                state_obs_new = None
                done = 1
                a_s_new = None
            else:
                look_forward = s_a_pair_hand[j+1]
                state_obs_new = (
                    look_forward.player_show,
                    look_forward.house_show,
                    2*int(look_forward.useable_ace)-1,
                    2*int(look_forward.can_split)-1,
                    2*int(look_forward.can_double)-1
                )
                a_s_new = action_space[i][j+1]
            
            buffer.append(
                (state_obs, a_s, move, reward, done, state_obs_new, a_s_new)
            )


def play_round(blackjack: type[Game], model: type[Net], wagers: List[float]):

    model.eval()
    
    blackjack.init_round(wagers)
    blackjack.deal_init()

    house_show = blackjack.get_house_show(show_value=True)

    for player in blackjack.players:
        player: type[Player]
        while not player.is_done():

            player_total, useable_ace = player.get_value()

            policy = player.get_valid_moves()
            policy = [p for p in policy if p != "surrender"]

            can_split = "split" in policy
            can_double = "double" in policy

            obs = (player_total, house_show, 2*int(useable_ace)-1, 2*int(can_split)-1, 2*int(can_double)-1)

            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            # _, _, action_ind = model.act(obs=obs_t, method="argmax", avail_actions=[policy])
            _, _, action_ind = model.act(obs=obs_t, method="argmax")
            move = model.moves[action_ind[0][0].item()]
            if move not in policy:
                return [[-1]] # I can tinker with this value.

            blackjack.step_player(player, move)

    blackjack.step_house()
    _, players_winnings = blackjack.get_results()

    return players_winnings


async def play_rounds(blackjack: type[Game], model: type[Net], n_rounds: int, wagers: List[float]):
    rewards = [[] for _ in wagers]

    for i in range(n_rounds):
        players_rewards = play_round(
            blackjack=blackjack,
            model=model,
            wagers=wagers
        )

        for i,reward in enumerate(players_rewards):
            # reward is a list which represents the reward for each hand of a single player due to splitting.
            rewards[i].append(sum(reward))

    return rewards


async def play_games(model: type[Net], n_games: int, n_rounds: int, wagers: List[float], game_hyperparams: object):

    tasks = []
    for _ in range(n_games):
        blackjack = Game(**game_hyperparams)
        tasks.append(
            asyncio.create_task(
            play_rounds(blackjack=blackjack, model=model, n_rounds=n_rounds, wagers=wagers)
            ))
        
    rewards = await asyncio.gather(*tasks)

    return np.array(rewards)