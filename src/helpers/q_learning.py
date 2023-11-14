from typing import List, Tuple

from src.pydantic_types import StateActionPair, ConditionalActionSpace
from src.modules.game import Game
from src.modules.player import Player
from src.helpers.runner import select_action


def gen_episode(
        game: Game,
        player_ind: int,
        q: object,
        epsilon: float,
        method: str
    ) -> Tuple[List[List[StateActionPair]], ConditionalActionSpace]:
    
    """
    Given the blackjack module, index of Player, Q values object, epsilon value, and method;
    generate the state-action pairs and action-space (conditional action space based off state)
    """

    assert method in ["epsilon", "thompson"], "invalid method selected"
    
    s_a_pairs = [[]]
    conditional_action_spaces = [[]]
    
    player = game.players[player_ind]
    house_card_show = game.get_house_show()
    house_value = house_card_show.value if house_card_show.value > 1 else 11

    while not player.is_done() :

        player_total, useable_ace = player.get_value()
        nHand = player.i_hand # need this for isolating "split" moves. Can access since decorator to update i_hand is called in get_value() method.

        policy = player.get_valid_moves()
        conditional_action_spaces[nHand].append((policy))

        can_split = "split" in policy

        q_dict = q[(player_total, house_value, useable_ace, can_split)]
        move = select_action(state=q_dict, policy=policy, epsilon=epsilon, method=method)

        s_a_pair = StateActionPair(
            player_show=player_total,
            house_show=house_value,
            useable_ace=useable_ace,
            can_split=can_split,
            move=move
        )
        s_a_pairs[nHand].append(s_a_pair)

        if move == "split" :
            s_a_pairs.append(s_a_pairs[nHand].copy())
            conditional_action_spaces.append(conditional_action_spaces[nHand].copy())

        game.step_player(player_ind, move)
        
    return s_a_pairs, conditional_action_spaces

def learn_policy(
        game: type[Game],
        q: object,
        epsilon: float,
        gamma: float,
        lr: float,
        method: str="epsilon"
    ) -> None:
    
    """
    Given the Game module, learn an optimal policy inplace:
    - epsilon : e-greedy hyperparameter
    - gamma : decay factor, which I use to discount rewards for earlier moves in a round
    - lr : learning rate to update Q function
    - method: Q value selection method

    Some additional logic is required to account for splits.

    In blackjack, we don't care about how many moves are performed. We only care about final outcome.
    Rewards for non-terminal states receive a reward of 0.

    If we are in a terminal state, there is no s`, so we define it as 0.
    """

    assert method in ["epsilon", "thompson"], "invalid method selected"

    s_a_pairs: List[List[List[StateActionPair]]] = []
    conditional_action_space: List[ConditionalActionSpace] = []

    for ind in range(len(game.players)):
        s_a, action_space = gen_episode(
            game=game,
            player_ind=ind,
            q=q,
            epsilon=epsilon,
            method=method
        )
        s_a_pairs.append(s_a)
        conditional_action_space.append(action_space)

    game.step_house()
    # _, player_winnings = blackjack.get_results()

    player: Player
    for i,player in enumerate(game.players):
        _, player_winnings = player.get_result(game.house.cards[0])
        j = 0 # move number in state-action pair
        hand = 0 # hand number (to account for splits)
        while (hand < len(s_a_pairs[i])):
            if s_a_pairs[i][hand]: #otherwise, player blackjack, and we can't learn from this since no moves are taken.
                s_a_pair = s_a_pairs[i][hand][j]

                old_q = q[(
                    s_a_pair.player_show,
                    s_a_pair.house_show,
                    s_a_pair.useable_ace,
                    s_a_pair.can_split
                )]

                # r = player_winnings[hand]
                r = sum(player_winnings)
                max_q_p = 0
                if (j+1) < len(s_a_pairs[i][hand]):
                    s_a_pair_p = s_a_pairs[i][hand][j+1]
                    action_space = conditional_action_space[i][hand][j+1]

                    q_dict = q[(
                        s_a_pair_p.player_show,
                        s_a_pair_p.house_show,
                        s_a_pair_p.useable_ace,
                        s_a_pair_p.can_split,
                    )]

                    max_q_p = max([v for k,v in q_dict.items() if k in action_space])
                    r = 0
                old_q[s_a_pair.move] = old_q[s_a_pair.move] + lr*(r + gamma * max_q_p - old_q[s_a_pair.move])

            if j < len(s_a_pairs[i][hand])-1:
                j += 1
            else: 
                hand += 1
                j = 0
