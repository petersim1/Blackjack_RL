from typing import List, Tuple

from src.helpers.runner import select_action
from src.modules.game import Game
from src.pydantic_types import ConditionalActionSpace, StateActionPair


def gen_episode(
    game: Game, player_ind: int, q: object, epsilon: float, method: str
) -> Tuple[List[List[StateActionPair]], ConditionalActionSpace]:
    """
    Given the blackjack module, index of Player, Q values object, epsilon
    value, and method;
    generate the state-action pairs and action-space (conditional action space
    based off state)
    """

    assert method in ["epsilon", "thompson"], "invalid method selected"
    s_a_pairs = [[]]
    conditional_action_spaces = [[]]

    player = game.players[player_ind]
    house_card_show = game.get_house_show()
    # casts it to an integer -> Ace == 11 here.
    house_value = house_card_show.value if house_card_show.value > 1 else 11

    while not player.is_done():
        player_total, useable_ace = player.get_value()
        nHand = (
            player.i_hand
        )  # need this for isolating "split" moves. Can access since decorator to update i_hand is called in get_value() method. # noqa: E501

        policy = player.get_valid_moves()
        conditional_action_spaces[nHand].append((policy))

        q_dict = q[(player_total, house_value, useable_ace)]
        move = select_action(
            state=q_dict, policy=policy, epsilon=epsilon, method=method
        )

        s_a_pair = StateActionPair(
            player_show=player_total,
            house_show=house_value,
            useable_ace=useable_ace,
            move=move,
        )
        s_a_pairs[nHand].append(s_a_pair)

        if move == "split":
            """
            Determines how to handle state-action pairs + action-space
            Options:
            - Push an empty array. This means "split" is only accounted for once.
            - Push a copy of previous state-action pairs + action-space.
                This means that "split" is accounted for twice.
            """
            # s_a_pairs.append([])
            # conditional_action_spaces.append([])
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
    method: str = "epsilon",
) -> None:
    """
    Given the Game module, learn an optimal policy inplace:
    - epsilon : e-greedy hyperparameter
    - gamma : decay factor, which I use to discount rewards for earlier moves in a round
    - lr : learning rate to update Q function
    - method: Q value selection method

    Some additional logic is required to account for splits.

    Learning step is skipped if Player or House has blackjack,
    as round would have immediately ended and no state-action pairs would exist

    Rewards for non-terminal states receive a reward of 0, except for splits.

    If we are in a terminal state, there is no s`, so we define it as 0.
    """

    assert method in ["epsilon", "thompson"], "invalid method selected"

    s_a_pairs: List[List[List[StateActionPair]]] = []
    conditional_action_space: List[ConditionalActionSpace] = []

    for ind in range(len(game.players)):
        s_a, action_space = gen_episode(
            game=game, player_ind=ind, q=q, epsilon=epsilon, method=method
        )
        s_a_pairs.append(s_a)
        conditional_action_space.append(action_space)

    game.step_house(only_reveal_card=True)
    while not game.house_done():
        game.step_house()

    _, results = game.get_results()
    for i, player_winnings in enumerate(results):
        j = 0  # move number in state-action pair
        hand = 0  # hand number (to account for splits)
        while hand < len(s_a_pairs[i]):
            if s_a_pairs[i][hand]:
                # otherwise, player blackjack, and we can't learn from this since no moves are taken. # noqa: E501
                s_a_pair = s_a_pairs[i][hand][j]

                old_q = q[
                    (
                        s_a_pair.player_show,
                        s_a_pair.house_show,
                        s_a_pair.useable_ace,
                    )
                ]
                # This is an important component, as it dicates how to aggregate
                # rewards for when a split occurred.
                # if s_a_pair.move == "split":
                #     r = sum(player_winnings[hand:(hand+2)]) / 2
                # else:
                #     r = player_winnings[hand]
                r = player_winnings[hand]
                max_q_p = 0
                if (j + 1) < len(s_a_pairs[i][hand]):
                    # If not a terminal state...
                    s_a_pair_p = s_a_pairs[i][hand][j + 1]
                    action_space = conditional_action_space[i][hand][j + 1]

                    q_dict = q[
                        (
                            s_a_pair_p.player_show,
                            s_a_pair_p.house_show,
                            s_a_pair_p.useable_ace,
                        )
                    ]

                    max_q_p = max([v for k, v in q_dict.items() if k in action_space])
                    """
                    If not a terminal state, assume reward is zero.
                    However, maybe I want to treat splits differently?
                    """
                    # POTENTIALLY REMOVE THIS IF STATEMENT!!
                    # if s_a_pair.move != "split":
                    # for splits, this condition is always met unless it's splitting Aces. # noqa: E501
                    # we completely lose information that a split occurred.
                    # I can tell it to retain a reward based on final outcome, if split. # noqa: E501
                    r = 0
                old_q[s_a_pair.move] = old_q[s_a_pair.move] + lr * (
                    r + gamma * max_q_p - old_q[s_a_pair.move]
                )

            if j < len(s_a_pairs[i][hand]) - 1:
                j += 1
            else:
                hand += 1
                j = 0
