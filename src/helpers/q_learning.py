import numpy as np
from typing import List, Tuple
from src.constants import card_map
from src.pydantic_types import QMovesI, StateActionPair, ConditionalActionSpace
from src.modules.game import Game
from src.modules.player import Player

def init_q(moves: List[str]) -> object:
    """Initialize the Q value object. Isolates splittable vs. non-splittable"""
    
    moves_no_split = [m for m in moves if m!="split"]

    Q = {"can_split": {}, "no_split": {}}
    
    for p in range(5,22) :
        for h in range(2,12) :
            if (21 > p > 11) :
                for a in [True,False] :
                    Q["no_split"][(p,h,a)] = {m:0 for m in moves_no_split}
            else :
                Q["no_split"][(p,h,False)] = {m:0 for m in moves_no_split}
    
    for c in card_map.values() :
        if c in ["J","Q","K"] :
            continue
        for h in range(2,12) :
            a = False if c!="A" else True
            Q["can_split"][(c,h,a)] = {m:0 for m in moves}

    return Q

def get_best_action(state: QMovesI, policy: List[str], epsilon: float, method: str) -> str :
    """
    Get the best action according to a state, policy, epsilon value, and method.
    - Can use epsilon = -1 to serve as greedy.
    - Can use epsilon = 1 to serve as random.
    """
    assert method in ["epsilon", "thompson"], "invalid method selected"

    # masking of invalid states
    q_dict = {k:v for k,v in state.items() if k in policy}

    # softmax
    if method == "thompson" :
        exp = np.exp(np.array(list(q_dict.values())))
        p = exp / exp.sum()
        move = np.random.choice(list(q_dict.keys()), p=p)
        return move
    
    # epsilon-greedy
    if method == "epsilon" :
        n = np.random.rand()
        if n < epsilon :
            move = np.random.choice(policy)
        else :
            # possible that there are multiple "best" moves, sample from them.
            best_move = [k for k,v in q_dict.items() if v==max(list(q_dict.values()))]
            move = np.random.choice(best_move)

    return move


def gen_episode(
        blackjack: type[Game],
        i_player: int,
        q: object,
        epsilon: float,
        method: str
    ) -> Tuple[List[List[StateActionPair]], ConditionalActionSpace]:
    
    """
    Given the blackjack module, index of Player, Q values object, epsilon value, and method;
    generate the state-action pairs and action-space (conditional action space based off state),
    while updating the blackjack module)
    """

    assert method in ["epsilon", "thompson"], "invalid method selected"
    
    s_a_pairs = [[]]
    conditional_action_spaces = [[]]
    
    player: type[Player] = blackjack.players[i_player]
    house_show = blackjack.get_house_show(show_value=True)

    while not player.is_done() :

        player_total,can_split,useable_ace,card1 = player.get_value()
        nHand = player._get_cur_hand()

        policy = player.get_valid_moves(house_show)
        policy = [p for p in policy if p!="insurance"]
        conditional_action_spaces[nHand].append((policy))
        
        if can_split :
            q_dict = q["can_split"][(card1, house_show, useable_ace)]
        else :
            q_dict = q["no_split"][(player_total, house_show, useable_ace)]
        move = get_best_action(state=q_dict, policy=policy, epsilon=epsilon, method=method)

        s_a_pair = StateActionPair(
            player_show=player_total,
            house_show=house_show,
            useable_ace=useable_ace,
            can_split=can_split,
            card1=card1,
            move=move
        )

        s_a_pairs[nHand].append(s_a_pair)

        if move == "split" :
            s_a_pairs.append(s_a_pairs[nHand].copy())
            conditional_action_spaces.append(conditional_action_spaces[nHand].copy())

        blackjack.step_player(player,move)
        
    return s_a_pairs, conditional_action_spaces

def learn_policy(
        blackjack: type[Game],
        q: object,
        n_players: int,
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

    for i in range(n_players) :
        s_a, action_space = gen_episode(
            blackjack=blackjack,
            i_player=i,
            q=q,
            epsilon=epsilon,
            method=method
        )
        s_a_pairs.append(s_a)
        conditional_action_space.append(action_space)

    blackjack.step_house()
    _, player_winnings = blackjack.get_results()

    for i,w in enumerate(player_winnings) :
        j = 0
        hand = 0
        while (hand < len(s_a_pairs[i])) :
            if not s_a_pairs[i][hand] : break # means that blackjack was drawn.
            s_a_pair = s_a_pairs[i][hand][j]
            if s_a_pair.can_split:
                old_q = q["can_split"][(s_a_pair.card1, s_a_pair.house_show, s_a_pair.useable_ace)]
            else :
                old_q = q["no_split"][(s_a_pair.player_show, s_a_pair.house_show, s_a_pair.useable_ace)]
            r = w/len(s_a_pairs[i])
            max_q_p = 0
            if (j+1) < len(s_a_pairs[i][hand]) :
                s_a_pair_p = s_a_pairs[i][hand][j+1]
                action_space = conditional_action_space[i][hand][j+1]
                if s_a_pair_p.can_split:
                    q_dict = q["can_split"][(s_a_pair_p.card1, s_a_pair_p.house_show, s_a_pair_p.useable_ace)]
                else :
                    q_dict = q["no_split"][(s_a_pair_p.player_show, s_a_pair_p.house_show, s_a_pair_p.useable_ace)]
                max_q_p = max([v for k,v in q_dict.items() if k in action_space])
                r = 0
            old_q[s_a_pair.move] = old_q[s_a_pair.move] + lr*(r + gamma * max_q_p - old_q[s_a_pair.move])
            if j < len(s_a_pairs[i][hand])-1 :
                j += 1 # move to next state-action pair within a hand.
            else : 
                hand += 1 # move to next hand for a player due to splitting.
                j = 0

def evaluate_policy(blackjack: type[Game], q: object, wagers: List[float], n_rounds: int) :
    
    rewards = [[] for _ in wagers]
        
    for _ in range(n_rounds) :
        blackjack.init_round(wagers) # must call this before dealing a round
        blackjack.deal_init() # initial deal, before players decide what to do.
        house_show = blackjack.get_house_show(show_value=True)
        for i,player in enumerate(blackjack.players) :
            player: type[Player]
            while not player.is_done() :
                playerShow,canSplit,useableAce,card1 = player.get_value()
                policy = player.get_valid_moves(house_show)
                policy = [p for p in policy if p!="insurance"]
                if canSplit:
                    move = get_best_action(
                        state=q["can_split"][(card1, house_show, useableAce)],
                        policy=policy,
                        epsilon=-1,
                        method="epsilon"
                    )
                else :
                    move = get_best_action(
                        state=q["no_split"][(playerShow, house_show, useableAce)],
                        policy=policy,
                        epsilon=-1,
                        method="epsilon"
                    )
                blackjack.step_player(player,move)
        blackjack.step_house() #play the house complete hand.
        _, player_winnings = blackjack.get_results()

        for i,w in enumerate(player_winnings) :
            rewards[i].append(w)

    return rewards
