from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple

from src.modules.game import Game
from src.q.utils.create_q_dict import init_q
from src.q.utils.evaluation import (compare_to_accepted, mean_cum_rewards,
                                    q_value_assessment)
from src.q.utils.runner import play_n_games, select_action

'''
I'll use this to learn the Q function

I'll pass in an already-initialized Game module, which controls the gameplay
'''


@dataclass
class QAgent:
    selection_criteria: str
    gamma: float
    epsilon: float = 1
    learning_rate: float = 0.01
    counter: int = 0
    moves_blacklist: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.q = init_q(moves_blacklist=self.moves_blacklist)
        self.accepted_q = init_q(mode="accepted")

    def update_lr(self, lr):
        self.learning_rate = lr

    def update_eps(self, eps):
        self.epsilon = eps

    def update_q(
            self,
            state: Tuple[int, int, bool],
            action: str,
            reward: float,
            max_q_next: float) -> None:

        old_q = self.q[state]
        old_q[action] = old_q[action] + self.learning_rate *\
            (reward + self.gamma * max_q_next - old_q[action])

    def learn(self, game: Game):
        '''
        Learn a single iteration. Control the number of episodes outside
        of this function call.

        For now, I'll just assume a single player. Otherwise, you could
        iterate through game.players

        Since player moves before house does, and we don't observe a reward
        until that happens, I need to capture terminal states so I can back fill
        q updates once the reward is seen.

        With splitting,
        '''

        player_ind = 0
        player = game.players[player_ind]

        house_card_show = game.get_house_show()
        # casts it to an integer -> Ace == 11 here.
        house_value = house_card_show.value if house_card_show.value > 1 else 11

        terminal_states = []

        while not player.is_done():
            player_total, useable_ace = player.get_value()
            policy = player.get_valid_moves()
            i_hand = player.i_hand

            state = (player_total, house_value, useable_ace)
            q_dict = self.q[state]
            move = select_action(
                state=q_dict,
                policy=policy,
                epsilon=self.epsilon,
                method=self.selection_criteria,
            )

            game.step_player(player_ind, move)

            if not player.complete[i_hand]:
                # If there is a next state, get the info from it.
                player_total_next, useable_ace_next = player.get_value()
                policy_next = player.get_valid_moves()

                state_next = (player_total_next, house_value, useable_ace_next)

                # get best next move, which will allow us to get q_max
                q_dict = self.q[state_next]
                move_next = select_action(
                    state=q_dict,
                    policy=policy_next,
                    epsilon=-1,
                    method="epsilon",
                )
                max_q_next = self.q[state_next][move_next]

                self.update_q(
                    state=state,
                    action=move,
                    reward=0,
                    max_q_next=max_q_next
                )
            else:
                # need this for looking at at most recent action and updating
                # q values accordingly once the house moves.
                terminal_states.append((state, move))

            if move == "split":
                if player.complete[i_hand + 1]:
                    # this will rescue instances where hands were split,
                    # but the following hand cannot be acted on
                    # think of (A,A), or (10,10), where an Ace was then dealt.
                    terminal_states.append((state, move))

        game.step_house(only_reveal_card=True)
        while not game.house_done():
            game.step_house()

        _, results = game.get_results()
        result = results[player_ind]

        # this is skipped if house blackjack, or player blackjack.
        for i, (terminal_state, terminal_move) in enumerate(terminal_states):
            reward = result[i]

            self.update_q(
                state=terminal_state,
                action=terminal_move,
                reward=reward,
                max_q_next=0
            )

        self.counter += 1

    async def evaluate(self, n_rounds: int, n_games: int, game_hyperparams: object):
        rewards = await play_n_games(
            q=self.q,
            wagers=[1],
            n_rounds=n_rounds,
            n_games=n_games,
            game_hyperparams=game_hyperparams,
        )
        mean_reward = mean_cum_rewards(rewards)[0]

        percent_correct_baseline = compare_to_accepted(
            q=self.q,
            accepted_q=self.accepted_q,
        )

        avg_max_q = q_value_assessment(
            q=self.q, game_hyperparams=game_hyperparams, n_rounds=n_rounds
        )

        return mean_reward, percent_correct_baseline, avg_max_q

    def get_q(self):
        return deepcopy(self.q)
