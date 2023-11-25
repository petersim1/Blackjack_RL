from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

from src.modules.cards import Card
from src.modules.game import Game
from src.q.utils.create_q_dict import init_q
from src.q.utils.evaluation import (compare_to_accepted, mean_cum_rewards,
                                    q_value_assessment)
from src.q.utils.q_learning import learn_policy
from src.q.utils.runner import play_n_games


@dataclass(kw_only=True)
class EarlyStop:
    leniency: int
    counter = 0
    min_performance = -10
    stop = False

    def _update(self, performance: float):
        if performance < self.min_performance:
            self.counter += 1
        else:
            self.counter = 0

        self.min_performance = max(self.min_performance, performance)

        if self.counter == self.leniency:
            self.stop = True


@dataclass(kw_only=True)
class Trainer(EarlyStop):
    early_stop: bool
    method: str
    gamma: float
    leniency: int = 10
    moves_blacklist: List[str] = field(default_factory=list)

    def __post_init__(self):
        EarlyStop.__init__(self, leniency=self.leniency)
        self.q = init_q(moves_blacklist=self.moves_blacklist)
        self.best_q = init_q(moves_blacklist=self.moves_blacklist)
        self.accepted_q = init_q(mode="accepted")

    def step(
        self,
        game: Game,
        wagers: List[float],
        lr: float,
        force_cards: List[Card] = [],
        eps: Optional[float] = None,
    ) -> None:
        """
        To allow for decayed learning rate and decayed epsilon,
        I pass lr and eps as variables here, where decay is managed outside of
        the module.
        """
        game.init_round(wagers)
        game.deal_init(force_cards=force_cards)
        learn_policy(
            game=game,
            q=self.q,
            epsilon=eps or -1,
            lr=lr,
            gamma=self.gamma,
            method=self.method,
        )

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

        if self.early_stop:
            self._update(mean_reward)
            if not self.counter:
                self.best_q = deepcopy(self.q)

        return mean_reward, percent_correct_baseline, avg_max_q

    def get_q(self, backtrack: bool = False):
        if self.early_stop and backtrack:
            return self.best_q
        return self.q
