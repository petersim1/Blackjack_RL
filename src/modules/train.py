from dataclasses import dataclass, field
from typing import List, Optional
from copy import deepcopy

from src.helpers.create_q_dict import init_q
from src.helpers.q_learning import learn_policy
from src.helpers.runner import play_n_games
from src.helpers.evaluation import mean_cum_rewards, compare_to_accepted
from src.modules.game import Game

@dataclass
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

@dataclass
class Trainer(EarlyStop):
    early_stop: bool
    method: str
    lr: float
    gamma: float
    eps_decay: float=-1
    eps_min: float=-1
    leniency: int=10
    moves_blacklist: List[str] = field(default_factory=lambda : ["surrender"])
    eval = List[float] = field(default_factory=lambda : [])
    correctness = List[float] = field(default_factory=lambda : [])

    def __post_init__(self):
        EarlyStop.__init__(self, self.leniency)
        self.q = init_q(moves_blacklist=self.moves_blacklist)
        self.best_q = init_q(moves_blacklist=self.moves_blacklist)
        self.accepted_q = init_q(mode="accepted")
        self.eval = []
        self.correctness = []


    def step(self, blackjack: Game, wagers: List[float], eps: Optional[float]=None, reset_deck: bool=False):

        blackjack.init_round(wagers)
        blackjack.deal_init()
        if not blackjack.house_blackjack:
            learn_policy(
                blackjack=blackjack,
                q=self.q,
                epsilon=eps or -1,
                lr=self.lr,
                gamma=self.gamma,
                method=self.method
            )

        if reset_deck:
            blackjack.reset_game()

    async def evaluate(self, n_rounds: int, n_games: int, game_hyperparams: object):

        rewards = await play_n_games(
            q=self.q,
            wagers=[1],
            n_rounds=n_rounds,
            n_games=n_games,
            game_hyperparams=game_hyperparams
        )
        mean_reward = mean_cum_rewards(rewards)[0]
        self.eval.append(mean_reward)

        percent_correct_baseline = compare_to_accepted(
            q=self.q,
            accepted_q=self.accepted_q,
            game_hyperparams=game_hyperparams,
            n_rounds=n_rounds
        )
        self.correctness.append(percent_correct_baseline)

        if self.early_stop:
            self._update(mean_reward)
            if not self.counter:
                self.best_q = deepcopy(self.q)

        return mean_reward, percent_correct_baseline

    def get_q(self, backtrack: bool=False):
        if self.early_stop and backtrack:
            return self.best_q
        return self.q
