from dataclasses import dataclass, field
from typing import List, Optional
from copy import deepcopy

from src.helpers.create_q_dict import init_q
from src.helpers.q_learning import learn_policy
from src.helpers.runner import play_n_games
from src.helpers.evaluation import mean_cum_rewards, compare_to_accepted
from src.modules.game import Game

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
    leniency: int=10
    moves_blacklist: List[str] = field(default_factory=lambda : ["surrender"])

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
            eps: Optional[float]=None,
            reset_deck: bool=False
        ) -> None:
        '''
        To allow for decayed learning rate and decayed epsilon,
        I pass lr and eps as variables here, where decay is managed outside of the module.
        '''
        game.init_round(wagers)
        game.deal_init()
        if not game.house_blackjack:
            learn_policy(
                game=game,
                q=self.q,
                epsilon=eps or -1,
                lr=lr,
                gamma=self.gamma,
                method=self.method
            )

        if reset_deck:
            game.reset_game()

    async def evaluate(self, n_rounds: int, n_games: int, game_hyperparams: object):

        rewards = await play_n_games(
            q=self.q,
            wagers=[1],
            n_rounds=n_rounds,
            n_games=n_games,
            game_hyperparams=game_hyperparams
        )
        mean_reward = mean_cum_rewards(rewards)[0]

        percent_correct_baseline = compare_to_accepted(
            q=self.q,
            accepted_q=self.accepted_q,
            game_hyperparams=game_hyperparams,
            n_rounds=n_rounds
        )

        if self.early_stop:
            self._update(mean_reward)
            if not self.counter:
                self.best_q = deepcopy(self.q)

        return mean_reward, percent_correct_baseline

    def get_q(self, backtrack: bool=False):
        if self.early_stop and backtrack:
            return self.best_q
        return self.q
