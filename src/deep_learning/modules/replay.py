from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.pydantic_types import ReplayBufferI


@dataclass
class ReplayBuffer:
    capacity: int
    memory: deque[ReplayBufferI] = field(init=False)

    def __post_init__(self):
        self.memory = deque([], maxlen=self.capacity)

    def push(self, item: ReplayBufferI):
        self.memory.append(item)

    def sample(self, batch_size: int):
        inds = np.random.choice(self.memory.__len__(), batch_size, replace=True)

        return [self.memory[i] for i in inds]
