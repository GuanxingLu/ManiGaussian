from abc import ABC, abstractmethod
from typing import Any

from yarr.replay_buffer.replay_buffer import ReplayBuffer


class WrappedReplayBuffer(ABC):

    def __init__(self, replay_buffer: ReplayBuffer):
        """Initializes WrappedReplayBuffer.

        Raises:
          ValueError: If update_horizon is not positive.
          ValueError: If discount factor is not in [0, 1].
        """
        self._replay_buffer = replay_buffer

    @property
    def replay_buffer(self):
        return self._replay_buffer

    @abstractmethod
    def dataset(self) -> Any:
        pass