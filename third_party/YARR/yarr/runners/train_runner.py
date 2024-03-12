from abc import abstractmethod, ABC
from typing import Union, List

from yarr.agents.agent import Agent
from yarr.replay_buffer.wrappers import WrappedReplayBuffer
from yarr.runners.env_runner import EnvRunner
from yarr.utils.stat_accumulator import StatAccumulator


class TrainRunner(ABC):

    def __init__(self,
                 agent: Agent,
                 env_runner: EnvRunner,
                 wrapped_replay_buffer: WrappedReplayBuffer,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 iterations: int = int(1e6),
                 logdir: str = '/tmp/yarr/logs',
                 log_freq: int = 500,
                 transitions_before_train: int = 1000,
                 weightsdir: str = '/tmp/yarr/weights',
                 save_freq: int = 100,
                 ):
        self._agent = agent
        self._env_runner = env_runner
        self._wrapped_buffer = wrapped_replay_buffer
        self._stat_accumulator = stat_accumulator
        self._iterations = iterations
        self._logdir = logdir
        self._log_freq = log_freq
        self._transitions_before_train = transitions_before_train
        self._weightsdir = weightsdir
        self._save_freq = save_freq

    @abstractmethod
    def start(self):
        pass
