from abc import ABC, abstractmethod
from typing import Any, List


class Summary(object):
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value


class ScalarSummary(Summary):
    pass


class HistogramSummary(Summary):
    pass


class ImageSummary(Summary):
    pass


class TextSummary(Summary):
    pass


class VideoSummary(Summary):
    def __init__(self, name: str, value: Any, fps: int = 30):
        super(VideoSummary, self).__init__(name, value)
        self.fps = fps


class ActResult(object):

    def __init__(self, action: Any,
                 observation_elements: dict = None,
                 replay_elements: dict = None,
                 info: dict = None):
        self.action = action
        self.observation_elements = observation_elements or {}
        self.replay_elements = replay_elements or {}
        self.info = info or {}


class Agent(ABC):

    @abstractmethod
    def build(self, training: bool, device=None) -> None:
        pass

    @abstractmethod
    def update(self, step: int, replay_sample: dict) -> dict:
        pass

    @abstractmethod
    def act(self, step: int, observation: dict, deterministic: bool) -> ActResult:
        # returns dict of values that get put in the replay.
        # One of these must be 'action'.
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    def update_summaries(self) -> List[Summary]:
        pass

    @abstractmethod
    def act_summaries(self) -> List[Summary]:
        pass

    @abstractmethod
    def load_weights(self, savedir: str) -> None:
        pass

    @abstractmethod
    def save_weights(self, savedir: str) -> None:
        pass
