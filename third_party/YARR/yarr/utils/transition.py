from typing import List

import numpy as np
from yarr.agents.agent import Summary


class Transition(object):

    def __init__(self, observation: dict, reward: float, terminal: bool,
                 info: dict = None, summaries: List[Summary] = None):
        self.observation = observation
        self.reward = reward
        self.terminal = terminal
        self.info = info or {}
        self.summaries = summaries or []


class ReplayTransition(object):

    def __init__(self, observation: dict, action: np.ndarray,
                 reward: float, terminal: bool, timeout: bool,
                 final_observation: dict = None,
                 summaries: List[Summary] = None,
                 info: dict = None):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.terminal = terminal
        self.timeout = timeout
        # final only populated on last timestep
        self.final_observation = final_observation
        self.summaries = summaries or []
        self.info = info
