from multiprocessing import Value

import numpy as np

from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.envs.multi_task_env import MultiTaskEnv
from yarr.utils.transition import ReplayTransition


class RolloutGenerator(object):

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: MultiTaskEnv, agent: Agent,
                  episode_length: int, timesteps: int, eval: bool):
        obs = env.reset()
        agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        for step in range(episode_length):

            prepped_data = {k: np.array([v]) for k, v in obs_history.items()}

            act_result = agent.act(step_signal.value, prepped_data,
                                   deterministic=eval)

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            agent_extra_elems = {k: np.array(v) for k, v in
                                 act_result.replay_elements.items()}

            transition = env.step(act_result)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs.update(agent_obs_elems)
            obs_tp1 = dict(transition.observation)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs, act_result.action, transition.reward,
                transition.terminal,
                timeout, obs_tp1, agent_extra_elems,
                transition.info)

            obs = transition.observation
            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
