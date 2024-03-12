import collections
import logging
import os
import signal
import time
from multiprocessing import Value
from threading import Thread
from typing import List
from typing import Union

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.agents.agent import ScalarSummary
from yarr.agents.agent import Summary
from yarr.envs.env import Env
from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.runners._env_runner import _EnvRunner
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import StatAccumulator, SimpleAccumulator
from yarr.utils.process_str import change_case
from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv

class EnvRunner(object):

    def __init__(self,
                 train_env: Env,
                 agent: Agent,
                 train_replay_buffer: Union[ReplayBuffer, List[ReplayBuffer]],
                 num_train_envs: int,
                 num_eval_envs: int,
                 rollout_episodes: int,
                 eval_episodes: int,
                 training_iterations: int,
                 eval_from_eps_number: int,
                 episode_length: int,
                 eval_env: Union[Env, None] = None,
                 eval_replay_buffer: Union[ReplayBuffer, List[ReplayBuffer], None] = None,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 rollout_generator: RolloutGenerator = None,
                 weightsdir: str = None,
                 logdir: str = None,
                 max_fails: int = 10,
                 num_eval_runs: int = 1,
                 env_device: torch.device = None,
                 multi_task: bool = False):
        self._train_env = train_env
        self._eval_env = eval_env if eval_env else train_env
        self._agent = agent
        self._train_envs = num_train_envs
        self._eval_envs = num_eval_envs
        self._train_replay_buffer = train_replay_buffer if isinstance(train_replay_buffer, list) else [train_replay_buffer]
        self._timesteps = self._train_replay_buffer[0].timesteps if self._train_replay_buffer[0] is not None else 1

        if eval_replay_buffer is not None:
            eval_replay_buffer = eval_replay_buffer if isinstance(eval_replay_buffer, list) else [eval_replay_buffer]
        self._eval_replay_buffer = eval_replay_buffer
        self._rollout_episodes = rollout_episodes
        self._eval_episodes = eval_episodes
        self._num_eval_runs = num_eval_runs
        self._training_iterations = training_iterations
        self._eval_from_eps_number = eval_from_eps_number
        self._episode_length = episode_length
        self._stat_accumulator = stat_accumulator
        self._rollout_generator = (
            RolloutGenerator() if rollout_generator is None
            else rollout_generator)
        self._rollout_generator._env_device = env_device
        self._weightsdir = weightsdir
        self._logdir = logdir
        self._max_fails = max_fails
        self._env_device = env_device
        self._previous_loaded_weight_folder = ''
        self._p = None
        self._kill_signal = Value('b', 0)
        self._step_signal = Value('i', -1)
        self._num_eval_episodes_signal = Value('i', 0)
        self._eval_epochs_signal = Value('i', 0)
        self._eval_report_signal = Value('b', 0)
        self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
        self._total_transitions = {'train_envs': 0, 'eval_envs': 0}
        self.log_freq = 1000  # Will get overridden later
        self.target_replay_ratio = None  # Will get overridden later
        self.current_replay_ratio = Value('f', -1)
        self._current_task_id = -1
        self._multi_task = multi_task

    def summaries(self) -> List[Summary]:
        summaries = []
        if self._stat_accumulator is not None:
            summaries.extend(self._stat_accumulator.pop())
        for key, value in self._new_transitions.items():
            summaries.append(ScalarSummary('%s/new_transitions' % key, value))
        for key, value in self._total_transitions.items():
            summaries.append(ScalarSummary('%s/total_transitions' % key, value))
        self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
        summaries.extend(self._agent_summaries)

        # add current task_name to eval summaries .... argh this should be inside a helper function
        if hasattr(self._eval_env, '_task_class'):
            eval_task_name = change_case(self._eval_env._task_class.__name__)
        elif hasattr(self._eval_env, '_task_classes'):
            if self._current_task_id != -1:
                task_id = (self._current_task_id) % len(self._eval_env._task_classes)
                eval_task_name = change_case(self._eval_env._task_classes[task_id].__name__)
            else:
                eval_task_name = ''
        else:
            raise Exception('Neither task_class nor task_classes found in eval env')

        # multi-task summaries
        if eval_task_name and self._multi_task:
            for s in summaries:
                if 'eval' in s.name:
                    s.name = '%s/%s' % (s.name, eval_task_name)

        return summaries

    def _update(self):
        # Move the stored transitions to the replay and accumulate statistics.
        new_transitions = collections.defaultdict(int)
        with self._internal_env_runner.write_lock:
            self._agent_summaries = list(
                self._internal_env_runner.agent_summaries)
            if self._num_eval_episodes_signal.value % self._eval_episodes == 0 and self._num_eval_episodes_signal.value > 0:
                self._internal_env_runner.agent_summaries[:] = []
            for name, transition, eval in self._internal_env_runner.stored_transitions:
                add_to_buffer = (not eval) or self._eval_replay_buffer is not None
                if add_to_buffer:
                    kwargs = dict(transition.observation)
                    replay_index = transition.info["active_task_id"]
                    rb = self._eval_replay_buffer[replay_index] if eval else self._train_replay_buffer[replay_index]
                    rb.add(
                        np.array(transition.action), transition.reward,
                        transition.terminal,
                        transition.timeout, **kwargs)
                    if transition.terminal:
                        rb.add_final(
                            **transition.final_observation)
                new_transitions[name] += 1
                self._new_transitions[
                    'eval_envs' if eval else 'train_envs'] += 1
                self._total_transitions[
                    'eval_envs' if eval else 'train_envs'] += 1
                if self._stat_accumulator is not None:
                    self._stat_accumulator.step(transition, eval)
                self._current_task_id = transition.info["active_task_id"] if eval else -1
            self._internal_env_runner.stored_transitions[:] = []  # Clear list
        return new_transitions

    def _run(self, save_load_lock):
        self._internal_env_runner = _EnvRunner(
            self._train_env, self._eval_env, self._agent, self._timesteps, self._train_envs,
            self._eval_envs, self._rollout_episodes, self._eval_episodes,
            self._training_iterations, self._eval_from_eps_number, self._episode_length, self._kill_signal,
            self._step_signal, self._num_eval_episodes_signal,
            self._eval_epochs_signal, self._eval_report_signal,
            self.log_freq, self._rollout_generator, save_load_lock,
            self.current_replay_ratio, self.target_replay_ratio,
            self._weightsdir, self._logdir,
            self._env_device, self._previous_loaded_weight_folder,
            num_eval_runs=self._num_eval_runs)
        training_envs = self._internal_env_runner.spin_up_envs('train_env', self._train_envs, False)
        eval_envs = self._internal_env_runner.spin_up_envs('eval_env', self._eval_envs, True)
        envs = training_envs + eval_envs
        no_transitions = {env.name: 0 for env in envs}
        while True:
            for p in envs:
                if p.exitcode is not None:
                    envs.remove(p)
                    if p.exitcode != 0:
                        self._internal_env_runner.p_failures[p.name] += 1
                        n_failures = self._internal_env_runner.p_failures[p.name]
                        if n_failures > self._max_fails:
                            logging.error('Env %s failed too many times (%d times > %d)' %
                                          (p.name, n_failures, self._max_fails))
                            raise RuntimeError('Too many process failures.')
                        logging.warning('Env %s failed (%d times <= %d). restarting' %
                                        (p.name, n_failures, self._max_fails))
                        p = self._internal_env_runner.restart_process(p.name)
                        envs.append(p)

            if not self._kill_signal.value:
                new_transitions = self._update()
                for p in envs:
                    if new_transitions[p.name] == 0:
                        no_transitions[p.name] += 1
                    else:
                        no_transitions[p.name] = 0
                    if no_transitions[p.name] > 1200: #600:  # 10min
                        logging.warning("Env %s hangs, so restarting" % p.name)
                        envs.remove(p)
                        os.kill(p.pid, signal.SIGTERM)
                        p = self._internal_env_runner.restart_process(p.name)
                        envs.append(p)
                        no_transitions[p.name] = 0

            if len(envs) == 0:
                break
            time.sleep(1)

    def start(self, save_load_lock):
        self._p = Thread(target=self._run, args=(save_load_lock,), daemon=True)
        self._p.name = 'EnvRunnerThread'
        self._p.start()

    def wait(self):
        if self._p.is_alive():
            self._p.join()

    def stop(self):
        if self._p.is_alive():
            self._kill_signal.value = True
            self._p.join()

    def set_step(self, step):
        self._step_signal.value = step

    def set_eval_report(self, report):
        self._eval_report_signal.value = report

    def set_eval_epochs(self, epochs):
        self._eval_epochs_signal.value = epochs

