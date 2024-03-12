import copy
import logging
import os
import time
import pandas as pd

from multiprocessing import Process, Manager
from multiprocessing import get_start_method, set_start_method
from typing import Any

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.agents.agent import ScalarSummary
from yarr.agents.agent import Summary
from yarr.envs.env import Env
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.log_writer import LogWriter
from yarr.utils.process_str import change_case
from yarr.utils.video_utils import CircleCameraMotion, TaskRecorder

from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import RenderMode

from yarr.runners._env_runner import _EnvRunner


class _IndependentEnvRunner(_EnvRunner):

    def __init__(self,
                 train_env: Env,
                 eval_env: Env,
                 agent: Agent,
                 timesteps: int,
                 train_envs: int,
                 eval_envs: int,
                 rollout_episodes: int,
                 eval_episodes: int,
                 training_iterations: int,
                 eval_from_eps_number: int,
                 episode_length: int,
                 kill_signal: Any,
                 step_signal: Any,
                 num_eval_episodes_signal: Any,
                 eval_epochs_signal: Any,
                 eval_report_signal: Any,
                 log_freq: int,
                 rollout_generator: RolloutGenerator,
                 save_load_lock,
                 current_replay_ratio,
                 target_replay_ratio,
                 weightsdir: str = None,
                 logdir: str = None,
                 env_device: torch.device = None,
                 previous_loaded_weight_folder: str = '',
                 num_eval_runs: int = 1,
                 ):

            super().__init__(train_env, eval_env, agent, timesteps,
                             train_envs, eval_envs, rollout_episodes, eval_episodes,
                             training_iterations, eval_from_eps_number, episode_length,
                             kill_signal, step_signal, num_eval_episodes_signal,
                             eval_epochs_signal, eval_report_signal, log_freq,
                             rollout_generator, save_load_lock, current_replay_ratio,
                             target_replay_ratio, weightsdir, logdir, env_device,
                             previous_loaded_weight_folder, num_eval_runs)

    def _load_save(self):
        if self._weightsdir is None:
            logging.info("'weightsdir' was None, so not loading weights.")
            return
        while True:
            weight_folders = []
            with self._save_load_lock:
                if os.path.exists(self._weightsdir):
                    weight_folders = os.listdir(self._weightsdir)
                if len(weight_folders) > 0:
                    weight_folders = sorted(map(int, weight_folders))
                    # only load if there has been a new weight saving
                    if self._previous_loaded_weight_folder != weight_folders[-1]:
                        self._previous_loaded_weight_folder = weight_folders[-1]
                        d = os.path.join(self._weightsdir, str(weight_folders[-1]))
                        try:
                            self._agent.load_weights(d)
                        except FileNotFoundError:
                            # rare case when agent hasn't finished writing.
                            time.sleep(1)
                            self._agent.load_weights(d)
                        logging.info('Agent %s: Loaded weights: %s' % (self._name, d))
                        self._new_weights = True
                    else:
                        self._new_weights = False
                    break
            logging.info('Waiting for weights to become available.')
            time.sleep(1)

    def _get_task_name(self):
        if hasattr(self._eval_env, '_task_class'):
            eval_task_name = change_case(self._eval_env._task_class.__name__)
            multi_task = False
        elif hasattr(self._eval_env, '_task_classes'):
            if self._eval_env.active_task_id != -1:
                task_id = (self._eval_env.active_task_id) % len(self._eval_env._task_classes)
                eval_task_name = change_case(self._eval_env._task_classes[task_id].__name__)
            else:
                eval_task_name = ''
            multi_task = True
        else:
            raise Exception('Neither task_class nor task_classes found in eval env')
        return eval_task_name, multi_task

    def _run_eval_independent(self, name: str,
                              stats_accumulator,
                              weight,
                              writer_lock,
                              eval=True,
                              device_idx=0,
                              save_metrics=True,
                              cinematic_recorder_cfg=None,
                              novel_command=None):

        self._name = name
        self._save_metrics = save_metrics
        self._is_test_set = type(weight) == dict

        self._agent = copy.deepcopy(self._agent)

        # device = torch.device('cuda:%d' % device_idx) if torch.cuda.device_count() > 1 else torch.device('cuda:0')
        device = torch.device('cuda:%d' % device_idx)
        print("Device count: %d" % torch.cuda.device_count())
        print('Using device: %s' % device)

        with writer_lock: # hack to prevent multiple CLIP downloads ... argh should use a separate lock
            self._agent.build(training=False, device=device)

        logging.info('%s: Launching env.' % name)
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self._agent)

        env = self._eval_env
        env.eval = eval
        env.launch()

        # initialize cinematic recorder if specified
        rec_cfg = cinematic_recorder_cfg
        if rec_cfg.enabled:
            cam_placeholder = Dummy('cam_cinematic_placeholder')

            cam = VisionSensor.create(rec_cfg.camera_resolution, render_mode=RenderMode.OPENGL)
            cam.set_pose(cam_placeholder.get_pose())
            cam.set_parent(cam_placeholder)

            cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), rec_cfg.rotate_speed)
            task_recorder = TaskRecorder(env, cam_motion, fps=rec_cfg.fps)
            for _ in range(200):
                task_recorder._cam_motion.step()
            task_recorder._cam_motion.save_pose()

            env.env._action_mode.arm_action_mode.set_callable_each_step(task_recorder.take_snap)

        if not os.path.exists(self._weightsdir):
            raise Exception('No weights directory found.')

        # to save or not to save evaluation metrics (set as False for recording videos)
        if self._save_metrics:
            csv_file = 'eval_data_novel_command.csv' if not self._is_test_set else 'test_data.csv'
            writer = LogWriter(self._logdir, True, True,
                               env_csv=csv_file)

        # one weight for all tasks (used for validation)
        if type(weight) == int:
            logging.info('Evaluating weight %s' % weight)
            weight_path = os.path.join(self._weightsdir, str(weight))
            seed_path = self._weightsdir.replace('/weights', '')
            self._agent.load_weights(weight_path)
            weight_name = str(weight)

        new_transitions = {'train_envs': 0, 'eval_envs': 0}
        total_transitions = {'train_envs': 0, 'eval_envs': 0}
        current_task_id = -1

        for n_eval in range(self._num_eval_runs):
            if rec_cfg.enabled:
                task_recorder._cam_motion.save_pose()

            # best weight for each task (used for test evaluation)
            if type(weight) == dict:
                task_name = list(weight.keys())[n_eval]
                task_weight = weight[task_name]
                weight_path = os.path.join(self._weightsdir, str(task_weight))
                seed_path = self._weightsdir.replace('/weights', '')
                self._agent.load_weights(weight_path)
                weight_name = str(task_weight)
                print('Evaluating weight %s for %s' % (weight_name, task_name))

            # evaluate on N tasks * M episodes per task = total eval episodes
            for ep in range(self._eval_episodes):
                start_time = time.time()
                eval_demo_seed = ep + self._eval_from_eps_number
                logging.info('%s: Starting episode %d, seed %d.' % (name, ep, eval_demo_seed))

                # the current task gets reset after every M episodes
                episode_rollout = []

                generator = self._rollout_generator.generator(
                    self._step_signal, env, self._agent,
                    self._episode_length, self._timesteps,
                    eval, eval_demo_seed=eval_demo_seed,
                    record_enabled=rec_cfg.enabled,
                    novel_command=novel_command)
                try:
                    for replay_transition in generator:
                        while True:
                            if self._kill_signal.value:
                                env.shutdown()
                                return
                            if (eval or self._target_replay_ratio is None or
                                    self._step_signal.value <= 0 or (
                                            self._current_replay_ratio.value >
                                            self._target_replay_ratio)):
                                break
                            time.sleep(1)
                            logging.debug(
                                'Agent. Waiting for replay_ratio %f to be more than %f' %
                                (self._current_replay_ratio.value, self._target_replay_ratio))

                        with self.write_lock:
                            if len(self.agent_summaries) == 0:
                                # Only store new summaries if the previous ones
                                # have been popped by the main env runner.
                                for s in self._agent.act_summaries():
                                    self.agent_summaries.append(s)
                        episode_rollout.append(replay_transition)
                        # print("len(episode_rollout): ", len(episode_rollout))
                        
                except StopIteration as e:
                    continue
                except Exception as e:
                    env.shutdown()
                    raise e

                with self.write_lock:
                    for transition in episode_rollout:
                        self.stored_transitions.append((name, transition, eval))

                        new_transitions['eval_envs'] += 1
                        total_transitions['eval_envs'] += 1
                        stats_accumulator.step(transition, eval)
                        current_task_id = transition.info['active_task_id']


                self._num_eval_episodes_signal.value += 1

                task_name, _ = self._get_task_name()
                reward = episode_rollout[-1].reward
                lang_goal = env._lang_goal
                time_cost = time.time() - start_time
                print(f"Evaluating {task_name} | Episode {ep} | Score: {reward} | Lang Goal: {lang_goal} | Time: {time_cost:.2f}s")
                
                # # for debug
                # if reward > 0.99:
                #     exit()
                    
                # save recording
                if rec_cfg.enabled:
                    success = reward > 0.99
                    record_file = os.path.join(seed_path, 'videos',
                                               '%s_w%s_s%s_%s.mp4' % (task_name,
                                                                      weight_name,
                                                                      eval_demo_seed,
                                                                      'succ' if success else 'fail'))

                    lang_goal = self._eval_env._lang_goal

                    task_recorder.save(record_file, lang_goal, reward)
                    task_recorder._cam_motion.restore_pose()

            # report summaries
            summaries = []
            summaries.extend(stats_accumulator.pop())

            eval_task_name, multi_task = self._get_task_name()

            if eval_task_name and multi_task:
                for s in summaries:
                    if 'eval' in s.name:
                        s.name = '%s/%s' % (s.name, eval_task_name)

            if len(summaries) > 0:
                try:
                    if multi_task:
                        task_score = [s.value for s in summaries if f'eval_envs/return/{eval_task_name}' in s.name][0]
                    else:
                        task_score = [s.value for s in summaries if f'eval_envs/return' in s.name][0]
                except:
                    task_score = "unknown"
            else:
                task_score = "unknown"

            print(f"Finished {eval_task_name} | Final Score: {task_score}\n")

            if self._save_metrics:
                with writer_lock:
                    writer.add_summaries(weight_name, summaries)

            self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
            self.agent_summaries[:] = []
            self.stored_transitions[:] = []

        if self._save_metrics:
            with writer_lock:
                writer.end_iteration()

        logging.info('Finished evaluation.')
        env.shutdown()

    def kill(self):
        self._kill_signal.value = True
