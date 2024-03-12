import copy
import logging
import os
import shutil
import signal
import sys
import threading
import time
from typing import Optional, List
from typing import Union

from omegaconf import DictConfig
import gc
import numpy as np
import psutil
import torch
import pandas as pd
from yarr.agents.agent import Agent
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from yarr.runners.env_runner import EnvRunner
from yarr.runners.train_runner import TrainRunner
from yarr.utils.log_writer import LogWriter
from yarr.utils.stat_accumulator import StatAccumulator
from yarr.replay_buffer.prioritized_replay_buffer import PrioritizedReplayBuffer

import wandb

from termcolor import colored, cprint

os.environ['WANDB_SILENT'] = 'true'

class OfflineTrainRunnerSingleProcess():

    def __init__(self,
                 agent: Agent,
                 wrapped_replay_buffer: PyTorchReplayBuffer,
                 train_device: torch.device,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 iterations: int = int(6e6),
                 logdir: str = '/tmp/yarr/logs',
                 logging_level: int = logging.INFO,
                 log_freq: int = 10,
                 weightsdir: str = '/tmp/yarr/weights',
                 num_weights_to_keep: int = 60,
                 save_freq: int = 100,
                 tensorboard_logging: bool = True,
                 csv_logging: bool = False,
                 load_existing_weights: bool = True,
                 wandb_logger: object = None,
                 world_size: int = None,
                 cfg: DictConfig = None):
        self._agent = agent
        self._wrapped_buffer = wrapped_replay_buffer
        self._stat_accumulator = stat_accumulator
        self._iterations = iterations
        self._logdir = logdir
        self._logging_level = logging_level
        self._log_freq = log_freq
        self._weightsdir = weightsdir
        self._num_weights_to_keep = num_weights_to_keep
        self._save_freq = save_freq

        self._wrapped_buffer = wrapped_replay_buffer
        self._train_device = train_device
        self._tensorboard_logging = tensorboard_logging
        self._csv_logging = csv_logging
        self._load_existing_weights = load_existing_weights
        self._rank = 0
        self._world_size = world_size

        self._writer = None
        # self.wandb_logger = wandb_logger
        self.use_wandb = cfg.framework.use_wandb

        if self.use_wandb:
            
            wandb.init(project=cfg.framework.wandb_project, group=str(cfg.framework.wandb_group), name=str(cfg.framework.seed), 
                    config=cfg)
            cprint(f'[wandb] init in {cfg.framework.wandb_project}/{cfg.framework.wandb_group}/{cfg.framework.seed}', 'cyan')

        self.cfg = cfg

        
        self._use_online_evaluation = cfg.framework.use_online_evaluation
        cprint(f'[wandb] use_online_evaluation: {self._use_online_evaluation}', 'cyan')
        # if logdir is None:
        #     logging.info("'logdir' was None. No logging will take place.")
        # else:
        #     self._writer = LogWriter(
        #         self._logdir, tensorboard_logging, csv_logging)

        if weightsdir is None:
            logging.info(
                "'weightsdir' was None. No weight saving will take place.")
        else:
            os.makedirs(self._weightsdir, exist_ok=True)

    def _save_model(self, i):
        d = os.path.join(self._weightsdir, str(i))
        os.makedirs(d, exist_ok=True)
        self._agent.save_weights(d)

        # remove oldest save
        prev_dir = os.path.join(self._weightsdir, str(
            i - self._save_freq * self._num_weights_to_keep))
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)

    def _step(self, i, sampled_batch):
        update_dict = self._agent.update(i, sampled_batch)
        total_losses = update_dict['total_losses'].item()
        return total_losses

    def _get_resume_eval_epoch(self):
        starting_epoch = 0
        eval_csv_file = self._weightsdir.replace('weights', 'eval_data.csv') # TODO(mohit): check if it's supposed be 'env_data.csv'
        if os.path.exists(eval_csv_file):
             eval_dict = pd.read_csv(eval_csv_file).to_dict()
             epochs = list(eval_dict['step'].values())
             return epochs[-1] if len(epochs) > 0 else starting_epoch
        else:
            return starting_epoch
    
    def preprocess_data(self, data_iter, SILENT=True):
        try:
            sampled_batch = next(data_iter)
        except Exception as e:
            # this is a pretty bad hack...
            cprint(e, 'red')
            cprint("data iter bug. let's try again...", 'red')
            return self.preprocess_data(data_iter)
        
        batch = {k: v.to(self._train_device) for k, v in sampled_batch.items() if type(v) == torch.Tensor}
        batch['nerf_multi_view_rgb'] = sampled_batch['nerf_multi_view_rgb']
        batch['nerf_multi_view_depth'] = sampled_batch['nerf_multi_view_depth']
        batch['nerf_multi_view_camera'] = sampled_batch['nerf_multi_view_camera'] # must!!!
        batch['lang_goal'] = sampled_batch['lang_goal']
        
        if len(batch['nerf_multi_view_rgb'].shape) == 3:
            batch['nerf_multi_view_rgb'] = batch['nerf_multi_view_rgb'][0]
            batch['nerf_multi_view_depth'] = batch['nerf_multi_view_depth'][0]
            batch['nerf_multi_view_camera'] = batch['nerf_multi_view_camera'][0]
        
        if batch['nerf_multi_view_rgb'] is None or batch['nerf_multi_view_rgb'][0,0] is None:
            if not SILENT:
                cprint('batch[nerf_multi_view_rgb] is None. find next data iter', 'red')
            return self.preprocess_data(data_iter)
        return batch
    
    def start(self):
        logging.getLogger().setLevel(self._logging_level)
        self._agent = copy.deepcopy(self._agent)
        self._agent.build(training=True, device=self._train_device, use_ddp=False)

        if self._weightsdir is not None:
            existing_weights = sorted([int(f) for f in os.listdir(self._weightsdir)])
            if (not self._load_existing_weights) or len(existing_weights) == 0:
                self._save_model(0)
                start_iter = 0
            else:
                resume_iteration = existing_weights[-1]
                self._agent.load_weights(os.path.join(self._weightsdir, str(resume_iteration)))
                start_iter = resume_iteration + 1

                logging.info(f"Resuming training from iteration {resume_iteration} ...")

        dataset = self._wrapped_buffer.dataset()
        data_iter = iter(dataset)
        

        process = psutil.Process(os.getpid())
        num_cpu = psutil.cpu_count()
        

        # sampled_batch = next(data_iter) # debug

        # use this to compute pvr-style success rate
        self.pvr_success_rate = {}
        for task in self.cfg.rlbench.tasks:
            self.pvr_success_rate[task] = [0., 0., 0.]

        every_100_iter_time = time.time()
        for i in range(start_iter, self._iterations):

            log_iteration = i % self._log_freq == 0 and i > 0
            eval_iteration = i % self.cfg.evaluation.eval_freq == 0 and i > 0
            eval_episodes = self.cfg.evaluation.eval_episodes

            if log_iteration:
                process.cpu_percent(interval=None)
            
           
            t = time.time()
            

            batch = self.preprocess_data(data_iter)
            

            sample_time = time.time() - t


            t = time.time()
            
            loss = self._step(i, batch)

            step_time = time.time() - t


            if log_iteration:
                logging.info(f"Train Step {i:06d} | Loss: {loss:0.5f} | Sample time: {sample_time:0.6f} | Step time: {step_time:0.4f}.")


            if log_iteration and self.use_wandb:
                wandb.log({'train/loss': loss, 'train/step_time': step_time}, step=i)


            if i % self._save_freq == 0 and self._weightsdir is not None:
                self._save_model(i)
            
            if eval_iteration and self._use_online_evaluation:
                self.evaluate_agent(agent=self._agent, iteration=i)
            

        # if self._writer is not None:
            # self._writer.close()
        logging.info('Stopping envs ...')

        self._wrapped_buffer.replay_buffer.shutdown()


    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype
    

    def evaluate_agent(self, agent, iteration):
        """
        a simple evaluation function
        """
        
        from rlbench.backend import task as rlbench_task
        from rlbench.backend.utils import task_file_to_task_class
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import Discrete
        from helpers import utils

        """
        prepare some configs
        """
        eval_cfg = self.cfg.evaluation

        task_files = [t.replace('.py', '') for t in os.listdir(rlbench_task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
        
        obs_config = utils.create_obs_config(self.cfg.rlbench.cameras,
                                        self.cfg.rlbench.camera_resolution,
                                        self.cfg.method.name)
        
        gripper_mode = Discrete()
        arm_action_mode = EndEffectorPoseViaPlanning()
        action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

        # single-task or multi-task
        if len(self.cfg.rlbench.tasks) > 1:
            tasks = self.cfg.rlbench.tasks
            multi_task = True

            task_classes = []
            for task in tasks:
                if task not in task_files:
                    raise ValueError('Task %s not recognised!.' % task)
                task_classes.append(task_file_to_task_class(task))

            env_config = (task_classes,
                        obs_config,
                        action_mode,
                        self.cfg.rlbench.demo_path,
                        self.cfg.rlbench.episode_length,
                        self.cfg.rlbench.headless,
                        eval_cfg.eval_episodes,
                        self.cfg.rlbench.include_lang_goal_in_obs,
                        eval_cfg.time_in_state,
                        eval_cfg.record_every_n)
        else:
            task = self.cfg.rlbench.tasks[0]
            multi_task = False

            if task not in task_files:
                raise ValueError('Task %s not recognised!.' % task)
            task_class = task_file_to_task_class(task)

            env_config = (task_class,
                        obs_config,
                        action_mode,
                        self.cfg.rlbench.demo_path,
                        eval_cfg.episode_length,
                        self.cfg.rlbench.headless,
                        self.cfg.rlbench.include_lang_goal_in_obs,
                        eval_cfg.time_in_state,
                        eval_cfg.record_every_n)
        
        """
        create env
        """
        from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
        multi_task = isinstance(env_config[0], list)
        if multi_task:
            eval_env = CustomMultiTaskRLBenchEnv(
                task_classes=env_config[0],
                observation_config=env_config[1],
                action_mode=env_config[2],
                dataset_root=env_config[3],
                episode_length=env_config[4],
                headless=env_config[5],
                swap_task_every=env_config[6],
                include_lang_goal_in_obs=env_config[7],
                time_in_state=env_config[8],
                record_every_n=env_config[9])
        else:
            eval_env = CustomRLBenchEnv(
                task_class=env_config[0],
                observation_config=env_config[1],
                action_mode=env_config[2],
                dataset_root=env_config[3],
                episode_length=env_config[4],
                headless=env_config[5],
                include_lang_goal_in_obs=env_config[6],
                time_in_state=env_config[7],
                record_every_n=env_config[8])
        
        eval_env.launch()
        cprint("Evaluation env launched!", "cyan")


        """
        run evaluation
        """
        num_episodes = eval_cfg.eval_episodes
        episode_length = eval_cfg.episode_length
        timesteps = 1

        env_device = utils.get_device(eval_cfg.gpu)
        device_idx = eval_cfg.gpu
        cprint("Evaluation device: %s" % env_device, "cyan")
        # during training, we do not load clip. so we need to load it here
        agent.load_clip() 
        
        success_rate_dict = {}
        all_task_time = time.time()
        for task_id in range(len(eval_env._task_classes)):
            
            cur_task_name = self.cfg.rlbench.tasks[task_id]
            eval_env.set_task(cur_task_name)
            cur_task_name = eval_env._task.get_name()
            avg_SR = 0.

            task_eval_time = time.time()
            for episode_id in range(num_episodes):
                episode_start_time = time.time()

                # set variation change
                possible_variations = eval_env._task.variation_count()
                variation = episode_id % possible_variations
                eval_env._task.set_variation(variation)
                
                # reset env
                obs = eval_env.reset() # TODO: use demo seed to reset
                
                agent.reset()
                obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}

                
                for step in range(episode_length):
                    """
                    prepped_data time w/o np: 0.012056589126586914
                    prepped_data time w np: 0.000331878662109375

                    use np to convert list, it's much faster!
                    """

                    # prepped_data = {k:torch.tensor([v], device=env_device) for k, v in obs_history.items()}
                    prepped_data = {k:torch.tensor(np.array([v]), device=env_device) for k, v in obs_history.items()}

                    with torch.no_grad():
                        act_result = agent.act(step=666, observation=prepped_data, deterministic=True)

                    # main cost come from env
                    # start_time = time.time()
                    transition = eval_env.step(act_result)
                    # cprint(f"step time: {time.time() - start_time}", "cyan")


                    for k in obs_history.keys():
                        obs_history[k].append(transition.observation[k])
                        obs_history[k].pop(0)


                    timeout = False
                    if step == episode_length - 1:
                        # If last transition, and not terminal, then we timed out
                        timeout = not transition.terminal
                        if timeout:
                            transition.terminal = True
                            if "needs_reset" in transition.info:
                                transition.info["needs_reset"] = True

                # we use last state's reward as the success measure of the episode
                reward = transition.reward
                avg_SR += reward
                time_cost = time.time() - episode_start_time
                lang_goal = eval_env._lang_goal
                cprint(f"Task {cur_task_name} | Episode {episode_id} | Reward: {reward} | Time cost: {time_cost:.2f} | Lang goal: {lang_goal}", "cyan")
        

            task_eval_time = time.time() - task_eval_time
            success_rate_dict[cur_task_name] = avg_SR/num_episodes

            self.pvr_success_rate[cur_task_name].append(avg_SR/num_episodes)
            
            cprint(f"Task {cur_task_name} finished.", "cyan")

            if self.use_wandb:
                wandb.log({f'eval/{cur_task_name}_SR': avg_SR/num_episodes}, step=iteration)
                wandb.log({f'eval/{cur_task_name}_time': task_eval_time}, step=iteration)

                # pick best 3 SR and log
                best_3_SR = sorted(self.pvr_success_rate[cur_task_name], reverse=True)[:3]
                avg_SR = sum(best_3_SR) / len(best_3_SR)
                wandb.log({f'eval/{cur_task_name}_SR_pvr': avg_SR}, step=iteration)


        all_task_time = time.time() - all_task_time
        # second to minute
        all_task_time = all_task_time / 60
        cprint(f"iter {iteration} | time cost: {all_task_time:.2f} min | success rate: {success_rate_dict}", "cyan")
        # unload clip after evaluation
        agent.unload_clip()
        eval_env.shutdown()

