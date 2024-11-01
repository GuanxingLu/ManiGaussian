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

from termcolor import cprint
from tqdm import tqdm
import wandb
from lightning.fabric import Fabric


class OfflineTrainRunner():

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
                 rank: int = None,
                 world_size: int = None,
                 cfg: DictConfig = None,
                 fabric: Fabric = None):
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
        self._rank = rank
        self._world_size = world_size
        self._fabric = fabric
        
        self.tqdm_mininterval = cfg.framework.tqdm_mininterval
        self.use_wandb = cfg.framework.use_wandb

        if self.use_wandb and rank == 0:
            print(f"wandb init in {cfg.framework.wandb_project}/{cfg.framework.wandb_group}/{cfg.framework.seed}")
            # wandb.init(project=cfg.framework.wandb_project, group=str(cfg.framework.wandb_group), name=str(cfg.framework.seed), 
            #         config=cfg)
            wandb_name = str(cfg.framework.seed) if cfg.framework.wandb_name is None else cfg.framework.wandb_name
            wandb.init(project=cfg.framework.wandb_project, group=str(cfg.framework.wandb_group), name=wandb_name, 
                    config=cfg)
            cprint(f'[wandb] init in {cfg.framework.wandb_project}/{cfg.framework.wandb_group}/{wandb_name}', 'cyan')

    
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

    def _step(self, i, sampled_batch, **kwargs):
        update_dict = self._agent.update(i, sampled_batch, **kwargs)
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
        # try:
        sampled_batch = next(data_iter) # may raise StopIteration
        # print error and restart data iter
        # except Exception as e:
        #     cprint(e, 'red')
        #     # FIXME: this is a pretty bad hack...
        #     cprint("restarting data iter...", 'red')
        #     return self.preprocess_data(data_iter)
        
        batch = {k: v.to(self._train_device) for k, v in sampled_batch.items() if type(v) == torch.Tensor}
        batch['nerf_multi_view_rgb'] = sampled_batch['nerf_multi_view_rgb'] # [bs, 1, 21]
        batch['nerf_multi_view_depth'] = sampled_batch['nerf_multi_view_depth']
        batch['nerf_multi_view_camera'] = sampled_batch['nerf_multi_view_camera'] # must!!!
        batch['lang_goal'] = sampled_batch['lang_goal']

        if 'nerf_next_multi_view_rgb' in sampled_batch:
            batch['nerf_next_multi_view_rgb'] = sampled_batch['nerf_next_multi_view_rgb']
            batch['nerf_next_multi_view_depth'] = sampled_batch['nerf_next_multi_view_depth']
            batch['nerf_next_multi_view_camera'] = sampled_batch['nerf_next_multi_view_camera']
        
        if len(batch['nerf_multi_view_rgb'].shape) == 3:
            batch['nerf_multi_view_rgb'] = batch['nerf_multi_view_rgb'].squeeze(1)
            batch['nerf_multi_view_depth'] = batch['nerf_multi_view_depth'].squeeze(1)
            batch['nerf_multi_view_camera'] = batch['nerf_multi_view_camera'].squeeze(1)

            if 'nerf_next_multi_view_rgb' in batch and batch['nerf_next_multi_view_rgb'] is not None:
                batch['nerf_next_multi_view_rgb'] = batch['nerf_next_multi_view_rgb'].squeeze(1)
                batch['nerf_next_multi_view_depth'] = batch['nerf_next_multi_view_depth'].squeeze(1)
                batch['nerf_next_multi_view_camera'] = batch['nerf_next_multi_view_camera'].squeeze(1)
        
        if batch['nerf_multi_view_rgb'] is None or batch['nerf_multi_view_rgb'][0,0] is None:
            if not SILENT:
                cprint('batch[nerf_multi_view_rgb] is None. find next data iter', 'red')
            return self.preprocess_data(data_iter)
        
        return batch

    def start(self):
        logging.getLogger().setLevel(self._logging_level)
        self._agent = copy.deepcopy(self._agent)
        # DDP setup model and optimizer
        if self._fabric is not None:
            self._agent.build(training=True, device=self._train_device, fabric=self._fabric)
        else:
            self._agent.build(training=True, device=self._train_device)

        if self._weightsdir is not None:
            existing_weights = sorted([int(f) for f in os.listdir(self._weightsdir)])
            if (not self._load_existing_weights) or len(existing_weights) == 0:
                self._save_model(0)
                start_iter = 0
            else:
                resume_iteration = existing_weights[-1]
                if self._fabric is not None:
                    self._agent.load_weights(os.path.join(self._weightsdir, str(resume_iteration)), fabric=self._fabric)
                else:
                    self._agent.load_weights(os.path.join(self._weightsdir, str(resume_iteration)))
                start_iter = resume_iteration + 1
                if self._rank == 0:
                    logging.info(f"load weights from {os.path.join(self._weightsdir, str(resume_iteration))} ...")
                    logging.info(f"Resuming training from iteration {resume_iteration} ...")

        dataset = self._wrapped_buffer.dataset()    # <class 'torch.utils.data.dataloader.DataLoader'>

        # DDP setup dataloader
        if self._fabric is not None:
            dataset = self._fabric.setup_dataloaders(dataset)

        data_iter = iter(dataset)

        process = psutil.Process(os.getpid())
        num_cpu = psutil.cpu_count()    # e.g., 255

        for i in tqdm(range(start_iter, self._iterations), mininterval=self.tqdm_mininterval):
            log_iteration = i % self._log_freq == 0 and i > 0

            if log_iteration:
                process.cpu_percent(interval=None)


            t = time.time()

            try:
                batch = self.preprocess_data(data_iter)
            except StopIteration:
                cprint('StopIteration', 'red')
                data_iter = iter(dataset)  # recreate the iterator
                batch = self.preprocess_data(data_iter)
            
            t = time.time()
            if self._fabric is not None:
                loss = self._step(i, batch, fabric=self._fabric)
            else:
                loss = self._step(i, batch)
            step_time = time.time() - t

            if self._rank == 0:
                if log_iteration:

                    logging.info(f"Train Step {i:06d} | Loss: {loss:0.5f} | Step time: {step_time:0.4f} | CWD: {os.getcwd()}")
                    # message = f"Train Step {i:06d} | Loss: {loss:0.5f} | Step time: {step_time:0.4f} | CWD: {os.getcwd()}"
                    # tqdm.write(message)  # Use tqdm.write to log without interrupting the progress bar

                    # SUMMARY =  True # TODO: add summaries here
                    # if SUMMARY:
                    #     agent_summaries = self._agent.update_summaries()
                    #     self._writer.add_summaries(i, agent_summaries)
                    
                if i % self._save_freq == 0 and self._weightsdir is not None:
                    self._save_model(i)

        if self._rank == 0:
            logging.info('Stopping envs ...')

            # self._wrapped_buffer.replay_buffer.shutdown() # HACK: remove this so that your replay files are safe.
