import os
import pickle
import gc
import logging
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from rlbench import CameraConfig, ObservationConfig
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from yarr.runners.offline_train_runner import OfflineTrainRunner
from yarr.utils.stat_accumulator import SimpleAccumulator

from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
import torch.distributed as dist

from termcolor import cprint
import lightning as L


def run_seed(
            rank,
            cfg: DictConfig,
            obs_config: ObservationConfig,
            cams,
            multi_task,
            seed,
            world_size,
            fabric: L.Fabric = None,
            ) -> None:
    
    if fabric is not None:
        rank = fabric.global_rank
    else:
        dist.init_process_group("gloo",
                        rank=rank,
                        world_size=world_size)

    task = cfg.rlbench.tasks[0]
    tasks = cfg.rlbench.tasks

    replay_path = os.path.join(cfg.replay.path, 'seed%d' % seed)
    
    if cfg.method.name == 'GNFACTOR_BC':
        from agents import gnfactor_bc
        replay_buffer = gnfactor_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution,
            cfg=cfg)

        gnfactor_bc.launch_utils.fill_multi_task_replay(
            cfg, obs_config, 0,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams, cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes, cfg.method.bounds_offset,
            cfg.method.rotation_resolution, cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method
        )

        agent = gnfactor_bc.launch_utils.create_agent(cfg)
        
    elif cfg.method.name == 'ManiGaussian_BC':
        from agents import manigaussian_bc
        replay_buffer = manigaussian_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution,
            cfg=cfg)

        manigaussian_bc.launch_utils.fill_multi_task_replay(
            cfg, obs_config, 0,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams, cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes, cfg.method.bounds_offset,
            cfg.method.rotation_resolution, cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method,
            fabric=fabric,
        )

        agent = manigaussian_bc.launch_utils.create_agent(cfg)
    
    elif cfg.method.name == 'PERACT_BC':
        from agents import peract_bc
        replay_buffer = peract_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution)

        peract_bc.launch_utils.fill_multi_task_replay(
            cfg, obs_config, rank,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams, cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes, cfg.method.bounds_offset,
            cfg.method.rotation_resolution, cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method)

        agent = peract_bc.launch_utils.create_agent(cfg)
    else:
        raise ValueError('Method %s does not exists.' % cfg.method.name)

    wrapped_replay = PyTorchReplayBuffer(replay_buffer, num_workers=cfg.framework.num_workers)
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(cwd, 'seed%d' % seed, 'weights')  # load from the last checkpoint

    logdir = os.path.join(cwd, 'seed%d' % seed)

    cprint(f'Project path: {weightsdir}', 'cyan')

    train_runner = OfflineTrainRunner(
        agent=agent,
        wrapped_replay_buffer=wrapped_replay,
        train_device=rank,
        stat_accumulator=stat_accum,
        iterations=cfg.framework.training_iterations,
        logdir=logdir,
        logging_level=cfg.framework.logging_level,
        log_freq=cfg.framework.log_freq,
        weightsdir=weightsdir,
        num_weights_to_keep=cfg.framework.num_weights_to_keep,
        save_freq=cfg.framework.save_freq,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        load_existing_weights=cfg.framework.load_existing_weights,
        rank=rank,
        world_size=world_size,
        cfg=cfg,
        fabric=fabric)
    cprint('Starting training!!', 'green')
    train_runner.start()

    del train_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()