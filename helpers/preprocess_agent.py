import torch
from typing import List
from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary, ImageSummary
from termcolor import cprint


class PreprocessAgent(Agent):
    '''
    normalize rgb, logging
    '''

    def __init__(self,
                 pose_agent: Agent,
                 norm_rgb: bool = True):
        self._pose_agent = pose_agent
        self._norm_rgb = norm_rgb

    def build(self, training: bool, device: torch.device = None, use_ddp: bool = True, **kwargs):
        try:
            self._pose_agent.build(training, device, use_ddp, **kwargs)
        except:
            self._pose_agent.build(training, device, **kwargs)

    def _norm_rgb_(self, x):
        return (x.float() / 255.0) * 2.0 - 1.0

    def update(self, step: int, replay_sample: dict, **kwargs) -> dict:


        nerf_multi_view_rgb = replay_sample['nerf_multi_view_rgb']
        nerf_multi_view_depth = replay_sample['nerf_multi_view_depth']
        nerf_multi_view_camera = replay_sample['nerf_multi_view_camera']

        if 'nerf_next_multi_view_rgb' in replay_sample:
            nerf_next_multi_view_rgb = replay_sample['nerf_next_multi_view_rgb']
            nerf_next_multi_view_depth = replay_sample['nerf_next_multi_view_depth']
            nerf_next_multi_view_camera = replay_sample['nerf_next_multi_view_camera']
        lang_goal = replay_sample['lang_goal']

        if replay_sample['nerf_multi_view_rgb'] is None or replay_sample['nerf_multi_view_rgb'][0,0] is None:
            cprint("preprocess agent no nerf rgb 1", "red")

        replay_sample = {k: v[:, 0] if len(v.shape) > 2 else v for k, v in replay_sample.items()}

        for k, v in replay_sample.items():
            if self._norm_rgb and 'rgb' in k and 'nerf' not in k:
                replay_sample[k] = self._norm_rgb_(v)
            elif 'nerf' in k:
                replay_sample[k] = v
            else:
                try:
                    replay_sample[k] = v.float()
                except:
                    replay_sample[k] = v
                    pass # some elements are not tensors/arrays
        replay_sample['nerf_multi_view_rgb'] = nerf_multi_view_rgb
        replay_sample['nerf_multi_view_depth'] = nerf_multi_view_depth
        replay_sample['nerf_multi_view_camera'] = nerf_multi_view_camera

        if 'nerf_next_multi_view_rgb' in replay_sample:
            replay_sample['nerf_next_multi_view_rgb'] = nerf_next_multi_view_rgb
            replay_sample['nerf_next_multi_view_depth'] = nerf_next_multi_view_depth
            replay_sample['nerf_next_multi_view_camera'] = nerf_next_multi_view_camera
        
        replay_sample['lang_goal'] = lang_goal
        self._replay_sample = replay_sample


        if replay_sample['nerf_multi_view_rgb'] is None or replay_sample['nerf_multi_view_rgb'][0,0] is None:
            cprint("preprocess agent no nerf rgb 2", "red")

        return self._pose_agent.update(step, replay_sample, **kwargs)

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:

        for k, v in observation.items():
            if self._norm_rgb and 'rgb' in k:
                observation[k] = self._norm_rgb_(v)
            else:
                try:
                    observation[k] = v.float()
                except:
                    observation[k] = v
                    pass # some elements are not tensors/arrays
        act_res = self._pose_agent.act(step, observation, deterministic)
        act_res.replay_elements.update({'demo': False})
        return act_res

    def update_summaries(self) -> List[Summary]:
        prefix = 'inputs'
        demo_f = self._replay_sample['demo'].float()
        demo_proportion = demo_f.mean()
        tile = lambda x: torch.squeeze(
            torch.cat(x.split(1, dim=1), dim=-1), dim=1)
        sums = [
            ScalarSummary('%s/demo_proportion' % prefix, demo_proportion),
            HistogramSummary('%s/low_dim_state' % prefix,
                    self._replay_sample['low_dim_state']),
            HistogramSummary('%s/low_dim_state_tp1' % prefix,
                    self._replay_sample['low_dim_state_tp1']),
            ScalarSummary('%s/low_dim_state_mean' % prefix,
                    self._replay_sample['low_dim_state'].mean()),
            ScalarSummary('%s/low_dim_state_min' % prefix,
                    self._replay_sample['low_dim_state'].min()),
            ScalarSummary('%s/low_dim_state_max' % prefix,
                    self._replay_sample['low_dim_state'].max()),
            ScalarSummary('%s/timeouts' % prefix,
                    self._replay_sample['timeout'].float().mean()),
        ]

        if 'sampling_probabilities' in self._replay_sample:
            sums.extend([
                HistogramSummary('replay/priority',
                                 self._replay_sample['sampling_probabilities']),
            ])
        sums.extend(self._pose_agent.update_summaries())
        return sums

    def update_wandb_summaries(self):
        return self._pose_agent.update_wandb_summaries()
        
    def act_summaries(self) -> List[Summary]:
        return self._pose_agent.act_summaries()

    def load_weights(self, savedir: str):
        self._pose_agent.load_weights(savedir)

    def save_weights(self, savedir: str):
        self._pose_agent.save_weights(savedir)

    def reset(self) -> None:
        self._pose_agent.reset()

    def load_clip(self):
        self._pose_agent.load_clip()

    def unload_clip(self):
        self._pose_agent.unload_clip()
