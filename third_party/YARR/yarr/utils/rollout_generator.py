from multiprocessing import Value

import numpy as np
import torch, torchvision
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition

# for cam view change
from yarr.utils.video_utils import CircleCameraMotion, StaticCameraMotion
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

class RolloutGenerator(object):

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False, novel_command=None, cam_view_change=False):
        if eval:
            obs = env.reset_to_demo(eval_demo_seed, novel_command=novel_command)
        else:
            env.set_variation(eval_demo_seed)
            obs = env.reset(novel_command=novel_command)

        
        if cam_view_change:
            cam_placeholder = Dummy('cam_cinematic_base')
            cam_front = VisionSensor('cam_front')
            env._task._scene._cam_front.set_pose(cam_front.get_pose())
            env._task._scene._cam_front.set_parent(cam_placeholder)
            rotate_speed = 0.005
            cam_motion = CircleCameraMotion(cam=env._task._scene._cam_front, origin=Dummy('cam_cinematic_base'), speed=rotate_speed)
            cam_motion.step()
        
        agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        lang_goal = env._task.get_task_descriptions()[0]
        for step in range(episode_length):

            # prepped_data = {k:torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
            prepped_data = {k:torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}
            prepped_data['lang_goal'] = lang_goal
            # save img
            # torchvision.utils.save_image(prepped_data['front_rgb'][0].div(255), f'{step}_camview.png')

            act_result = agent.act(step_signal.value, prepped_data,
                                   deterministic=eval)  # step_signal.value is step

            # print("action: ", act_result.action)


            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            transition = env.step(act_result)
            if cam_view_change:
                cam_motion.step()
                # save img
                # rgb = cam_motion.cam.capture_rgb()
                # torchvision.utils.save_image(torch.tensor(rgb).permute(2,0,1), f'{step}_camview2.png')
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    # prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    prepped_data = {k: torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}
                    prepped_data['lang_goal'] = lang_goal
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                steps=60, step_scene=True)

            obs = dict(transition.observation)
            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
