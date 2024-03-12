from typing import Type, List

import numpy as np
try:
    from rlbench import ObservationConfig, Environment, CameraConfig
except (ModuleNotFoundError, ImportError) as e:
    print("You need to install RLBench: 'https://github.com/stepjam/RLBench'")
    raise e
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task

from clip import tokenize

from yarr.envs.env import Env, MultiTaskEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition
from yarr.utils.process_str import change_case


ROBOT_STATE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
                        'gripper_open', 'gripper_pose',
                        'gripper_joint_positions', 'gripper_touch_forces',
                        'task_low_dim_state', 'misc']

def _extract_obs(obs: Observation, channels_last: bool, observation_config):
    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    robot_state = obs.get_low_dim_data()
    # Remove all of the individual state elements
    obs_dict = {k: v for k, v in obs_dict.items()
                if k not in ROBOT_STATE_KEYS}
    if not channels_last:
        # Swap channels from last dim to 1st dim
        obs_dict = {k: np.transpose(
            v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                    for k, v in obs_dict.items()}
    else:
        # Add extra dim to depth data
        obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
                    for k, v in obs_dict.items()}
    obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)
    obs_dict['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)
    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
        obs_dict[k] = v.astype(np.float32)

    for config, name in [
        (observation_config.left_shoulder_camera, 'left_shoulder'),
        (observation_config.right_shoulder_camera, 'right_shoulder'),
        (observation_config.front_camera, 'front'),
        (observation_config.wrist_camera, 'wrist'),
        (observation_config.overhead_camera, 'overhead')]:
        if config.point_cloud:
            obs_dict['%s_camera_extrinsics' % name] = obs.misc['%s_camera_extrinsics' % name]
            obs_dict['%s_camera_intrinsics' % name] = obs.misc['%s_camera_intrinsics' % name]
    return obs_dict


def _get_cam_observation_elements(camera: CameraConfig, prefix: str, channels_last):
    elements = []
    img_s = list(camera.image_size)
    shape = img_s + [3] if channels_last else [3] + img_s
    if camera.rgb:
        elements.append(
            ObservationElement('%s_rgb' % prefix, shape, np.uint8))
    if camera.point_cloud:
        elements.append(
            ObservationElement('%s_point_cloud' % prefix, shape, np.float32))
        elements.append(
            ObservationElement('%s_camera_extrinsics' % prefix, (4, 4),
                               np.float32))
        elements.append(
            ObservationElement('%s_camera_intrinsics' % prefix, (3, 3),
                               np.float32))
    if camera.depth:
        shape = img_s + [1] if channels_last else [1] + img_s
        elements.append(
            ObservationElement('%s_depth' % prefix, shape, np.float32))
    if camera.mask:
        raise NotImplementedError()

    return elements


def _observation_elements(observation_config, channels_last) -> List[ObservationElement]:
    elements = []
    robot_state_len = 0
    if observation_config.joint_velocities:
        robot_state_len += 7
    if observation_config.joint_positions:
        robot_state_len += 7
    if observation_config.joint_forces:
        robot_state_len += 7
    if observation_config.gripper_open:
        robot_state_len += 1
    if observation_config.gripper_pose:
        robot_state_len += 7
    if observation_config.gripper_joint_positions:
        robot_state_len += 2
    if observation_config.gripper_touch_forces:
        robot_state_len += 2
    if observation_config.task_low_dim_state:
        raise NotImplementedError()
    if robot_state_len > 0:
        elements.append(ObservationElement(
            'low_dim_state', (robot_state_len,), np.float32))
    elements.extend(_get_cam_observation_elements(
        observation_config.left_shoulder_camera, 'left_shoulder', channels_last))
    elements.extend(_get_cam_observation_elements(
        observation_config.right_shoulder_camera, 'right_shoulder', channels_last))
    elements.extend(_get_cam_observation_elements(
        observation_config.front_camera, 'front', channels_last))
    elements.extend(_get_cam_observation_elements(
        observation_config.wrist_camera, 'wrist', channels_last))
    return elements


class RLBenchEnv(Env):

    def __init__(self, task_class: Type[Task],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 dataset_root: str = '',
                 channels_last=False,
                 headless=True,
                 include_lang_goal_in_obs=False):
        super(RLBenchEnv, self).__init__()
        self._task_class = task_class
        self._observation_config = observation_config
        self._channels_last = channels_last
        self._include_lang_goal_in_obs = include_lang_goal_in_obs
        self._rlbench_env = Environment(
            action_mode=action_mode, obs_config=observation_config,
            dataset_root=dataset_root, headless=headless)
        self._task = None
        self._lang_goal = 'unknown goal'

    def extract_obs(self, obs: Observation):
        extracted_obs = _extract_obs(obs, self._channels_last, self._observation_config)
        if self._include_lang_goal_in_obs:
            extracted_obs['lang_goal_tokens'] = tokenize([self._lang_goal])[0].numpy()
        return extracted_obs

    def launch(self):
        self._rlbench_env.launch()
        self._task = self._rlbench_env.get_task(self._task_class)

    def shutdown(self):
        self._rlbench_env.shutdown()

    def reset(self, novel_command=None) -> dict:
        descriptions, obs = self._task.reset()
        self._lang_goal = descriptions[0] if novel_command is None else novel_command
        extracted_obs = self.extract_obs(obs)
        return extracted_obs

    def step(self, action: np.ndarray) -> Transition:
        obs, reward, terminal = self._task.step(action)
        obs = self.extract_obs(obs)
        return Transition(obs, reward, terminal)

    @property
    def observation_elements(self) -> List[ObservationElement]:
        return _observation_elements(self._observation_config, self._channels_last)

    @property
    def action_shape(self):
        return (self._rlbench_env.action_size, )

    @property
    def env(self) -> Environment:
        return self._rlbench_env


class MultiTaskRLBenchEnv(MultiTaskEnv):

    def __init__(self,
                 task_classes: List[Type[Task]],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 dataset_root: str = '',
                 channels_last=False,
                 headless=True,
                 swap_task_every: int = 1,
                 include_lang_goal_in_obs=False):
        super(MultiTaskRLBenchEnv, self).__init__()
        self._task_classes = task_classes
        self._observation_config = observation_config
        self._channels_last = channels_last
        self._include_lang_goal_in_obs = include_lang_goal_in_obs
        self._rlbench_env = Environment(
            action_mode=action_mode, obs_config=observation_config,
            dataset_root=dataset_root, headless=headless)
        self._task = None
        self._task_name = ''
        self._lang_goal = 'unknown goal'
        self._swap_task_every = swap_task_every
        self._rlbench_env
        self._episodes_this_task = 0
        self._active_task_id = -1

        self._task_name_to_idx = {change_case(tc.__name__):i for i, tc in enumerate(self._task_classes)}

    def _set_new_task(self, shuffle=False):
        if shuffle:
            self._active_task_id = np.random.randint(0, len(self._task_classes))
        else:
            self._active_task_id = (self._active_task_id + 1) % len(self._task_classes)
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)

    def set_task(self, task_name: str):
        self._active_task_id = self._task_name_to_idx[task_name]
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)
        descriptions, _ = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant

    def extract_obs(self, obs: Observation):
        extracted_obs = _extract_obs(obs, self._channels_last, self._observation_config)
        if self._include_lang_goal_in_obs:
            extracted_obs['lang_goal_tokens'] = tokenize([self._lang_goal])[0].numpy()
        return extracted_obs

    def launch(self):
        self._rlbench_env.launch()
        self._set_new_task()

    def shutdown(self):
        self._rlbench_env.shutdown()

    def reset(self, novel_command=None) -> dict:
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        descriptions, obs = self._task.reset()
        self._lang_goal = descriptions[0] if novel_command is None else novel_command
        extracted_obs = self.extract_obs(obs)

        return extracted_obs

    def step(self, action: np.ndarray) -> Transition:
        obs, reward, terminal = self._task.step(action)
        obs = self.extract_obs(obs)
        return Transition(obs, reward, terminal)

    @property
    def observation_elements(self) -> List[ObservationElement]:
        return _observation_elements(self._observation_config, self._channels_last)

    @property
    def action_shape(self):
        return (self._rlbench_env.action_size, )

    @property
    def env(self) -> Environment:
        return self._rlbench_env

    @property
    def num_tasks(self) -> int:
        return len(self._task_classes)