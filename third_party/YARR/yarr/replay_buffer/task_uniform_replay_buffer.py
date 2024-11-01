import numpy as np
import os
from os.path import join
import pickle
import math
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import invalid_range

from yarr.replay_buffer.replay_buffer import ReplayBuffer, ReplayElement
from yarr.utils.observation_type import ObservationElement

ACTION = 'action'
REWARD = 'reward'
TERMINAL = 'terminal'
TIMEOUT = 'timeout'
INDICES = 'indices'
TASK = 'task'


class TaskUniformReplayBuffer(UniformReplayBuffer):
    """
    A uniform with uniform task sampling for each batch
    """

    def __init__(self, *args, **kwargs):
        """Initializes OutOfGraphPrioritizedReplayBuffer."""
        super(TaskUniformReplayBuffer, self).__init__(*args, **kwargs)
        self._task_idxs = dict()

    def _add(self, kwargs: dict):
        """Internal add method to add to the storage arrays.

        Args:
          kwargs: All the elements in a transition.
        """
        with self._lock:
            cursor = self.cursor()

            if self._disk_saving:   # default: True
                term = self._store[TERMINAL]
                term[cursor] = kwargs[TERMINAL]
                self._store[TERMINAL] = term
                
                ## reduce size
                # Training Speed-Up and Storage Memory Reduction: Ishika found that switching from fp32 to fp16 for storing pickle files dramatically speeds-up training time and significantly reduces memory usage. Checkout her modifications to YARR here.
                for k, v in kwargs.items():
                    try:
                        if 'float' in v.dtype.name and v.size > 100:
                            v = v.astype(np.float16)
                            kwargs[k] = v
                    except:
                        pass

                with open(join(self._save_dir, '%d.replay' % cursor), 'wb') as f:
                    pickle.dump(kwargs, f)
                # If first add, then pad for correct wrapping
                if self._add_count.value == 0:
                    self._add_initial_to_disk(kwargs)
            else:
                for name, data in kwargs.items():
                    item = self._store[name]
                    item[cursor] = data
                    self._store[name] = item
            with self._add_count.get_lock():
                task = kwargs[TASK]
                if task not in self._task_idxs:
                    self._task_idxs[task] = [cursor]
                else:
                    self._task_idxs[task] =  self._task_idxs[task] + [cursor]
                self._add_count.value += 1

            self.invalid_range = invalid_range(
                self.cursor(), self._replay_capacity, self._timesteps,
                self._update_horizon)

    def sample_index_batch(self,
                           batch_size):
        """Returns a batch of valid indices sampled uniformly.

        Args:
          batch_size: int, number of indices returned.

        Returns:
          list of ints, a batch of valid indices sampled uniformly across tasks.

        Raises:
          RuntimeError: If the batch was not constructed after maximum number of
            tries.
        """
        if self.is_full():
            min_id = (self.cursor() - self._replay_capacity +
                      self._timesteps - 1)
            max_id = self.cursor() - self._update_horizon
        else:
            min_id = 0
            max_id = self.cursor() - self._update_horizon
            if max_id <= min_id:    # FIXME: the exception is not true
                raise RuntimeError(
                    'Cannot sample a batch with fewer than stack size '
                    '({}) + update_horizon ({}) transitions.'.
                    format(self._timesteps, self._update_horizon))

        tasks = list(self._task_idxs.keys())
        attempt_count = 0
        found_indicies = False

        # uniform distribution of tasks
        while not found_indicies and attempt_count < 1000:
            # sample random tasks of batch_size length
            sampled_tasks = list(np.random.choice(tasks, batch_size, replace=(batch_size > len(tasks))))
            potential_indices = []
            for task in sampled_tasks:
                # DDP setting where each GPU only sees a fraction of the data
                # reference: https://github.com/pytorch/pytorch/blob/master/torch/utils/data/distributed.py
                task_data_size = len(self._task_idxs[task])
                num_samples = math.ceil(task_data_size / self._num_replicas)
                total_size = num_samples * self._num_replicas
                task_indices = self._task_idxs[task][self._rank:total_size:self._num_replicas]

                sampled_task_idx = np.random.choice(task_indices, 1)[0]
                per_task_attempt_count = 0

                # Argh.. this is slow
                while not self.is_valid_transition(sampled_task_idx) and \
                    per_task_attempt_count < self._max_sample_attempts:
                    sampled_task_idx = np.random.choice(task_indices, 1)[0]
                    per_task_attempt_count += 1

                if not self.is_valid_transition(sampled_task_idx):
                    attempt_count += 1
                    continue
                else:
                    potential_indices.append(sampled_task_idx)
            found_indicies = len(potential_indices) == batch_size
        indices = potential_indices

        if len(indices) != batch_size:
            raise RuntimeError(
                'Max sample attempts: Tried {} times but only sampled {}'
                ' valid indices. Batch size is {}'.
                    format(self._max_sample_attempts, len(indices), batch_size))

        return indices

    def get_transition_elements(self, batch_size=None):
        """Returns a 'type signature' for sample_transition_batch.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
        Returns:
          signature: A namedtuple describing the method's return type signature.
        """
        batch_size = self._batch_size if batch_size is None else batch_size

        transition_elements = [
            ReplayElement(ACTION, (batch_size, self._timesteps) + self._action_shape,
                          self._action_dtype),
            ReplayElement(REWARD, (batch_size, self._timesteps) + self._reward_shape,
                          self._reward_dtype),
            ReplayElement(TERMINAL, (batch_size, self._timesteps), np.int8),
            ReplayElement(TIMEOUT, (batch_size, self._timesteps), np.bool),
            ReplayElement(INDICES, (batch_size, self._timesteps), np.int32),
        ]

        for element in self._observation_elements:
            transition_elements.append(ReplayElement(
                element.name,
                (batch_size, self._timesteps) + tuple(element.shape),
                element.type, True))
            transition_elements.append(ReplayElement(
                element.name + '_tp1',
                (batch_size, self._timesteps) + tuple(element.shape),
                element.type, True))

        for element in self._extra_replay_elements:
            transition_elements.append(ReplayElement(
                element.name,
                (batch_size,) + tuple(element.shape),
                element.type))
        return transition_elements