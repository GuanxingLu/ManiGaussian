# From https://github.com/stepjam/YARR/blob/main/yarr/replay_buffer/uniform_replay_buffer.py

"""The standard DQN replay memory.

This implementation is an out-of-graph replay memory + in-graph wrapper. It
supports vanilla n-step updates of the form typically found in the literature,
i.e. where rewards are accumulated for n steps and the intermediate trajectory
is not exposed to the agent. This does not allow, for example, performing
off-policy corrections.
"""
import ctypes
import collections
import concurrent.futures
import os
from os.path import join
import pickle
from typing import List, Tuple, Type
import time
import math
# from threading import Lock
import multiprocessing as mp
from multiprocessing import Lock
import numpy as np
import logging

from natsort import natsort

from yarr.replay_buffer.replay_buffer import ReplayBuffer, ReplayElement
from yarr.utils.observation_type import ObservationElement

from termcolor import colored

# Defines a type describing part of the tuple returned by the replay
# memory. Each element of the tuple is a tensor of shape [batch, ...] where
# ... is defined the 'shape' field of ReplayElement. The tensor type is
# given by the 'type' field. The 'name' field is for convenience and ease of
# debugging.


# String constants for storage
ACTION = 'action'
REWARD = 'reward'
TERMINAL = 'terminal'
TIMEOUT = 'timeout'
INDICES = 'indices'


def invalid_range(cursor, replay_capacity, stack_size, update_horizon):
    """Returns a array with the indices of cursor-related invalid transitions.

    There are update_horizon + stack_size invalid indices:
      - The update_horizon indices before the cursor, because we do not have a
        valid N-step transition (including the next state).
      - The stack_size indices on or immediately after the cursor.
    If N = update_horizon, K = stack_size, and the cursor is at c, invalid
    indices are:
      c - N, c - N + 1, ..., c, c + 1, ..., c + K - 1.

    It handles special cases in a circular buffer in the beginning and the end.

    Args:
      cursor: int, the position of the cursor.
      replay_capacity: int, the size of the replay memory.
      stack_size: int, the size of the stacks returned by the replay memory.
      update_horizon: int, the agent's update horizon.
    Returns:
      np.array of size stack_size with the invalid indices.
    """
    assert cursor < replay_capacity
    return np.array(
        [(cursor - update_horizon + i) % replay_capacity
         for i in range(stack_size + update_horizon)])


class UniformReplayBufferSingleProcess(ReplayBuffer):
    """A simple out-of-graph Replay Buffer.

    Stores transitions, state, action, reward, next_state, terminal (and any
    extra contents specified) in a circular buffer and provides a uniform
    transition sampling function.

    When the states consist of stacks of observations storing the states is
    inefficient. This class writes observations and constructs the stacked states
    at sample time.

    Attributes:
      _add_count: int, counter of how many transitions have been added (including
        the blank ones at the beginning of an episode).
      invalid_range: np.array, an array with the indices of cursor-related invalid
        transitions
    """

    def __init__(self,
                 batch_size: int = 32,
                 timesteps: int = 1,
                 replay_capacity: int = int(1e6),
                 update_horizon: int = 1,
                 gamma: float = 0.99,
                 max_sample_attempts: int = 10000,
                 action_shape: tuple = (),
                 action_dtype: Type[np.dtype] = np.float32,
                 reward_shape: tuple = (),
                 reward_dtype: Type[np.dtype] = np.float32,
                 observation_elements: List[ObservationElement] = None,
                 extra_replay_elements: List[ReplayElement] = None,
                 save_dir: str = None,
                 purge_replay_on_shutdown: bool = True
                 ):
        """Initializes OutOfGraphReplayBuffer.

        Args:
          batch_size: int.
          timesteps: int, number of frames to use in state stack.
          replay_capacity: int, number of transitions to keep in memory.
          update_horizon: int, length of update ('n' in n-step update).
          gamma: int, the discount factor.
          max_sample_attempts: int, the maximum number of attempts allowed to
            get a sample.
          action_shape: tuple of ints, the shape for the action vector.
            Empty tuple means the action is a scalar.
          action_dtype: np.dtype, type of elements in the action.
          reward_shape: tuple of ints, the shape of the reward vector.
            Empty tuple means the reward is a scalar.
          reward_dtype: np.dtype, type of elements in the reward.
          observation_elements: list of ObservationElement defining the type of
            the extra contents that will be stored and returned.
          extra_storage_elements: list of ReplayElement defining the type of
            the extra contents that will be stored and returned.

        Raises:
          ValueError: If replay_capacity is too small to hold at least one
            transition.
        """

        if observation_elements is None:
            observation_elements = []
        if extra_replay_elements is None:
            extra_replay_elements = []

        if replay_capacity < update_horizon + timesteps:
            raise ValueError('There is not enough capacity to cover '
                             'update_horizon and stack_size.')

        logging.info(
            'Creating a %s replay memory with the following parameters:',
            self.__class__.__name__)
        logging.info('\t timesteps: %d', timesteps)
        logging.info('\t replay_capacity: %d', replay_capacity)
        logging.info('\t batch_size: %d', batch_size)
        logging.info('\t update_horizon: %d', update_horizon)
        logging.info('\t gamma: %f', gamma)

        self._disk_saving = save_dir is not None
        self._save_dir = save_dir
        self._purge_replay_on_shutdown = purge_replay_on_shutdown
        if self._disk_saving:
            logging.info('\t saving to disk: %s', self._save_dir)
            os.makedirs(save_dir, exist_ok=True)
        else:
            logging.info('\t saving to RAM')


        self._action_shape = action_shape
        self._action_dtype = action_dtype
        self._reward_shape = reward_shape
        self._reward_dtype = reward_dtype
        self._timesteps = timesteps
        self._replay_capacity = replay_capacity
        self._batch_size = batch_size
        self._update_horizon = update_horizon
        self._gamma = gamma
        self._max_sample_attempts = max_sample_attempts

        self._observation_elements = observation_elements
        self._extra_replay_elements = extra_replay_elements

        self._storage_signature, self._obs_signature = self.get_storage_signature()
        self._create_storage()

        self._lock = Lock()
        self._add_count = mp.Value('i', 0)

        self._replay_capacity = replay_capacity

        self.invalid_range = np.zeros((self._timesteps))

        # When the horizon is > 1, we compute the sum of discounted rewards as a dot
        # product using the precomputed vector <gamma^0, gamma^1, ..., gamma^{n-1}>.
        self._cumulative_discount_vector = np.array(
            [math.pow(self._gamma, n) for n in range(update_horizon)],
            dtype=np.float32)

    @property
    def timesteps(self):
        return self._timesteps

    @property
    def replay_capacity(self):
        return self._replay_capacity

    @property
    def batch_size(self):
        return self._batch_size

    def _create_storage(self, store=None):
        """Creates the numpy arrays used to store transitions.
        """
        self._store = {} if store is None else store
        for storage_element in self._storage_signature:
            array_shape = [self._replay_capacity] + list(storage_element.shape)
            if storage_element.name == TERMINAL:
                self._store[storage_element.name] = np.full(
                    array_shape, -1, dtype=storage_element.type)
            elif not self._disk_saving:
                # If saving to disk, we don't need to store anything else.
                self._store[storage_element.name] = np.empty(
                    array_shape, dtype=storage_element.type)

    def get_storage_signature(self) -> Tuple[List[ReplayElement],
                                             List[ReplayElement]]:
        """Returns a default list of elements to be stored in this replay memory.

        Note - Derived classes may return a different signature.

        Returns:
          dict of ReplayElements defining the type of the contents stored.
        """
        storage_elements = [
            ReplayElement(ACTION, self._action_shape, self._action_dtype),
            ReplayElement(REWARD, self._reward_shape, self._reward_dtype),
            ReplayElement(TERMINAL, (), np.int8),
            ReplayElement(TIMEOUT, (), np.bool),
        ]

        obs_elements = []
        for obs_element in self._observation_elements:
            obs_elements.append(
                ReplayElement(
                    obs_element.name, obs_element.shape, obs_element.type))
        storage_elements.extend(obs_elements)

        for extra_replay_element in self._extra_replay_elements:
            storage_elements.append(extra_replay_element)

        return storage_elements, obs_elements

    def add(self, action, reward, terminal, timeout, **kwargs):
        """Adds a transition to the replay memory.

        WE ONLY STORE THE TPS1s on the final frame

        This function checks the types and handles the padding at the beginning of
        an episode. Then it calls the _add function.

        Since the next_observation in the transition will be the observation added
        next there is no need to pass it.

        If the replay memory is at capacity the oldest transition will be discarded.

        Args:
          action: int, the action in the transition.
          reward: float, the reward received in the transition.
          terminal: A uint8 acting as a boolean indicating whether the transition
                    was terminal (1) or not (0).
          **kwargs: The remaining args
        """

        # If previous transition was a terminal, then add_final wasn't called
        # if not self.is_empty() and self._store['terminal'][self.cursor() - 1] == 1:
        #     raise ValueError('The previous transition was a terminal, '
        #                      'but add_final was not called.')

        kwargs[ACTION] = action
        kwargs[REWARD] = reward
        kwargs[TERMINAL] = terminal
        kwargs[TIMEOUT] = timeout
        self._check_add_types(kwargs, self._storage_signature)
        # check here for nerf data.
        # print(kwargs['nerf_multi_view_rgb'])
        self._add(kwargs)

    def add_final(self, **kwargs):
        """Adds a transition to the replay memory.
        Args:
          **kwargs: The remaining args
        """
        # if self.is_empty() or self._store['terminal'][self.cursor() - 1] != 1:
        #     raise ValueError('The previous transition was not terminal.')
        self._check_add_types(kwargs, self._obs_signature)
        transition = self._final_transition(kwargs)
        self._add(transition)

    def _final_transition(self, kwargs):
        transition = {}
        for element_type in self._storage_signature:
            if element_type.name in kwargs:
                transition[element_type.name] = kwargs[element_type.name]
            elif element_type.name == TERMINAL:
                # Used to check that user is correctly adding transitions
                transition[element_type.name] = -1
            else:
                transition[element_type.name] = np.empty(
                    element_type.shape, dtype=element_type.type)
        return transition

    def _add_initial_to_disk(self ,kwargs: dict):
        for i in range(self._timesteps - 1):
            with open(join(self._save_dir, '%d.replay' % (
                    self._replay_capacity - 1 - i)), 'wb') as f:
                pickle.dump(kwargs, f)

    def _add(self, kwargs: dict):
        """Internal add method to add to the storage arrays.

        Args:
          kwargs: All the elements in a transition.
        """
        with self._lock:
            cursor = self.cursor()

            if self._disk_saving:
                term = self._store[TERMINAL]
                term[cursor] = kwargs[TERMINAL]
                self._store[TERMINAL] = term
                # self._store[TERMINAL][cursor] = kwargs[TERMINAL]
                
                ## reduce size
                for k, v in kwargs.items():
                    try:
                        if 'float' in v.dtype.name and v.size > 100:
                            v = v.astype(np.float16)
                            kwargs[k] = v
                    except:
                        pass
                    

                with open(join(self._save_dir, '%d.replay' % cursor), 'wb') as f:
                    # # check nerf data. found
                    # print(kwargs['nerf_multi_view_rgb'])
                    # print("save path:", join(self._save_dir, '%d.replay' % cursor)) # /tmp/replay/open_drawer/NERFINACT_BC/seed0/0.replay
                    pickle.dump(kwargs, f)
                # If first add, then pad for correct wrapping
                if self._add_count.value == 0:
                    self._add_initial_to_disk(kwargs)
            else:
                for name, data in kwargs.items():
                    item = self._store[name]
                    item[cursor] = data
                    self._store[name] = item

                    # self._store[name][cursor] = data
            with self._add_count.get_lock():
                self._add_count.value += 1
            self.invalid_range = invalid_range(
                self.cursor(), self._replay_capacity, self._timesteps,
                self._update_horizon)

    def _get_from_disk(self, start_index, end_index):
        """Returns the range of array at the index handling wraparound if necessary.

        Args:
          start_index: int, index to the start of the range to be returned. Range
            will wraparound if start_index is smaller than 0.
          end_index: int, exclusive end index. Range will wraparound if end_index
            exceeds replay_capacity.

        Returns:
          np.array, with shape [end_index - start_index, array.shape[1:]].
        """
        assert end_index > start_index, 'end_index must be larger than start_index'
        assert end_index >= 0
        assert start_index < self._replay_capacity
        if not self.is_full():
            assert end_index <= self.cursor(), (
                'Index {} has not been added.'.format(start_index))

        # Here we fake a mini store (buffer)
        store = {store_element.name: {}
                 for store_element in self._storage_signature}
        if start_index % self._replay_capacity < end_index % self._replay_capacity:
            for i in range(start_index, end_index):
                with open(join(self._save_dir, '%d.replay' % i), 'rb') as f:
                    d = pickle.load(f)
                    # print(d['nerf_multi_view_rgb'])
                    # d = pickle.load(join(self._save_dir, '%d.replay' % 0))
                    for k, v in d.items():
                        store[k][i] = v
        else:
            for i in range(end_index - start_index):
                idx = (start_index + i) % self._replay_capacity
                with open(join(self._save_dir, '%d.replay' % idx), 'rb') as f:
                    d = pickle.load(f)
                    for k, v in d.items():
                        store[k][idx] = v
        # check nerf data. found in some.
        # if store['nerf_multi_view_rgb'] is not None:
        #     print(store['nerf_multi_view_camera'])
        return store

    def _check_add_types(self, kwargs, signature):
        """Checks if args passed to the add method match those of the storage.

        Args:
          *args: Args whose types need to be validated.

        Raises:
          ValueError: If args have wrong shape or dtype.
        """

        if (len(kwargs)) != len(signature):
            expected = str(natsort.natsorted([e.name for e in signature]))
            actual = str(natsort.natsorted(list(kwargs.keys())))
            error_list = '\nList of expected:\n{}\nList of actual:\n{}'.format(
                expected, actual)
            raise ValueError('Add expects {} elements, received {}.'.format(
                len(signature), len(kwargs)) + error_list)

        for store_element in signature:
            arg_element = kwargs[store_element.name]
            if isinstance(arg_element, np.ndarray):
                arg_shape = arg_element.shape
            elif isinstance(arg_element, tuple) or isinstance(arg_element, list):
                # TODO: This is not efficient when arg_element is a list.
                arg_shape = np.array(arg_element).shape
            elif isinstance(arg_element, int) or isinstance(arg_element, float) or isinstance(arg_element, str):
                # Assume it is scalar.
                arg_shape = tuple()
            else:
                # print(f"Mismatched type: {type(arg_element)} from {store_element.name}. we assume the program will go on.")
                arg_shape = tuple(store_element.shape) # to avoid error we assume it is the right shape. debug this.
            store_element_shape = tuple(store_element.shape)
            if arg_shape != store_element_shape:
                raise ValueError('arg has shape {}, expected {}'.format(
                    arg_shape, store_element_shape))

    def is_empty(self):
        """Is the Replay Buffer empty?"""
        return self._add_count.value == 0

    def is_full(self):
        """Is the Replay Buffer full?"""
        return self._add_count.value >= self._replay_capacity

    def cursor(self):
        """Index to the location where the next transition will be written."""
        return self._add_count.value % self._replay_capacity

    @property
    def add_count(self):
        return np.array(self._add_count.value) #self._add_count.copy()

    @add_count.setter
    def add_count(self, count):
        if isinstance(count, int):
            self._add_count = mp.Value('i', count)
        else:
            self._add_count = count


    def get_range(self, array, start_index, end_index):
        """Returns the range of array at the index handling wraparound if necessary.

        Args:
          array: np.array, the array to get the stack from.
          start_index: int, index to the start of the range to be returned. Range
            will wraparound if start_index is smaller than 0.
          end_index: int, exclusive end index. Range will wraparound if end_index
            exceeds replay_capacity.

        Returns:
          np.array, with shape [end_index - start_index, array.shape[1:]].
        """
        assert end_index > start_index, 'end_index must be larger than start_index'
        assert end_index >= 0
        assert start_index < self._replay_capacity
        if not self.is_full():
            assert end_index <= self.cursor(), (
                'Index {} has not been added.'.format(start_index))

        # Fast slice read when there is no wraparound.
        if start_index % self._replay_capacity < end_index % self._replay_capacity:
            return_array = np.array(
                [array[i] for i in range(start_index, end_index)])
        # Slow list read.
        else:
            indices = [(start_index + i) % self._replay_capacity
                       for i in range(end_index - start_index)]
            return_array = np.array([array[i] for i in indices])

        return return_array

    def get_range_stack(self, array, start_index, end_index, terminals=None):
        """Returns the range of array at the index handling wraparound if necessary.

        Args:
          array: np.array, the array to get the stack from.
          start_index: int, index to the start of the range to be returned. Range
            will wraparound if start_index is smaller than 0.
          end_index: int, exclusive end index. Range will wraparound if end_index
            exceeds replay_capacity.

        Returns:
          np.array, with shape [end_index - start_index, array.shape[1:]].
        """
        return_array = np.array(self.get_range(array, start_index, end_index))
        if terminals is None:
            terminals = self.get_range(
                self._store[TERMINAL], start_index, end_index)

        terminals = terminals[:-1]

        # Here we now check if we need to pad the front episodes
        # If any have a terminal of -1, then we have spilled over
        # into the the previous transition
        if np.any(terminals == -1):
            padding_item = return_array[-1]
            _array = list(return_array)[:-1]
            arr_len = len(_array)
            pad_from_now = False
            for i, (ar, term) in enumerate(
                    zip(reversed(_array), reversed(terminals))):
                if term == -1 or pad_from_now:
                    # The first time we see a -1 term, means we have hit the
                    # beginning of this episode, so pad from now.
                    # pad_from_now needed because the next transition (reverse)
                    # will not be a -1 terminal.
                    pad_from_now = True
                    return_array[arr_len - 1 - i] = padding_item
                else:
                    # After we hit out first -1 terminal, we never reassign.
                    padding_item = ar

        return return_array

    def _get_element_stack(self, array, index, terminals=None):
        state = self.get_range_stack(array,
                                     index - self._timesteps + 1, index + 1,
                                     terminals=terminals)
        return state

    def get_terminal_stack(self, index):
        terminal_stack = self.get_range(self._store[TERMINAL],
                              index - self._timesteps + 1,
                              index + 1)
        return terminal_stack

    def is_valid_transition(self, index):
        """Checks if the index contains a valid transition.

        Checks for collisions with the end of episodes and the current position
        of the cursor.

        Args:
          index: int, the index to the state in the transition.

        Returns:
          Is the index valid: Boolean.

        """
        # Check the index is in the valid range
        if index < 0 or index >= self._replay_capacity:
            return False
        if not self.is_full():
            # The indices and next_indices must be smaller than the cursor.
            if index >= self.cursor() - self._update_horizon:
                return False

        # Skip transitions that straddle the cursor.
        if index in set(self.invalid_range):
            return False

        term_stack = self.get_terminal_stack(index)
        if term_stack[-1] == -1:
            return False

        return True

    def _create_batch_arrays(self, batch_size):
        """Create a tuple of arrays with the type of get_transition_elements.

        When using the WrappedReplayBuffer with staging enabled it is important
        to create new arrays every sample because StaginArea keeps a pointer to
        the returned arrays.

        Args:
          batch_size: (int) number of transitions returned. If None the default
            batch_size will be used.

        Returns:
          Tuple of np.arrays with the shape and type of get_transition_elements.
        """
        transition_elements = self.get_transition_elements(batch_size)
        batch_arrays = []
        for element in transition_elements:
            batch_arrays.append(np.empty(element.shape, dtype=element.type))
        return tuple(batch_arrays)

    def sample_index_batch(self, batch_size):
        """Returns a batch of valid indices sampled uniformly.

        Args:
          batch_size: int, number of indices returned.

        Returns:
          list of ints, a batch of valid indices sampled uniformly.

        Raises:
          RuntimeError: If the batch was not constructed after maximum number of
            tries.
        """
        if self.is_full():
            # add_count >= self._replay_capacity > self._stack_size
            min_id = (self.cursor() - self._replay_capacity +
                      self._timesteps - 1)
            max_id = self.cursor() - self._update_horizon
        else:
            min_id = 0
            max_id = self.cursor() - self._update_horizon
            if max_id <= min_id:
                raise RuntimeError(
                    'Cannot sample a batch with fewer than stack size '
                    '({}) + update_horizon ({}) transitions.'.
                    format(self._timesteps, self._update_horizon))

        indices = []
        attempt_count = 0
        while (len(indices) < batch_size and
                       attempt_count < self._max_sample_attempts):
            index = np.random.randint(min_id, max_id) % self._replay_capacity
            if self.is_valid_transition(index):
                indices.append(index)
            else:
                attempt_count += 1
        if len(indices) != batch_size:
            raise RuntimeError(
                'Max sample attempts: Tried {} times but only sampled {}'
                ' valid indices. Batch size is {}'.
                    format(self._max_sample_attempts, len(indices), batch_size))

        return indices

    def unpack_transition(self, transition_tensors, transition_type):
        """Unpacks the given transition into member variables.

        Args:
          transition_tensors: tuple of tf.Tensors.
          transition_type: tuple of ReplayElements matching transition_tensors.
        """
        self.transition = collections.OrderedDict()
        for element, element_type in zip(transition_tensors, transition_type):
            self.transition[element_type.name] = element
        return self.transition

    def sample_transition_batch(self, batch_size=None, indices=None,
                                pack_in_dict=True):
        """Returns a batch of transitions (including any extra contents).

        If get_transition_elements has been overridden and defines elements not
        stored in self._store, an empty array will be returned and it will be
        left to the child class to fill it. For example, for the child class
        OutOfGraphPrioritizedReplayBuffer, the contents of the
        sampling_probabilities are stored separately in a sum tree.

        When the transition is terminal next_state_batch has undefined contents.

        NOTE: This transition contains the indices of the sampled elements.
        These are only valid during the call to sample_transition_batch,
        i.e. they may  be used by subclasses of this replay buffer but may
        point to different data as soon as sampling is done.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
          indices: None or list of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.

        Returns:
          transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().

        Raises:
          ValueError: If an element to be sampled is missing from the
            replay buffer.
        """
        
        if batch_size is None:
            batch_size = self._batch_size
        with self._lock:
            if indices is None:
                indices = self.sample_index_batch(batch_size)
            assert len(indices) == batch_size

            transition_elements = self.get_transition_elements(batch_size)
            batch_arrays = self._create_batch_arrays(batch_size)

            for batch_element, state_index in enumerate(indices):

                if not self.is_valid_transition(state_index):
                    raise ValueError('Invalid index %d.' % state_index)

                trajectory_indices = [(state_index + j) % self._replay_capacity
                                      for j in range(self._update_horizon)]
                trajectory_terminals = self._store['terminal'][
                    trajectory_indices]
                is_terminal_transition = trajectory_terminals.any()
                if not is_terminal_transition:
                    trajectory_length = self._update_horizon
                else:
                    # np.argmax of a bool array returns index of the first True.
                    trajectory_length = np.argmax(
                        trajectory_terminals.astype(np.bool),
                        0) + 1

                next_state_index = state_index + trajectory_length

                store = self._store
                if self._disk_saving:
                    store = self._get_from_disk(
                        state_index - (self._timesteps - 1),
                        next_state_index + 1)

                trajectory_discount_vector = (
                    self._cumulative_discount_vector[:trajectory_length])
                trajectory_rewards = self.get_range(store['reward'],
                                                    state_index,
                                                    next_state_index)

                terminal_stack = self.get_terminal_stack(state_index)
                terminal_stack_tp1 = self.get_terminal_stack(
                    next_state_index % self._replay_capacity)

                # Fill the contents of each array in the sampled batch.
                assert len(transition_elements) == len(batch_arrays)
                for element_array, element in zip(batch_arrays,
                                                  transition_elements):
                    if element.is_observation:
                        if element.name.endswith('tp1'):
                            element_array[
                                batch_element] = self._get_element_stack(
                                store[element.name[:-4]],
                                next_state_index % self._replay_capacity,
                                terminal_stack_tp1)
                        else:
                            element_array[
                                batch_element] = self._get_element_stack(
                                store[element.name],
                                state_index, terminal_stack)
                    elif element.name == REWARD:
                        # compute discounted sum of rewards in the trajectory.
                        element_array[batch_element] = np.sum(
                            trajectory_discount_vector * trajectory_rewards,
                            axis=0)
                    elif element.name == TERMINAL:
                        element_array[batch_element] = is_terminal_transition
                    elif element.name == INDICES:
                        element_array[batch_element] = state_index
                    elif element.name in store.keys():
                        element_array[batch_element] = (
                            store[element.name][state_index])

        if pack_in_dict:
            batch_arrays = self.unpack_transition(
                batch_arrays, transition_elements)

        # TODO: make a proper fix for this
        if 'task' in batch_arrays:
            del batch_arrays['task']
        if 'task_tp1' in batch_arrays:
            del batch_arrays['task_tp1']
            
        return batch_arrays

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
            ReplayElement(ACTION, (batch_size,) + self._action_shape,
                          self._action_dtype),
            ReplayElement(REWARD, (batch_size,) + self._reward_shape,
                          self._reward_dtype),
            ReplayElement(TERMINAL, (batch_size,), np.int8),
            ReplayElement(TIMEOUT, (batch_size,), np.bool),
            ReplayElement(INDICES, (batch_size,), np.int32),
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

    def shutdown(self):
        if self._purge_replay_on_shutdown:
            # Safely delete replay
            logging.info('Clearing disk replay buffer.')
            for f in [f for f in os.listdir(self._save_dir) if '.replay' in f]:
                os.remove(join(self._save_dir, f))

    def using_disk(self):
        return self._disk_saving
