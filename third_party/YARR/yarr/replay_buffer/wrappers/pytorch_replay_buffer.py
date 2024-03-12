import time
from threading import Thread

from torch.utils.data import IterableDataset, DataLoader

from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.replay_buffer.wrappers import WrappedReplayBuffer


class PyTorchIterableReplayDataset(IterableDataset):

    def __init__(self, replay_buffer: ReplayBuffer):
        self._replay_buffer = replay_buffer

    def _generator(self):
        while True:
            yield self._replay_buffer.sample_transition_batch(pack_in_dict=True)

    def __iter__(self):
        return iter(self._generator())

# class PyTorchIterableReplayDataset(IterableDataset):
#
#     BUFFER = 4
#
#     def __init__(self, replay_buffer: ReplayBuffer, num_workers: int):
#         self._replay_buffer = replay_buffer
#         self._num_wokers = num_workers
#         self._samples = []
#         self._lock = Lock()
#
#     def _run(self):
#         while True:
#             # Check if replay buffer is ig enough to be sampled
#             while self._replay_buffer.add_count < self._replay_buffer.batch_size:
#                 time.sleep(1.)
#             s = self._replay_buffer.sample_transition_batch(pack_in_dict=True)
#             while len(self._samples) >= PyTorchIterableReplayDataset.BUFFER:
#                 time.sleep(0.25)
#             with self._lock:
#                 self._samples.append(s)
#
#     def _generator(self):
#         ts = [Thread(
#             target=self._run, args=()) for _ in range(self._num_wokers)]
#         [t.start() for t in ts]
#         while True:
#             while len(self._samples) == 0:
#                 time.sleep(0.1)
#             with self._lock:
#                 s = self._samples.pop(0)
#             yield s
#
#     def __iter__(self):
#         i = iter(self._generator())
#         return i


class PyTorchReplayBuffer(WrappedReplayBuffer):
    """Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism.

    Usage:
      To add a transition:  call the add function.

      To sample a batch:    Construct operations that depend on any of the
                            tensors is the transition dictionary. Every sess.run
                            that requires any of these tensors will sample a new
                            transition.
    """

    def __init__(self, replay_buffer: ReplayBuffer, num_workers: int = 2):
        super(PyTorchReplayBuffer, self).__init__(replay_buffer)
        self._num_workers = num_workers

    def dataset(self, batch_size=None, drop_last=False) -> DataLoader:
        d = PyTorchIterableReplayDataset(self._replay_buffer)

        # Batch size None disables automatic batching
        return DataLoader(d, batch_size=batch_size,
                          drop_last=drop_last,
                          num_workers=self._num_workers, pin_memory=True)
