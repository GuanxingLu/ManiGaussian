from multiprocessing import Lock
from typing import List

import numpy as np
from yarr.agents.agent import Summary, ScalarSummary
from yarr.utils.transition import ReplayTransition


class StatAccumulator(object):

    def step(self, transition: ReplayTransition, eval: bool):
        pass

    def pop(self) -> List[Summary]:
        pass

    def peak(self) -> List[Summary]:
        pass

    def reset(self) -> None:
        pass


class Metric(object):

    def __init__(self):
        self._previous = []
        self._current = 0

    def update(self, value):
        self._current += value

    def next(self):
        self._previous.append(self._current)
        self._current = 0

    def reset(self):
        self._previous.clear()

    def min(self):
        return np.min(self._previous)

    def max(self):
        return np.max(self._previous)

    def mean(self):
        return np.mean(self._previous)

    def median(self):
        return np.median(self._previous)

    def std(self):
        return np.std(self._previous)

    def __len__(self):
        return len(self._previous)

    def __getitem__(self, i):
        return self._previous[i]


class _SimpleAccumulator(StatAccumulator):

    def __init__(self, prefix, eval_video_fps: int = 30,
                 mean_only: bool = True):
        self._prefix = prefix
        self._eval_video_fps = eval_video_fps
        self._mean_only = mean_only
        self._lock = Lock()
        self._episode_returns = Metric()
        self._episode_lengths = Metric()
        self._summaries = []
        self._transitions = 0

    def _reset_data(self):
        with self._lock:
            self._episode_returns.reset()
            self._episode_lengths.reset()
            self._summaries.clear()

    def step(self, transition: ReplayTransition, eval: bool):
        with self._lock:
            self._transitions += 1
            self._episode_returns.update(transition.reward)
            self._episode_lengths.update(1)
            if transition.terminal:
                self._episode_returns.next()
                self._episode_lengths.next()
            self._summaries.extend(list(transition.summaries))

    def _get(self) -> List[Summary]:
        sums = []

        if self._mean_only:
            stat_keys = ["mean"]
        else:
            stat_keys = ["min", "max", "mean", "median", "std"]
        names = ["return", "length"]
        metrics = [self._episode_returns, self._episode_lengths]
        for name, metric in zip(names, metrics):
            for stat_key in stat_keys:
                if self._mean_only:
                    assert stat_key == "mean"
                    sum_name = '%s/%s' % (self._prefix, name)
                else:
                    sum_name = '%s/%s/%s' % (self._prefix, name, stat_key)
                sums.append(
                    ScalarSummary(sum_name, getattr(metric, stat_key)()))
        sums.append(ScalarSummary(
            '%s/total_transitions' % self._prefix, self._transitions))
        sums.extend(self._summaries)
        return sums

    def pop(self) -> List[Summary]:
        data = []
        if len(self._episode_returns) > 1:
            data = self._get()
            self._reset_data()
        return data

    def peak(self) -> List[Summary]:
        return self._get()
    
    def reset(self):
        self._transitions = 0
        self._reset_data()


class SimpleAccumulator(StatAccumulator):

    def __init__(self, eval_video_fps: int = 30, mean_only: bool = True):
        self._train_acc = _SimpleAccumulator(
            'train_envs', eval_video_fps, mean_only=mean_only)
        self._eval_acc = _SimpleAccumulator(
            'eval_envs', eval_video_fps, mean_only=mean_only)

    def step(self, transition: ReplayTransition, eval: bool):
        if eval:
            self._eval_acc.step(transition, eval)
        else:
            self._train_acc.step(transition, eval)

    def pop(self) -> List[Summary]:
        return self._train_acc.pop() + self._eval_acc.pop()

    def peak(self) -> List[Summary]:
        return self._train_acc.peak() + self._eval_acc.peak()
    
    def reset(self) -> None:
        self._train_acc.reset()
        self._eval_acc.reset()


class MultiTaskAccumulator(StatAccumulator):

    def __init__(self, num_tasks,
                 eval_video_fps: int = 30, mean_only: bool = True,
                 train_prefix: str = 'train_task',
                 eval_prefix: str = 'eval_task'):
        self._train_accs = [_SimpleAccumulator(
            '%s%d/envs' % (train_prefix, i), eval_video_fps, mean_only=mean_only)
            for i in range(num_tasks)]
        self._eval_accs = [_SimpleAccumulator(
            '%s%d/envs' % (eval_prefix, i), eval_video_fps, mean_only=mean_only)
            for i in range(num_tasks)]
        self._train_accs_mean = _SimpleAccumulator(
            '%s_summary/envs' % train_prefix, eval_video_fps,
            mean_only=mean_only)

    def step(self, transition: ReplayTransition, eval: bool):
        replay_index = transition.info["active_task_id"]
        if eval:
            self._eval_accs[replay_index].step(transition, eval)
        else:
            self._train_accs[replay_index].step(transition, eval)
            self._train_accs_mean.step(transition, eval)

    def pop(self) -> List[Summary]:
        combined = self._train_accs_mean.pop()
        for acc in self._train_accs + self._eval_accs:
            combined.extend(acc.pop())
        return combined

    def peak(self) -> List[Summary]:
        combined = self._train_accs_mean.peak()
        for acc in self._train_accs + self._eval_accs:
            combined.extend(acc.peak())
        return combined

    def reset(self) -> None:
        self._train_accs_mean.reset()
        [acc.reset() for acc in self._train_accs + self._eval_accs]
