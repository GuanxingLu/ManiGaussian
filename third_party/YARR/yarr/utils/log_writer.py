import csv
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from yarr.agents.agent import ScalarSummary, HistogramSummary, ImageSummary, \
    VideoSummary, TextSummary
# from torch.utils.tensorboard import SummaryWriter


class LogWriter(object):

    def __init__(self,
                 logdir: str,
                 tensorboard_logging: bool,
                 csv_logging: bool,
                 train_csv: str = 'train_data.csv',
                 env_csv: str = 'env_data.csv'):
        # self._tensorboard_logging = tensorboard_logging
        self._csv_logging = csv_logging
        os.makedirs(logdir, exist_ok=True)
        self._tensorboard_logging = False
        # if tensorboard_logging:
            # self._tf_writer = SummaryWriter(logdir)
        if csv_logging:
            self._train_prev_row_data = self._train_row_data = OrderedDict()
            self._train_csv_file = os.path.join(logdir, train_csv)
            self._env_prev_row_data = self._env_row_data = OrderedDict()
            self._env_csv_file = os.path.join(logdir, env_csv)
            self._train_field_names = None
            self._env_field_names = None

    def add_scalar(self, i, name, value):
        # if self._tensorboard_logging:
            # self._tf_writer.add_scalar(name, value, i)
        if self._csv_logging:
            if 'env' in name or 'eval' in name or 'test' in name:
                if len(self._env_row_data) == 0:
                    self._env_row_data['step'] = i
                self._env_row_data[name] = value.item() if isinstance(
                    value, torch.Tensor) else value
            else:
                if len(self._train_row_data) == 0:
                    self._train_row_data['step'] = i
                self._train_row_data[name] = value.item() if isinstance(
                    value, torch.Tensor) else value

    def add_summaries(self, i, summaries):
        for summary in summaries:
            try:
                if isinstance(summary, ScalarSummary):
                    self.add_scalar(i, summary.name, summary.value)
                elif self._tensorboard_logging:
                    if isinstance(summary, HistogramSummary):
                        self._tf_writer.add_histogram(
                            summary.name, summary.value, i)
                    elif isinstance(summary, ImageSummary):
                        # Only grab first item in batch
                        v = (summary.value if summary.value.ndim == 3 else
                             summary.value[0])
                        self._tf_writer.add_image(summary.name, v, i)
                    elif isinstance(summary, VideoSummary):
                        # Only grab first item in batch
                        v = (summary.value if summary.value.ndim == 5 else
                             np.array([summary.value]))
                        self._tf_writer.add_video(
                            summary.name, v, i, fps=summary.fps)
                    elif isinstance(summary, TextSummary):
                        self._tf_writer.add_text(summary.name, summary.value, i)
            except Exception as e:
                logging.error('Error on summary: %s' % summary.name)
                raise e

    def end_iteration(self):
        # write train data
        if self._csv_logging and len(self._train_row_data) > 0:
            should_write_train_header = not os.path.exists(self._train_csv_file)
            with open(self._train_csv_file, mode='a+') as csv_f:
                names = self._train_row_data.keys()
                writer = csv.DictWriter(csv_f, fieldnames=names)
                if should_write_train_header:
                    if self._train_field_names is None:
                        writer.writeheader()
                    else:
                        if not np.array_equal(self._train_field_names, self._train_row_data.keys()):
                            # Special case when we are logging faster than new
                            # summaries are coming in.
                            missing_keys = list(set(self._train_field_names) - set(
                                self._train_row_data.keys()))
                            for mk in missing_keys:
                                self._train_row_data[mk] = self._train_prev_row_data[mk]
                self._train_field_names = names
                try:
                    writer.writerow(self._train_row_data)
                except Exception as e:
                    print(e)
            self._train_prev_row_data = self._train_row_data
            self._train_row_data = OrderedDict()

        # write env data (also eval or test during evaluation)
        if self._csv_logging and len(self._env_row_data) > 0:
            should_write_env_header = not os.path.exists(self._env_csv_file)
            with open(self._env_csv_file, mode='a+') as csv_f:
                names = self._env_row_data.keys()
                writer = csv.DictWriter(csv_f, fieldnames=names)
                if should_write_env_header:
                    if self._env_field_names is None:
                        writer.writeheader()
                    else:
                        if not np.array_equal(self._env_field_names, self._env_row_data.keys()):
                            # Special case when we are logging faster than new
                            # summaries are coming in.
                            missing_keys = list(set(self._env_field_names) - set(
                                self._env_row_data.keys()))
                            for mk in missing_keys:
                                self._env_row_data[mk] = self._env_prev_row_data[mk]
                self._env_field_names = names
                try:
                    writer.writerow(self._env_row_data)
                except Exception as e:
                    print(e)
            self._env_prev_row_data = self._env_row_data
            self._env_row_data = OrderedDict()

    def close(self):
        return
        if self._tensorboard_logging:
            self._tf_writer.close()
