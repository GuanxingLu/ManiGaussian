# ------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# To view a copy of this license, visit
# https://github.com/facebookresearch/detectron2/blob/main/LICENSE
#
# Modified by Jiarui Xu
# ------------------------------------------------------------------------------

"""
Model Zoo API for ODISE: a collection of functions to create common model architectures
listed in `MODEL_ZOO.md <https://github.com/NVlabs/ODISE/blob/master/README.md#model-zoo>`_,
and optionally load their pre-trained weights.
"""

from .model_zoo import get, get_config_file, get_checkpoint_url, get_config

__all__ = ["get_checkpoint_url", "get", "get_config_file", "get_config"]
