import torch

from detectron2.config import LazyCall as L
from detectron2.solver.build import get_default_optimizer_params


AdamW = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0,
        weight_decay_bias=0.0,
    ),
    lr="???",
    weight_decay="???",
)
