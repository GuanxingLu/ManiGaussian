from fvcore.common.param_scheduler import CosineParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

cosine_lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(start_value=1.0, end_value=0.01),
    warmup_length="???",
    warmup_method="linear",
    warmup_factor=0.001,
)
