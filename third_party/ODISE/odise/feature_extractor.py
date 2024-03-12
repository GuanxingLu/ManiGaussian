from detectron2.config import LazyConfig, instantiate
from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise

MY_CKPT_PATH = "/home/yanjieze/projects/nerf-act/archive/odise_caption_coco_50e-853cc971.pth"
MY_CONFIG_PATH = "/home/yanjieze/projects/nerf-act/ODISE_PRERELEASE/configs/Panoptic/odise_caption_coco_50e.py"

def instantiate_odise_feature_extractor(ckpt_path=MY_CKPT_PATH, cfg_path=MY_CONFIG_PATH):
    cfg = LazyConfig.load(cfg_path)
    cfg.model.overlap_threshold = 0
    model = instantiate_odise(cfg.model)
    ODISECheckpointer(model).load(ckpt_path)
    return model.backbone.cuda()