'''
Modified from: https://github.com/Kunhao-Liu/3D-OVS/blob/main/models/DINO_extractor.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class VitExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vitl14'):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.last_block = None
        self.feature_output = None
        self.last_block = self.model.blocks[-1]
        self.last_block.register_forward_hook(self._get_block_hook())
        self.patch_size = 14
        self.feature_dims = 1024

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.feature_output = output
        return _get_block_output

    def forward(self, input_img):
        B, C, H, W = input_img.shape
        dino_ret = self.model.forward_features(input_img)['x_norm_patchtokens'] # [B, 256, 1024]
        dino_ret = dino_ret.reshape([B, -1, H//self.patch_size, W//self.patch_size])    # [B, 1024, 16, 16]
        return dino_ret
