'''
Modified from: https://github.com/Kunhao-Liu/3D-OVS/blob/main/models/DINO_extractor.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class VitExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vitl14'):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.patch_size = 14
        self.feature_dims = 1024
        self.preprocess = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet
        ])

        self._freeze()

    def _freeze(self):
        super().train(mode=False)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, input_img):
        B, C, H, W = input_img.shape
        input_img = self.preprocess(input_img)
        dino_ret = self.model.forward_features(input_img)['x_norm_patchtokens']
        dino_ret = dino_ret.transpose(1, 2).reshape([B, -1, H//self.patch_size, W//self.patch_size])    # [B, 1024, 128, 128]
        return dino_ret
