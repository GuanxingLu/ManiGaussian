import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import torchvision.transforms as T

from termcolor import colored, cprint
from dotmap import DotMap

import agents.manigaussian_bc.utils as utils
from agents.manigaussian_bc.models_embed import GeneralizableGSEmbedNet
from agents.manigaussian_bc.loss import l1_loss, l2_loss, cosine_loss, ssim
from agents.manigaussian_bc.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from agents.manigaussian_bc.gaussian_renderer import render

import visdom
import logging
import einops


def PSNR_torch(img1, img2, max_val=1, mask=None):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0).to(img1.device)
    PIXEL_MAX = max_val
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


class NeuralRenderer(nn.Module):
    """
    take a voxel, camera pose, and camera intrinsics as input,
    and output a rendered image
    """
    def __init__(self, cfg):
        super(NeuralRenderer, self).__init__()

        self.cfg = cfg
        self.coordinate_bounds = cfg.coordinate_bounds # bounds of voxel grid
        self.W = cfg.image_width
        self.H = cfg.image_height
        self.bg_color = cfg.dataset.bg_color

        self.znear = cfg.dataset.znear
        self.zfar = cfg.dataset.zfar
        self.trans = cfg.dataset.trans # default: [0, 0, 0]
        self.scale = cfg.dataset.scale

        # gs regressor
        self.gs_model = GeneralizableGSEmbedNet(cfg, with_gs_render=True)
        print(colored("[NeuralRenderer] GeneralizableGSEmbedNet is build", "cyan"))

        self.model_name = cfg.foundation_model_name
        self.d_embed = cfg.d_embed
        self.loss_embed_fn = cfg.loss_embed_fn

        if self.model_name == "diffusion":
            from odise.modeling.meta_arch.ldm import LdmFeatureExtractor
            import torchvision.transforms as T
            self.feature_extractor = LdmFeatureExtractor(
                            encoder_block_indices=(5, 7),
                            unet_block_indices=(2, 5, 8, 11),
                            decoder_block_indices=(2, 5),
                            steps=(0,),
                            captioner=None,
                        )
            self.diffusion_preprocess = T.Resize(512, antialias=True)
            cprint("diffusion feature dims: "+str(self.feature_extractor.feature_dims), "yellow")
        elif self.model_name == "dinov2":
            from agents.manigaussian_bc.dino_extractor import VitExtractor
            import torchvision.transforms as T
            self.feature_extractor = VitExtractor(
                model_name='dinov2_vitl14',
            )
            self.dino_preprocess = T.Compose([
                T.Resize(224 * 8, antialias=True),  # must be a multiple of 14
            ])
            cprint("dinov2 feature dims: "+str(self.feature_extractor.feature_dims), "yellow")
        else:
            cprint(f"foundation model {self.model_name} is not implemented", "yellow")

        self.lambda_embed = cfg.lambda_embed
        print(colored(f"[NeuralRenderer] foundation model {self.model_name} is build. loss weight: {self.lambda_embed}", "cyan"))

        self.lambda_rgb = 1.0 if cfg.lambda_rgb is None else cfg.lambda_rgb
        print(colored(f"[NeuralRenderer] rgb loss weight: {self.lambda_rgb}", "cyan"))

        self.use_dynamic_field = cfg.use_dynamic_field

    def _embed_loss_fn(self, render_embed, gt_embed):
        """
        render_embed: [bs, h, w, 3]
        gt_embed: [bs, h, w, 3]
        """
        if self.loss_embed_fn == "l2_norm":
            # label normalization
            MIN_DENOMINATOR = 1e-12
            gt_embed = (gt_embed - gt_embed.min()) / (gt_embed.max() - gt_embed.min() + MIN_DENOMINATOR)
            loss_embed = l2_loss(render_embed, gt_embed)
        elif self.loss_embed_fn == "l2":
            loss_embed = l2_loss(render_embed, gt_embed)
        elif self.loss_embed_fn == "cosine":
            loss_embed = cosine_loss(render_embed, gt_embed)
        else:
            cprint(f"loss_embed_fn {self.loss_embed_fn} is not implemented", "yellow")
        return loss_embed

    def _save_gradient(self, name):
        """
        for debugging language feature rendering
        """
        def hook(grad):
            print(f"name={name}, grad={grad}")
            return grad
        return hook

    def extract_foundation_model_feature(self, gt_rgb, lang_goal):
        """
        we use the last layer of the diffusion feature extractor
        since we reshape 128x128 img to 512x512, the last layer's feature is just 128x128
        thus, no need to resize the feature map
        lang_goal: numpy.ndarray, [bs, 1, 1]
        """
        
        if self.model_name == "diffusion":
            """
            we support multiple captions for batched input here
            """
            if lang_goal.shape[0] > 1:
                caption = ['a robot arm ' + cap.item() for cap in lang_goal]
            else:
                caption = "a robot arm " + lang_goal.item()
            batched_input = {'img': self.diffusion_preprocess(gt_rgb.permute(0, 3, 1, 2)), 'caption': caption}
            feature_list, lang_embed = self.feature_extractor(batched_input) # list of visual features, and 77x768 language embedding
            used_feature_idx = -1  
            gt_embed = feature_list[used_feature_idx]   # [bs,512,128,128]

            # NOTE: dimensionality reduction with PCA, which is used to satisfy the output dimension of the Gaussian Renderer
            bs = gt_rgb.shape[0]
            A = gt_embed.reshape(bs, 512, -1).permute(0, 2, 1)  # [bs, 128*128, 512]
            gt_embed_list = []
            for i in range(bs):
                U, S, V = torch.pca_lowrank(A[i], q=np.maximum(6, self.d_embed))
                reconstructed_embed = torch.matmul(A[i], V[:, :self.d_embed])
                gt_embed_list.append(reconstructed_embed)

            gt_embed = torch.stack(gt_embed_list, dim=0).permute(0, 2, 1).reshape(bs, self.d_embed, 128, 128)
            return gt_embed
        
        elif self.model_name == "dinov2":
            batched_input = self.dino_preprocess(gt_rgb.permute(0, 3, 1, 2))    # resize
            feature = self.feature_extractor(batched_input)
            gt_embed = F.interpolate(feature, size=(128, 128), mode='bilinear', align_corners=False)    # [b, 1024, 128, 128]

            # NOTE: dimensionality reduction with PCA, which is used to satisfy the output dimension of the Gaussian Renderer
            bs = gt_rgb.shape[0]
            A = gt_embed.reshape(bs, 1024, -1).permute(0, 2, 1)  # [bs, 128*128, 1024]
            gt_embed_list = []
            for i in range(bs):
                U, S, V = torch.pca_lowrank(A[i], q=np.maximum(6, self.d_embed))
                reconstructed_embed = torch.matmul(A[i], V[:, :self.d_embed])
                gt_embed_list.append(reconstructed_embed)
            gt_embed = torch.stack(gt_embed_list, dim=0).permute(0, 2, 1).reshape(bs, self.d_embed, 128, 128)
            return gt_embed
        else:
            return None

    def encode_data(self, pcd, dec_fts, lang, 
                    rgb=None, depth=None, focal=None, c=None, lang_goal=None, tgt_pose=None, tgt_intrinsic=None,
                    next_tgt_pose=None, next_tgt_intrinsic=None, action=None, step=None):
        '''prepare data dict'''
        bs = pcd.shape[0]
        data = {}
        # format input
        data['img'] = rgb
        data['dec_fts'] = dec_fts
        data['depth'] = depth
        data['lang'] = lang
        data['action'] = action
        data['step'] = step

        # novel pose
        data['novel_view'] = {}
        data['intr'] = tgt_intrinsic
        data['extr'] = tgt_pose
        data['xyz'] = einops.rearrange(pcd, 'b c h w -> b (h w) c')

        # use extrinsic pose to generate gaussain parameters
        if data['intr'] is not None:
            data_novel = self.get_novel_calib(data)
            data['novel_view'].update(data_novel)

        if self.use_dynamic_field:
            data['next'] = {
                'extr': next_tgt_pose,
                'intr': next_tgt_intrinsic,
                'novel_view': {},
            }
            if data['next']['intr'] is not None:
                data_novel = self.get_novel_calib(data['next'])
                data['next']['novel_view'].update(data_novel)

        return data

    def get_novel_calib(self, data):
        """
        get readable camera state for gaussian renderer from gt_pose
        :param data: dict
        :param data['intr']: intrinsic matrix
        :param data['extr']: c2w matrix

        :return: dict
        """
        bs = data['intr'].shape[0]
        device = data['intr'].device
        fovx_list, fovy_list, world_view_transform_list, full_proj_transform_list, camera_center_list = [], [], [], [], []
        for i in range(bs):
            intr = data['intr'][i, ...].cpu().numpy()
            extr = data['extr'][i, ...].cpu().numpy()
            extr = np.linalg.inv(extr)  # the saved extrinsic is actually cam2world matrix, so turn it to world2cam matrix

            width, height = self.W, self.H
            R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)    # inverse
            T = np.array(extr[:3, 3], np.float32)
            FovX = focal2fov(intr[0, 0], width)
            FovY = focal2fov(intr[1, 1], height)
            projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=intr, h=height, w=width).transpose(0, 1)
            world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.trans), self.scale)).transpose(0, 1) # [4, 4], w2c
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)    # [4, 4]
            camera_center = world_view_transform.inverse()[3, :3]   # inverse is c2w

            fovx_list.append(FovX)
            fovy_list.append(FovY)
            world_view_transform_list.append(world_view_transform.unsqueeze(0))
            full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
            camera_center_list.append(camera_center.unsqueeze(0))

        novel_view_data = {
            'FovX': torch.FloatTensor(np.array(fovx_list)).to(device),
            'FovY': torch.FloatTensor(np.array(fovy_list)).to(device),
            'width': torch.tensor([width] * bs).to(device),
            'height': torch.tensor([height] * bs).to(device),
            'world_view_transform': torch.concat(world_view_transform_list).to(device),
            'full_proj_transform': torch.concat(full_proj_transform_list).to(device),
            'camera_center': torch.concat(camera_center_list).to(device),
        }

        return novel_view_data

    def forward(self, pcd, dec_fts, language, gt_rgb=None, gt_pose=None, gt_intrinsic=None, rgb=None, depth=None, camera_intrinsics=None, camera_extrinsics=None, 
                focal=None, c=None, lang_goal=None, gt_depth=None,
                next_gt_pose=None, next_gt_intrinsic=None, next_gt_rgb=None, step=None, action=None,
                training=True):
        '''
        main forward function
        Return:
        :loss_dict: dict, loss values
        :ret_dict: dict, rendered images
        '''
        bs = rgb.shape[0]

        data = self.encode_data(
            rgb=rgb, depth=depth, pcd=pcd, focal=focal, c=c, lang_goal=None, tgt_pose=gt_pose, tgt_intrinsic=gt_intrinsic,
            dec_fts=dec_fts, lang=language, next_tgt_pose=next_gt_pose, next_tgt_intrinsic=next_gt_intrinsic, 
            action=action, step=step,
        )

        render_novel = None
        next_render_novel = None
        render_embed = None
        gt_embed = None

        # create gt feature from foundation models
        with torch.no_grad():
            gt_embed = self.extract_foundation_model_feature(gt_rgb, lang_goal)

        # if gt_rgb is not None:
        if training:
            # Gaussian Generator
            data = self.gs_model(data)

            # Gaussian Render
            data = self.pts2render(data, bg_color=self.bg_color) # default: [0, 0, 0]

            # Loss
            render_novel = data['novel_view']['img_pred'].permute(0, 2, 3, 1)   # [1, 128, 128, 3]

            if self.cfg.visdom:
                vis = visdom.Visdom()
                rgb_vis = data['img'][0].detach().cpu().numpy() * 0.5 + 0.5
                vis.image(rgb_vis, win='front_rgb', opts=dict(title='front_rgb'))

                depth_vis = data['depth'][0].detach().cpu().numpy()#/255.0
                # convert 128x128 0-255 depth map to 3x128x128 0-1 colored map
                vis.image(depth_vis, win='front_depth', opts=dict(title='front_depth'))
                vis.image(render_novel[0].permute(2, 0, 1).detach().cpu().numpy(), win='render_novel', opts=dict(title='render_novel'))
                vis.image(gt_rgb[0].permute(2, 0, 1).detach().cpu().numpy(), win='gt_novel', opts=dict(title='gt_novel'))

            # Ll1 = l1_loss(render_novel, gt_rgb)
            Ll1 = l2_loss(render_novel, gt_rgb)
            # Lssim = 1.0 - ssim(render_novel, gt_rgb)
            Lssim = 0.
            psnr = PSNR_torch(render_novel, gt_rgb)

            loss = 0.
            # loss_rgb = self.cfg.lambda_l1 * Ll1 + self.cfg.lambda_ssim * Lssim
            loss_rgb = Ll1
            loss += loss_rgb

            if gt_embed is not None:
                gt_embed = gt_embed.permute(0, 2, 3, 1) # channel last
                render_embed = data['novel_view']['embed_pred'].permute(0, 2, 3, 1)

                # DEBUG gradient
                # render_embed_grad = render_embed.register_hook(self._save_gradient('render_embed'))

                loss_embed = self._embed_loss_fn(render_embed, gt_embed)
                loss += self.cfg.lambda_embed * loss_embed
            else:
                loss_embed = torch.tensor(0.)

            # next frame prediction
            if self.use_dynamic_field and (next_gt_rgb is not None) and ('xyz_maps' in data['next']):
                data['next'] = self.pts2render(data['next'], bg_color=self.bg_color)
                next_render_novel = data['next']['novel_view']['img_pred'].permute(0, 2, 3, 1)
                # loss_dyna = l1_loss(next_render_novel, next_gt_rgb)
                loss_dyna = l2_loss(next_render_novel, next_gt_rgb)
                lambda_dyna = self.cfg.lambda_dyna if step >= self.cfg.next_mlp.warm_up else 0.
                loss += lambda_dyna * loss_dyna

                loss_reg = torch.tensor(0.)
                # TODO: regularization on deformation
                # if self.cfg.lambda_reg > 0:
                #     loss_reg = l2_loss(data['next']['xyz_maps'], data['xyz_maps'].detach())
                #     lambda_reg = self.cfg.lambda_reg if step >= self.cfg.next_mlp.warm_up else 0.
                #     loss += lambda_reg * loss_reg

                # TODO: local rigid loss

            else:
                loss_dyna = torch.tensor(0.)
                loss_reg = torch.tensor(0.)

            loss_dict = {
                'loss': loss,
                'loss_rgb': loss_rgb.item(),
                'loss_embed': loss_embed.item(),
                'loss_dyna': loss_dyna.item(),
                'loss_reg': loss_reg.item(),
                'l1': Ll1.item(),
                'psnr': psnr.item(),
                }
        else:
            # no ground-truth given, rendering (inference)
            with torch.no_grad():
                # Gaussian Generator
                data = self.gs_model(data)
                # Gaussian Render
                data = self.pts2render(data, bg_color=self.bg_color) # default: [0, 0, 0]
                render_novel = data['novel_view']['img_pred'].permute(0, 2, 3, 1) # channel last
                render_embed = data['novel_view']['embed_pred'].permute(0, 2, 3, 1)
                
                if self.use_dynamic_field and 'xyz_maps' in data['next']:
                    data['next'] = self.pts2render(data['next'], bg_color=self.bg_color)
                    next_render_novel = data['next']['novel_view']['img_pred'].permute(0, 2, 3, 1)

                loss_dict = {
                    'loss': 0.,
                    'loss_rgb': 0.,
                    'loss_embed': 0.,
                    'loss_dyna': 0.,
                    'loss_reg': 0.,
                    'l1': 0.,
                    'psnr': 0.,
                }

        # get Gaussian embedding
        ret_dict = DotMap(render_novel=render_novel, next_render_novel=next_render_novel,
                          render_embed=render_embed, gt_embed=gt_embed)

        return loss_dict, ret_dict
    
    def pts2render(self, data: dict, bg_color=[0,0,0]):
        '''use render function in GS'''
        bs = data['intr'].shape[0]
        assert bs == 1, "batch size should be 1"
        i = 0
        xyz_i = data['xyz_maps'][i, :, :]
        feature_i = data['sh_maps'][i, :, :, :] # [16384, 4, 3]
        rot_i = data['rot_maps'][i, :, :]
        scale_i = data['scale_maps'][i, :, :]
        opacity_i = data['opacity_maps'][i, :, :]
        feature_language_i = data['feature_maps'][i, :, :]

        render_return_dict = render(
            data, i, xyz_i, rot_i, scale_i, opacity_i, 
            bg_color=bg_color, pts_rgb=None, features_color=feature_i, features_language=feature_language_i
            )

        data['novel_view']['img_pred'] = render_return_dict['render'].unsqueeze(0)
        data['novel_view']['embed_pred'] = render_return_dict['render_embed'].unsqueeze(0)
        return data
