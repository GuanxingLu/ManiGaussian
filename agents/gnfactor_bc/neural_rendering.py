import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import torchvision.transforms as T
import agents.gnfactor_bc.utils as utils

from termcolor import colored, cprint
from dotmap import DotMap
from agents.gnfactor_bc.models_embed import GeneralizableNeRFEmbedNet


def PSNR_torch(img1, img2, max_val=1):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
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
        self.z_near = cfg.z_near
        self.z_far = cfg.z_far
        self.regress_coord = cfg.regress_coord
        self.regress_attention = cfg.regress_attention

        self.n_coarse = cfg.n_coarse
        self.n_fine = cfg.n_fine
        self.n_fine_depth = cfg.n_fine_depth
        self.lindisp = cfg.lindisp
        self.using_fine = self.n_fine > 0
        self.eval_batch_size = cfg.eval_batch_size
        self.ret_last_feat = cfg.ret_last_feat

        self.noise_std = cfg.noise_std
        self.white_bkgd = cfg.white_bkgd
        self.depth_std = cfg.depth_std


        self.nerf_model = GeneralizableNeRFEmbedNet(cfg)
        print(colored("[NeuralRenderer] GeneralizableNeRFEmbedNet is build", "cyan"))

        self.model_name = cfg.foundation_model_name
    
        if self.model_name == "diffusion":
            from odise.modeling.meta_arch.ldm import LdmFeatureExtractor
            import torchvision.transforms as T
            self.diffusion_extractor = LdmFeatureExtractor(
                            encoder_block_indices=(5, 7),
                            unet_block_indices=(2, 5, 8, 11),
                            decoder_block_indices=(2, 5),
                            steps=(0,),
                            captioner=None,
                        )
            self.diffusion_preprocess = T.Resize(512, antialias=True)
            cprint("diffusion feature dims: "+str(self.diffusion_extractor.feature_dims), "yellow")
        else:
            cprint(f"foundation model {self.model_name} is not implemented", "yellow")
            # raise NotImplementedError(f"foundation model {self.model_name} is not implemented")

        self.lambda_embed = cfg.lambda_embed
        self.lambda_rgb = 1.0 if cfg.lambda_rgb is None else cfg.lambda_rgb
        print(colored(f"[NeuralRenderer] foundation model {self.model_name} is build. loss weight: {self.lambda_embed}", "cyan"))
        print(colored(f"[NeuralRenderer] rgb loss weight: {self.lambda_rgb}", "cyan"))


    def sample_coarse(self, rays):
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :return (B, Kc)
        """
        device = rays.device
        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)

        step = 1.0 / self.n_coarse
        B = rays.shape[0]
        z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)  # (Kc)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
        z_steps += torch.rand_like(z_steps) * step
        if not self.lindisp:  # Use linear sampling in depth space
            return near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            return 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)


    def sample_fine(self, rays, weights):
        """
        Weighted stratified (importance) sample
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param weights (B, Kc)
        :return (B, Kf-Kfd)
        """
        device = rays.device
        B = rays.shape[0]

        weights = weights.detach() + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (B, Kc)
        cdf = torch.cumsum(pdf, -1)  # (B, Kc)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (B, Kc+1)

        u = torch.rand(
            B, self.n_fine - self.n_fine_depth, dtype=torch.float32, device=device
        )  # (B, Kf)
        inds = torch.searchsorted(cdf, u, right=True).float() - 1.0  # (B, Kf)
        inds = torch.clamp_min(inds, 0.0)

        z_steps = (inds + torch.rand_like(inds)) / self.n_coarse  # (B, Kf)

        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)
        if not self.lindisp:  # Use linear sampling in depth space
            z_samp = near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            z_samp = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)
        return z_samp


    def sample_fine_depth(self, rays, depth):
        """
        Sample around specified depth
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param depth (B)
        :return (B, Kfd)
        """
        z_samp = depth.unsqueeze(1).repeat((1, self.n_fine_depth))
        z_samp += torch.randn_like(z_samp) * self.depth_std
        # Clamp does not support tensor bounds
        z_samp = torch.max(torch.min(z_samp, rays[:, -1:]), rays[:, -2:-1])
        return z_samp


    def composite(self, model, rays, z_samp, coarse=True, sb=0):
        """
        Render RGB and depth for each ray using NeRF alpha-compositing formula,
        given sampled positions along each ray (see sample_*)
        :param model should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
        should also support 'coarse' boolean argument
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param z_samp z positions sampled for each ray (B, K)
        :param coarse whether to evaluate using coarse NeRF
        :param sb super-batch dimension; 0 = disable
        :return weights (B, K), rgb (B, 3), depth (B)
        """
        with profiler.record_function("renderer_composite"):
            B, K = z_samp.shape # B=512, K=64

            deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
            delta_inf = rays[:, -1:] - z_samp[:, -1:]
            deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)=[512,64]

            # (B, K, 3)
            points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
            points = points.reshape(-1, 3)  # (B*K, 3)

            use_viewdirs = hasattr(model, "use_viewdirs") and model.use_viewdirs

            val_all, last_feat_all, point_density_all = [], [], []
            if sb > 0:  # default: 1
                points = points.reshape(
                    sb, -1, 3
                )  # (SB, B'*K, 3) B' is real ray batch size, [1, 32768, 3]
                eval_batch_size = (self.eval_batch_size - 1) // sb + 1  # 4096
                eval_batch_dim = 1
            else:
                eval_batch_size = self.eval_batch_size
                eval_batch_dim = 0

            split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)
            if use_viewdirs:    # True
                dim1 = K
                viewdirs = rays[:, None, 3:6].expand(-1, dim1, -1)  # (B, K, 3)
                if sb > 0:
                    viewdirs = viewdirs.reshape(sb, -1, 3)  # (SB, B'*K, 3)
                else:
                    viewdirs = viewdirs.reshape(-1, 3)  # (B*K, 3)
                split_viewdirs = torch.split(
                    viewdirs, eval_batch_size, dim=eval_batch_dim
                )
                for pnts, dirs in zip(split_points, split_viewdirs):
                    # pnts: [1, 4096, 3], dirs: [1, 4096, 3]
                    if not self.ret_last_feat:  # default: False
                        val, point_density = model(pnts, coarse=coarse, viewdirs=dirs)
                        val_all.append(val)
                        point_density_all.append(point_density)
                    else:
                        val, last_feat, point_density = model(pnts, coarse=coarse, viewdirs=dirs, ret_last_feat=True)
                        val_all.append(val)
                        last_feat_all.append(last_feat)
                        point_density_all.append(point_density)
            else:
                raise NotImplementedError

            points = None
            viewdirs = None
            # (B*K, 4) OR (SB, B'*K, 4)
            out = torch.cat(val_all, dim=eval_batch_dim)
            out = out.reshape(B, K, -1)  # (B, K, 4 or 5)

            if point_density is not None:
                point_density_all = torch.cat(point_density_all, dim=eval_batch_dim)
                point_density_all = point_density_all.reshape(B, K)  # (B, K)

            rgbs = out[..., :3]  # (B, K, 3)
            sigmas = out[..., 3]  # (B, K)
            if not self.regress_coord and not self.regress_attention:   # default
                embeds = out[..., 4:]   # [512,64,512]
            elif self.regress_coord and not self.regress_attention:
                embeds = out[..., 4:-3]
                coords = out[..., -3:]
            elif not self.regress_coord and self.regress_attention:
                embeds = out[..., 4:-6]
                attention = out[..., -6:]
            elif self.regress_coord and self.regress_attention:
                embeds = out[..., 4:-9]
                coords = out[..., -9:-6]
                attention = out[..., -6:]
            

            if self.ret_last_feat:
                embeds = torch.cat(last_feat_all, dim=eval_batch_dim)
                embeds = embeds.reshape(B, K, -1)

            if self.training and self.noise_std > 0.0:
                sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std

            # volumetric rendering
            alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (B, K)
            deltas = None
            sigmas = None
            alphas_shifted = torch.cat(
                [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
            )  # (B, K+1) = [1, a1, a2, ...]
            T = torch.cumprod(alphas_shifted, -1)  # (B)
            weights = alphas * T[:, :-1]  # (B, K)
            alphas = None
            alphas_shifted = None

            rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (B, 3)
            embed_final = torch.sum(weights.unsqueeze(-1) * embeds, -2)  # (B, D)
            
            if self.regress_attention:
                attention_final = torch.sum(weights.unsqueeze(-1) * attention, -2)

            if self.regress_coord:
                coord_final = torch.mean(coords, -2)

            depth_final = torch.sum(weights * z_samp, -1)  # (B)
            
            if self.white_bkgd:
                # White background
                pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
                rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)  # (B, 3)

            if not self.regress_coord and not self.regress_attention:   # True
                return weights, rgb_final, embed_final, depth_final  # make sure depth_final is the last one
            elif self.regress_coord and not self.regress_attention:
                return weights, rgb_final, embed_final, coord_final, depth_final
            elif not self.regress_coord and self.regress_attention:
                return weights, rgb_final, embed_final, attention_final, depth_final
            elif self.regress_coord and self.regress_attention:
                return weights, rgb_final, embed_final, coord_final, attention_final, depth_final


    def _format_outputs(
        self, rendered_outputs, superbatch_size, want_weights=False,
    ):
        if not self.regress_coord and not self.regress_attention:
            weights, rgb, embed, depth = rendered_outputs
        elif self.regress_coord and not self.regress_attention:
            weights, rgb, embed, coord, depth = rendered_outputs
        elif not self.regress_coord and self.regress_attention:
            weights, rgb, embed, attention, depth = rendered_outputs
        elif self.regress_coord and self.regress_attention:
            weights, rgb, embed, coord, attention, depth = rendered_outputs

        if superbatch_size > 0:
            rgb = rgb.reshape(superbatch_size, -1, 3)
            embed = embed.reshape(superbatch_size, -1, embed.shape[-1])
            depth = depth.reshape(superbatch_size, -1)
            weights = weights.reshape(superbatch_size, -1, weights.shape[-1])
            if self.regress_coord:
                coord = coord.reshape(superbatch_size, -1, 3)
            if self.regress_attention:
                attention = attention.reshape(superbatch_size, -1, attention.shape[-1])
        ret_dict = DotMap(rgb=rgb, embed=embed, depth=depth)
        if want_weights:
            ret_dict.weights = weights
        if self.regress_coord:
            ret_dict.coord = coord
        if self.regress_attention:
            ret_dict.attention = attention
        return ret_dict


    def encode(self, multi_scale_voxel_list, voxel_density, lang, voxel_feat, poses, focal, c=None):
        self.nerf_model.encode(multi_scale_voxel_list=multi_scale_voxel_list,
                               voxel_density=voxel_density,
                                lang=lang, voxel_feat=voxel_feat,
                                poses=poses, focal=focal, c=c)


    def forward_nerf(self, rays, want_weights=False):
        assert len(rays.shape) == 3
        superbatch_size = rays.shape[0]
        rays = rays.reshape(-1, 8)  # (SB * B, 8)

        z_coarse = self.sample_coarse(rays)  # (B, Kc)


        coarse_composite = self.composite(
                self.nerf_model, rays, z_coarse, coarse=True, sb=superbatch_size,
            )
        
        outputs = DotMap(
                coarse=self._format_outputs(
                    coarse_composite, superbatch_size, want_weights=want_weights,
                ),
            )
        if self.using_fine: # True
                all_samps = [z_coarse]
                if self.n_fine - self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine(rays, coarse_composite[0].detach())
                    )  # (B, Kf - Kfd)
                if self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine_depth(rays, coarse_composite[-1])
                    )  # (B, Kfd)
                z_combine = torch.cat(all_samps, dim=-1)  # (B, Kc + Kf)
                z_combine_sorted, argsort = torch.sort(z_combine, dim=-1)

                fine_composite = self.composite(
                    self.nerf_model, rays, z_combine_sorted, coarse=False, sb=superbatch_size,
                )
                outputs.fine = self._format_outputs(
                    fine_composite, superbatch_size, want_weights=want_weights,
                )
        return outputs


    @torch.no_grad()
    def rendering(self, voxel_feat, language, multi_scale_voxel_list, voxel_density, voxel_pose, focal, tgt_pose, c=None):
        rays = utils.gen_rays(tgt_pose, self.W, self.H, focal, self.z_near, self.z_far, c=c)    # [B, H, W, 8]
        self.encode(multi_scale_voxel_list=multi_scale_voxel_list, \
                    voxel_density=voxel_density,
                    lang=language, voxel_feat=voxel_feat, poses=voxel_pose, focal=focal, c=c)
        B, H, W, DimRay = rays.shape
        rays = rays.reshape(B*H*W, DimRay)
        chunk_size = 4096
        rgbs = []
        embeds = []
        depths = []
        for i in range(0, rays.shape[0], chunk_size):
            output = self.forward_nerf(rays[i:i+chunk_size].unsqueeze(0))

            coarse = output.coarse
            fine = output.fine

            rgb = fine.rgb
            embed = fine.embed
            depth = fine.depth
            rgbs.append(rgb.squeeze(0))
            embeds.append(embed.squeeze(0))
            depths.append(depth.squeeze(0))
        
        rgbs = torch.cat(rgbs, dim=0).reshape(B, H, W, 3)
        embeds = torch.cat(embeds, dim=0).reshape(B, H, W, -1)
        depths = torch.cat(depths, dim=0).reshape(B, H, W)
        return rgbs, embeds, depths
    
    
    def extract_foundation_model_feature(self, gt_rgb, lang_goal):
        """
        we use the last layer of the diffusion feature extractor
        since we reshape 128x128 img to 512x512, the last layer's feature is just 128x128
        thus, no need to resize the feature map
        """
        
        if self.model_name == "diffusion":
            """
            we support multiple captions for batched input here
            """
            if isinstance(lang_goal, list):
                caption = ['a robot arm ' + cap.item() for cap in lang_goal]
            else:
                caption = "a robot arm " + lang_goal.item()
            batched_input = {'img': self.diffusion_preprocess(gt_rgb.permute(0, 3, 1, 2)), 'caption': caption}
            feature_list, lang_embed = self.diffusion_extractor(batched_input) # list of visual features, and 77x768 language embedding
            used_feature_idx = -1  
            gt_embed = feature_list[used_feature_idx]
            
        else:
            return None
            # raise NotImplementedError(f"foundation model {self.model_name} is not implemented")
        
        return gt_embed

     
    def compute_rendering_loss(self, multi_scale_voxel_list, voxel_density, language, voxel_feat, voxel_poses, focal, gt_rgb, gt_depth, gt_pose, c=None, lang_goal=None):

        rays = utils.gen_rays(gt_pose, self.W, self.H, focal, self.z_near, self.z_far, c=c)
        self.encode(multi_scale_voxel_list=multi_scale_voxel_list,\
                    voxel_density=voxel_density,\
                    lang=language, voxel_feat=voxel_feat, \
                    poses=voxel_poses, focal=focal, c=c)
        B, H, W, DimRay = rays.shape
        rays = rays.reshape(B, H*W, DimRay)

        # sample rays from H*W
        chunk_size = self.cfg.ray_chunk_size    # 512
        sampled_rays_idx = torch.randint(H*W, (chunk_size, ), device=rays.device)
        sampled_rays = rays[:, sampled_rays_idx, :]

        # render
        outputs = self.forward_nerf(sampled_rays)

        loss = 0.

        # create feature
        with torch.no_grad():
            gt_embed = self.extract_foundation_model_feature(gt_rgb, lang_goal)
            if gt_embed is not None:
                gt_embed = gt_embed.permute(0, 2, 3, 1) # channel last, [bs,128,128,512]

        # rgb loss
        gt_rgb = gt_rgb.reshape(B, H*W, 3)
        gt_rgb = gt_rgb[:, sampled_rays_idx, :]
        lambda_rgb = self.lambda_rgb
        loss_rgb_coarse = lambda_rgb * F.mse_loss(outputs.coarse.rgb, gt_rgb)
        loss_rgb_fine = lambda_rgb * F.mse_loss(outputs.fine.rgb, gt_rgb)
        loss += loss_rgb_coarse + loss_rgb_fine

        psnr = PSNR_torch(outputs.fine.rgb, gt_rgb)

        if gt_embed is not None:
            # embed loss
            lambda_embed = self.lambda_embed
            gt_embed = gt_embed.reshape(B, H*W, -1) # [bs,16384,512]
            gt_embed = gt_embed[:, sampled_rays_idx, :] # [bs,512,512], randomly sample 512 pixels
            
            # we use MSE
            loss_embed_coarse = lambda_embed * F.mse_loss(outputs.coarse.embed, gt_embed)
            loss_embed_fine = lambda_embed * F.mse_loss(outputs.fine.embed, gt_embed)
            loss +=  loss_embed_coarse +  loss_embed_fine
        else:
            loss_embed_coarse = torch.tensor(0.)
            loss_embed_fine = torch.tensor(0.)

        return {'loss': loss,
            'loss_rgb_coarse': loss_rgb_coarse.item(),
            'loss_rgb_fine': loss_rgb_fine.item(),
            'loss_rgb': (loss_rgb_coarse.item() + loss_rgb_fine.item()),
            'loss_embed_coarse': loss_embed_coarse.item(),
            'loss_embed_fine': loss_embed_fine.item(),
            'loss_embed': (loss_embed_coarse.item() + loss_embed_fine.item()),
             'psnr': psnr.item()}


    def forward(self, multi_scale_voxel_list, voxel_density, language, voxel_feat, voxel_poses, focal, gt_rgb, gt_depth, gt_pose, c=None, lang_goal=None):
        return self.compute_rendering_loss(multi_scale_voxel_list, voxel_density, language, voxel_feat, voxel_poses, focal, gt_rgb, gt_depth, gt_pose, c, lang_goal)

