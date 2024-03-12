import logging
import os
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from yarr.agents.agent import Agent, ActResult, ScalarSummary, \
    HistogramSummary, ImageSummary, Summary
from termcolor import colored, cprint
import io

from helpers.utils import visualise_voxel
from voxel.voxel_grid import VoxelGrid
from voxel.augmentation import apply_se3_augmentation_with_camera_pose
from helpers.clip.core.clip import build_model, load_clip
import PIL.Image as Image
import transformers
from helpers.optim.lamb import Lamb
from torch.nn.parallel import DistributedDataParallel as DDP
from agents.gnfactor_bc.neural_rendering import NeuralRenderer
from helpers.language_model import create_language_model

import wandb


NAME = 'QAttentionAgent'


def visualize_feature_map_by_clustering(features, num_cluster=4, return_cluster_center=False):
    from sklearn.cluster import KMeans
    features = features.cpu().detach().numpy()
    B, D, H, W = features.shape
    features_1d = features.reshape(B, D, H*W).transpose(0, 2, 1).reshape(-1, D)
    kmeans = KMeans(n_clusters=num_cluster, random_state=0, n_init=10).fit(features_1d)
    labels = kmeans.labels_
    labels = labels.reshape(H, W)

    cluster_colors = [
        np.array([255, 0, 0]),   # red
        np.array([0, 255, 0]),       # green
        np.array([0, 0, 255]),      # blue
        np.array([255, 255, 0]),   # yellow
        np.array([255, 0, 255]),  # magenta
    ]

    segmented_img = np.zeros((H, W, 3))
    for i in range(num_cluster):
        segmented_img[labels==i] = cluster_colors[i]
        
    if return_cluster_center:
        cluster_centers = []
        for i in range(num_cluster):
            cluster_pixels = np.argwhere(labels == i)
            cluster_center = cluster_pixels.mean(axis=0)
            cluster_centers.append(cluster_center)
        return labels, cluster_centers
        
    return segmented_img
 
 
def PSNR_torch(img1, img2, max_val=1):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = max_val
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


def parse_camera_file(file_path):
    """
    Parse our camera format.

    The format is (*.txt):
    
    4x4 matrix (camera extrinsic)
    space
    3x3 matrix (camera intrinsic)

    focal is extracted from the intrinsc matrix
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    camera_extrinsic = []
    for x in lines[0:4]:
        camera_extrinsic += [float(y) for y in x.split()]
    camera_extrinsic = np.array(camera_extrinsic).reshape(4, 4)

    camera_intrinsic = []
    for x in lines[5:8]:
        camera_intrinsic += [float(y) for y in x.split()]
    camera_intrinsic = np.array(camera_intrinsic).reshape(3, 3)

    focal = camera_intrinsic[0, 0]

    return camera_extrinsic, camera_intrinsic, focal


def parse_img_file(file_path):
    """
    return np.array of RGB image with range [0, 1]
    """
    rgb = Image.open(file_path).convert('RGB')
    rgb = np.asarray(rgb).astype(np.float32) / 255.0
    return rgb


class QFunction(nn.Module):

    def __init__(self,
                 perceiver_encoder: nn.Module,
                 voxelizer: VoxelGrid,
                 bounds_offset: float,
                 rotation_resolution: float,
                 device,
                 training,
                 use_ddp=True,  # default: True
                 cfg=None):
        super(QFunction, self).__init__()
        self._rotation_resolution = rotation_resolution
        self._voxelizer = voxelizer
        self._bounds_offset = bounds_offset
        self._qnet = perceiver_encoder.to(device)
        self._coord_trans = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32)).to(device)
        self.cfg = cfg
        if cfg.use_neural_rendering:
            self._neural_renderer = NeuralRenderer(cfg.neural_renderer).to(device)
            if training and use_ddp:
                self._neural_renderer = DDP(self._neural_renderer, device_ids=[device], find_unused_parameters=True)
        else:
            self._neural_renderer = None
        print(colored(f"[NeuralRenderer]: {cfg.use_neural_rendering}", "cyan"))
        
        # distributed training
        if training and use_ddp:
            self._qnet = DDP(self._qnet, device_ids=[device], find_unused_parameters=True)
        
        self.device = device


    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices


    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision
    

    def forward(self, rgb_pcd, proprio, pcd, camera_extrinsics, camera_intrinsics, lang_goal_emb, lang_token_embs,
                bounds=None, prev_bounds=None, prev_layer_voxel_grid=None,
                use_neural_rendering=False, nerf_target_rgb=None, 
                nerf_target_pose=None,
                lang_goal=None,
                gt_embed=None):
        # rgb_pcd will be list of list (list of [rgb, pcd])

        b = rgb_pcd[0][0].shape[0]
        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1)  # [1, 16384, 3]
        
        # flatten RGBs and Pointclouds
        rgb = [rp[0] for rp in rgb_pcd]
        feat_size = rgb[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in rgb], 1)  # [1, 16384, 3]

        # construct voxel grid
        voxel_grid, voxel_density = self._voxelizer.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds, return_density=True)

        # swap to channels fist
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach() # Bx10x100x100x100

        # batch bounds if necessary
        if bounds.shape[0] != b:
            bounds = bounds.repeat(b, 1)

        # forward pass
        q_trans, \
        q_rot_and_grip,\
        q_ignore_collisions,\
        voxel_grid_feature, \
        multi_scale_voxel_list, \
        lang_embedd = self._qnet(voxel_grid,  # [1,10,100^3]
                                proprio, # [1,4]
                                lang_goal_emb, # [1,1024]
                                lang_token_embs, # [1,77,512]
                                None,
                                bounds, # [1,6]
                                None)


        rendering_loss_dict = {}

        if use_neural_rendering:    # train default: True; eval default: False
            # prepare nerf rendering
            focal = camera_intrinsics[0][:, 0, 0]  # [SB]
            focal = torch.tensor((focal, focal), dtype=torch.float32).unsqueeze(0)
            cx = 128 / 2
            cy = 128 / 2
            c = torch.tensor([cx, cy], dtype=torch.float32).unsqueeze(0)

            if nerf_target_rgb is not None:
                gt_rgb = nerf_target_rgb    # [1,128,128,3]
                gt_pose = nerf_target_pose @ self._coord_trans # remember to do this

                # render
                rendering_loss_dict =  self._neural_renderer(voxel_feat=voxel_grid_feature, \
                                                language=lang_embedd, \
                                                multi_scale_voxel_list=multi_scale_voxel_list, \
                                                voxel_density=voxel_density, \
                                                voxel_poses=None, \
                                                gt_depth=None, \
                                                focal=focal, c=c, gt_rgb=gt_rgb,
                                                gt_pose=gt_pose, lang_goal=lang_goal, 
                                                )

            else:
                # # if we do not have additional multi-view data, we use input view as reconstruction target
                rendering_loss_dict = {'loss': 0., 'loss_rgb_coarse': 0., 'loss_rgb_fine': 0.,
                        'loss_rgb': 0., 'loss_embed_coarse': 0., 'loss_embed_fine': 0., 'loss_embed': 0.,   'psnr':0. }

            

        return q_trans, q_rot_and_grip, q_ignore_collisions, voxel_grid, rendering_loss_dict


    @torch.no_grad()
    def render(self, rgb_pcd, proprio, pcd, camera_extrinsics, camera_intrinsics, lang_goal_emb, lang_token_embs,
                tgt_pose=None,
                bounds=None, prev_bounds=None, prev_layer_voxel_grid=None):
        """
        rgb_pcd: list of [(1,3,128,128), (1,3,128,128)]
        proprio: (1,4)
        pcd: list of (1,3,128,128)
        camera_intrinsics: list of (1,3,3)
        tgt_pose: (1,4,4)
        """
        # rgb_pcd will be list of list (list of [rgb, pcd])
        b = rgb_pcd[0][0].shape[0]
        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1)

        # flatten RGBs and Pointclouds
        rgb = [rp[0] for rp in rgb_pcd]
        feat_size = rgb[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in rgb], 1)

        # construct voxel grid
        voxel_grid, voxel_density = self._voxelizer.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds, return_density=True)

        # swap to channels fist
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()

        # batch bounds if necessary
        if bounds.shape[0] != b:
            bounds = bounds.repeat(b, 1)

        # forward pass
        if isinstance(self._qnet, DDP):
            q_trans, \
            q_rot_and_grip,\
            q_ignore_collisions,\
            voxel_grid_feature,\
            multi_scale_voxel_list,\
            lang_embedd = self._qnet.module(voxel_grid, 
                                            proprio,
                                            lang_goal_emb, 
                                            lang_token_embs,
                                            prev_layer_voxel_grid,
                                            bounds, 
                                            prev_bounds)
        else:
            q_trans, \
            q_rot_and_grip,\
            q_ignore_collisions,\
            voxel_grid_feature,\
            multi_scale_voxel_list,\
            lang_embedd = self._qnet(voxel_grid, 
                                            proprio,
                                            lang_goal_emb, 
                                            lang_token_embs,
                                            prev_layer_voxel_grid,
                                            bounds, 
                                            prev_bounds)
        rendering_loss_dict = {}

        # prepare nerf rendering
        focal = camera_intrinsics[0][:, 0, 0]  # [SB]
        focal = torch.tensor((focal, focal), dtype=torch.float32).unsqueeze(0)
        cx = 128 / 2
        cy = 128 / 2
        c = torch.tensor([cx, cy], dtype=torch.float32).unsqueeze(0)
        rgb_gt = rgb[0].permute(0, 2, 3, 1) # support only one camera. channel last
        tgt_pose = tgt_pose @ self._coord_trans.to(tgt_pose.device)

        if isinstance(self._neural_renderer, DDP):
            rgb_render, embed_render, depth_render = self._neural_renderer.module.rendering(voxel_feat=voxel_grid_feature, \
                                            language=lang_embedd, \
                                            multi_scale_voxel_list=multi_scale_voxel_list, \
                                            voxel_density=voxel_density, \
                                            voxel_pose=None, \
                                            tgt_pose=tgt_pose, focal=focal, c=c,)
        else:
            rgb_render, embed_render, depth_render = self._neural_renderer.rendering(voxel_feat=voxel_grid_feature, \
                                                language=lang_embedd, \
                                                multi_scale_voxel_list=multi_scale_voxel_list, \
                                                voxel_density=voxel_density, \
                                                voxel_pose=None, \
                                                tgt_pose=tgt_pose, focal=focal, c=c,)
                                                                    

        return rgb_render, embed_render, depth_render


class QAttentionPerActBCAgent(Agent):

    def __init__(self,
                 layer: int,
                 coordinate_bounds: list,
                 perceiver_encoder: nn.Module,
                 camera_names: list,
                 batch_size: int,
                 voxel_size: int,
                 bounds_offset: float,
                 voxel_feature_size: int,
                 image_crop_size: int,
                 num_rotation_classes: int,
                 rotation_resolution: float,
                 lr: float = 0.0001,
                 lr_scheduler: bool = False,
                 training_iterations: int = 100000,
                 num_warmup_steps: int = 20000,
                 trans_loss_weight: float = 1.0,
                 rot_loss_weight: float = 1.0,
                 grip_loss_weight: float = 1.0,
                 collision_loss_weight: float = 1.0,
                 include_low_dim_state: bool = False,
                 image_resolution: list = None,
                 lambda_weight_l2: float = 0.0,
                 transform_augmentation: bool = True,   # True
                 transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
                 transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
                 transform_augmentation_rot_resolution: int = 5,
                 optimizer_type: str = 'adam',
                 num_devices: int = 1,
                 cfg = None,
                 ):
        self._layer = layer
        self._coordinate_bounds = coordinate_bounds
        self._perceiver_encoder = perceiver_encoder
        self._voxel_feature_size = voxel_feature_size
        self._bounds_offset = bounds_offset
        self._image_crop_size = image_crop_size
        self._lr = lr
        self._lr_scheduler = lr_scheduler
        self._training_iterations = training_iterations
        self._num_warmup_steps = num_warmup_steps
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._include_low_dim_state = include_low_dim_state
        self._image_resolution = image_resolution or [128, 128]
        self._voxel_size = voxel_size
        self._camera_names = camera_names
        self._num_cameras = len(camera_names)
        self._batch_size = batch_size
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = torch.from_numpy(np.array(transform_augmentation_xyz))
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._optimizer_type = optimizer_type
        self._num_devices = num_devices
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution

        self.cfg = cfg
        print(colored(f"[agent] nerf weight step: {self.cfg.neural_renderer.lambda_nerf}", "red"))

        self.use_neural_rendering = self.cfg.use_neural_rendering
        print(colored(f"use_neural_rendering: {self.use_neural_rendering}", "red"))

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self._name = NAME + '_layer' + str(self._layer)

    def build(self, training: bool, device: torch.device = None, use_ddp=True):
        self._training = training
        self._device = device

        if device is None:
            device = torch.device('cpu')

        self._voxelizer = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=self._batch_size if training else 1,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod(self._image_resolution) * self._num_cameras,
        )

        self._q = QFunction(self._perceiver_encoder,
                            self._voxelizer,
                            self._bounds_offset,
                            self._rotation_resolution,
                            device,
                            training,
                            use_ddp,
                            self.cfg).to(device).train(training)

        grid_for_crop = torch.arange(0,
                                     self._image_crop_size,
                                     device=device).unsqueeze(0).repeat(self._image_crop_size, 1).unsqueeze(-1)
        self._grid_for_crop = torch.cat([grid_for_crop.transpose(1, 0),
                                         grid_for_crop], dim=2).unsqueeze(0)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds,
                                               device=device).unsqueeze(0)

        if self._training:
            # optimizer
            if self._optimizer_type == 'lamb':
                self._optimizer = Lamb(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
            elif self._optimizer_type == 'adam':
                self._optimizer = torch.optim.Adam(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            else:
                raise Exception('Unknown optimizer type')

            # learning rate scheduler
            if self._lr_scheduler:
                self._scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                    self._optimizer,
                    num_warmup_steps=self._num_warmup_steps,
                    num_training_steps=self._training_iterations,
                    num_cycles=self._training_iterations // 10000,
                )

            # one-hot zero tensors
            self._action_trans_one_hot_zeros = torch.zeros((self._batch_size,
                                                            1,
                                                            self._voxel_size,
                                                            self._voxel_size,
                                                            self._voxel_size),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_x_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_y_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_z_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_grip_one_hot_zeros = torch.zeros((self._batch_size,
                                                           2),
                                                           dtype=int,
                                                           device=device)
            self._action_ignore_collisions_one_hot_zeros = torch.zeros((self._batch_size,
                                                                        2),
                                                                        dtype=int,
                                                                        device=device)

            # print total params
            logging.info('# Q Params: %d M' % (sum(
                p.numel() for name, p in self._q.named_parameters() \
                if p.requires_grad and 'clip' not in name)/1e6) )
        else:
            for param in self._q.parameters():
                param.requires_grad = False

            # # load CLIP for encoding language goals during evaluation
            # model, _ = load_clip("RN50", jit=False)
            # self._clip_rn50 = build_model(model.state_dict())
            # self._clip_rn50 = self._clip_rn50.float().to(device)
            # self._clip_rn50.eval()
            # del model

            self._optimizer = None
            self._lr_scheduler = None

            self.language_model =  create_language_model(self.cfg.language_model)
            self._voxelizer.to(device)
            self._q.to(device)

    def _extract_crop(self, pixel_action, observation):
        '''
        NOT USED
        '''
        # Pixel action will now be (B, 2)
        # observation = stack_on_channel(observation)
        h = observation.shape[-1]
        top_left_corner = torch.clamp(
            pixel_action - self._image_crop_size // 2, 0,
            h - self._image_crop_size)
        grid = self._grid_for_crop + top_left_corner.unsqueeze(1)
        grid = ((grid / float(h)) * 2.0) - 1.0  # between -1 and 1
        # Used for cropping the images across a batch
        # swap fro y x, to x, y
        grid = torch.cat((grid[:, :, :, 1:2], grid[:, :, :, 0:1]), dim=-1)
        crop = F.grid_sample(observation, grid, mode='nearest',
                             align_corners=True)
        return crop

    def _preprocess_inputs(self, replay_sample, sample_id=None):
        obs = []
        pcds = []
        exs = []
        ins = []
        self._crop_summary = []
        for n in self._camera_names:    # default: [front,left_shoulder,right_shoulder,wrist] or [front]
            if sample_id is not None:
                rgb = replay_sample['%s_rgb' % n][sample_id:sample_id+1]
                pcd = replay_sample['%s_point_cloud' % n][sample_id:sample_id+1]
                extin = replay_sample['%s_camera_extrinsics' % n][sample_id:sample_id+1]
                intin = replay_sample['%s_camera_intrinsics' % n][sample_id:sample_id+1]
            else:
                rgb = replay_sample['%s_rgb' % n]
                pcd = replay_sample['%s_point_cloud' % n]
                extin = replay_sample['%s_camera_extrinsics' % n]
                intin = replay_sample['%s_camera_intrinsics' % n]
            obs.append([rgb, pcd])
            pcds.append(pcd)
            exs.append(extin)
            ins.append(intin)
        return obs, pcds, exs, ins

    def _act_preprocess_inputs(self, observation):
        obs, pcds, exs, ins = [], [], [], []
        for n in self._camera_names:
            rgb = observation['%s_rgb' % n]
            # [-1,1] to [0,1]
            # rgb = (rgb + 1) / 2
            pcd = observation['%s_point_cloud' % n]
            extin = observation['%s_camera_extrinsics' % n].squeeze(0)
            intin = observation['%s_camera_intrinsics' % n].squeeze(0)

            obs.append([rgb, pcd])
            pcds.append(pcd)
            exs.append(extin)
            ins. append(intin)
        return obs, pcds, exs, ins

    def _get_value_from_voxel_index(self, q, voxel_idx):
        b, c, d, h, w = q.shape
        q_trans_flat = q.view(b, c, d * h * w)
        flat_indicies = (voxel_idx[:, 0] * d * h + voxel_idx[:, 1] * h + voxel_idx[:, 2])[:, None].int()
        highest_idxs = flat_indicies.unsqueeze(-1).repeat(1, c, 1)
        chosen_voxel_values = q_trans_flat.gather(2, highest_idxs)[..., 0]  # (B, trans + rot + grip)
        return chosen_voxel_values

    def _get_value_from_rot_and_grip(self, rot_grip_q, rot_and_grip_idx):
        q_rot = torch.stack(torch.split(
            rot_grip_q[:, :-2], int(360 // self._rotation_resolution),
            dim=1), dim=1)  # B, 3, 72
        q_grip = rot_grip_q[:, -2:]
        rot_and_grip_values = torch.cat(
            [q_rot[:, 0].gather(1, rot_and_grip_idx[:, 0:1]),
             q_rot[:, 1].gather(1, rot_and_grip_idx[:, 1:2]),
             q_rot[:, 2].gather(1, rot_and_grip_idx[:, 2:3]),
             q_grip.gather(1, rot_and_grip_idx[:, 3:4])], -1)
        return rot_and_grip_values

    def _celoss(self, pred, labels):
        return self._cross_entropy_loss(pred, labels.argmax(-1))

    def _softmax_q_trans(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

    def _softmax_q_rot_grip(self, q_rot_grip):
        q_rot_x_flat = q_rot_grip[:, 0*self._num_rotation_classes: 1*self._num_rotation_classes]
        q_rot_y_flat = q_rot_grip[:, 1*self._num_rotation_classes: 2*self._num_rotation_classes]
        q_rot_z_flat = q_rot_grip[:, 2*self._num_rotation_classes: 3*self._num_rotation_classes]
        q_grip_flat  = q_rot_grip[:, 3*self._num_rotation_classes:]

        q_rot_x_flat_softmax = F.softmax(q_rot_x_flat, dim=1)
        q_rot_y_flat_softmax = F.softmax(q_rot_y_flat, dim=1)
        q_rot_z_flat_softmax = F.softmax(q_rot_z_flat, dim=1)
        q_grip_flat_softmax = F.softmax(q_grip_flat, dim=1)

        return torch.cat([q_rot_x_flat_softmax,
                          q_rot_y_flat_softmax,
                          q_rot_z_flat_softmax,
                          q_grip_flat_softmax], dim=1)

    def _softmax_ignore_collision(self, q_collision):
        q_collision_softmax = F.softmax(q_collision, dim=1)
        return q_collision_softmax


    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
            
        
    def update(self, step: int, replay_sample: dict) -> dict:
        action_trans = replay_sample['trans_action_indicies'][:, self._layer * 3:self._layer * 3 + 3].int()
        action_rot_grip = replay_sample['rot_grip_action_indicies'].int()
        action_gripper_pose = replay_sample['gripper_pose']
        action_ignore_collisions = replay_sample['ignore_collisions'].int()
        lang_goal_emb = replay_sample['lang_goal_emb'].float()
        lang_token_embs = replay_sample['lang_token_embs'].float()
        prev_layer_voxel_grid = replay_sample.get('prev_layer_voxel_grid', None)
        prev_layer_bounds = replay_sample.get('prev_layer_bounds', None)
        lang_goal = replay_sample['lang_goal']

        device = self._device

        # get rank by device id
        rank = self._q.device

        # for nerf multi-view training
        nerf_multi_view_rgb_path = replay_sample['nerf_multi_view_rgb'] # only succeed to get path sometime
        nerf_multi_view_depth_path = replay_sample['nerf_multi_view_depth']
        nerf_multi_view_camera_path = replay_sample['nerf_multi_view_camera']



        if nerf_multi_view_rgb_path is None or nerf_multi_view_rgb_path[0,0] is None:
            cprint(nerf_multi_view_rgb_path, 'red')
            cprint(replay_sample['indices'], 'red')
            nerf_target_rgb = None
            nerf_target_camera_extrinsic = None
            print(colored('warn: one iter not use additional multi view', 'cyan'))
            raise ValueError('nerf_multi_view_rgb_path is None')
        else:
            # control the number of views by the following code
            num_view = nerf_multi_view_rgb_path.shape[-1]
            num_view_by_user = self.cfg.num_view_for_nerf
            # compute interval first
            assert num_view_by_user <= num_view, f'num_view_by_user {num_view_by_user} should be less than num_view {num_view}'
            interval = num_view // num_view_by_user
            nerf_multi_view_rgb_path = nerf_multi_view_rgb_path[:, ::interval]
            
            # sample one target img
            view_dix = np.random.randint(0, num_view_by_user)
            nerf_multi_view_rgb_path = nerf_multi_view_rgb_path[0, view_dix]  # assume batch size is 1
            nerf_multi_view_depth_path = nerf_multi_view_depth_path[0, view_dix]
            nerf_multi_view_camera_path = nerf_multi_view_camera_path[0, view_dix]
            # load img and camera
            nerf_target_camera_extrinsic, nerf_target_camera_intrinsic, nerf_target_focal = parse_camera_file(nerf_multi_view_camera_path)
            
            nerf_target_rgb = parse_img_file(nerf_multi_view_rgb_path)
            nerf_target_rgb = torch.from_numpy(nerf_target_rgb).float().unsqueeze(0).to(device) # [1, H, W, 3], [0,1]
            
            nerf_target_camera_extrinsic = torch.from_numpy(nerf_target_camera_extrinsic).float().unsqueeze(0).to(device)



        bounds = self._coordinate_bounds.to(device)
        if self._layer > 0:
            cp = replay_sample['attention_coordinate_layer_%d' % (self._layer - 1)]
            bounds = torch.cat([cp - self._bounds_offset, cp + self._bounds_offset], dim=1)

        proprio = None
        if self._include_low_dim_state:
            proprio = replay_sample['low_dim_state']

        obs, pcd, extrinsics, intrinsics = self._preprocess_inputs(replay_sample)
        # batch size
        bs = pcd[0].shape[0]


        # ### debug
        # # vis point cloud for debug
        # import visdom
        # pc_vis = pcd[0].reshape(3, -1).permute(1, 0).cpu().numpy()
        # # random sample half of the points
        # pc_vis = pc_vis[np.random.choice(pc_vis.shape[0], pc_vis.shape[0]//2, replace=False), :]
        # label_vis = np.zeros((pc_vis.shape[0], 1)) +1
        # vis = visdom.Visdom()
        # vis.scatter(X=pc_vis, Y=label_vis, win='pc_vis', opts=dict(markersize=1, title='pc_vis',color='red'))
        

        # SE(3) augmentation of point clouds and actions
        if self._transform_augmentation:
            action_trans, \
            action_rot_grip, \
            pcd,\
            extrinsics = apply_se3_augmentation_with_camera_pose(pcd,
                                         extrinsics,
                                         action_gripper_pose,
                                         action_trans,
                                         action_rot_grip,
                                         bounds,
                                         self._layer,
                                         self._transform_augmentation_xyz,
                                         self._transform_augmentation_rpy,
                                         self._transform_augmentation_rot_resolution,
                                         self._voxel_size,
                                         self._rotation_resolution,
                                         self._device)

        # voxelization

        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:# register hook for grad cam
            features_hook = self._q._qnet.final.register_forward_hook(self.save_feature)
            gradients_hook = self._q._qnet.final.register_backward_hook(self.save_gradient)
            # features_hook = self._q._qnet.encoder_3d.register_forward_hook(self.save_feature)
            # gradients_hook = self._q._qnet.encoder_3d.register_backward_hook(self.save_gradient)



        # forward pass
        q_trans, q_rot_grip, \
        q_collision, \
        voxel_grid, \
        rendering_loss_dict = self._q(obs,
                                proprio,
                                pcd,
                                extrinsics,
                                intrinsics,
                                lang_goal_emb,
                                lang_token_embs,
                                bounds,
                                prev_layer_bounds,
                                prev_layer_voxel_grid,
                                use_neural_rendering=self.use_neural_rendering,
                                nerf_target_rgb=nerf_target_rgb,
                                nerf_target_pose=nerf_target_camera_extrinsic,
                                lang_goal=lang_goal)
        # argmax to choose best action
        coords, \
        rot_and_grip_indicies, \
        ignore_collision_indicies = self._q.choose_highest_action(q_trans, q_rot_grip, q_collision)

        q_trans_loss, q_rot_loss, q_grip_loss, q_collision_loss = 0., 0., 0., 0.

        # translation one-hot
        action_trans_one_hot = self._action_trans_one_hot_zeros.clone()
        for b in range(bs):
            gt_coord = action_trans[b, :].int()
            action_trans_one_hot[b, :, gt_coord[0], gt_coord[1], gt_coord[2]] = 1
            
        # translation loss
        q_trans_flat = q_trans.view(bs, -1)
        action_trans_one_hot_flat = action_trans_one_hot.view(bs, -1)
        q_trans_loss = self._celoss(q_trans_flat, action_trans_one_hot_flat)

        with_rot_and_grip = rot_and_grip_indicies is not None
        if with_rot_and_grip:
            # rotation, gripper, and collision one-hots
            action_rot_x_one_hot = self._action_rot_x_one_hot_zeros.clone()
            action_rot_y_one_hot = self._action_rot_y_one_hot_zeros.clone()
            action_rot_z_one_hot = self._action_rot_z_one_hot_zeros.clone()
            action_grip_one_hot = self._action_grip_one_hot_zeros.clone()
            action_ignore_collisions_one_hot = self._action_ignore_collisions_one_hot_zeros.clone()

            for b in range(bs):
                gt_rot_grip = action_rot_grip[b, :].int()
                action_rot_x_one_hot[b, gt_rot_grip[0]] = 1
                action_rot_y_one_hot[b, gt_rot_grip[1]] = 1
                action_rot_z_one_hot[b, gt_rot_grip[2]] = 1
                action_grip_one_hot[b, gt_rot_grip[3]] = 1

                gt_ignore_collisions = action_ignore_collisions[b, :].int()
                action_ignore_collisions_one_hot[b, gt_ignore_collisions[0]] = 1

            # flatten predictions
            q_rot_x_flat = q_rot_grip[:, 0*self._num_rotation_classes:1*self._num_rotation_classes]
            q_rot_y_flat = q_rot_grip[:, 1*self._num_rotation_classes:2*self._num_rotation_classes]
            q_rot_z_flat = q_rot_grip[:, 2*self._num_rotation_classes:3*self._num_rotation_classes]
            q_grip_flat =  q_rot_grip[:, 3*self._num_rotation_classes:]
            q_ignore_collisions_flat = q_collision

            # rotation loss
            q_rot_loss += self._celoss(q_rot_x_flat, action_rot_x_one_hot)
            q_rot_loss += self._celoss(q_rot_y_flat, action_rot_y_one_hot)
            q_rot_loss += self._celoss(q_rot_z_flat, action_rot_z_one_hot)

            # gripper loss
            q_grip_loss += self._celoss(q_grip_flat, action_grip_one_hot)

            # collision loss
            q_collision_loss += self._celoss(q_ignore_collisions_flat, action_ignore_collisions_one_hot)

        combined_losses = (q_trans_loss * self._trans_loss_weight) + \
                          (q_rot_loss * self._rot_loss_weight) + \
                          (q_grip_loss * self._grip_loss_weight) + \
                          (q_collision_loss * self._collision_loss_weight)
        total_loss = combined_losses.mean()

        if self.use_neural_rendering:
            
            lambda_nerf = self.cfg.neural_renderer.lambda_nerf
            lambda_BC = 1.0

            total_loss =  lambda_BC * total_loss + lambda_nerf * rendering_loss_dict['loss']
            # for print
            loss_rgb_item = rendering_loss_dict['loss_rgb']
            loss_embed_item = rendering_loss_dict['loss_embed']
            psnr = rendering_loss_dict['psnr']

            if step % 10 == 0 and rank == 0:
                cprint(f"total L: {total_loss.item():.4f} | L_BC: {combined_losses.item():.3f} x {lambda_BC:.3f} | L_rgb: {loss_rgb_item:.3f} x {lambda_nerf:.3f} | L_embed: {loss_embed_item:.3f} | psnr: {psnr:.3f}", 'green')
                if self.cfg.use_wandb:
                    wandb.log({'train/BC_loss':combined_losses.item(), 
                                'train/psnr':psnr, 
                                'train/rgb_loss':loss_rgb_item,
                                'train/embed_loss':loss_embed_item,
                                }, step=step)

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        ################################

        render_freq = 2000
        to_render = (step % render_freq == 0 and self.use_neural_rendering and nerf_target_camera_extrinsic is not None)
        if to_render:
            if isinstance(self._q, DDP):
                rgb_render, embed_render, depth_render = self._q.module.render(rgb_pcd=obs,
                                proprio=proprio,
                                pcd=pcd,
                                camera_extrinsics=extrinsics,
                                camera_intrinsics=intrinsics,
                                lang_goal_emb=lang_goal_emb,
                                lang_token_embs=lang_token_embs,
                                bounds=bounds,
                                prev_bounds=prev_layer_bounds,
                                prev_layer_voxel_grid=prev_layer_voxel_grid,
                                tgt_pose=nerf_target_camera_extrinsic,)
            else:
                rgb_render, embed_render, depth_render = self._q.render(rgb_pcd=obs,
                                proprio=proprio,
                                pcd=pcd,
                                camera_extrinsics=extrinsics,
                                camera_intrinsics=intrinsics,
                                lang_goal_emb=lang_goal_emb,
                                lang_token_embs=lang_token_embs,
                                bounds=bounds,
                                prev_bounds=prev_layer_bounds,
                                prev_layer_voxel_grid=prev_layer_voxel_grid,
                                tgt_pose=nerf_target_camera_extrinsic,)


            rgb_gt = nerf_target_rgb
            psnr = PSNR_torch(rgb_render, rgb_gt)

            os.makedirs('recon', exist_ok=True)
            import matplotlib.pyplot as plt
            # plot three images in one row with subplots:
            # src, tgt, pred
            rgb_src =  obs[0][0].squeeze(0).permute(1, 2, 0)  / 2 + 0.5
            fig, axs = plt.subplots(1, 5, figsize=(15, 3))
            # src
            axs[0].imshow(rgb_src.cpu().numpy())
            axs[0].title.set_text('src')
            # tgt
            axs[1].imshow(rgb_gt[0].cpu().numpy())
            axs[1].title.set_text('tgt')
            # pred rgb
            axs[2].imshow(rgb_render[0].cpu().numpy())
            axs[2].title.set_text('psnr={:.2f}'.format(psnr))
            # pred embed
            embed_render = visualize_feature_map_by_clustering(embed_render.permute(0,3,1,2),4)
            axs[3].imshow(embed_render)
            axs[3].title.set_text('embed seg')
            # pred depth
            axs[4].imshow(depth_render[0].cpu().numpy())
            axs[4].title.set_text('depth')
            # remove axis
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            
            if rank == 0:
                if self.cfg.use_wandb:
                    # save to buffer and write to wandb
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)

                    image = Image.open(buf)
                    wandb.log({"eval/recon_img": wandb.Image(image)}, step=step)

                    buf.close()
                    cprint(f'Saved recon/{step}_rgb.png to wandb.', 'cyan')

                    # log depth mean, min, max
                    wandb.log({'eval/depth_mean': depth_render.mean(),
                                'eval/depth_min': depth_render.min(),
                                'eval/depth_max': depth_render.max(),
                                }, step=step)
                else:
                    plt.savefig(f'recon/{step}_rgb.png')
                    cprint(f'Saved recon/{step}_rgb.png locally.', 'cyan')


        self._summaries = {
            'losses/total_loss': total_loss.item(),
            'losses/trans_loss': q_trans_loss.mean(),
            'losses/rot_loss': q_rot_loss.mean() if with_rot_and_grip else 0.,
            'losses/grip_loss': q_grip_loss.mean() if with_rot_and_grip else 0.,
            'losses/collision_loss': q_collision_loss.mean() if with_rot_and_grip else 0.,
        }

        self._wandb_summaries = {
            'losses/total_loss': total_loss.item(),
            'losses/trans_loss': q_trans_loss.mean(),
            'losses/rot_loss': q_rot_loss.mean() if with_rot_and_grip else 0.,
            'losses/grip_loss': q_grip_loss.mean() if with_rot_and_grip else 0.,
            'losses/collision_loss': q_collision_loss.mean() if with_rot_and_grip else 0.,

            # for visualization
            'point_cloud': None,
            'coord_pred': coords,
            'coord_gt': gt_coord,
        }

        if self._lr_scheduler:
            self._scheduler.step()
            self._summaries['learning_rate'] = self._scheduler.get_last_lr()[0]

        self._vis_voxel_grid = voxel_grid[0]        
        
        self._vis_translation_qvalue = self._softmax_q_trans(q_trans[0])
        self._vis_max_coordinate = coords[0]
        self._vis_gt_coordinate = action_trans[0]

        # Note: PerAct doesn't use multi-layer voxel grids like C2FARM
        # stack prev_layer_voxel_grid(s) from previous layers into a list
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [voxel_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [voxel_grid]

        # stack prev_layer_bound(s) from previous layers into a list
        if prev_layer_bounds is None:
            prev_layer_bounds = [self._coordinate_bounds.repeat(bs, 1)]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        return {
            'total_loss': total_loss,
            'prev_layer_voxel_grid': prev_layer_voxel_grid,
            'prev_layer_bounds': prev_layer_bounds,
        }


    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        deterministic = True
        bounds = self._coordinate_bounds
        prev_layer_voxel_grid = observation.get('prev_layer_voxel_grid', None)
        prev_layer_bounds = observation.get('prev_layer_bounds', None)
        lang_goal_tokens = observation.get('lang_goal_tokens', None).long()
        lang_goal = observation['lang_goal']

        # extract language embs
        with torch.no_grad():
            lang_goal_emb, lang_token_embs = self.language_model.extract(lang_goal)
            # lang_goal_tokens = lang_goal_tokens.to(device=self._device)
            # lang_goal_emb, lang_token_embs = self._clip_rn50.encode_text_with_embeddings(lang_goal_tokens[0])

        # voxelization resolution
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        max_rot_index = int(360 // self._rotation_resolution)
        proprio = None

        if self._include_low_dim_state:
            proprio = observation['low_dim_state']

        obs, pcd, extrinsics, intrinsics = self._act_preprocess_inputs(observation)

        ### debug
        # vis point cloud for debug
        # import visdom
        # pc_vis = pcd[0].reshape(3, -1).permute(1, 0).cpu().numpy()
        # wrist_pcd = observation['wrist_rgb'].reshape(3, -1).permute(1, 0).cpu().numpy()
        # # pc_vis2 = pcd[1].reshape(3, -1).permute(1, 0).cpu().numpy()
        # # concat
        # pc_vis = np.concatenate((pc_vis, wrist_pcd), axis=0)
        # # # random sample half of the points
        # # pc_vis = pc_vis[np.random.choice(pc_vis.shape[0], pc_vis.shape[0]//2, replace=False), :]
        # label_vis = np.zeros((pc_vis.shape[0], 1)) +1
        # vis = visdom.Visdom()
        # vis.scatter(X=wrist_pcd, win='pc_vis', opts=dict(markersize=1, title='pc_vis',color='red'))


        # correct batch size and device
        obs = [[o[0][0].to(self._device), o[1][0].to(self._device)] for o in obs]
        proprio = proprio[0].to(self._device)
        pcd = [p[0].to(self._device) for p in pcd]
        lang_goal_emb = lang_goal_emb.to(self._device)
        lang_token_embs = lang_token_embs.to(self._device)
        bounds = torch.as_tensor(bounds, device=self._device)
        prev_layer_voxel_grid = prev_layer_voxel_grid.to(self._device) if prev_layer_voxel_grid is not None else None
        prev_layer_bounds = prev_layer_bounds.to(self._device) if prev_layer_bounds is not None else None

        # inference
        q_trans, \
        q_rot_grip, \
        q_ignore_collisions, \
        vox_grid,\
        rendering_loss_dict  = self._q(obs,
                            proprio,
                            pcd,
                            extrinsics,
                            intrinsics,
                            lang_goal_emb,
                            lang_token_embs,
                            bounds,
                            prev_layer_bounds,
                            prev_layer_voxel_grid, 
                            use_neural_rendering=False)

        # softmax Q predictions
        q_trans = self._softmax_q_trans(q_trans)
        q_rot_grip =  self._softmax_q_rot_grip(q_rot_grip) if q_rot_grip is not None else q_rot_grip
        q_ignore_collisions = self._softmax_ignore_collision(q_ignore_collisions) \
            if q_ignore_collisions is not None else q_ignore_collisions

        # argmax Q predictions
        coords, \
        rot_and_grip_indicies, \
        ignore_collisions = self._q.choose_highest_action(q_trans, q_rot_grip, q_ignore_collisions)

        rot_grip_action = rot_and_grip_indicies if q_rot_grip is not None else None
        ignore_collisions_action = ignore_collisions.int() if ignore_collisions is not None else None

        coords = coords.int()
        attention_coordinate = bounds[:, :3] + res * coords + res / 2

        # stack prev_layer_voxel_grid(s) into a list
        # NOTE: PerAct doesn't used multi-layer voxel grids like C2FARM
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [vox_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [vox_grid]

        if prev_layer_bounds is None:
            prev_layer_bounds = [bounds]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        observation_elements = {
            'attention_coordinate': attention_coordinate,
            'prev_layer_voxel_grid': prev_layer_voxel_grid,
            'prev_layer_bounds': prev_layer_bounds,
        }
        info = {
            'voxel_grid_depth%d' % self._layer: vox_grid,
            'q_depth%d' % self._layer: q_trans,
            'voxel_idx_depth%d' % self._layer: coords
        }
        self._act_voxel_grid = vox_grid[0]
        self._act_max_coordinate = coords[0]
        self._act_qvalues = q_trans[0].detach()
        return ActResult((coords, rot_grip_action, ignore_collisions_action),
                         observation_elements=observation_elements,
                         info=info)

    
    def update_summaries(self) -> List[Summary]:
        summaries = []

        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (self._name, n), v))

        for (name, crop) in (self._crop_summary):
            crops = (torch.cat(torch.split(crop, 3, dim=1), dim=3) + 1.0) / 2.0
            summaries.extend([
                ImageSummary('%s/crops/%s' % (self._name, name), crops)])

        for tag, param in self._q.named_parameters():
            # assert not torch.isnan(param.grad.abs() <= 1.0).all()
            if param.grad is None:
                continue

            summaries.append(
                HistogramSummary('%s/gradient/%s' % (self._name, tag),
                                 param.grad))
            summaries.append(
                HistogramSummary('%s/weight/%s' % (self._name, tag),
                                 param.data))

        return summaries
    
    
    def update_wandb_summaries(self):
        summaries = dict()

        for k, v in self._wandb_summaries.items():
            summaries[k] = v
        return summaries


    def act_summaries(self) -> List[Summary]:
        return [
            ImageSummary('%s/act_Qattention' % self._name,
                         transforms.ToTensor()(visualise_voxel(
                             self._act_voxel_grid.cpu().numpy(),
                             self._act_qvalues.cpu().numpy(),
                             self._act_max_coordinate.cpu().numpy())))]


    def load_weights(self, savedir: str):
        device = self._device if not self._training else torch.device('cuda:%d' % self._device)
        # if device is str, convert it to torch.device
        if isinstance(device, int):
            device = torch.device('cuda:%d' % self._device)

        weight_file = os.path.join(savedir, '%s.pt' % self._name)
        state_dict = torch.load(weight_file, map_location=device)

        # load only keys that are in the current model
        merged_state_dict = self._q.state_dict()
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace('_qnet.module', '_qnet')
                k = k.replace('_neural_renderer.module', '_neural_renderer')
            if k in merged_state_dict:
                merged_state_dict[k] = v
            else:
                if '_voxelizer' not in k: # and '_neural_renderer' not in k:
                    # logging.warning("key %s not found in checkpoint" % k)
                    logging.warning(f"key {k} is found in checkpoint, but not found in current model.")
        msg = self._q.load_state_dict(merged_state_dict, strict=False)
        if msg.missing_keys:
            print("missing some keys...")
        if msg.unexpected_keys:
            print("unexpected some keys...")
        print("loaded weights from %s" % weight_file)


    def save_weights(self, savedir: str):
        torch.save(
            self._q.state_dict(), os.path.join(savedir, '%s.pt' % self._name))
    
    
    def load_clip(self):
        model, _ = load_clip("RN50", jit=False)
        self._clip_rn50 = build_model(model.state_dict())
        self._clip_rn50 = self._clip_rn50.float().to(self._device)
        self._clip_rn50.eval() 
        del model


    def unload_clip(self):
        del self._clip_rn50