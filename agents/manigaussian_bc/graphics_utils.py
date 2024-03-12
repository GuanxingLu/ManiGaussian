#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, K, h, w):
    near_fx = znear / K[0, 0]
    near_fy = znear / K[1, 1]
    left = - (w - K[0, 2]) * near_fx
    right = K[0, 2] * near_fx
    bottom = (K[1, 2] - h) * near_fy
    top = K[1, 2] * near_fy

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
    # return 2*math.atan(pixels/(2*np.abs(focal)))  # no impact


def depth2pc(depth, extrinsic, intrinsic):
    B, C, S, S = depth.shape    # S=128
    depth = depth[:, 0, :, :]
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device), torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B S S 3

    pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)
    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)


@torch.no_grad()
def world_to_canonical(xyz, coordinate_bounds=[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]):
    """
    :param xyz (B, N, 3) or (B, 3, N)
    :return (B, N, 3) or (B, 3, N)

    transform world coordinate to canonical coordinate with bounding box
    """
    xyz = xyz.clone()
    bb_min = coordinate_bounds[:3]
    bb_max = coordinate_bounds[3:]
    bb_min = torch.tensor(bb_min, device=xyz.device).unsqueeze(0).unsqueeze(0) if xyz.shape[-1] == 3 \
        else torch.tensor(bb_min, device=xyz.device).unsqueeze(-1).unsqueeze(0)
    bb_max = torch.tensor(bb_max, device=xyz.device).unsqueeze(0).unsqueeze(0) if xyz.shape[-1] == 3 \
        else torch.tensor(bb_max, device=xyz.device).unsqueeze(-1).unsqueeze(0)
    xyz -= bb_min
    xyz /= (bb_max - bb_min)

    return xyz

# @torch.no_grad()  # training
def canonical_to_world(xyz, coordinate_bounds=[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]):
    """
    :param xyz (B, 3)
    :return (B, 3)

    inverse process of world_to_canonical
    """
    xyz = xyz.clone()
    bb_min = coordinate_bounds[:3]
    bb_max = coordinate_bounds[3:]
    bb_min = torch.tensor(bb_min, device=xyz.device).unsqueeze(0)
    bb_max = torch.tensor(bb_max, device=xyz.device).unsqueeze(0)
    xyz *= (bb_max - bb_min)
    xyz += bb_min

    return xyz


if __name__ == '__main__':
    # test world_to_canonical
    # xyz = torch.rand(2, 128, 3)
    xyz = torch.rand(2, 3, 128)
    xyz = world_to_canonical(xyz)
    xyz = canonical_to_world(xyz)
    print(xyz.shape)
