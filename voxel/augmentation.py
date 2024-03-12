import numpy as np
import torch
from helpers import utils
from pytorch3d import transforms as torch3d_tf
from termcolor import cprint

import einops
from scipy.spatial.transform import Rotation as R


def perturb_se3(pcd,
                trans_shift_4x4,
                rot_shift_4x4,
                action_gripper_4x4,
                bounds):
    """ Perturb point clouds with given transformation.
    :param pcd: list of point clouds [[bs, 3, N], ...] for N cameras
    :param trans_shift_4x4: translation matrix [bs, 4, 4]
    :param rot_shift_4x4: rotation matrix [bs, 4, 4]
    :param action_gripper_4x4: original keyframe action gripper pose [bs, 4, 4]
    :param bounds: metric scene bounds [bs, 6]
    :return: peturbed point clouds
    """
    # baatch bounds if necessary
    bs = pcd[0].shape[0]
    if bounds.shape[0] != bs:
        bounds = bounds.repeat(bs, 1)

    perturbed_pcd = []
    for p in pcd:
        p_shape = p.shape
        num_points = p_shape[-1] * p_shape[-2]

        action_trans_3x1 = action_gripper_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        trans_shift_3x1 = trans_shift_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)

        # flatten point cloud
        p_flat = p.reshape(bs, 3, -1)
        p_flat_4x1_action_origin = torch.ones(bs, 4, p_flat.shape[-1]).to(p_flat.device)

        # shift points to have action_gripper pose as the origin
        p_flat_4x1_action_origin[:, :3, :] = p_flat - action_trans_3x1

        # apply rotation
        perturbed_p_flat_4x1_action_origin = torch.bmm(p_flat_4x1_action_origin.transpose(2, 1),
                                                       rot_shift_4x4).transpose(2, 1)

        # apply bounded translations
        bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
        bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
        bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

        action_then_trans_3x1 = action_trans_3x1 + trans_shift_3x1
        action_then_trans_3x1_x = torch.clamp(action_then_trans_3x1[:, 0],
                                              min=bounds_x_min, max=bounds_x_max)
        action_then_trans_3x1_y = torch.clamp(action_then_trans_3x1[:, 1],
                                              min=bounds_y_min, max=bounds_y_max)
        action_then_trans_3x1_z = torch.clamp(action_then_trans_3x1[:, 2],
                                              min=bounds_z_min, max=bounds_z_max)
        action_then_trans_3x1 = torch.stack([action_then_trans_3x1_x,
                                             action_then_trans_3x1_y,
                                             action_then_trans_3x1_z], dim=1)

        # shift back the origin
        perturbed_p_flat_3x1 = perturbed_p_flat_4x1_action_origin[:, :3, :] + action_then_trans_3x1

        perturbed_p = perturbed_p_flat_3x1.reshape(p_shape)
        perturbed_pcd.append(perturbed_p)
    return perturbed_pcd


def perturb_se3_camera_pose(camera_pose,
                trans_shift_4x4,
                rot_shift_4x4,
                action_gripper_4x4,
                bounds):
    """ Perturb point clouds with given transformation.
    :param pcd: list of point clouds [[bs, 3, N], ...] for N cameras
    :param trans_shift_4x4: translation matrix [bs, 4, 4]
    :param rot_shift_4x4: rotation matrix [bs, 4, 4]
    :param action_gripper_4x4: original keyframe action gripper pose [bs, 4, 4]
    :param bounds: metric scene bounds [bs, 6]
    :return: peturbed point clouds
    """
    # batch bounds if necessary
    bs = camera_pose[0].shape[0]
    if bounds.shape[0] != bs:
        bounds = bounds.repeat(bs, 1)

    perturbed_camera_pose = []
    for cam_pose in camera_pose:
        
        cam_R, cam_T = cam_pose[:, :3, :3], cam_pose[:, :3, 3:]

        # action_trans_3x1 = action_gripper_4x4[:, 0:3, 3].unsqueeze(-1).repeat(bs, 1, 1)
        # trans_shift_3x1 = trans_shift_4x4[:, 0:3, 3].unsqueeze(-1).repeat(bs, 1, 1)
        action_trans_3x1 = action_gripper_4x4[:, 0:3, 3].unsqueeze(-1)
        trans_shift_3x1 = trans_shift_4x4[:, 0:3, 3].unsqueeze(-1)

        cam_T = cam_T - action_trans_3x1    # [bs, 3, 1]
        cam_T_4x1 = torch.ones(bs, 4, 1).to(cam_T.device)
        cam_T_4x1[:, :3, :] = cam_T
        cam_T_4x1 = torch.bmm(cam_T_4x1.transpose(2, 1), rot_shift_4x4).transpose(2, 1)

        cam_R = torch.bmm(cam_R.transpose(2, 1), rot_shift_4x4[:, :3, :3]).transpose(2, 1)

        # apply bounded translations
        bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
        bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
        bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

        action_then_trans_3x1 = action_trans_3x1 + trans_shift_3x1
        action_then_trans_3x1_x = torch.clamp(action_then_trans_3x1[:, 0],
                                              min=bounds_x_min, max=bounds_x_max)
        action_then_trans_3x1_y = torch.clamp(action_then_trans_3x1[:, 1],
                                              min=bounds_y_min, max=bounds_y_max)
        action_then_trans_3x1_z = torch.clamp(action_then_trans_3x1[:, 2],
                                              min=bounds_z_min, max=bounds_z_max)
        action_then_trans_3x1 = torch.stack([action_then_trans_3x1_x,
                                             action_then_trans_3x1_y,
                                             action_then_trans_3x1_z], dim=1)

        # shift back the origin
        cam_T_4x1[:, :3]  = cam_T_4x1[:, :3] + action_then_trans_3x1

        # get new camera pose
        cam_T = cam_T_4x1[:, :3, :]
        cam_pose[:, :3, :3], cam_pose[:, :3, 3:] = cam_R, cam_T
        perturbed_camera_pose.append(cam_pose)

    return perturbed_camera_pose


def apply_se3_augmentation(pcd,
                           action_gripper_pose,
                           action_trans,
                           action_rot_grip,
                           bounds,
                           layer,
                           trans_aug_range,
                           rot_aug_range,
                           rot_aug_resolution,
                           voxel_size,
                           rot_resolution,
                           device):
    """ Apply SE3 augmentation to a point clouds and actions.
    :param pcd: list of point clouds [[bs, 3, H, W], ...] for N cameras
    :param action_gripper_pose: 6-DoF pose of keyframe action [bs, 7]
    :param action_trans: discretized translation action [bs, 3]
    :param action_rot_grip: discretized rotation and gripper action [bs, 4]
    :param bounds: metric scene bounds of voxel grid [bs, 6]
    :param layer: voxelization layer (always 1 for PerAct)
    :param trans_aug_range: range of translation augmentation [x_range, y_range, z_range]
    :param rot_aug_range: range of rotation augmentation [x_range, y_range, z_range]
    :param rot_aug_resolution: degree increments for discretized augmentation rotations
    :param voxel_size: voxelization resoltion
    :param rot_resolution: degree increments for discretized rotations
    :param device: torch device
    :return: perturbed action_trans, action_rot_grip, pcd
    """

    # batch size
    bs = pcd[0].shape[0]

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose
    action_gripper_trans = action_gripper_pose[:, :3]
    action_gripper_quat_wxyz = torch.cat((action_gripper_pose[:, 6].unsqueeze(1),
                                          action_gripper_pose[:, 3:6]), dim=1)
    action_gripper_rot = torch3d_tf.quaternion_to_matrix(action_gripper_quat_wxyz)
    action_gripper_4x4 = identity_4x4.detach().clone()
    action_gripper_4x4[:, :3, :3] = action_gripper_rot
    action_gripper_4x4[:, 0:3, 3] = action_gripper_trans

    perturbed_trans = torch.full_like(action_trans, -1.)
    perturbed_rot_grip = torch.full_like(action_rot_grip, -1.)

    # perturb the action, check if it is within bounds, if not, try another perturbation
    perturb_attempts = 0
    while torch.any(perturbed_trans < 0):
        # might take some repeated attempts to find a perturbation that doesn't go out of bounds
        perturb_attempts += 1
        if perturb_attempts > 10:
            # raise Exception('Failing to perturb action and keep it within bounds.')
            cprint('Failing to perturb action and keep it within bounds. use non-perturbed value.', 'red')
            # return original action
            return action_trans, action_rot_grip, pcd

        # sample translation perturbation with specified range
        trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(device=device)
        trans_shift = trans_range * utils.rand_dist((bs, 3)).to(device=device)
        trans_shift_4x4 = identity_4x4.detach().clone()
        trans_shift_4x4[:, 0:3, 3] = trans_shift

        # sample rotation perturbation at specified resolution and range
        roll_aug_steps = int(rot_aug_range[0] // rot_aug_resolution)
        pitch_aug_steps = int(rot_aug_range[1] // rot_aug_resolution)
        yaw_aug_steps = int(rot_aug_range[2] // rot_aug_resolution)

        roll = utils.rand_discrete((bs, 1),
                                   min=-roll_aug_steps,
                                   max=roll_aug_steps) * np.deg2rad(rot_aug_resolution)
        pitch = utils.rand_discrete((bs, 1),
                                    min=-pitch_aug_steps,
                                    max=pitch_aug_steps) * np.deg2rad(rot_aug_resolution)
        yaw = utils.rand_discrete((bs, 1),
                                  min=-yaw_aug_steps,
                                  max=yaw_aug_steps) * np.deg2rad(rot_aug_resolution)
        rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(torch.cat((roll, pitch, yaw), dim=1), "XYZ")
        rot_shift_4x4 = identity_4x4.detach().clone()
        rot_shift_4x4[:, :3, :3] = rot_shift_3x3

        # rotate then translate the 4x4 keyframe action
        perturbed_action_gripper_4x4 = torch.bmm(action_gripper_4x4, rot_shift_4x4)
        perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        perturbed_action_trans = perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(perturbed_action_gripper_4x4[:, :3, :3])
        perturbed_action_quat_xyzw = torch.cat([perturbed_action_quat_wxyz[:, 1:],
                                                perturbed_action_quat_wxyz[:, 0].unsqueeze(1)],
                                               dim=1).cpu().numpy()

        # discretize perturbed translation and rotation
        # TODO(mohit): do this in torch without any numpy.
        trans_indicies, rot_grip_indicies = [], []
        for b in range(bs):
            bounds_idx = b if layer > 0 else 0
            bounds_np = bounds[bounds_idx].cpu().numpy()

            trans_idx = utils.point_to_voxel_index(perturbed_action_trans[b], voxel_size, bounds_np)
            trans_indicies.append(trans_idx.tolist())

            quat = perturbed_action_quat_xyzw[b]
            quat = utils.normalize_quaternion(perturbed_action_quat_xyzw[b])
            if quat[-1] < 0:
                quat = -quat
            disc_rot = utils.quaternion_to_discrete_euler(quat, rot_resolution)
            rot_grip_indicies.append(disc_rot.tolist() + [int(action_rot_grip[b, 3].cpu().numpy())])

        # if the perturbed action is out of bounds,
        # the discretized perturb_trans should have invalid indicies
        perturbed_trans = torch.from_numpy(np.array(trans_indicies)).to(device=device)
        perturbed_rot_grip = torch.from_numpy(np.array(rot_grip_indicies)).to(device=device)

    action_trans = perturbed_trans
    action_rot_grip = perturbed_rot_grip

    # apply perturbation to pointclouds
    pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)

    return action_trans, action_rot_grip, pcd

def apply_se3_augmentation_with_camera_pose(pcd,
                            camera_pose,
                           action_gripper_pose,
                           action_trans,
                           action_rot_grip,
                           bounds,
                           layer,
                           trans_aug_range,
                           rot_aug_range,
                           rot_aug_resolution,
                           voxel_size,
                           rot_resolution,
                           device):
    """ Apply SE3 augmentation to a point clouds and actions.
    :param pcd: list of point clouds [[bs, 3, H, W], ...] for N cameras
    :param action_gripper_pose: 6-DoF pose of keyframe action [bs, 7]
    :param action_trans: discretized translation action [bs, 3]
    :param action_rot_grip: discretized rotation and gripper action [bs, 4]
    :param bounds: metric scene bounds of voxel grid [bs, 6]
    :param layer: voxelization layer (always 1 for PerAct)
    :param trans_aug_range: range of translation augmentation [x_range, y_range, z_range]
    :param rot_aug_range: range of rotation augmentation [x_range, y_range, z_range]
    :param rot_aug_resolution: degree increments for discretized augmentation rotations
    :param voxel_size: voxelization resoltion
    :param rot_resolution: degree increments for discretized rotations
    :param device: torch device
    :return: perturbed action_trans, action_rot_grip, pcd
    """

    # batch size
    bs = pcd[0].shape[0]

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose
    action_gripper_trans = action_gripper_pose[:, :3]
    action_gripper_quat_wxyz = torch.cat((action_gripper_pose[:, 6].unsqueeze(1),
                                          action_gripper_pose[:, 3:6]), dim=1)
    action_gripper_rot = torch3d_tf.quaternion_to_matrix(action_gripper_quat_wxyz)
    action_gripper_4x4 = identity_4x4.detach().clone()
    action_gripper_4x4[:, :3, :3] = action_gripper_rot
    action_gripper_4x4[:, 0:3, 3] = action_gripper_trans

    perturbed_trans = torch.full_like(action_trans, -1.)
    perturbed_rot_grip = torch.full_like(action_rot_grip, -1.)

    # perturb the action, check if it is within bounds, if not, try another perturbation
    perturb_attempts = 0
    while torch.any(perturbed_trans < 0):
        # might take some repeated attempts to find a perturbation that doesn't go out of bounds
        perturb_attempts += 1
        if perturb_attempts > 10:
            # raise Exception('Failing to perturb action and keep it within bounds.')
            cprint('Failing to perturb action and keep it within bounds. use non-perturbed value.', 'red')
            # return original action
            return action_trans, action_rot_grip, pcd, camera_pose

        # sample translation perturbation with specified range
        trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(device=device)
        trans_shift = trans_range * utils.rand_dist((bs, 3)).to(device=device)
        trans_shift_4x4 = identity_4x4.detach().clone()
        trans_shift_4x4[:, 0:3, 3] = trans_shift

        # sample rotation perturbation at specified resolution and range
        roll_aug_steps = int(rot_aug_range[0] // rot_aug_resolution)
        pitch_aug_steps = int(rot_aug_range[1] // rot_aug_resolution)
        yaw_aug_steps = int(rot_aug_range[2] // rot_aug_resolution)

        roll = utils.rand_discrete((bs, 1),
                                   min=-roll_aug_steps,
                                   max=roll_aug_steps) * np.deg2rad(rot_aug_resolution)
        pitch = utils.rand_discrete((bs, 1),
                                    min=-pitch_aug_steps,
                                    max=pitch_aug_steps) * np.deg2rad(rot_aug_resolution)
        yaw = utils.rand_discrete((bs, 1),
                                  min=-yaw_aug_steps,
                                  max=yaw_aug_steps) * np.deg2rad(rot_aug_resolution)
        rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(torch.cat((roll, pitch, yaw), dim=1), "XYZ")
        rot_shift_4x4 = identity_4x4.detach().clone()
        rot_shift_4x4[:, :3, :3] = rot_shift_3x3

        # rotate then translate the 4x4 keyframe action
        perturbed_action_gripper_4x4 = torch.bmm(action_gripper_4x4, rot_shift_4x4)
        perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        perturbed_action_trans = perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(perturbed_action_gripper_4x4[:, :3, :3])
        perturbed_action_quat_xyzw = torch.cat([perturbed_action_quat_wxyz[:, 1:],
                                                perturbed_action_quat_wxyz[:, 0].unsqueeze(1)],
                                               dim=1).cpu().numpy()

        # discretize perturbed translation and rotation
        # TODO(mohit): do this in torch without any numpy.
        trans_indicies, rot_grip_indicies = [], []
        for b in range(bs):
            bounds_idx = b if layer > 0 else 0
            bounds_np = bounds[bounds_idx].cpu().numpy()

            trans_idx = utils.point_to_voxel_index(perturbed_action_trans[b], voxel_size, bounds_np)
            trans_indicies.append(trans_idx.tolist())

            quat = perturbed_action_quat_xyzw[b]
            quat = utils.normalize_quaternion(perturbed_action_quat_xyzw[b])
            if quat[-1] < 0:
                quat = -quat
            disc_rot = utils.quaternion_to_discrete_euler(quat, rot_resolution)
            rot_grip_indicies.append(disc_rot.tolist() + [int(action_rot_grip[b, 3].cpu().numpy())])

        # if the perturbed action is out of bounds,
        # the discretized perturb_trans should have invalid indicies
        perturbed_trans = torch.from_numpy(np.array(trans_indicies)).to(device=device)
        perturbed_rot_grip = torch.from_numpy(np.array(rot_grip_indicies)).to(device=device)

    action_trans = perturbed_trans
    action_rot_grip = perturbed_rot_grip

    # apply perturbation to pointclouds
    pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)
    camera_pose = perturb_se3_camera_pose(camera_pose, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)
    return action_trans, action_rot_grip, pcd, camera_pose


### ref: https://github.com/vlc-robot/polarnet
# NOT USED
def random_rotate_pcd_and_action(pcd, action, rot_range, rot=None):
    '''
    pcd: (B, 3, npoints)
    action: (B, 8)
    shift_range: float
    '''

    if rot is None:
        rot = np.random.uniform(-rot_range, rot_range)
    r = R.from_euler('z', rot, degrees=True)

    pos_ori = einops.rearrange(pcd, 'b c n -> (b n) c')
    pos_new = r.apply(pos_ori)
    pcd = einops.rearrange(pos_new, '(b n) c -> b c n', b=pcd.shape[0], n=pcd.shape[2])
    action[..., :3] = r.apply(action[..., :3])
    
    a_ori = R.from_quat(action[..., 3:7])
    a_new = r * a_ori
    action[..., 3:7] = a_new.as_quat()

    return pcd, action

def random_shift_pcd_and_action(pcd, action, shift_range, shift=None):
    '''
    pcd: (B, 3, npoints)
    action: (B, 8)
    shift_range: float
    '''
    if shift is None:
        shift = np.random.uniform(-shift_range, shift_range, size=(3, ))

    pcd = pcd + shift[None, :, None]
    action[..., :3] += shift[None, :]

    return pcd, action
