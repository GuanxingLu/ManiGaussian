import numpy as np
import pyrender
import torch
import trimesh
from pyrender.trackball import Trackball
from rlbench.backend.const import DEPTH_SCALE
from scipy.spatial.transform import Rotation
from rlbench.backend.observation import Observation
from rlbench import CameraConfig, ObservationConfig
from pyrep.const import RenderMode
from typing import List

REMOVE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
               'gripper_open', 'gripper_pose',
               'gripper_joint_positions', 'gripper_touch_forces',
               'task_low_dim_state', 'misc']

SCALE_FACTOR = DEPTH_SCALE
DEFAULT_SCENE_SCALE = 2.0


def loss_weights(replay_sample, beta=1.0):
    loss_weights = 1.0
    if 'sampling_probabilities' in replay_sample:
        probs = replay_sample['sampling_probabilities']
        loss_weights = 1.0 / torch.sqrt(probs + 1e-10)
        loss_weights = (loss_weights / torch.max(loss_weights)) ** beta
    return loss_weights


def save_tensor(x, save_path):
    import pickle
    from termcolor import cprint
    with open(save_path, 'wb') as f:
        pickle.dump(x, f)
    cprint(f'save tensor with shape {x.shape} to {save_path}', 'green')
    
def soft_updates(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def stack_on_channel(x):
    # expect (B, T, C, ...)
    return torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)


def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)


def correct_rotation_instability(disc, resolution):
    return disc

def check_gimbal_lock(pred_rot_and_grip, gt_rot_and_grip, resolution):
    pred_rot_and_grip_np = pred_rot_and_grip.detach().cpu().numpy()
    gt_rot_and_grip_np = gt_rot_and_grip.detach().cpu().numpy()

    pred_rot = discrete_euler_to_quaternion(pred_rot_and_grip_np[:,:3], resolution)
    gt_rot = discrete_euler_to_quaternion(gt_rot_and_grip_np[:,:3], resolution)
    gimbal_lock_matches = [np.all(np.abs(pred_rot[i] - gt_rot[i]) < 1e-10) and
                           np.any(pred_rot_and_grip_np[i,:3] != gt_rot_and_grip_np[i, :3])
                           for i in range(pred_rot.shape[0])]
    return 0

def quaternion_to_discrete_euler(quaternion, resolution):
    euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler('xyz', euluer, degrees=True).as_quat()

def point_to_voxel_index(
        point: np.ndarray,
        voxel_size: np.ndarray,
        coord_bounds: np.ndarray):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(
            np.int32), dims_m_one)
    return voxel_indicy

def voxel_index_to_point(
        voxel_index: torch.Tensor,
        voxel_size: int,
        coord_bounds: np.ndarray):
    res = (coord_bounds[:, 3:] - coord_bounds[:, :3]) / voxel_size
    points = (voxel_index * res) + coord_bounds[:, :3]
    return points

def point_to_pixel_index(
        point: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray):
    point = np.array([point[0], point[1], point[2], 1])
    world_to_cam = np.linalg.inv(extrinsics)
    point_in_cam_frame = world_to_cam.dot(point)
    px, py, pz = point_in_cam_frame[:3]
    px = 2 * intrinsics[0, 2] - int(-intrinsics[0, 0] * (px / pz) + intrinsics[0, 2])
    py = 2 * intrinsics[1, 2] - int(-intrinsics[1, 1] * (py / pz) + intrinsics[1, 2])
    return px, py


def _compute_initial_camera_pose(scene):
    # Adapted from:
    # https://github.com/mmatl/pyrender/blob/master/pyrender/viewer.py#L1032
    centroid = scene.centroid
    # print("centroid is ", centroid)
    scale = scene.scale
    if scale == 0.0:
        scale = DEFAULT_SCENE_SCALE
    s2 = 1.0 / np.sqrt(2.0)
    cp = np.eye(4)
    cp[:3, :3] = np.array([[0.0, -s2, s2], [1.0, 0.0, 0.0], [0.0, s2, s2]])
    hfov = np.pi / 6.0
    dist = scale / (2.0 * np.tan(hfov))
    cp[:3, 3] = dist * np.array([1.0, 0.0, 1.0]) + centroid
    return cp


def _from_trimesh_scene(
        trimesh_scene, bg_color=None, ambient_light=None):
    # convert trimesh geometries to pyrender geometries
    geometries = {name: pyrender.Mesh.from_trimesh(geom, smooth=False)
                  for name, geom in trimesh_scene.geometry.items()}
    # create the pyrender scene object
    scene_pr = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)
    # add every node with geometry to the pyrender scene
    for node in trimesh_scene.graph.nodes_geometry:
        pose, geom_name = trimesh_scene.graph[node]
        scene_pr.add(geometries[geom_name], pose=pose)
    return scene_pr


def _create_bounding_box(scene, voxel_size, res):
    l = voxel_size * res
    T = np.eye(4)
    w = 0.01
    for trans in [[0, 0, l / 2], [0, l, l / 2], [l, l, l / 2], [l, 0, l / 2]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(trimesh.creation.box(
            [w, w, l], T, face_colors=[0, 0, 0, 255]))
    for trans in [[l / 2, 0, 0], [l / 2, 0, l], [l / 2, l, 0], [l / 2, l, l]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(trimesh.creation.box(
            [l, w, w], T, face_colors=[0, 0, 0, 255]))
    for trans in [[0, l / 2, 0], [0, l / 2, l], [l, l / 2, 0], [l, l / 2, l]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(trimesh.creation.box(
            [w, l, w], T, face_colors=[0, 0, 0, 255]))


def create_voxel_scene(
        voxel_grid: np.ndarray,
        q_attention: np.ndarray = None,
        highlight_coordinate: np.ndarray = None,
        highlight_gt_coordinate: np.ndarray = None,
        highlight_alpha: float = 1.0,
        voxel_size: float = 0.1,
        show_bb: bool = False,
        alpha: float = 0.5):
    _, d, h, w = voxel_grid.shape
    v = voxel_grid.transpose((1, 2, 3, 0))
    occupancy = v[:, :, :, -1] != 0
    alpha = np.expand_dims(np.full_like(occupancy, alpha, dtype=np.float32), -1)
    rgb = np.concatenate([(v[:, :, :, 3:6] + 1)/ 2.0, alpha], axis=-1)

    if q_attention is not None:
        q = np.max(q_attention, 0)
        q = q / np.max(q)
        show_q = (q > 0.75)
        occupancy = (show_q + occupancy).astype(bool)
        q = np.expand_dims(q - 0.5, -1)  # Max q can be is 0.9
        q_rgb = np.concatenate([
            q, np.zeros_like(q), np.zeros_like(q),
            np.clip(q, 0, 1)], axis=-1)
        rgb = np.where(np.expand_dims(show_q, -1), q_rgb, rgb)

    if highlight_coordinate is not None:
        x, y, z = highlight_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [1.0, 0.0, 0.0, highlight_alpha]

    if highlight_gt_coordinate is not None:
        x, y, z = highlight_gt_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [0.0, 0.0, 1.0, highlight_alpha]

    transform = trimesh.transformations.scale_and_translate(
        scale=voxel_size, translate=(0.0, 0.0, 0.0))
    trimesh_voxel_grid = trimesh.voxel.VoxelGrid(
        encoding=occupancy, transform=transform)
    geometry = trimesh_voxel_grid.as_boxes(colors=rgb)
    scene = trimesh.Scene()
    scene.add_geometry(geometry)
    if show_bb:
        assert d == h == w
        _create_bounding_box(scene, voxel_size, d)
    return scene


def visualise_voxel(voxel_grid: np.ndarray,
                    q_attention: np.ndarray = None,
                    highlight_coordinate: np.ndarray = None,
                    highlight_gt_coordinate: np.ndarray = None,
                    highlight_alpha: float = 1.0,
                    rotation_amount: float = 0.0,
                    show: bool = False,
                    voxel_size: float = 0.1,
                    offscreen_renderer: pyrender.OffscreenRenderer = None,
                    show_bb: bool = False,
                    alpha: float = 0.5):
    scene = create_voxel_scene(
        voxel_grid, q_attention, highlight_coordinate, highlight_gt_coordinate,
        highlight_alpha, voxel_size,
        show_bb, alpha)
    if show:
        scene.show()
    else:
        r = offscreen_renderer or pyrender.OffscreenRenderer(
            viewport_width=640, viewport_height=480, point_size=1.0)
        s = _from_trimesh_scene(
            scene, ambient_light=[0.8, 0.8, 0.8],
            bg_color=[1.0, 1.0, 1.0])
        cam = pyrender.PerspectiveCamera(
            yfov=np.pi / 4.0, aspectRatio=r.viewport_width/r.viewport_height)
        p = _compute_initial_camera_pose(s)
        t = Trackball(p, (r.viewport_width, r.viewport_height), s.scale, s.centroid)
        t.rotate(rotation_amount, np.array([0.0, 0.0, 1.0]))
        s.add(cam, pose=t.pose)
        color, depth = r.render(s)
        return color.copy()


def preprocess(img, dist='transporter'):
    """Pre-process input (subtract mean, divide by std)."""

    transporter_color_mean = [0.18877631, 0.18877631, 0.18877631]
    transporter_color_std = [0.07276466, 0.07276466, 0.07276466]
    transporter_depth_mean = 0.00509261
    transporter_depth_std = 0.00903967

    franka_color_mean = [0.622291933, 0.628313992, 0.623031488]
    franka_color_std = [0.168154213, 0.17626014, 0.184527364]
    franka_depth_mean = 0.872146842
    franka_depth_std = 0.195743116

    clip_color_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_color_std = [0.26862954, 0.26130258, 0.27577711]

    # choose distribution
    if dist == 'clip':
        color_mean = clip_color_mean
        color_std = clip_color_std
    elif dist == 'franka':
        color_mean = franka_color_mean
        color_std = franka_color_std
    else:
        color_mean = transporter_color_mean
        color_std = transporter_color_std

    if dist == 'franka':
        depth_mean = franka_depth_mean
        depth_std = franka_depth_std
    else:
        depth_mean = transporter_depth_mean
        depth_std = transporter_depth_std

    # convert to pytorch tensor (if required)
    if type(img) == torch.Tensor:
        def cast_shape(stat, img):
            tensor = torch.from_numpy(np.array(stat)).to(device=img.device, dtype=img.dtype)
            tensor = tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            tensor = tensor.repeat(img.shape[0], 1, img.shape[-2], img.shape[-1])
            return tensor

        color_mean = cast_shape(color_mean, img)
        color_std = cast_shape(color_std, img)
        depth_mean = cast_shape(depth_mean, img)
        depth_std = cast_shape(depth_std, img)

        # normalize
        img = img.clone()
        img[:, :3, :, :] = ((img[:, :3, :, :] / 255 - color_mean) / color_std)
        img[:, 3:, :, :] = ((img[:, 3:, :, :] - depth_mean) / depth_std)
    else:
        # normalize
        img[:, :, :3] = (img[:, :, :3] / 255 - color_mean) / color_std
        img[:, :, 3:] = (img[:, :, 3:] - depth_mean) / depth_std
    return img


def rand_dist(size, min=-1.0, max=1.0):
    return (max-min) * torch.rand(size) + min


def rand_discrete(size, min=0, max=1):
    if min == max:
        return torch.zeros(size)
    return torch.randint(min, max+1, size)


def split_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def extract_obs(obs: Observation,
                cameras,
                t: int = 0,
                prev_action=None,
                channels_last: bool = False,
                episode_length: int = 10,
                next_obs: Observation = None,
                ):
    obs.joint_velocities = None
    grip_mat = obs.gripper_matrix
    grip_pose = obs.gripper_pose
    joint_pos = obs.joint_positions
    obs.gripper_pose = None
    obs.gripper_matrix = None
    obs.wrist_camera_matrix = None
    obs.joint_positions = None
    if obs.gripper_joint_positions is not None:
        obs.gripper_joint_positions = np.clip(
            obs.gripper_joint_positions, 0., 0.04)
    

    if obs.nerf_multi_view_rgb is not None:
        nerf_multi_view_rgb = obs.nerf_multi_view_rgb
    else:
        nerf_multi_view_rgb = None

    if obs.nerf_multi_view_depth is not None:
        nerf_multi_view_depth = obs.nerf_multi_view_depth
    else:
        nerf_multi_view_depth = None

    if obs.nerf_multi_view_camera is not None:
        nerf_multi_view_camera = obs.nerf_multi_view_camera
    else:
        nerf_multi_view_camera = None


    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    robot_state = np.array([
        obs.gripper_open,
        *obs.gripper_joint_positions])
    # remove low-level proprioception variables that are not needed
    obs_dict = {k: v for k, v in obs_dict.items()
                if k not in REMOVE_KEYS}

    if not channels_last:
        # swap channels from last dim to 1st dim
        obs_dict = {k: np.transpose(
            v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                    for k, v in obs_dict.items() if type(v) == np.ndarray or type(v) == list}
    else:
        # add extra dim to depth data
        obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
                    for k, v in obs_dict.items()}
    obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)
    # binary variable indicating if collisions are allowed or not while planning paths to reach poses
    obs_dict['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)
    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
        obs_dict[k] = v.astype(np.float32)

    for camera_name in cameras:
        obs_dict['%s_camera_extrinsics' % camera_name] = obs.misc['%s_camera_extrinsics' % camera_name]
        obs_dict['%s_camera_intrinsics' % camera_name] = obs.misc['%s_camera_intrinsics' % camera_name]

    # add timestep to low_dim_state
    time = (1. - (t / float(episode_length - 1))) * 2. - 1.
    obs_dict['low_dim_state'] = np.concatenate(
        [obs_dict['low_dim_state'], [time]]).astype(np.float32)

    obs.gripper_matrix = grip_mat
    obs.joint_positions = joint_pos
    obs.gripper_pose = grip_pose

    obs_dict['nerf_multi_view_rgb'] = nerf_multi_view_rgb
    obs_dict['nerf_multi_view_depth'] = nerf_multi_view_depth
    obs_dict['nerf_multi_view_camera'] = nerf_multi_view_camera

    # for next frame prediction
    if next_obs is not None:
        if next_obs.nerf_multi_view_rgb is not None:
            obs_dict['nerf_next_multi_view_rgb'] = next_obs.nerf_multi_view_rgb
            obs_dict['nerf_next_multi_view_depth'] = next_obs.nerf_multi_view_depth
            obs_dict['nerf_next_multi_view_camera'] = next_obs.nerf_multi_view_camera
        else:
            obs_dict['nerf_next_multi_view_rgb'] = None
            obs_dict['nerf_next_multi_view_depth'] = None
            obs_dict['nerf_next_multi_view_camera'] = None

    # if next_obs is None, we do not add the next frame prediction

    return obs_dict


def create_obs_config(camera_names: List[str],
                       camera_resolution: List[int],
                       method_name: str,
                       use_nerf_multi_view=True,
                       use_depth=False):
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=False,
        depth=use_depth,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL)

    cam_obs = []
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams
        cam_obs.append('%s_rgb' % n)
        cam_obs.append('%s_depth' % n)
        cam_obs.append('%s_pointcloud' % n)

    # Some of these obs are only used for keypoint detection.
    obs_config = ObservationConfig(
        front_camera=kwargs.get('front', unused_cams),
        left_shoulder_camera=kwargs.get('left_shoulder', unused_cams),
        right_shoulder_camera=kwargs.get('right_shoulder', unused_cams),
        wrist_camera=kwargs.get('wrist', unused_cams),
        overhead_camera=kwargs.get('overhead', unused_cams),
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
        nerf_multi_view=use_nerf_multi_view,
    )
    return obs_config


def get_device(gpu):
    if gpu is not None and gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:%d" % gpu)
        torch.backends.cudnn.enabled = torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    return device