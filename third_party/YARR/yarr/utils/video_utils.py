import os
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench import Environment
from rlbench.backend.observation import Observation
from rlbench.backend import utils
from rlbench.backend.const import *

class CameraMotion(object):
    def __init__(self, cam: VisionSensor):
        self.cam = cam

    def step(self):
        raise NotImplementedError()

    def save_pose(self):
        self._prev_pose = self.cam.get_pose()

    def restore_pose(self):
        self.cam.set_pose(self._prev_pose)


class CircleCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor, origin: Dummy,
                 speed: float, init_rotation: float = np.deg2rad(180)):
        super().__init__(cam)
        self.origin = origin
        self.speed = speed  # in radians
        self.origin.rotate([0, 0, init_rotation])

    def step(self):
        self.origin.rotate([0, 0, self.speed])


class StaticCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor, origin: Dummy,
                 speed: float, init_rotation: float = np.deg2rad(180)):
        super().__init__(cam)
        self.origin = origin
        self.speed = speed  # in radians
        self.origin.rotate([0, 0, init_rotation])

    def step(self):
        # self.origin.rotate([0, 0, 0])
        pass


class TaskRecorder(object):

    def __init__(self, env: Environment, cam_motion: CameraMotion, fps=30):
        self._env = env
        self._cam_motion = cam_motion
        self._fps = fps
        self._snaps = []
        self._current_snaps = []
    

    def take_snap(self, obs: Observation):
        self._cam_motion.step()
        self._current_snaps.append(
            (self._cam_motion.cam.capture_rgb() * 255.).astype(np.uint8))


    def save(self, path, lang_goal=None, reward=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print('Converting to video and saving to {}'.format(path))
        assert len(self._current_snaps) > 0, 'No snaps to save'
        # OpenCV QT version can conflict with PyRep, so import here
        import cv2, torchvision, torch
        image_size = self._cam_motion.cam.get_resolution()
        fourcc = cv2.VideoWriter_fourcc(*'vp80')

        # video = cv2.VideoWriter(
        #         path, fourcc, self._fps,
        #         tuple(image_size))
        video_frames = []

        for image in self._current_snaps:
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if lang_goal is not None:
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = (0.45 * image_size[0]) / 640
                font_thickness = 2

                lang_textsize = cv2.getTextSize(lang_goal, font, font_scale, font_thickness)[0]
                lang_textX = (image_size[0] - lang_textsize[0]) // 2

                frame = cv2.putText(frame, lang_goal, org=(lang_textX, image_size[1] - 35),
                                    fontScale=font_scale, fontFace=font, color=(0, 0, 0),
                                    thickness=font_thickness, lineType=cv2.LINE_AA)

            # video.write(frame)
            video_frames.append(frame)
        # video.release()
        # make it a video
        video_frames = np.stack(video_frames, axis=0)
        # change BGR to RGB
        video_frames = torch.from_numpy(video_frames)
        video_frames = video_frames[:, :, :, [2, 1, 0]]
        torchvision.io.write_video(path, video_frames, self._fps)


        self._current_snaps = []



class NeRFTaskRecorder(object):
    """
    for nerf data generation
    """

    def __init__(self, env: Environment, cam_motion: CameraMotion, fps=30, num_views=50):
        self._env = env
        self._cam_motion = cam_motion
        self._fps = fps
        self._num_views = num_views

        self._snaps_episode = []
        self._depths_episode = []
        self._poses_episode = []
        self._intrinsics_episode = []
        self._near_far_episode = []
        self.t = 0

        from termcolor import colored
        print(colored('[NeRFTaskRecorder] num_views: {}'.format(num_views), 'red'))

        # create a progress bar
        from tqdm import tqdm
        self.pbar = tqdm(total=200)


    def record_task_description(self, task_description):
        self._task_description = task_description
    

    def take_snap(self, scene=None, obs: Observation=None):
        # save start pose
        self._cam_motion.save_pose()
        
        # sparse sampling along time
        self.t += 1        
        self.pbar.update(1) # update progress bar

        # not every 10 steps
        # if self.t % 10 != 0:
        #     return

        # get views
        all_views = []
        all_depths = []
        all_poses = []
        all_intrinsics = []
        all_near_far = []
        # t=0, which we don't need
        # all_views.append((self._cam_motion.cam.capture_rgb() * 255.).astype(np.uint8))
        # all_poses.append(self._cam_motion.cam.get_matrix())
        # all_intrinsics.append(self._cam_motion.cam.get_intrinsic_matrix())

        for i in range(self._num_views):
            self._cam_motion.step()

            # sparse sampling along views
            if i < 20 or i > 40:
                continue
            # if i%2 == 0: # squeeze to 10 views
            #     continue
            

            scene.step() # step the simulation environment
            all_views.append(
                (self._cam_motion.cam.capture_rgb() * 255.).astype(np.uint8))
            all_poses.append(self._cam_motion.cam.get_matrix())
            all_depths.append(self._cam_motion.cam.capture_depth(in_meters=False))
            all_intrinsics.append(self._cam_motion.cam.get_intrinsic_matrix())
            all_near_far.append((self._cam_motion.cam.get_near_clipping_plane(),
                                 self._cam_motion.cam.get_far_clipping_plane()))
        
        self._snaps_episode.append(all_views)
        self._depths_episode.append(all_depths)
        self._poses_episode.append(all_poses)
        self._intrinsics_episode.append(all_intrinsics)
        self._near_far_episode.append(all_near_far)

        # restore start pose
        self._cam_motion.restore_pose()
        scene.step()
        

    def make_video(self, path, img_list):
        import cv2
        image_size = self._cam_motion.cam.get_resolution()
        video = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self._fps,
                tuple(image_size))
        for image in img_list:
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.write(frame)
        video.release()


    def save_extrinsic_and_intrinsic(self, path, extrinsic, intrinsic):

        with open(path, 'w') as f:
            for row in extrinsic:
                for ele in row:
                    f.write('{:.6f}'.format(ele) + ' ')
                f.write('\n')
            f.write('\n')
            for row in intrinsic:
                for ele in row:
                    f.write('{:.6f}'.format(ele) + ' ')
                f.write('\n')


    def save(self, path_dir):
        """
        save imgs and poses for nerf
        """
        os.makedirs(path_dir, exist_ok=True)
        print('saving imgs, extrinsic, intrinsic to {}'.format(path_dir))

        # reset progress bar
        self.pbar.reset()

        assert len(self._snaps_episode) > 0, 'No imgs to save'
        
        import cv2
        
        for t, all_views in enumerate(self._snaps_episode):
            # save all views and poses of this time step in a folder
            timestep_dir = os.path.join(path_dir, str(t))
            timestep_img_dir = os.path.join(timestep_dir, 'images')
            timestep_depth_dir = os.path.join(timestep_dir, 'depths')
            timestep_pose_dir = os.path.join(timestep_dir, 'poses')

            os.makedirs(timestep_img_dir, exist_ok=True)
            os.makedirs(timestep_depth_dir, exist_ok=True)
            os.makedirs(timestep_pose_dir, exist_ok=True)

            all_poses = self._poses_episode[t]
            all_intrinsics = self._intrinsics_episode[t]
            for i, view in enumerate(all_views):
                # save the image
                img_path = os.path.join(timestep_img_dir, str(i) + '.png')
                view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, view)

                # save the depth
                depth_path = os.path.join(timestep_depth_dir, str(i) + '.png')
                depth = self._depths_episode[t][i]
                depth = utils.float_array_to_rgb_image(depth, scale_factor=DEPTH_SCALE)
                depth.save(depth_path)
                
                # save the pose and intrinsic
                pose_path = os.path.join(timestep_pose_dir, str(i) + '.txt')
                transformation_matrix =  all_poses[i]
                intrinsic_matrix = all_intrinsics[i]

                self.save_extrinsic_and_intrinsic(pose_path, transformation_matrix, intrinsic_matrix)

                # save the pose and intrinsic in one file
                # np.savetxt(pose_path, transformation_matrix, fmt='%.6f')
                # np.savetxt(pose_path, intrinsic_matrix, fmt='%.6f')
                
            # save description
            with open(os.path.join(timestep_dir, 'description.txt'), 'w') as f:
                # save each line
                for desc in self._task_description:
                    f.write(desc + '\n')
        print('Successfully saved {} time steps'.format(self.t))
        self.t = 0 # reset time
        self._current_snaps = [] # clear current snaps
        self._poses_episode = []
        self._intrinsics_episode = []