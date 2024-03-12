"""
Main NeRF implementation
"""
import torch
from agents.gnfactor_bc.utils import PositionalEncoding
from agents.gnfactor_bc.resnetfc import ResnetFC
import torch.autograd.profiler as profiler
from agents.gnfactor_bc.utils import repeat_interleave
import os
import os.path as osp
import warnings
from termcolor import colored, cprint
import torch.nn.functional as F


class GeneralizableNeRFEmbedNet(torch.nn.Module):
    def __init__(self, conf, stop_encoder_grad=False):
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        self.conf = conf
        self.coordinate_bounds = conf.coordinate_bounds # default: [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
        print(colored(f"[GeneralizableNeRFEmbedNet] coordinate_bounds: {self.coordinate_bounds}", "red"))
        # for voxel sampling
        self._voxel_shape = conf.voxel_shape
        self._voxel_shape_spec = torch.tensor(self._voxel_shape,
                                              ).unsqueeze(
            0) + 2  # +2 because we crop the edges.
        self._dims_orig = self._voxel_shape_spec.int() - 2
        self._res = torch.tensor([[0.0100, 0.0100, 0.0100]]).float().cuda()


        self.image_shape = (conf.image_height, conf.image_width)

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will
        # be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = conf.normalize_z

        self.canon_xyz = True
        print(colored(f"[GeneralizableNeRFEmbedNet] canon_xyz: {self.canon_xyz}", "red"))

        self.stop_encoder_grad = (
            stop_encoder_grad  # Stop ConvNet gradient (freeze weights)
        )
        self.use_code = conf.use_code  # Positional encoding, default: True
        self.use_code_viewdirs = conf.use_code_viewdirs # default: False


        # Enable view directions
        self.use_viewdirs = conf.use_viewdirs
        
        self.use_xyz = conf.use_xyz


        # Regress coordinates
        self.regress_coord = conf.regress_coord # default: False
        print(colored(f"[GeneralizableNeRFEmbedNet] regress_coord: {self.regress_coord}", "red"))

        # Regress attention
        self.regress_attention = conf.regress_attention # default: False
        print(colored(f"[GeneralizableNeRFEmbedNet] regress_attention: {self.regress_attention}", "red"))

        self.d_latent = d_latent = conf.d_latent

        self.use_multi_scale_voxel = conf.use_multi_scale_voxel # False
        
        if self.use_multi_scale_voxel:
            self.d_latent = d_latent = conf.d_multi_scale_latent
        cprint(f"[GeneralizableNeRFEmbedNet] use_multi_scale_voxel: {self.use_multi_scale_voxel} with d_latent: {self.d_latent}", "red")


        self.use_depth_supervision = conf.use_depth_supervision
        cprint(f"[GeneralizableNeRFEmbedNet] use_depth_supervision: {self.use_depth_supervision}", "red")


        self.d_lang = d_lang = conf.d_lang
        d_in = 3 if self.use_xyz else 1

        if self.use_viewdirs and self.use_code_viewdirs:
            # Apply positional encoding to viewdirs
            d_in += 3
        if self.use_code and d_in > 0:
            # Positional encoding for x,y,z OR view z
            self.code = PositionalEncoding.from_conf(conf["code"], d_in=d_in)
            d_in = self.code.d_out
        if self.use_viewdirs and not self.use_code_viewdirs:
            # Don't apply positional encoding to viewdirs (concat after encoded)
            d_in += 3


        d_out = 4 + conf["d_embed"]

        if self.regress_coord:
            d_out += 3
        
        if self.regress_attention:
            d_out += 6

        self.share_mlp = conf.share_mlp

        self.mlp_coarse = ResnetFC(d_in=d_in, d_latent=d_latent, d_lang=d_lang, d_out=d_out, 
                                    d_hidden=conf.mlp.d_hidden, 
                                    n_blocks=conf.mlp.n_blocks, 
                                    combine_layer=conf.mlp.combine_layer,
                                    beta=conf.mlp.beta, use_spade=conf.mlp.use_spade)
        if self.share_mlp:  # True
            self.mlp_fine = self.mlp_coarse
        else:
            self.mlp_fine = ResnetFC(d_in=d_in, d_latent=d_latent, d_lang=d_lang, d_out=d_out,
                                     d_hidden=conf.mlp.d_hidden, 
                                    n_blocks=conf.mlp.n_blocks, 
                                    combine_layer=conf.mlp.combine_layer,
                                    beta=conf.mlp.beta, use_spade=conf.mlp.use_spade)
        # Note: this is world -> camera, and bottom row is omitted
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)

        self.d_in = d_in
        self.d_out = d_out
        self.d_latent = d_latent
        self.d_embed = conf["d_embed"]
        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        # Principal point
        self.register_buffer("c", torch.empty(1, 2), persistent=False)

        self.num_objs = 0
        self.num_views_per_obj = 1


    def encode(self, voxel_feat, lang=None, multi_scale_voxel_list=None, voxel_density=None, poses=None, focal=None,  c=None):
        """
        voxels: channel first
        :param images (NS, 3, H, W)
        NS is number of input (aka source or reference) views
        :param poses (NS, 4, 4)
        :param focal focal length () or (2) or (NS) or (NS, 2) [fx, fy]
        :param z_bounds ignored argument (used in the past)
        :param c principal point None or () or (2) or (NS) or (NS, 2) [cx, cy],
        default is center of image
        """
        self.voxel_feat = voxel_feat # B x 128 x 20^3
        self.multi_scale_voxel_list = multi_scale_voxel_list # multi scale voxel
        if not self.use_multi_scale_voxel:  # default: False
            self.multi_scale_voxel_list = None
        
        self.voxel_density = voxel_density
        if self.voxel_density is not None:
            self.voxel_density = voxel_density.permute(0, 4, 1, 2, 3) # channel first
        if not self.use_depth_supervision:
            self.voxel_density = None
            
        self.language = lang # B x L x 128 or None


    @torch.no_grad()
    def world_to_canonical(self, xyz):
        """
        :param xyz (B, 3)
        :return (B, 3)

        transform world coordinate to canonical coordinate with bounding box
        """
        xyz = xyz.clone()
        bb_min = self.coordinate_bounds[:3]
        bb_max = self.coordinate_bounds[3:]

        bb_min = torch.tensor(bb_min).unsqueeze(0).unsqueeze(0).to(xyz.device)
        bb_max = torch.tensor(bb_max).unsqueeze(0).unsqueeze(0).to(xyz.device)

        xyz -= bb_min
        xyz /= (bb_max - bb_min)

        return xyz
    

    def sample_in_voxel(self, xyz): # NOT USED (voxel_feat)
        """
        :param xyz (B, 3)
        :return (B, Feat)
        """
        xyz_voxel_space = xyz.clone()

        
        coord_bounds = torch.tensor(self.coordinate_bounds).cuda().unsqueeze(0).repeat(xyz.shape[0], 1)
        bb_mins = coord_bounds[..., 0:3]
        bb_maxs = coord_bounds[..., 3:6]
        bb_ranges = bb_maxs - bb_mins
        MIN_DENOMINATOR = 1e-12
        res = bb_ranges / (self._dims_orig.float().cuda() + MIN_DENOMINATOR)
        voxel_indicy_denmominator = res + MIN_DENOMINATOR
        
        # align point cloud to voxel grid
        bb_mins_shifted = bb_mins - self._res
        xyz_voxel_space -= bb_mins_shifted
        xyz_voxel_space /= voxel_indicy_denmominator

        # normalize to [0, 1]
        xyz_voxel_space = xyz_voxel_space / self._voxel_shape

        # bounding box [0, 1] to [-1, 1]
        xyz_voxel_space = xyz_voxel_space * 2 - 1.0


        #  unsqeeuze the point cloud to also have 5 dim
        xyz_voxel_space = xyz_voxel_space.unsqueeze(1).unsqueeze(1) # (B, 1, 1, N,  3)

        
        # sample in voxel space
        point_feature = F.grid_sample(self.voxel_feat, xyz_voxel_space, align_corners=True, mode='bilinear')
        # point_density = F.grid_sample(self.voxel_density, xyz_voxel_space, align_corners=True, mode='bilinear')
        
        # post activation on density (Ref: DVGO), using softplus
        # point_density = torch.log(1 + torch.exp(point_density))


        # squeeze back to point cloud shape 
        point_feature = point_feature.squeeze(2).squeeze(2).permute(0, 2, 1) 
        # point_density = point_density.squeeze(2).squeeze(2).permute(0, 2, 1)

        # concat density to feature
        # point_feature = torch.cat((point_feature, point_density), dim=-1)

        return point_feature, None
    

    def sample_in_canonical_voxel(self, xyz):   # USED
        """
        :param xyz (B, 3)
        :return (B, Feat)
        """
        xyz_voxel_space = xyz.clone()

        xyz_voxel_space = xyz_voxel_space * 2 - 1.0


        #  unsqeeuze the point cloud to also have 5 dim
        xyz_voxel_space = xyz_voxel_space.unsqueeze(1).unsqueeze(1) # (B, 1, 1, N,  3)

        # xyz_voxel_space = xyz_voxel_space.permute(0, 4, 1, 2, 3).contiguous() # (B, 3, 1, 1, N)

        # sample in voxel space
        point_feature = F.grid_sample(self.voxel_feat, xyz_voxel_space, align_corners=True, mode='bilinear')
        # squeeze back to point cloud shape 
        point_feature = point_feature.squeeze(2).squeeze(2).permute(0, 2, 1) 
        
        if self.multi_scale_voxel_list is not None:
            multi_scale_point_feat_list = []
            for voxel_feat in self.multi_scale_voxel_list:
                sampled_feature = F.grid_sample(voxel_feat, xyz_voxel_space, align_corners=True, mode='bilinear')
                sampled_feature = sampled_feature.squeeze(2).squeeze(2).permute(0, 2, 1)
                multi_scale_point_feat_list.append(sampled_feature)
            multi_scale_point_feat_list.append(point_feature)
            point_feature = torch.cat(multi_scale_point_feat_list, dim=-1)
        
        if self.use_depth_supervision:  # default: False
            point_density = F.grid_sample(self.voxel_density, xyz_voxel_space, align_corners=True, mode='bilinear')
            point_density = point_density.squeeze(2).squeeze(2).permute(0, 2, 1)
            return point_feature, point_density

        return point_feature, None
    
    def forward(self, xyz, coarse=True, viewdirs=None, far=False, ret_last_feat=False):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """

        with profiler.record_function("model_inference"):
            SB, B, _ = xyz.shape
            NS = self.num_views_per_obj # 1

            # normalize the point cloud bounding box into voxel space ranging from [0, 1]. outlier points are outside of [0, 1]
            xyz = self.world_to_canonical(xyz)

            canon_xyz = xyz.clone() # [1,4096,3], min:-2.28, max:1.39

            # Transform query points into the camera spaces of the input views
            xyz = repeat_interleave(xyz, NS)  # (SB*NS, B, 3), [1,4096,3]


            # # debug
            # self.points = xyz

            if self.canon_xyz:  # default: True
                xyz_rot = xyz

            # else:
            #     xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[
            #         ..., 0
            #     ]
            #     xyz = xyz_rot + self.poses[:, None, :3, 3]

            if self.d_in > 0:   # default: d_in=42
                # * Encode the xyz coordinates
                if self.use_xyz:    # True
                    if self.normalize_z:
                        z_feature = xyz_rot.reshape(-1, 3)  # (SB*B, 3)
                    else:   # True
                        z_feature = xyz.reshape(-1, 3)  # (SB*B, 3)
                else:
                    if self.normalize_z:
                        z_feature = -xyz_rot[..., 2].reshape(-1, 1)  # (SB*B, 1)
                    else:
                        z_feature = -xyz[..., 2].reshape(-1, 1)  # (SB*B, 1)

                if self.use_code and not self.use_code_viewdirs:    # True
                    # Positional encoding (no viewdirs)
                    z_feature = self.code(z_feature)    # [4096, 39]

                if self.use_viewdirs:   # default: True
                    # * Encode the view directions
                    assert viewdirs is not None
                    # Viewdirs to input view space
                    viewdirs = viewdirs.reshape(SB, B, 3, 1)
                    viewdirs = repeat_interleave(viewdirs, NS)  # (SB*NS, B, 3, 1)
                    # if not self.canon_xyz:
                    #     viewdirs = torch.matmul(
                    #         self.poses[:, None, :3, :3], viewdirs
                    #     )  # (SB*NS, B, 3, 1)
                    viewdirs = viewdirs.reshape(-1, 3)  # (SB*B, 3)
                    z_feature = torch.cat(
                        (z_feature, viewdirs), dim=1
                    )  # (SB*B, 4 or 6)

                if self.use_code and self.use_code_viewdirs:    # False
                    # Positional encoding (with viewdirs)
                    z_feature = self.code(z_feature)

                mlp_input = z_feature   # [4096,42]


            # volumetric sampling
            latent, point_density = self.sample_in_canonical_voxel(canon_xyz)
            if self.stop_encoder_grad:  # False
                latent = latent.detach()
            latent = latent.reshape(
                -1, self.d_latent
            )  # (SB * NS * B, latent)  [4096, 128]

            if self.d_in == 0:
                # z_feature not needed
                mlp_input = latent
            else:   # True
                mlp_input = torch.cat((latent, z_feature), dim=-1)  # [4096,170]


            # Camera frustum culling stuff, currently disabled
            combine_index = None
            dim_size = None
            bs = self.voxel_feat.shape[0]
            # Run main NeRF network
            if coarse or self.mlp_fine is None:
                mlp_output, prev_output = self.mlp_coarse(
                    mlp_input,
                    combine_inner_dims=(self.num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                    ret_last_feat=ret_last_feat,
                    language_embed=self.language,
                    batch_size=bs
                )   # mlp_output: [B,4096,516]
            else:
                mlp_output, prev_output = self.mlp_fine(
                    mlp_input,
                    combine_inner_dims=(self.num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                    ret_last_feat=ret_last_feat,
                    language_embed=self.language,
                    batch_size=bs
                )

            # Interpret the output
            if ret_last_feat:
                last_feat = mlp_output[..., self.d_out:]
                mlp_output = mlp_output[..., :self.d_out]
            mlp_output = mlp_output.reshape(-1, B, self.d_out)

            if ret_last_feat:
                last_feat = last_feat.reshape(mlp_output.shape[0], B, -1)

            rgb = mlp_output[..., :3]
            sigma = mlp_output[..., 3:4]

            if self.regress_coord and not self.regress_attention:
                embed = mlp_output[..., 4:-3]
                coord = mlp_output[..., -3:]
                coord_residual = coord - canon_xyz
                output_list = [torch.sigmoid(rgb), torch.relu(sigma), embed, coord_residual]
            elif self.regress_coord and self.regress_attention: # attent includes last 6 dim
                embed = mlp_output[..., 4:-9]
                coord = mlp_output[..., -9:-6]
                coord_residual = coord - canon_xyz
                attention = mlp_output[..., -6:]
                output_list = [torch.sigmoid(rgb), torch.relu(sigma), embed, coord_residual, attention]
            elif not self.regress_coord and self.regress_attention:
                embed = mlp_output[..., 4:-6]
                attention = mlp_output[..., -6:]
                output_list = [torch.sigmoid(rgb), torch.relu(sigma), embed, attention]
            else:   # True
                embed = mlp_output[..., 4:]
                output_list = [torch.sigmoid(rgb), torch.relu(sigma), embed]
            output = torch.cat(output_list, dim=-1)
            output = output.reshape(SB, B, -1)

        if not ret_last_feat:
            return output, point_density
        else:
            return output, last_feat, point_density


    def load_weights(self, args, opt_init=False, strict=True, device=None):
        """
        Helper for loading weights according to argparse arguments.
        Your can put a checkpoint at checkpoints/<exp>/pixel_nerf_init to use as initialization.
        :param opt_init if true, loads from init checkpoint instead of usual even when resuming
        """
        # TODO: make backups
        if opt_init and not args.resume:
            return
        ckpt_name = (
            "pixel_nerf_init" if opt_init or not args.resume else "pixel_nerf_latest"
        )
        model_path = "%s/%s/%s" % (args.checkpoints_path, args.name, ckpt_name)

        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print("Load", model_path)
            self.load_state_dict(
                torch.load(model_path, map_location=device), strict=strict
            )
        elif not opt_init:
            warnings.warn(
                (
                    "WARNING: {} does not exist, not loaded!! Model will be re-initialized.\n"
                    + "If you are trying to load a pretrained model, STOP since it's not in the right place. "
                    + "If training, unless you are startin a new experiment, please remember to pass --resume."
                ).format(model_path)
            )
        return self


    def save_weights(self, args, opt_init=False):
        """
        Helper for saving weights according to argparse arguments
        :param opt_init if true, saves from init checkpoint instead of usual
        """
        from shutil import copyfile

        ckpt_name = "pixel_nerf_init" if opt_init else "pixel_nerf_latest"
        backup_name = "pixel_nerf_init_backup" if opt_init else "pixel_nerf_backup"

        ckpt_path = osp.join(args.checkpoints_path, args.name, ckpt_name)
        ckpt_backup_path = osp.join(args.checkpoints_path, args.name, backup_name)

        if osp.exists(ckpt_path):
            copyfile(ckpt_path, ckpt_backup_path)
        torch.save(self.state_dict(), ckpt_path)
        return self
