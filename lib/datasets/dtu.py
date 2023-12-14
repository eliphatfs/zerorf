import os
import io
import cv2
import json
import numpy
import random
# import trimesh
import matplotlib.pyplot as plotlib
from typing import Optional, List
from torch.utils.data import Dataset
from mmgen.datasets.builder import DATASETS
from mmcv.parallel import DataContainer as DC
from multiprocessing.pool import ThreadPool
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math
from PIL import Image
import cv2


from .parallel_zip import ParallelZipFile as ZipFile


def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

# def create_spheric_poses(cameras, n_steps=120):
#     center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype, device=cameras.device)
#     cam_center = F.normalize(cameras.mean(0), p=2, dim=-1) * cameras.mean(0).norm(2)
#     eigvecs = torch.linalg.eig(cameras.T @ cameras).eigenvectors
#     rot_axis = F.normalize(eigvecs[:,1].real.float(), p=2, dim=-1)
#     up = rot_axis
#     rot_dir = torch.cross(rot_axis, cam_center)
#     max_angle = (F.normalize(cameras, p=2, dim=-1) * F.normalize(cam_center, p=2, dim=-1)).sum(-1).acos().max()

#     all_c2w = []
#     for theta in torch.linspace(-max_angle, max_angle, n_steps):
#         cam_pos = cam_center * math.cos(theta) + rot_dir * math.sin(theta)
#         l = F.normalize(center - cam_pos, p=2, dim=0)
#         s = F.normalize(l.cross(up), p=2, dim=0)
#         u = F.normalize(s.cross(l), p=2, dim=0)
#         c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
#         all_c2w.append(c2w)

#     all_c2w = torch.stack(all_c2w, dim=0)
    
#     return all_c2w
def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def get_spiral(c2ws_all, near_fars=np.array([0.1, 1000]), rads_scale=1.0, N_views=120):
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)

@DATASETS.register_module()
class DTUDataset(Dataset):

    def __init__(
        self, root_dir: str, split: str, world_scale: float = 1.0, rgba: bool = False
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.world_scale = world_scale
        self.rgba = rgba
        self.downsample = 2
        self.split = split

        cams = np.load(os.path.join(root_dir, "cameras_sphere.npz"))

        img_sample = cv2.imread(os.path.join(root_dir, 'image', '000000.png'))
        H, W = img_sample.shape[0], img_sample.shape[1]

        w, h = int(W / self.downsample + 0.5), int(H / self.downsample + 0.5)

        self.w, self.h = w, h
        self.img_wh = (w, h)
        self.factor = w / W

        mask_dir = os.path.join(self.root_dir, 'mask')
        self.has_mask = True
        self.apply_mask = True
        
        self.directions = []
        self.poses, self.imgs, self.intrinsic = [], [], []

        n_images = max([int(k.split('_')[-1]) for k in cams.keys()]) + 1

        for i in range(n_images):
            world_mat, scale_mat = cams[f'world_mat_{i}'], cams[f'scale_mat_{i}']
            P = (world_mat @ scale_mat)[:3,:4]
            K, c2w = load_K_Rt_from_P(P)
            fx, fy, cx, cy = K[0,0] * self.factor, K[1,1] * self.factor, K[0,2] * self.factor, K[1,2] * self.factor
            
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w[:3,:4])    

            self.intrinsic.append(torch.tensor([fx, fy, cx, cy], dtype=torch.float32))     

            if self.split in ['train', 'test']:
                img_path = os.path.join(self.root_dir, 'image', f'{i:06d}.png')
                img = Image.open(img_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]

                mask_path = os.path.join(mask_dir, f'{i:03d}.png')
                mask = Image.open(mask_path).convert('L') # (H, W, 1)
                mask = mask.resize(self.img_wh, Image.LANCZOS)
                mask = TF.to_tensor(mask)[0]

                img = img * mask[..., None] + (1 - mask[..., None])
                if rgba:
                    img = torch.cat([img, mask[..., None]], dim=-1)
                self.imgs.append(img)

        self.poses = torch.stack(self.poses, dim=0)
        # concat 0,0,0,1
        self.poses = torch.cat([self.poses, torch.tensor([0, 0, 0, 1.], dtype=torch.float32).repeat(self.poses.shape[0], 1, 1)], axis=-2)

        self.imgs = torch.stack(self.imgs, dim=0)
        self.intrinsic = torch.stack(self.intrinsic, dim=0)

        n_test_traj_steps = 60
        self.render_path = self.poses # get_spiral(self.poses[:,:3,:], N_views=n_test_traj_steps)
        self.render_intrinsic = self.intrinsic #self.intrinsic[[0]].repeat(n_test_traj_steps, 1)


    def __len__(self):
        return 1

    def __getitem__(self, index):
        return dict(
            scene_id=DC(index, cpu_only=True),
            scene_name=DC(self.root_dir, cpu_only=True),
            cond_imgs=numpy.array(self.imgs, numpy.float32),
            cond_poses=numpy.array(self.poses, numpy.float32),
            cond_intrinsics=numpy.array(self.intrinsic, numpy.float32)
        )