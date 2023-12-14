import os
import io
import cv2
import json
import numpy
import numpy as np
import random
# import trimesh
import matplotlib.pyplot as plotlib
from typing import Optional, List
from torch.utils.data import Dataset
from mmgen.datasets.builder import DATASETS
from mmcv.parallel import DataContainer as DC
from multiprocessing.pool import ThreadPool

import glob
import imageio

from .parallel_zip import ParallelZipFile as ZipFile


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

# b = numpy.random.randn(3, 32)
# x, y, z = b
# trimesh.transformations.affine_matrix_from_points(b, numpy.stack([x, z, -y]))
BLENDER_TO_OPENGL_MATRIX = numpy.array([
    [1,  0,  0,  0],
    [0,  0,  1,  0],
    [0, -1,  0,  0],
    [0,  0,  0,  1]
], dtype=numpy.float32)
BLENDER_TO_OPENCV_MATRIX = numpy.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1]
], dtype=numpy.float32)


def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes."""
    poses_ = poses.copy()
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean
    eigval, eigvec = np.linalg.eig(t.T @ t)

    # Sort eigenvectors in order of largest to the smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    bottom = np.broadcast_to([0, 0, 0, 1.], poses[..., :1, :4].shape)
    pad_poses = np.concatenate([poses[..., :3, :4], bottom], axis=-2)
    poses_recentered = transform @ pad_poses
    poses_recentered = poses_recentered[..., :3, :4]

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor

    poses_[:, :3, :4] = poses_recentered[:, :3, :4]
    poses_recentered = poses_
    return poses_recentered

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def generate_ellipse_path(poses: np.ndarray,
                          n_frames: int = 120,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
            low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])


@DATASETS.register_module()
class Unbounded(Dataset):

    def __init__(
        self, base_dir: str, world_scale: float = 1.0, rgba: bool = False, split: str ='train'
    ) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.world_scale = world_scale
        self.rgba = rgba
        self.hold_every = 8
        self.split = split

        # Enumerate all image files
        all_img = np.array([f for f in sorted(glob.glob(os.path.join(self.base_dir, "images_4", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')])
        imgs = np.array([imageio.imread(f) / 255. for f in all_img])

        # Load camera poses
        poses_bounds = np.load(os.path.join(self.base_dir, 'poses_bounds.npy'))
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)

        # Step 1: rescale focal length according to training resolution
        intrinsic = poses[:, :, -1]  # original intrinsics, same for all images
        # self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        intrinsic /= 4

        self.intrinsic = []
        for i in range(len(intrinsic)):
            self.intrinsic.append(np.array([intrinsic[i, 2], intrinsic[i, 2], intrinsic[i, 1]/2, intrinsic[i, 0]/2]))   # focal, focal, cx, cy
        self.intrinsic = np.array(self.intrinsic)

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        poses = transform_poses_pca(poses)

        # build rendering path
        N_views = 30
        render_path = generate_ellipse_path(poses, n_frames=N_views)
        self.render_path = render_path @ BLENDER_TO_OPENCV_MATRIX * self.world_scale
        self.render_intrinsic = self.intrinsic[:N_views]

        # concat 0,0,0,1
        poses = np.concatenate([poses, np.broadcast_to(np.array([0, 0, 0, 1.], dtype=np.float32), poses[..., :1, :4].shape)], axis=-2)
        poses = poses @ BLENDER_TO_OPENCV_MATRIX * self.world_scale
                
        i_test = np.arange(0, poses.shape[0], self.hold_every)
        self.img_list = i_test if self.split != 'train' else list(set(np.arange(len(poses))) - set(i_test))
        self.n_images = len(self.img_list)

        self.poses = poses[self.img_list]
        self.imgs = imgs[self.img_list]
        self.intrinsic = self.intrinsic[self.img_list]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return dict(
            scene_id=DC(index, cpu_only=True),
            scene_name=DC(self.base_dir, cpu_only=True),
            cond_imgs=numpy.array(self.imgs, numpy.float32),
            cond_poses=numpy.array(self.poses, numpy.float32),
            cond_intrinsics=numpy.array(self.intrinsic, numpy.float32)
        )
