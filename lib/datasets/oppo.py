import os
import io
import cv2
import json
import numpy as np
import random
import torch
from torchvision import transforms as T
from PIL import Image
# import trimesh
import matplotlib.pyplot as plotlib
from typing import Optional, List
from torch.utils.data import Dataset
from mmgen.datasets.builder import DATASETS
from mmcv.parallel import DataContainer as DC
from multiprocessing.pool import ThreadPool


from .parallel_zip import ParallelZipFile as ZipFile

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

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



@DATASETS.register_module()
class OppoDataset(Dataset):

    def __init__(
        self, root_dir: str, split: str, world_scale: float = 1.0, rgba: bool = False
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.world_scale = world_scale
        self.rgba = rgba
        self.split = split

        self.downsample = 4.0
        self.img_wh = (int(2656 / self.downsample), int(3984 / self.downsample))
        self.define_transforms()

        # self.scene_bbox = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        # self.near_far = [0.5, 1.5]

        camera_file = os.path.join(self.root_dir, f"../../transforms_alignz_{split}.json")
        with open(camera_file, 'r') as f:
            self.meta = json.load(f)['frames']

        self.poses = []
        self.imgs = []
        self.intrinsic = []
        w, h = self.img_wh

        for k, v in self.meta.items():
            imgid = v['file_path'].split('/')[-1]

            focal = 0.5 * v['calib_imgw'] / np.tan(0.5 * v['camera_angle_x'])  # original focal length
            if self.downsample != 1.0:
                focal = focal / self.downsample

            image_path = os.path.join(self.root_dir, f"../Lights/013/raw_undistorted/{imgid}.JPG")
            c2w = np.array(v['transform_matrix'])
            c2w = torch.FloatTensor(c2w)
            self.poses.append(c2w)

            self.intrinsic.append(torch.tensor([focal, focal, w / 2, h / 2]))  # focal, focal, cx, cy

            img = Image.open(image_path)

            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            if self.split == 'train':
                mask_path = os.path.join(self.root_dir, f"com_masks/{imgid}.png")
            else:
                # mask_path = os.path.join(self.root_dir, f"obj_masks/{imgid}.png")
                mask_path = os.path.join(self.root_dir, f"com_masks/{imgid}.png")
            mask = cv2.imread(mask_path, 2) > 0
            if self.downsample != 1.0:
                mask = cv2.resize(mask.astype(np.uint8), self.img_wh) > 0
            mask = torch.from_numpy(mask).bool()
            img = img.permute(1,2,0)
            img = img * mask[...,None].float() + (1 - mask[...,None].float())  # blend A to RGB
            if rgba:
                img = torch.cat([img, mask[..., None]], dim=-1)
            self.imgs += [img]

        self.poses = torch.stack(self.poses, dim=0) * self.world_scale
        # self.poses = transform_poses_pca(np.array(self.poses))
        self.imgs = torch.stack(self.imgs, dim=0)
        self.intrinsic = torch.stack(self.intrinsic, dim=0)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return dict(
            scene_id=DC(index, cpu_only=True),
            scene_name=DC(self.root_dir, cpu_only=True),
            cond_imgs=np.array(self.imgs, np.float32),
            cond_poses=np.array(self.poses, np.float32),
            cond_intrinsics=np.array(self.intrinsic, np.float32)
        )
