import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_redstone import supercat

from diffusers.models.vae import Decoder
from mmcv.cnn import xavier_init, constant_init
from mmgen.models.builder import MODULES, build_module
import matplotlib.pyplot as plt

from .base_volume_renderer import VolumeRenderer
from lib.ops import SHEncoder, TruncExp


class ResBlock3D(nn.Module):
    def __init__(self, c_in, c_out, num_blocks, stride) -> None:
        super().__init__()
        self.sub_blocks = nn.ModuleList()
        self.first = nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, stride, 1, bias=False),
            nn.GroupNorm(16, c_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_out, c_out, 3, 1, 1, bias=False),
            nn.GroupNorm(16, c_out)
        )
        if stride != 1 or c_in != c_out:
            self.projection = nn.Sequential(
                nn.Conv3d(c_in, c_out, 1, stride, bias=False),
                nn.GroupNorm(16, c_out)
            )
        else:
            self.projection = nn.Identity()
        for _ in range(num_blocks - 1):
            self.sub_blocks.append(nn.Sequential(
                nn.Conv3d(c_out, c_out, 3, 1, 1, bias=False),
                nn.GroupNorm(16, c_out),
                nn.ReLU(inplace=True),
                nn.Conv3d(c_out, c_out, 3, 1, 1, bias=False),
                nn.GroupNorm(16, c_out)
            ))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.relu(self.projection(x) + self.first(x))
        for sub_block in self.sub_blocks:
            x = torch.relu(x + sub_block(x))
        return x


class UpSamplingDecoder(nn.Module):
    def __init__(self, triplane_shape, upsample_blocks) -> None:
        super().__init__()
        if upsample_blocks == 3:
            block_out_channels = (128, 256, 256)
        elif upsample_blocks == 4:
            block_out_channels = (32, 64, 128, 256)
        self.decoder = Decoder(
            8, triplane_shape[1],
            up_block_types=("UpDecoderBlock2D",) * upsample_blocks,
            block_out_channels=block_out_channels,
            layers_per_block=1
        )

    def forward(self, z):
        return self.decoder(z)


class LittleUNet(nn.Module):
    def __init__(self, vol_shape) -> None:
        super().__init__()
        self.down_block = ResBlock3D(24, 128, 2, 2)
        self.mid_block = ResBlock3D(128, 256, 5, 2)
        self.up = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(256, 128, 3, 1, 1)])
        self.up_block = ResBlock3D(128, 128, 2, 1)
        self.conv_out = nn.Conv3d(128, vol_shape[0], 3, 1, 1)

    def forward(self, x):
        res = self.down_block(x)
        x = self.mid_block(res)
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)
        x = self.up[0](x)
        if dtype == torch.bfloat16:
            x = x.to(torch.bfloat16)
        x = self.up[1](x) + res
        x = self.up_block(x)
        return self.conv_out(x)


def xyz_transform(xyz):
    plane_cfg = ['yx', 'yz', 'xz']
    flip_z = True
    out = []
    for plane in plane_cfg:
        plane_axes = []
        for axis in plane:
            if axis == 'x':
                plane_axes.append(xyz[..., 0])
            elif axis == 'y':
                plane_axes.append(xyz[..., 1])
            elif axis == 'z':
                if flip_z:
                    plane_axes.append(-xyz[..., 2])
                else:
                    plane_axes.append(xyz[..., 2])
            else:
                raise ValueError
        out.append(torch.stack(plane_axes, dim=-1))
    if xyz.dim() == 2:
        out = torch.stack(out, dim=0).unsqueeze(1)  # (3, 1, num_points, 2)
    elif xyz.dim() == 3:
        num_scenes, num_points, _ = xyz.size()
        out = torch.stack(out, dim=1).reshape(num_scenes * 3, 1, num_points, 2)
    else:
        raise ValueError
    return out


@MODULES.register_module()
class HybridCodex(nn.Module):
    def __init__(self, vol_shape, triplane_shape, upsample_blocks=3, noise_siamese=True) -> None:
        super().__init__()
        self.triplane_codex = UpSamplingDecoder(triplane_shape, upsample_blocks)
        self.vol_codex = LittleUNet(vol_shape)
        self.noise_siamese = noise_siamese

    def create_volume(self, triplane):
        flip_z = True
        yx, yz, xz = torch.unbind(triplane, 1)  # bchw
        # bcyx, bczx, bczy -> bc'zyx
        vol = supercat([yx.transpose(-1, -2).unsqueeze(-3), yz.unsqueeze(-1), xz.unsqueeze(-2)], dim=1)
        if flip_z:
            vol = torch.flip(vol, [-3])
        return vol

    def forward(self, latents):
        if self.noise_siamese:
            latents = torch.cat([latents, torch.randn_like(latents)], dim=2)
        b = len(latents)
        volume = self.create_volume(latents)
        volume = self.vol_codex(volume).flatten(1)
        triplanes = self.triplane_codex(latents.flatten(0, 1)).reshape(b, -1)
        return torch.cat([volume, triplanes], dim=-1)


@MODULES.register_module()
class HybridDecoder(VolumeRenderer):
    def __init__(self, *args, vol_shape, triplane_shape, codeviz_decoder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dir_encoder = SHEncoder()
        self.base_net = nn.Linear(vol_shape[0] + triplane_shape[0] * triplane_shape[1], 64)
        self.base_activation = nn.SiLU()
        self.density_net = nn.Sequential(
            nn.Linear(64, 1),
            TruncExp()
        )
        self.dir_net = nn.Linear(16, 64)
        self.color_net = nn.Sequential(
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
        self.sigmoid_saturation = 0.001
        self.interp_mode = 'bilinear'
        self.vol_shape = vol_shape
        self.triplane_shape = triplane_shape

        if codeviz_decoder is not None:
            self.codeviz_decoder = build_module(codeviz_decoder)
        else:
            self.codeviz_decoder = None
    
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
        if self.dir_net is not None:
            constant_init(self.dir_net, 0)

    def get_point_code(self, volume, triplane, xyzs):
        num_scenes, _, n_channels, h, w = triplane.size()
        num_points = xyzs.size(-2)
        dtype = triplane.dtype
        assert dtype == volume.dtype
        point_code = F.grid_sample(
            triplane.reshape(num_scenes * 3, -1, h, w).float(),
            xyz_transform(xyzs),
            mode=self.interp_mode, padding_mode='border', align_corners=False
        ).reshape(num_scenes, 3, n_channels, num_points)
        point_code_vol = F.grid_sample(
            volume.float(),
            xyzs.reshape(num_scenes, 1, 1, num_points, 3),
            mode=self.interp_mode, padding_mode='border', align_corners=False
        ).reshape(num_scenes, 1, volume.size(1), num_points)
        point_code = point_code.to(dtype)
        point_code_vol = point_code_vol.to(dtype)
        point_code = point_code.permute(0, 3, 2, 1).reshape(num_scenes * num_points, n_channels * 3)
        point_code_vol = point_code_vol.permute(0, 3, 2, 1).reshape(num_scenes * num_points, volume.size(1))
        point_code = torch.cat([point_code, point_code_vol], dim=-1)
        return point_code

    def point_decode(self, xyzs, dirs, code, density_only=False):
        """
        Args:
            xyzs: Shape (num_scenes, (num_points_per_scene, 3))
            dirs: Shape (num_scenes, (num_points_per_scene, 3))
            code: Shape (num_scenes, vol_flat + tri_flat)
        """
        vol_numel = numpy.prod(self.vol_shape)
        volume, triplane = code[:, :vol_numel].reshape(-1, *self.vol_shape), code[:, vol_numel:].reshape(-1, *self.triplane_shape)
        num_scenes, _, n_channels, h, w = triplane.size()
        if isinstance(xyzs, torch.Tensor):
            assert xyzs.dim() == 3
            num_points = xyzs.size(-2)
            point_code = self.get_point_code(volume, triplane, xyzs)
            num_points = [num_points] * num_scenes
        else:
            num_points = []
            point_code = []
            for i, xyzs_single in enumerate(xyzs):
                num_points_per_scene = xyzs_single.size(-2)
                point_code_single = self.get_point_code(volume[i: i + 1], triplane[i: i + 1], xyzs_single[None])
                num_points.append(num_points_per_scene)
                point_code.append(point_code_single)
            point_code = torch.cat(point_code, dim=0) if len(point_code) > 1 else point_code[0]
        base_x = self.base_net(point_code)
        base_x_act = self.base_activation(base_x)
        sigmas = self.density_net(base_x_act).squeeze(-1)
        if density_only:
            rgbs = None
        else:
            dirs = torch.cat(dirs, dim=0) if num_scenes > 1 else dirs[0]
            sh_enc = self.dir_encoder(dirs).to(base_x.dtype)
            color_in = self.base_activation(base_x + self.dir_net(sh_enc))
            rgbs = self.color_net(color_in)
            if self.sigmoid_saturation > 0:
                rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
        return sigmas, rgbs, num_points

    def point_density_decode(self, xyzs, code, **kwargs):
        sigmas, _, num_points = self.point_decode(
            xyzs, None, code, density_only=True, **kwargs)
        return sigmas, num_points

    def visualize(self, code, scene_name, viz_dir, code_range=[-1, 1]):
        num_scenes, _, num_chn, h, w = code.size()
        code_viz = code.cpu().float().numpy()
        code_viz = code_viz.transpose(0, 1, 3, 2, 4).reshape(num_scenes, 3 * h, num_chn * w)
        for code_viz_single, scene_name_single in zip(code_viz, scene_name):
            plt.imsave(os.path.join(viz_dir, 'scene_' + scene_name_single + '.png'), code_viz_single,
                       vmin=code_range[0], vmax=code_range[1])
        if self.codeviz_decoder is not None:
            decoded = self.codeviz_decoder(code.transpose(2, 1).reshape(num_scenes, 4, 3 * h, w)) / 2 + 0.5
            decoded = decoded[0].permute(1, 2, 0)
            plt.imsave(os.path.join(viz_dir, 'scene_' + scene_name_single + '_dec.png'), decoded.cpu().clamp(0, 1).float().numpy())
