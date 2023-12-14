import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr

from lib.ops.edge_dilation import edge_dilation


def make_divisible(x, m=8):
    return int(math.ceil(x / m) * m)


def interpolate_hwc(x, scale_factor, mode='area'):
    batch_dim = x.shape[:-3]
    y = x.reshape(batch_dim.numel(), *x.shape[-3:]).permute(0, 3, 1, 2)
    y = F.interpolate(y, scale_factor=scale_factor, mode=mode).permute(0, 2, 3, 1)
    return y.reshape(*batch_dim, *y.shape[1:])


class MeshRenderer(nn.Module):
    def __init__(self,
                 near=0.1,
                 far=10,
                 ssaa=1,
                 texture_filter='linear-mipmap-linear',
                 opengl=False):
        super().__init__()
        self.near = near
        self.far = far
        assert isinstance(ssaa, int) and ssaa >= 1
        self.ssaa = ssaa
        self.texture_filter = texture_filter
        self.glctx = dr.RasterizeGLContext() if opengl else dr.RasterizeCudaContext()

    def forward(self, meshes, poses, intrinsics, h, w, reshade_fun=None, dilate_edges=0):
        """
        Args:
            meshes (list[Mesh]): list of Mesh objects
            poses: Shape (num_scenes, num_images, 3, 4)
            intrinsics: Shape (num_scenes, num_images, 4) in [fx, fy, cx, cy]
        """
        num_scenes, num_images, _, _ = poses.size()

        if self.ssaa > 1:
            h = h * self.ssaa
            w = w * self.ssaa
            intrinsics = intrinsics * self.ssaa

        r_mat_c2w = torch.cat(
            [poses[..., :3, :1], -poses[..., :3, 1:3]], dim=-1)  # opencv to opengl conversion

        proj = poses.new_zeros([num_scenes, num_images, 4, 4])
        proj[..., 0, 0] = 2 * intrinsics[..., 0] / w
        proj[..., 0, 2] = 2 * intrinsics[..., 2] / w - 1
        proj[..., 1, 1] = -2 * intrinsics[..., 1] / h
        proj[..., 1, 2] = -2 * intrinsics[..., 3] / h + 1
        proj[..., 2, 2] = -(self.far + self.near) / (self.far - self.near)
        proj[..., 2, 3] = -(2 * self.far * self.near) / (self.far - self.near)
        proj[..., 3, 2] = -1

        # (num_scenes, (num_images, num_vertices, 3))
        v_cam = [(mesh.v - poses[i, :, :3, 3].unsqueeze(-2)) @ r_mat_c2w[i] for i, mesh in enumerate(meshes)]
        # (num_scenes, (num_images, num_vertices, 4))
        v_clip = [F.pad(v, pad=(0, 1), mode='constant', value=1.0) @ proj[i].transpose(-1, -2) for i, v in enumerate(v_cam)]

        if num_scenes == 1:
            # (num_images, h, w, 4) in [u, v, z/w, triangle_id] & (num_images, h, w, 4 or 0)
            rast, rast_db = dr.rasterize(
                self.glctx, v_clip[0], meshes[0].f, (h, w), grad_db=torch.is_grad_enabled())

            valid = (rast[..., 3] > 0).unsqueeze(0)  # (num_scenes, num_images, h, w)

            depth = 1 / dr.interpolate(
                -v_cam[0][..., 2:3].contiguous(), rast, meshes[0].f)[0].reshape(num_scenes, num_images, h, w)
            depth.masked_fill_(~valid, 0)

            normal = dr.interpolate(
                meshes[0].vn.unsqueeze(0).contiguous(), rast, meshes[0].fn)[0].reshape(num_scenes, num_images, h, w, 3)
            normal = F.normalize(normal, dim=-1)
            # (num_scenes, num_images, h, w, 3) = (num_scenes, num_images, h, w, 3) @ (num_scenes, num_images, 1, 3, 3)
            rot_normal = normal @ r_mat_c2w.unsqueeze(2)
            rot_normal[~valid] = rot_normal.new_tensor([0, 0, 1])

            # (num_images, h, w, 2) & (num_images, h, w, 4)
            texc, texc_db = dr.interpolate(
                meshes[0].vt.unsqueeze(0).contiguous(), rast, meshes[0].ft, rast_db=rast_db, diff_attrs='all')
            # (num_scenes, num_images, h, w, 3)
            albedo = dr.texture(
                meshes[0].albedo.unsqueeze(0)[..., :3].contiguous(), texc, uv_da=texc_db, filter_mode=self.texture_filter).unsqueeze(0)

            prev_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
            if reshade_fun is not None:
                xyz = dr.interpolate(
                    meshes[0].v.unsqueeze(0).contiguous(), rast, meshes[0].f)[0].reshape(num_scenes, num_images, h, w, 3)
                rgb_reshade = reshade_fun(xyz[valid], albedo[valid])
                albedo = torch.zeros_like(albedo)
                albedo[valid] = rgb_reshade

            # (num_scenes, num_images, h, w, 4)
            rgba = torch.cat([albedo, valid.float().unsqueeze(-1)], dim=-1)

            if dilate_edges > 0:
                rgba = rgba.reshape(num_scenes * num_images, h, w, 4).permute(0, 3, 1, 2)
                rgba = edge_dilation(rgba, rgba[:, 3:], dilate_edges)
                rgba = rgba.permute(0, 2, 3, 1).reshape(num_scenes, num_images, h, w, 4)

            rgba = dr.antialias(rgba.squeeze(0), rast, v_clip[0], meshes[0].f).unsqueeze(0)

        else:  # concat and range mode
            # v_cat = []
            v_clip_cat = []
            v_cam_cat = []
            vn_cat = []
            vt_cat = []
            f_cat = []
            fn_cat = []
            ft_cat = []
            v_count = 0
            vn_count = 0
            vt_count = 0
            f_count = 0
            f_ranges = []
            for i, mesh in enumerate(meshes):
                num_v = v_clip[i].size(1)
                num_vn = mesh.vn.size(0)
                num_vt = mesh.vt.size(0)
                # v_cat.append(mesh.v.unsqueeze(0).expand(num_images, -1, -1).reshape(num_images * num_v, 3))
                v_clip_cat.append(v_clip[i].reshape(num_images * num_v, 4))
                v_cam_cat.append(v_cam[i].reshape(num_images * num_v, 3))
                vn_cat.append(mesh.vn.unsqueeze(0).expand(num_images, -1, -1).reshape(num_images * num_vn, 3))
                vt_cat.append(mesh.vt.unsqueeze(0).expand(num_images, -1, -1).reshape(num_images * num_vt, 2))
                for _ in range(num_images):
                    f_cat.append(mesh.f + v_count)
                    fn_cat.append(mesh.fn + vn_count)
                    ft_cat.append(mesh.ft + vt_count)
                    v_count += num_v
                    vn_count += num_vn
                    vt_count += num_vt
                    f_ranges.append([f_count, mesh.f.size(0)])
                    f_count += mesh.f.size(0)
            # v_cat = torch.cat(v_cat, dim=0)
            v_clip_cat = torch.cat(v_clip_cat, dim=0)
            v_cam_cat = torch.cat(v_cam_cat, dim=0)
            vn_cat = torch.cat(vn_cat, dim=0)
            f_cat = torch.cat(f_cat, dim=0)
            f_ranges = torch.tensor(f_ranges, device=poses.device, dtype=torch.int32)
            # (num_scenes * num_images, h, w, 4) in [u, v, z/w, triangle_id] & (num_scenes * num_images, h, w, 4 or 0)
            rast, rast_db = dr.rasterize(
                self.glctx, v_clip_cat, f_cat, (h, w), ranges=f_ranges, grad_db=torch.is_grad_enabled())

            valid = (rast[..., 3] > 0).reshape(num_scenes, num_images, h, w)

            depth = 1 / dr.interpolate(
                -v_cam_cat[..., 2:3].contiguous(), rast, f_cat)[0].reshape(num_scenes, num_images, h, w)
            depth.masked_fill_(~valid, 0)

            normal = dr.interpolate(
                vn_cat, rast, fn_cat)[0].reshape(num_scenes, num_images, h, w, 3)
            normal = F.normalize(normal, dim=-1)
            # (num_scenes, num_images, h, w, 3) = (num_scenes, num_images, h, w, 3) @ (num_scenes, num_images, 1, 3, 3)
            rot_normal = normal @ r_mat_c2w.unsqueeze(2)
            rot_normal[~valid] = rot_normal.new_tensor([0, 0, 1])

            # (num_scenes * num_images, h, w, 2) & (num_scenes * num_images, h, w, 4)
            texc, texc_db = dr.interpolate(
                vt_cat, rast, ft_cat, rast_db=rast_db, diff_attrs='all')
            albedo = dr.texture(
                torch.cat([mesh.albedo.unsqueeze(0)[..., :3].expand(num_images, -1, -1, -1) for mesh in meshes], dim=0),
                texc, uv_da=texc_db, filter_mode=self.texture_filter
            ).reshape(num_scenes, num_images, h, w, 3)

            prev_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
            if reshade_fun is not None:
                raise NotImplementedError

            # (num_scenes, num_images, h, w, 4)
            rgba = torch.cat([albedo, valid.float().unsqueeze(-1)], dim=-1)

            if dilate_edges > 0:
                rgba = rgba.reshape(num_scenes * num_images, h, w, 4).permute(0, 3, 1, 2)
                rgba = edge_dilation(rgba, rgba[:, 3:], dilate_edges)
                rgba = rgba.permute(0, 2, 3, 1).reshape(num_scenes, num_images, h, w, 4)

            rgba = dr.antialias(
                rgba.reshape(num_scenes * num_images, h, w, 4), rast, v_clip_cat, f_cat
            ).reshape(num_scenes, num_images, h, w, 4)

        if self.ssaa > 1:
            rgba = interpolate_hwc(rgba, 1 / self.ssaa)
            depth = interpolate_hwc(depth.unsqueeze(-1), 1 / self.ssaa).squeeze(-1)
            rot_normal = interpolate_hwc(rot_normal, 1 / self.ssaa)

        results = dict(
            rgba=rgba,
            depth=depth,
            normal=rot_normal / 2 + 0.5)

        torch.set_grad_enabled(prev_grad_enabled)

        return results

    def bake_reshade_fun(self, meshes, reshade_fun, dilate=10):
        assert len(meshes) == 1, 'only support one mesh'
        mesh = meshes[0]

        albedo_map = mesh.albedo
        h, w, _ = albedo_map.size()
        vt_clip = torch.cat([mesh.vt * 2 - 1, mesh.vt.new_tensor([[0., 1.]]).expand(mesh.vt.size(0), -1)], dim=-1)

        rast = dr.rasterize(self.glctx, vt_clip[None], mesh.ft, (h, w), grad_db=False)[0]
        valid = (rast[..., 3] > 0).reshape(h, w)

        xyz = dr.interpolate(mesh.v[None], rast, mesh.f)[0].reshape(h, w, 3)
        rgb_reshade = reshade_fun(xyz[valid], albedo_map[..., :3][valid])
        new_albedo_map = torch.zeros_like(albedo_map[..., :3])
        new_albedo_map[valid] = rgb_reshade
        new_albedo_map = edge_dilation(
            new_albedo_map.permute(2, 0, 1)[None], valid[None, None].float(), radius=dilate
        ).squeeze(0).permute(1, 2, 0)
        mesh.albedo = torch.cat([new_albedo_map.clamp(min=0, max=1), albedo_map[..., 3:]], dim=-1)

        return [mesh]
