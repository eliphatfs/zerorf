import torch
import torch.nn as nn

from mmgen.models.builder import build_module
from mmgen.models.architectures.common import get_module_device
from mmcv.cnn import xavier_init, constant_init

from ...core import custom_meshgrid
from lib.ops import SHEncoder, TruncExp
from lib.ops import (
    batch_near_far_from_aabb,
    march_rays_train, batch_composite_rays_train, march_rays, composite_rays,
    morton3D, morton3D_invert, packbits)
from ..decoders.samplers import sample_ray_unbounded


class VolumeRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 min_near=0.2,
                 bg_radius=-1,
                 max_steps=256,
                 decoder_reg_loss=None,
                 preprocessor=None,
                 code_permute=None,
                 code_reshape=None,
                 preproc_size=None,
                 unbounded=False,
                 occlusion_culling_th=0.0001,
                 has_time_dynamics=False):
        super().__init__()

        self.bound = bound
        self.min_near = min_near
        self.bg_radius = bg_radius  # radius of the background sphere.
        self.max_steps = max_steps
        self.decoder_reg_loss = build_module(decoder_reg_loss) if decoder_reg_loss is not None else None

        self.preprocessor = None if preprocessor is None else build_module(preprocessor)
        self.preproc_size = preproc_size
        self.code_permute = code_permute
        self.code_reshape = code_reshape
        self.code_reshape_inv = [self.preproc_size[axis] for axis in self.code_permute] if code_permute is not None \
            else self.preproc_size
        self.code_permute_inv = [self.code_permute.index(axis) for axis in range(len(self.code_permute))] \
            if code_permute is not None else None
        self.code_buffer = None
        self.code_proc_buffer = None

        self.pre_gamma = None
        self.post_gamma = None
        self.unbounded = unbounded
        self.occlusion_culling_th = occlusion_culling_th
        self.has_time_dynamics = has_time_dynamics

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        aabb = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        self.register_buffer('aabb', aabb)

    def point_decode(self, xyzs, dirs, code):
        raise NotImplementedError

    def point_density_decode(self, xyzs, code):
        raise NotImplementedError

    def loss(self, results):
        assert self.decoder_reg_loss is None
        return None

    def code_proc_pr(self, code):
        code_proc = code
        if self.code_permute is not None:
            code_proc = code_proc.permute([0] + [axis + 1 for axis in self.code_permute])  # add batch dimension
        if self.code_reshape is not None:
            code_proc = code_proc.reshape(code.size(0), *self.code_reshape)  # add batch dimension
        return code_proc

    def code_proc_pr_inv(self, code_proc):
        code = code_proc
        if self.code_reshape is not None:
            code = code.reshape(code.size(0), *self.code_reshape_inv)
        if self.code_permute_inv is not None:
            code = code.permute([0] + [axis + 1 for axis in self.code_permute_inv])
        return code

    def preproc(self, code):
        # Avoid preprocessing the same code more than once
        # This implementation is not very robust though (changes to the model
        # parameters or inplace changes to the code will not update the buffer)
        if self.preprocessor is None:
            return code
        else:
            dtype = code.dtype
            preproc_dtype = next(self.preprocessor.parameters()).dtype
            if self.code_buffer is not code:
                if code.requires_grad:
                    self.code_buffer = code
                    self.code_proc_buffer = self.code_proc_pr_inv(
                        self.preprocessor(self.code_proc_pr(code.to(preproc_dtype)))).to(dtype)
                else:
                    if isinstance(self.code_buffer, torch.Tensor) and torch.all(self.code_buffer.data == code):
                        return self.code_proc_buffer.data.to(dtype)
                    else:
                        self.code_buffer = code
                        self.code_proc_buffer = self.code_proc_pr_inv(
                            self.preprocessor(self.code_proc_pr(code.to(preproc_dtype)))).to(dtype)
            return self.code_proc_buffer

    def update_extra_state(self, code, density_grid, density_bitfield,
                           iter_density, density_thresh=0.01, decay=0.9, S=128):
        code = self.preproc(code)
        if self.has_time_dynamics:
            decay = decay ** 0.5

        with torch.no_grad():
            device = get_module_device(self)
            num_scenes = density_grid.size(0)
            tmp_grid = torch.full_like(density_grid, -1)
            grid_size = int(round(density_grid.size(-1) ** (1. / 3.)))

            # full update.
            if iter_density < 16:
                X = torch.arange(grid_size, dtype=torch.int32, device=device).split(S)
                Y = torch.arange(grid_size, dtype=torch.int32, device=device).split(S)
                Z = torch.arange(grid_size, dtype=torch.int32, device=device).split(S)

                for xs in X:
                    for ys in Y:
                        for zs in Z:
                            # construct points
                            xx, yy, zz = custom_meshgrid(xs, ys, zs)
                            coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                               dim=-1)  # [N, 3], in [0, 128)
                            indices = morton3D(coords).long()  # [N]
                            xyzs = (coords.float() - (grid_size - 1) / 2) * (2 * self.bound / grid_size)
                            # add noise
                            half_voxel_width = self.bound / grid_size
                            xyzs += torch.rand_like(xyzs) * (2 * half_voxel_width) - half_voxel_width
                            if self.has_time_dynamics:
                                xyzs = torch.cat([xyzs, torch.rand_like(xyzs[..., :1])], dim=-1)
                            # query density
                            sigmas = self.point_density_decode(
                                xyzs[None].expand(num_scenes, -1, 4 if self.has_time_dynamics else 3), code)[0].reshape(num_scenes, -1)  # (num_scenes, N)
                            # assign
                            tmp_grid[:, indices] = sigmas.clamp(
                                max=torch.finfo(tmp_grid.dtype).max).to(tmp_grid.dtype)

            # partial update (half the computation)
            else:
                N = grid_size ** 3 // 4  # H * H * H / 4
                # random sample some positions
                coords = torch.randint(0, grid_size, (N, 3), device=device)  # [N, 3], in [0, 128)
                indices = morton3D(coords).long()  # [N]
                # random sample occupied positions
                occ_indices_all = []
                for scene_id in range(num_scenes):
                    occ_indices = torch.nonzero(density_grid[scene_id] > 0).squeeze(-1)  # [Nz]
                    rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long,
                                              device=device)
                    occ_indices_all.append(occ_indices[rand_mask])  # [Nz] --> [N], allow for duplication
                occ_indices_all = torch.stack(occ_indices_all, dim=0)
                occ_coords_all = morton3D_invert(occ_indices_all.flatten()).reshape(num_scenes, N, 3)
                indices = torch.cat([indices[None].expand(num_scenes, N), occ_indices_all], dim=0)
                coords = torch.cat([coords[None].expand(num_scenes, N, 3), occ_coords_all], dim=0)
                # same below
                xyzs = (coords.float() - (grid_size - 1) / 2) * (2 * self.bound / grid_size)
                half_voxel_width = self.bound / grid_size
                xyzs += torch.rand_like(xyzs) * (2 * half_voxel_width) - half_voxel_width
                sigmas = self.point_density_decode(xyzs, code)[0].reshape(num_scenes, -1)  # (num_scenes, N + N)
                # assign
                tmp_grid[torch.arange(num_scenes, device=device)[:, None], indices] = sigmas.clamp(
                    max=torch.finfo(tmp_grid.dtype).max).to(tmp_grid.dtype)

            # ema update
            valid_mask = (density_grid >= 0) & (tmp_grid >= 0)
            density_grid[:] = torch.where(valid_mask, torch.maximum(density_grid * decay, tmp_grid), density_grid)
            # density_grid[valid_mask] = torch.maximum(density_grid[valid_mask] * decay, tmp_grid[valid_mask])
            mean_density = torch.mean(density_grid.clamp(min=0))  # -1 regions are viewed as 0 density.
            iter_density += 1

            # convert to bitfield
            density_thresh = min(mean_density, density_thresh)
            packbits(density_grid, density_thresh, density_bitfield)

        return
    
    def rays_to_points(self, M, rays: torch.Tensor, rays_attr: torch.Tensor):
        assert rays.ndim == rays_attr.ndim == 2
        assert len(rays) == len(rays_attr)
        indices = torch.full([M + 1], -1, device=rays.device, dtype=torch.int64)
        indices[rays[:, 0].long()] = torch.arange(len(rays), device=rays.device, dtype=torch.int64)
        indices = indices[:-1]
        index_mask = indices != -1
        indices_densified = indices[index_mask]
        indices = index_mask.cumsum(0) - 1
        return rays_attr[indices_densified[indices]]

    def forward(self, rays_o, rays_d, code, density_bitfield, grid_size,
                dt_gamma=0, perturb=False, return_loss=False,
                update_extra_state=0, extra_args=None, extra_kwargs=None,
                sample_inds=None, rays_time=None):
        """
        Args:
            rays_o: Shape (num_scenes, num_rays_per_scene, 3)
            rays_d: Shape (num_scenes, num_rays_per_scene, 3)
            code: Shape (num_scenes, *code_size)
            density_bitfield: Shape (num_scenes, griz_size**3 // 8)
        """
        code = code.to(next(self.parameters()).dtype)

        for _ in range(update_extra_state):
            self.update_extra_state(code, *extra_args, **extra_kwargs)

        code = self.preproc(code)

        num_scenes = len(rays_o)
        assert num_scenes > 0
        if isinstance(grid_size, int):
            grid_size = [grid_size] * num_scenes
        if isinstance(dt_gamma, float):
            dt_gamma = [dt_gamma] * num_scenes

        nears, fars = batch_near_far_from_aabb(rays_o, rays_d, self.aabb.to(rays_o), self.min_near)
        dtype = torch.float32

        if self.unbounded:
            assert self.occlusion_culling_th > 0
            assert rays_time is None, "dynamic unbounded sampling is not supported"
            xyzs, dirs, ts, rays = sample_ray_unbounded(rays_o, rays_d, 2.0, self.training, self.max_steps)
            xyzs = xyzs / 2.0
            xyzs, dirs, ts, rays = [xyzs], [dirs], [ts.contiguous()], [rays.contiguous()]
            with torch.no_grad():
                sigmas, rgbs, num_points = self.point_decode(xyzs, None, code)
            rgbs = sigmas.new_zeros(sigmas.shape[0], 3)
            with torch.no_grad():
                if sigmas.dtype != dtype or rgbs.dtype != dtype:
                    sigmas, rgbs = sigmas.to(dtype), rgbs.to(dtype)
                weights, weights_sum, depth, image = batch_composite_rays_train(sigmas, rgbs, ts, rays, num_points)
            occl = weights > self.occlusion_culling_th
            ptr = 0
            xyzs_filt = []
            dirs_filt = []
            occls_filt = []
            for xyzs_single, dirs_single in zip(xyzs, dirs):
                blk = occl[ptr: ptr + len(xyzs_single)].nonzero().squeeze(-1)
                ptr += len(xyzs_single)
                xyzs_filt.append(xyzs_single[blk])
                dirs_filt.append(dirs_single[blk])
                occls_filt.append(blk)
            sigmas_occl, rgbs_occl, _ = self.point_decode(xyzs_filt, dirs_filt, code)
            sigmas = torch.zeros_like(sigmas)
            occl = torch.cat(occls_filt)
            sigmas[occl] = sigmas_occl
            rgbs[occl] = rgbs_occl
            if sigmas.dtype != dtype or rgbs.dtype != dtype:
                sigmas, rgbs = sigmas.to(dtype), rgbs.to(dtype)
            weights, weights_sum, depth, image = batch_composite_rays_train(sigmas, rgbs, ts, rays, num_points)
        elif self.training:
            xyzs = []
            dirs = []
            ts = []
            rays = []
            if rays_time is None:
                rays_time = [None] * num_scenes
            for (rays_o_single, rays_d_single, density_bitfield_single,
                 nears_single, fars_single, grid_size_single, dt_gamma_single, rays_time_single) in zip(
                    rays_o, rays_d, density_bitfield, nears, fars, grid_size, dt_gamma, rays_time):
                xyzs_single, dirs_single, ts_single, rays_single = march_rays_train(
                    rays_o_single, rays_d_single, self.bound, density_bitfield_single,
                    1, grid_size_single, nears_single, fars_single,
                    perturb=perturb, dt_gamma=dt_gamma_single.item(), max_steps=self.max_steps)
                if rays_time_single is not None:
                    times_single = self.rays_to_points(len(xyzs_single), rays_single, rays_time_single)
                    xyzs_single = torch.cat([xyzs_single, times_single], dim=-1)
                xyzs.append(xyzs_single)
                dirs.append(dirs_single)
                ts.append(ts_single)
                rays.append(rays_single)
            if self.occlusion_culling_th > 0:
                with torch.no_grad():
                    sigmas, rgbs, num_points = self.point_decode(xyzs, None, code)
                rgbs = sigmas.new_zeros(sigmas.shape[0], 3)
                with torch.no_grad():
                    if sigmas.dtype != dtype or rgbs.dtype != dtype:
                        sigmas, rgbs = sigmas.to(dtype), rgbs.to(dtype)
                    weights, weights_sum, depth, image = batch_composite_rays_train(sigmas, rgbs, ts, rays, num_points)
                occl = weights > self.occlusion_culling_th
                ptr = 0
                xyzs_filt = []
                dirs_filt = []
                occls_filt = []
                for xyzs_single, dirs_single in zip(xyzs, dirs):
                    blk = occl[ptr: ptr + len(xyzs_single)].nonzero().squeeze(-1)
                    ptr += len(xyzs_single)
                    xyzs_filt.append(xyzs_single[blk])
                    dirs_filt.append(dirs_single[blk])
                    occls_filt.append(blk)
                sigmas_occl, rgbs_occl, _ = self.point_decode(xyzs_filt, dirs_filt, code)
                sigmas = torch.zeros_like(sigmas)
                occl = torch.cat(occls_filt)
                sigmas[occl] = sigmas_occl
                rgbs[occl] = rgbs_occl
                if sigmas.dtype != dtype or rgbs.dtype != dtype:
                    sigmas, rgbs = sigmas.to(dtype), rgbs.to(dtype)
                weights, weights_sum, depth, image = batch_composite_rays_train(sigmas, rgbs, ts, rays, num_points)
            else:
                sigmas, rgbs, num_points = self.point_decode(xyzs, dirs, code)
                weights, weights_sum, depth, image = batch_composite_rays_train(sigmas.to(dtype), rgbs.to(dtype), ts, rays, num_points)

        else:
            device = rays_o.device
            weights = None
            weights_sum = []
            depth = []
            image = []

            if rays_time is None:
                rays_time = [None] * num_scenes
            for (rays_o_single, rays_d_single,
                 code_single, density_bitfield_single,
                 nears_single, fars_single,
                 grid_size_single, dt_gamma_single, rays_time_single) in zip(
                    rays_o, rays_d, code, density_bitfield, nears, fars, grid_size, dt_gamma, rays_time):
                num_rays_per_scene = rays_o_single.size(0)

                weights_sum_single = torch.zeros(num_rays_per_scene, dtype=dtype, device=device)
                depth_single = torch.zeros(num_rays_per_scene, dtype=dtype, device=device)
                image_single = torch.zeros(num_rays_per_scene, 3, dtype=dtype, device=device)

                num_rays_alive = num_rays_per_scene
                rays_alive = torch.arange(num_rays_alive, dtype=torch.int32, device=device)  # (num_rays_alive,)
                rays_t = nears_single.clone()  # (num_rays_alive,)

                step = 0
                while step < self.max_steps:
                    # count alive rays
                    num_rays_alive = rays_alive.size(0)
                    # exit loop
                    if num_rays_alive == 0:
                        break
                    # decide compact_steps
                    n_step = min(max(num_rays_per_scene // num_rays_alive, 1), 8)
                    xyzs, dirs, ts = march_rays(
                        num_rays_alive, n_step, rays_alive, rays_t, rays_o_single, rays_d_single,
                        self.bound, density_bitfield_single, 1, grid_size_single, nears_single, fars_single,
                        perturb=perturb, dt_gamma=dt_gamma_single.item(), max_steps=self.max_steps)
                    if rays_time_single is not None:
                        times = rays_time_single.repeat(1, n_step).reshape(-1, 1)
                        xyzs = torch.cat([xyzs, times], dim=-1)
                    sigmas, rgbs, _ = self.point_decode([xyzs], [dirs], code_single[None])
                    if self.pre_gamma is not None:
                        rgbs = rgbs ** self.pre_gamma
                    composite_rays(
                        num_rays_alive, n_step, rays_alive, rays_t, sigmas.to(dtype), rgbs.to(dtype), ts,
                        weights_sum_single, depth_single, image_single)
                    if rays_time_single is not None:
                        rays_time_single = rays_time_single[rays_alive >= 0]
                    rays_alive = rays_alive[rays_alive >= 0]
                    step += n_step
                if self.post_gamma is not None:
                    image_single = image_single ** (self.post_gamma / self.pre_gamma)
                weights_sum.append(weights_sum_single)
                depth.append(depth_single)
                image.append(image_single)

        results = dict(
            weights=weights,
            weights_sum=weights_sum,
            depth=depth,
            image=image,
            sample_inds=sample_inds
        )

        if return_loss:
            results.update(sigmas=sigmas, rgbs=rgbs)
            results.update(decoder_reg_loss=self.loss(results))

        return results


class PointBasedVolumeRenderer(VolumeRenderer):

    def get_point_code(self, code, xyzs):
        raise NotImplementedError

    def point_code_render(self, point_code, dirs):
        raise NotImplementedError

    def point_decode(self, xyzs, dirs, code, density_only=False):
        """
        Args:
            xyzs: Shape (num_scenes, (num_points_per_scene, 3))
            dirs: Shape (num_scenes, (num_points_per_scene, 3))
            code: Shape (num_scenes, vol_flat + tri_flat)
        """
        num_scenes = code.size(0)
        if isinstance(xyzs, torch.Tensor):
            assert xyzs.dim() == 3
            num_points = xyzs.size(-2)
            point_code = self.get_point_code(code, xyzs)
            num_points = [num_points] * num_scenes
        else:
            num_points = []
            point_code = []
            for i, xyzs_single in enumerate(xyzs):
                num_points_per_scene = xyzs_single.size(-2)
                point_code_single = self.get_point_code(code[i: i + 1], xyzs_single[None])
                num_points.append(num_points_per_scene)
                point_code.append(point_code_single)
            point_code = torch.cat(point_code, dim=0) if len(point_code) > 1 else point_code[0]
        sigmas, rgbs = self.point_code_render(point_code, dirs)
        return sigmas, rgbs, num_points

    def point_density_decode(self, xyzs, code, **kwargs):
        sigmas, _, num_points = self.point_decode(
            xyzs, None, code, density_only=True, **kwargs)
        return sigmas, num_points

    def visualize(self, *args, **kwargs):
        pass


class PointBasedDecoder(PointBasedVolumeRenderer):

    def __init__(self, *args, point_channels, skip_base=False, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dir_encoder = SHEncoder()
        self.base_net = nn.Identity() if skip_base else nn.Linear(point_channels, 64)
        self.base_activation = nn.Identity() if skip_base else nn.SiLU()
        self.density_net = nn.Sequential(
            nn.Linear(point_channels if skip_base else 64, 1),
            TruncExp()
        )
        self.dir_net = nn.Linear(16, point_channels if skip_base else 64)
        self.color_net = nn.Sequential(
            nn.Linear(point_channels if skip_base else 64, 3),
            nn.Sigmoid()
        )
        self.sigmoid_saturation = 0.001
        self.interp_mode = 'bilinear'
    
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
        if self.dir_net is not None:
            constant_init(self.dir_net, 0)

    def point_code_render(self, point_code, dirs):
        base_x = self.base_net(point_code)
        base_x_act = self.base_activation(base_x)
        sigmas = self.density_net(base_x_act).squeeze(-1)
        if dirs is None:
            rgbs = None
        else:
            dirs = torch.cat(dirs, dim=0) if len(dirs) > 1 else dirs[0]
            sh_enc = self.dir_encoder(dirs).to(base_x.dtype)
            color_in = self.base_activation(base_x + self.dir_net(sh_enc))
            rgbs = self.color_net(color_in)
            if self.sigmoid_saturation > 0:
                rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
        return sigmas, rgbs
