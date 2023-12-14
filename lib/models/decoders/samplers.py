import torch
import torch.nn as nn

def intersect_sphere(ray_o, ray_d):
    # ray_o, ray_d: [..., 3]
    # compute the depth of the intersection point between this ray and unit sphere
    # note: d1 becomes negative if this midpoint is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    p_norm_sq = torch.clamp(p_norm_sq, max=1.0)

    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

    return d1 + d2

def s_to_t(t_min, t_max, s):
    t_min_inv = 1 / t_min
    t_max_inv = 1 / t_max
    return 1 / (s * t_max_inv + (1 - s) * t_min_inv)


def t_to_s(t_min, t_max, t):
    t_min_inv = 1 / t_min
    t_max_inv = 1 / t_max
    return (1 / t - t_min_inv) / (t_max_inv - t_min_inv)

def sample_ray_unbounded(rays_o, rays_d, bound=2.0, is_train=True, N_samples=-1):
    rays_o = rays_o.view(-1, 3)
    rays_d = rays_d.view(-1, 3)
    N_samples_inner = N_samples // 2
    N_samples_outer = N_samples - N_samples_inner

    # t_min = self.min_depth
    t_min = 0.1
    t_max = 1e4

    use_bg = True
    use_inf_norm = False  # From DVGO
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # 1. Find the intersection of the ray with the inner sphere (from Nerf++)
    t_s = intersect_sphere(rays_o, rays_d)
    t_s = t_s.max()

    # 2. Sample N/2 points within the inner sphere, from t_min to t_s.
    t_inner = torch.arange(N_samples_inner)[None].float().to(rays_o.device)
    t_inner = t_inner.repeat(rays_d.shape[-2], 1)
    if is_train:
        t_inner += torch.rand_like(t_inner[:, [0]])

    t_inner = t_inner / N_samples_inner
    t_inner = (t_s - t_min) * t_inner + t_min

    # 3. Convert t_s to s_s (from Mip-NeRF 360)
    s_s = t_to_s(t_min, t_max, t_s)

    # 4. Sample N/2 points outside the inner sphere, from s_s to 1.0.
    s_outer = torch.arange(N_samples_outer)[None].float().to(rays_o.device)
    s_outer = s_outer.repeat(rays_d.shape[-2], 1)
    if is_train:
        s_outer += torch.rand_like(s_outer[:, [0]])

    s_outer = s_outer / N_samples_outer
    s_outer = (1.0 - s_s) * s_outer + s_s

    # 5. Convert from s-space to t-space
    t_outer = s_to_t(t_min, t_max, s_outer)

    # 6. Compute sampled points in world coordinates
    interpx = torch.cat([t_inner, t_outer], dim=-1)
    delta_inner = t_inner[:,1] - t_inner[:,0]
    delta_inner = delta_inner.view(-1,1).repeat(1, t_inner.shape[1])
    t_outer_prev = t_outer[:,:-1]
    t_outer_curr = t_outer[:,1:]
    delta_outer = t_outer_curr - t_outer_prev
    delta_outer = torch.cat([delta_outer, delta_outer[:,-1:]], dim=-1)

    world_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]

    # 7. Perform contraction
    radius = bound

    if use_inf_norm:
        pts_norm = world_pts.abs().amax(dim=-1)
    else:
        pts_norm = torch.norm(world_pts, dim=-1)

    # mipnerf contraction
    # scale = (radius - 1.0 / pts_norm[..., None]) / pts_norm[..., None]
    # anpei contraction
    bg_len = radius - 1
    scale = 1 / pts_norm[..., None] * ((1 + bg_len) - bg_len / pts_norm[..., None])

    mask_inside_inner_sphere = (pts_norm <= 1.0)[..., None]
    contracted_pts = torch.where(mask_inside_inner_sphere, world_pts, scale * world_pts)

    # mask_outbbox = torch.norm(contracted_pts, dim=-1) > radius

    rays = torch.cat((torch.arange(rays_d.shape[0], device=contracted_pts.device)[:, None], torch.ones(rays_d.shape[0], device=contracted_pts.device)[:, None]), dim=1)*N_samples
    rays_d = rays_d[:, None, :].repeat(1, contracted_pts.shape[1], 1)

    inner = torch.cat([t_inner[..., None], delta_inner[..., None]], dim=2)
    outer = torch.cat([t_outer[..., None], delta_outer[..., None]], dim=2)
    ts = torch.cat([inner, outer], dim=1)

    return contracted_pts.view(-1, 3), rays_d.view(-1, 3), ts.view(-1, 2), rays.int()# ~mask_outbbox