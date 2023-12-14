import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from mmgen.models.builder import MODULES
from mmgen.models.losses.utils import weighted_loss


@weighted_loss
def tv_loss(tensor, dims, power=1):
    shape = list(tensor.size())
    diffs = []
    for dim in dims:
        pad_shape = deepcopy(shape)
        pad_shape[dim] = 1
        diffs.append(torch.cat([torch.diff(tensor, dim=dim), tensor.new_zeros(pad_shape)], dim=dim))
    return torch.stack(diffs, dim=0).norm(dim=0).pow(power).mean(dim=dims)


@MODULES.register_module()
class TVLoss(nn.Module):

    def __init__(self,
                 dims=[-2, -1],
                 power=1,
                 loss_weight=1.0):
        super().__init__()
        self.dims = dims
        self.power = power
        self.loss_weight = loss_weight

    def forward(self, tensor, weight=None, avg_factor=None):
        return tv_loss(
            tensor, self.dims, power=self.power,
            weight=weight, avg_factor=avg_factor
        ) * self.loss_weight


@MODULES.register_module()
class MaskedTVLoss(nn.Module):

    def __init__(self,
                 dims=[-2, -1],
                 power=1,
                 loss_weight=1.0):
        super().__init__()
        self.dims = dims
        self.power = power
        self.loss_weight = loss_weight

    def forward(self, alpha, tensor, weight=None, avg_factor=None):
        depth_mask = -F.max_pool2d(-alpha, 3, 1, 1)   # erode
        out_depth_fg = tensor / alpha.clamp(min=1e-6)   # foreground depth
        out_depth_fg = out_depth_fg * depth_mask + out_depth_fg.detach() * (1 - depth_mask)   # stop background gradient
        return (tv_loss(
            out_depth_fg[..., :, :-1] - out_depth_fg[..., :, 1:], self.dims, power=self.power,
            weight=weight, avg_factor=avg_factor
        ) + tv_loss(
            out_depth_fg[..., :-1, :] - out_depth_fg[..., 1:, :], self.dims, power=self.power,
            weight=weight, avg_factor=avg_factor
        )) * self.loss_weight
