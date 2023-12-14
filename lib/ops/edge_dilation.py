import torch
import torch.nn.functional as F


def edge_dilation(img, mask, radius=10):
    """
    Args:
        img (torch.Tensor): (n, c, h, w)
        mask (torch.Tensor): (n, 1, h, w)
        radius (float): Radius of dilation.

    Returns:
        torch.Tensor: Dilated image.
    """
    n, c, h, w = img.size()
    int_radius = round(radius)
    kernel_size = int(int_radius * 2 + 1)
    distance1d_sq = torch.linspace(-int_radius, int_radius, kernel_size, dtype=img.dtype, device=img.device).square()
    kernel_distance = (distance1d_sq.reshape(1, -1) + distance1d_sq.reshape(-1, 1)).sqrt()
    kernel_neg_distance = radius - kernel_distance

    # unfold the image and mask
    mask_unfold = F.unfold(mask, kernel_size, padding=int_radius).reshape(
        n, 1, kernel_size * kernel_size, h, w)

    nearest_val, nearest_ind = (
        mask_unfold * kernel_neg_distance.reshape(1, 1, kernel_size * kernel_size, 1, 1)).max(dim=2)
    # (num_fill, 3) in [ind_n, ind_h, ind_w]
    do_fill = ((nearest_val > 0) & (nearest_ind != int_radius * kernel_size + int_radius)).squeeze(1).nonzero()
    fill_ind = nearest_ind[do_fill[:, 0], 0, do_fill[:, 1], do_fill[:, 2]]

    img_out = img.clone()
    img_out[do_fill[:, 0], :, do_fill[:, 1], do_fill[:, 2]] = img[
        do_fill[:, 0],
        :,
        do_fill[:, 1] + fill_ind // kernel_size - int_radius,
        do_fill[:, 2] + fill_ind % kernel_size - int_radius]

    return img_out
