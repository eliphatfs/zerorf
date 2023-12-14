import torch
import torch.nn as nn
import einops
import transformers
from .arch_1d import Decoder1D
from .arch_3d import Decoder3D
from mmgen.models import build_module, MODULES


class CubemapGenerator(nn.Module):
    noise: torch.Tensor

    def __init__(self, noise_ch, out_ch, noise_res) -> None:
        super().__init__()
        self.noise_ch, self.out_ch, self.noise_res = noise_ch, out_ch, noise_res
        self.upx = 16
        self.register_buffer("noise", torch.randn(1, noise_ch, noise_res, noise_res))
        self.net = build_module(dict(
            type='VAEDecoder',
            in_channels=noise_ch,
            out_channels=out_ch,
            up_block_types=('UpDecoderBlock2D',) * 5,
            block_out_channels=(32, 64, 64, 128, 256),
            layers_per_block=1
        ))
        self.head = nn.Sequential(
            nn.Linear(out_ch, 32),
            nn.SiLU(True),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )
        self.sigmoid_saturation = 0.001

    def get_output_shape(self):
        return [self.out_ch, self.noise_res * self.upx, self.noise_res * self.upx]

    def forward(self, ray_dirs):
        # (1, N, 3) -> (1, N, 3)
        x = torch.atan2(ray_dirs[..., [1]], ray_dirs[..., [0]]) / 3.14159
        y = ray_dirs[..., [2]]
        cubemap = self.net(self.noise)
        samples = nn.functional.grid_sample(
            cubemap, torch.cat([x, y], dim=-1).unsqueeze(-2), mode='bilinear', padding_mode='border', align_corners=False
        ).squeeze(-1).transpose(-1, -2)
        return self.head(samples) * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation


class Tensorial2D(nn.Module):
    noise: torch.Tensor

    def __init__(self, noise_ch, out_ch, noise_res) -> None:
        super().__init__()
        self.noise_ch, self.out_ch, self.noise_res = noise_ch, out_ch, noise_res
        self.upx = 16
        self.register_buffer("noise", torch.randn(1, noise_ch, noise_res, noise_res))
        self.net = build_module(dict(
            type='VAEDecoder',
            in_channels=noise_ch,
            out_channels=out_ch,
            up_block_types=('UpDecoderBlock2D',) * 5,
            block_out_channels=(32, 64, 64, 128, 256),
            layers_per_block=1
        ))

    def get_output_shape(self):
        return [self.out_ch, self.noise.size(-2) * self.upx, self.noise.size(-1) * self.upx]

    def forward(self):
        return self.net(self.noise)


class Tensorial2DMLP(nn.Module):
    noise: torch.Tensor

    def __init__(self, noise_ch, out_ch, noise_res) -> None:
        super().__init__()
        self.noise_ch, self.out_ch, self.noise_res = noise_ch, out_ch, noise_res
        self.upx = 16
        grid_base = torch.linspace(-1, 1, noise_res * self.upx)
        grid = torch.stack(torch.meshgrid(grid_base, grid_base, indexing='xy'), dim=0)[None]
        # 1, 2, H, W
        self.register_buffer("noise", grid)
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 1),
            nn.GroupNorm(16, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 1),
            nn.GroupNorm(16, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 256, 1),
            nn.GroupNorm(16, 256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, out_ch, 1)
        )

    def get_output_shape(self):
        return [self.out_ch, self.noise.size(-2), self.noise.size(-1)]

    def forward(self):
        return self.net(self.noise)


class Tensorial2DViT(nn.Module):
    noise: torch.Tensor

    def __init__(self, noise_ch, out_ch, noise_res) -> None:
        super().__init__()
        self.noise_ch, self.out_ch, self.noise_res = noise_ch, out_ch, noise_res
        self.upx = 16
        self.register_buffer("noise", torch.randn(1, noise_ch, noise_res * self.upx, noise_res * self.upx))
        self.net = transformers.ViTModel(transformers.ViTConfig(
            256, 4, 4, 512, image_size=noise_res * self.upx, num_channels=noise_ch,
            patch_size=self.upx
        ))
        self.decode = nn.Linear(256, out_ch * 16 * 16, bias=False)

    def get_output_shape(self):
        return [self.out_ch, self.noise_res * self.upx, self.noise_res * self.upx]

    def forward(self):
        hidden = self.net(self.noise, return_dict=True).last_hidden_state[:, 1:]
        return einops.rearrange(
            self.decode(hidden),
            'b (w h) (pw ph c) -> b c (w pw) (h ph)',
            pw=self.upx, ph=self.upx, w=self.noise_res, h=self.noise_res, c=self.out_ch
        )


class Tensorial1D(nn.Module):
    noise: torch.Tensor

    def __init__(self, noise_ch, out_ch, noise_res) -> None:
        super().__init__()
        self.noise_ch, self.out_ch, self.noise_res = noise_ch, out_ch, noise_res
        self.upx = 16
        self.register_buffer("noise", torch.randn(1, noise_ch, noise_res))
        self.net = Decoder1D(
            noise_ch, out_ch,
            tuple(noise_res * i for i in [2, 4, 8, 16, 16]),
            (128, 128, 128, 64, 32, 32)
        )

    def get_output_shape(self):
        return [self.out_ch, self.noise_res * self.upx]

    def forward(self):
        return self.net(self.noise)


class Tensorial3D(nn.Module):
    noise: torch.Tensor

    def __init__(self, noise_ch, out_ch, noise_res) -> None:
        super().__init__()
        self.noise_ch, self.out_ch, self.noise_res = noise_ch, out_ch, noise_res
        self.upx = 8
        self.register_buffer("noise", torch.randn(1, noise_ch, noise_res, noise_res, noise_res))
        self.net = Decoder3D(
            noise_ch, out_ch,
            tuple(noise_res * i for i in [1, 1, 2, 4, 8]),
            (128, 128, 128, 64, 32, 32)
        )

    def get_output_shape(self):
        return [self.out_ch, self.noise_res * self.upx, self.noise_res * self.upx, self.noise_res * self.upx]

    def forward(self):
        return self.net(self.noise)


@MODULES.register_module()
class TensorialGenerator(nn.Module):
    def __init__(self, in_ch, out_ch, noise_res, tensor_config: list) -> None:
        super().__init__()
        self.in_ch, self.out_ch, self.noise_res = in_ch, out_ch, noise_res
        self.subs = nn.ModuleList([
            (Tensorial1D if len(sub) == 1 else Tensorial2D if len(sub) == 2 else Tensorial3D)(in_ch, out_ch, noise_res) for sub in tensor_config
        ])
        self.tensor_config = tensor_config
        self.sub_shapes = [sub.get_output_shape() for sub in self.subs]

    def forward(self, _):
        r = []
        for sub in self.subs:
            sub_out = sub()
            expected = list(sub.get_output_shape())
            assert list(sub_out.shape[1:]) == expected, [sub_out.shape[1:], expected]
            r.append(torch.flatten(sub_out, 1))
        return torch.cat(r, 1)
