import torch.nn as nn


class Decoder3D(nn.Module):
    def __init__(
        self,
        in_channels=8,
        out_channels=16,
        upsample_resolutions=(32, 64, 64, 128, 128, 256, 256),
        block_channels=(128, 128, 128, 128, 64, 64, 32, 32)
    ):
        super(Decoder3D, self).__init__()

        self.conv_in = nn.Conv3d(
            in_channels=in_channels,
            out_channels=block_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.conv_layers = nn.ModuleList()
        for i in range(len(upsample_resolutions)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=block_channels[i],
                        out_channels=block_channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(16, block_channels[i + 1]),
                    nn.SiLU(),
                )
            )

        self.upsample_layers = nn.ModuleList()
        for i in range(len(upsample_resolutions)):
            self.upsample_layers.append(
                nn.Upsample(
                    size=upsample_resolutions[i],
                    mode="nearest",
                )
            )

        self.conv_out = nn.Conv3d(
            in_channels=block_channels[-1],
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.act_fn = nn.SiLU()

    def forward(self, x):
        x = self.conv_in(x)

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.upsample_layers[i](x)

        x = self.conv_out(x)
        x = self.act_fn(x)

        return x
