import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm, remove_weight_norm

from ..utils import init_weights


LRELU_SLOPE = 0.1


class MRFLayer(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=(kernel_size * dilation - dilation) // 2,
                dilation=dilation,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                channels, channels, kernel_size, padding=kernel_size // 2, dilation=1
            )
        )

    def forward(self, x):
        y = F.leaky_relu(x, LRELU_SLOPE)
        y = self.conv1(y)
        y = F.leaky_relu(y, LRELU_SLOPE)
        y = self.conv2(y)
        return x + y

    def remove_weight_norm(self):
        remove_weight_norm(self.conv1)
        remove_weight_norm(self.conv2)


class MRFBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.layers = nn.ModuleList()
        for dilation in dilations:
            self.layers.append(MRFLayer(channels, kernel_size, dilation))
        self.layers.apply(init_weights)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def remove_weight_norm(self):
        for layer in self.layers:
            layer.remove_weight_norm()


class Vocoder(nn.Module):
    def __init__(
        self,
        in_channel,
        upsample_initial_channel,
        upsample_rates,
        upsample_kernel_sizes,
        resblock_kernel_sizes,
        resblock_dilations,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.upsample_rates = upsample_rates
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = nn.Conv1d(
            in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3
        )
        self.upsamples = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=k,
                        stride=u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.mrfs = nn.ModuleList()
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))
            self.mrfs.append(
                nn.ModuleList(
                    [
                        MRFBlock(channel, kernel_size=k, dilations=d)
                        for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                    ]
                )
            )

        self.conv_post = nn.Conv1d(
            upsample_initial_channel // (2 ** len(upsample_rates)),
            1,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.upsamples.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for up, mrf in zip(self.upsamples, self.mrfs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            xs = 0
            for layer in mrf:
                xs += layer(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.upsamples:
            remove_weight_norm(up)
        for mrf in self.resblocks:
            mrf.remove_weight_norm()
        remove_weight_norm(self.conv_post)
