from typing import Dict, Optional
import torch.nn as nn
import torch.nn.functional as F
from contrib.achievement_distillation.torch_util import FanInInitReLULayer
import torch
import torch.nn.init as init


class ConvolutionalResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvolutionalResBlock, self).__init__()
        mid_channels = out_channels // 2
        # self.conv1 = nn.Conv2d(
        #     in_channels,
        #     mid_channels,
        #     kernel_size=3,
        #     stride=stride,
        #     padding=1,
        #     bias=False,
        # )
        # self.bn1 = nn.GroupNorm(1, mid_channels)
        self.relu = nn.ReLU()

        self.conv1 = FanInInitReLULayer(
            inchan=in_channels,
            outchan=mid_channels,
            layer_type="conv",
            group_norm_groups=1,  # GroupNorm with 1 group for the first layer
            kernel_size=3,
            stride=stride,
            padding=1,
        )

        # self.conv2 = nn.Conv2d(
        #     mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        # )
        # self.bn2 = nn.GroupNorm(1, out_channels)
        self.conv2 = FanInInitReLULayer(
            inchan=mid_channels,
            outchan=out_channels,
            layer_type="conv",
            group_norm_groups=1,  # GroupNorm with 1 group for the second layer
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if stride != 1 or in_channels != out_channels:
            # self.downsample = nn.Sequential(
            #     nn.Conv2d(
            #         in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            #     ),
            #     nn.GroupNorm(1, out_channels),
            # )
            self.downsample = nn.Sequential(
                FanInInitReLULayer(
                    inchan=in_channels,
                    outchan=out_channels,
                    layer_type="conv",
                    group_norm_groups=1,  # GroupNorm with 1 group for the downsample layer
                    kernel_size=1,
                    stride=stride,
                )
            )
        else:
            self.downsample = None

        if stride != 1:
            # self.skip_out = nn.Conv2d(
            #     in_channels, in_channels, kernel_size=1, stride=stride, bias=False
            # )
            self.skip_out = FanInInitReLULayer(
                inchan=in_channels,
                outchan=in_channels,
                layer_type="conv",
                group_norm_groups=1,  # GroupNorm with 1 group for the skip_out layer
                kernel_size=1,
                stride=stride,
            )
        else:
            self.skip_out = None

    def forward(self, x):
        identity = x
        # out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv1(x)
        # out = self.bn2(self.conv2(out))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        # out = self.relu(out)
        if self.skip_out is not None:
            # x = self.skip_out(x)
            skip_out = self.skip_out(x)
        else:
            skip_out = x
        return out, skip_out


class DeconvolutionalResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DeconvolutionalResBlock, self).__init__()
        mid_channels = out_channels // 2
        # Adjust the deconv1 to match the expected input channels and desired output channels
        # self.deconv1 = nn.ConvTranspose2d(
        #     in_channels,  # This should match the actual incoming channels
        #     mid_channels,  # Intermediate channel size, which is half of out_channels
        #     kernel_size=3,
        #     stride=stride,
        #     padding=1,
        #     output_padding=0,
        #     bias=False,
        # )
        # self.bn1 = nn.GroupNorm(1, mid_channels)
        self.deconv1 = FanInInitReLULayer(
            inchan=in_channels,
            outchan=mid_channels,
            layer_type="deconv",
            group_norm_groups=1,  # GroupNorm with 1 group for the first layer
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=0,
        )

        # The second deconvolution layer to expand back to out_channels
        # self.deconv2 = nn.ConvTranspose2d(
        #     mid_channels,  # This matches the output of deconv1
        #     out_channels,  # Final output channels
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     output_padding=0,
        #     bias=False,
        # )
        # self.bn2 = nn.GroupNorm(1, out_channels)
        self.deconv2 = FanInInitReLULayer(
            inchan=mid_channels,
            outchan=out_channels,
            layer_type="deconv",
            group_norm_groups=1,  # GroupNorm with 1 group for the second layer
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=0,
        )

        # Adjusting upsample to match the stride and channel dimensions
        if stride != 1 or in_channels != out_channels:
            # self.upsample = nn.Sequential(
            #     nn.ConvTranspose2d(
            #         in_channels,  # This should match the actual incoming channels
            #         out_channels,  # This matches the final output channels
            #         kernel_size=1,
            #         stride=stride,
            #         output_padding=0,
            #         padding=0,
            #         bias=False,
            #     ),
            #     nn.GroupNorm(1, out_channels),
            # )
            self.upsample = FanInInitReLULayer(
                inchan=in_channels,
                outchan=out_channels,
                layer_type="deconv",
                group_norm_groups=1,  # GroupNorm with 1 group for the upsample layer
                kernel_size=1,
                stride=stride,
                padding=0,
                output_padding=0,
            )
        else:
            self.upsample = None
        # self.skip_in = nn.ConvTranspose2d(
        #     out_channels,  # This should match the actual incoming channels
        #     mid_channels,  # Intermediate channel size, which is half of out_channels
        #     kernel_size=1,
        #     stride=stride,
        #     padding=0,
        #     output_padding=0,
        #     bias=False,
        # )
        self.skip_in = FanInInitReLULayer(
            inchan=out_channels,
            outchan=mid_channels,
            layer_type="deconv",
            group_norm_groups=1,  # GroupNorm with 1 group for the skip_in layer
            kernel_size=1,
            stride=stride,
            padding=0,
            output_padding=0,
        )

    def forward(self, x, skip_in):
        identity = x
        # out = self.relu(self.bn1(self.deconv1(x)))
        out = self.deconv1(x)

        # skip in connection
        out = out + self.skip_in(skip_in)

        # out = self.bn2(self.deconv2(out))
        out = self.deconv2(out)

        if self.upsample is not None:
            identity = self.upsample(x)
        out = out + identity
        # out = self.relu(out)
        return out


class PyramidModule(nn.Module):
    def __init__(self, N, M, inshape):
        super(PyramidModule, self).__init__()
        # Initial convolution layer
        C, H, W = inshape
        self.init_conv = nn.Conv2d(C, 256, kernel_size=3, stride=1, padding=1)
        self.init_relu = nn.ReLU()
        # Outer convolution resblocks
        self.outer_conv_resblocks = torch.nn.ModuleList(
            [ConvolutionalResBlock(256, 256) for _ in range(N)]
        )
        # Strided convolution resblock
        self.strided_conv_resblock = ConvolutionalResBlock(256, 256, stride=2)
        # Inner convolution resblocks
        inner_conv_resblocks = []
        if M > 0:
            inner_conv_resblocks.append(ConvolutionalResBlock(256, 320))
        for _ in range(M - 1):
            inner_conv_resblocks.append(ConvolutionalResBlock(320, 320))
        self.inner_conv_resblocks = torch.nn.ModuleList(inner_conv_resblocks)
        # Inner deconvolution resblocks
        inner_deconv_resblocks = []
        if M > 0:
            for _ in range(M - 1):
                inner_deconv_resblocks.append(DeconvolutionalResBlock(320, 320))
            inner_deconv_resblocks.append(DeconvolutionalResBlock(320, 256))
        self.inner_deconv_resblocks = torch.nn.ModuleList(inner_deconv_resblocks)
        # Strided deconvolution resblock
        self.strided_deconv_resblock = DeconvolutionalResBlock(256, 256, stride=2)
        # # Outer deconvolution resblocks
        # self.outer_deconv_resblocks = nn.Sequential(
        #     *[DeconvolutionalResBlock(256, 256) for _ in range(N)]
        # )
        self.outer_deconv_resblocks = torch.nn.ModuleList(
            [DeconvolutionalResBlock(256, 256) for _ in range(N)]
        )
        # # Final convolution layer for action logits
        # self.final_conv = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.final_conv = FanInInitReLULayer(
            inchan=256,
            outchan=128,
            layer_type="conv",
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # self.linear_layers = nn.Sequential(
        #     nn.Linear(256 * 9 * 9, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1024),
        #     nn.ReLU(),
        # )
        dense_init_norm_kwargs = {"layer_norm": True}
        lin1 = FanInInitReLULayer(
            128 * 9 * 9,
            256,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )
        lin2 = FanInInitReLULayer(
            256,
            1024,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )
        self.linear_layers = nn.Sequential(lin1, lin2)

    def forward(self, x):
        x = self.init_relu(self.init_conv(x))

        ir_outer = []
        for block in self.outer_conv_resblocks:
            x, ir_i = block(x)
            ir_outer.append(ir_i)
        x, ir_strided = self.strided_conv_resblock(x)
        ir_inner = []
        for block in self.inner_conv_resblocks:
            x, ir_i = block(x)
            ir_inner.append(ir_i)
        # x = self.inner_deconv_resblocks(x)
        for block in self.inner_deconv_resblocks:
            x = block(x, ir_inner.pop())
        x = self.strided_deconv_resblock(x, ir_strided)
        for block in self.outer_deconv_resblocks:
            x = block(x, ir_outer.pop())
        # x = self.final_relu(self.final_conv(x))
        x = self.final_conv(x)
        x = x.flatten(start_dim=-3)
        x = self.linear_layers(x)
        return x


class OrthogonalInitLinear(nn.Linear):
    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.weight)

        if self.bias is not None:
            init.constant_(self.bias, val=0.0)
