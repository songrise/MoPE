# -*- coding : utf-8 -*-
# @FileName  : head.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Sep 07, 2023
# @Github    : https://github.com/songrise
# @Description: Decoder head for transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from mmcv.cnn import ConvModule
from transformers.models.upernet.modeling_upernet import UperNetHead, UperNetConfig


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    exit(-1)
                # warnings.warn(
                #     f'When align_corners={align_corners}, '
                #     'the output would more aligned if '
                #     f'input size {(input_h, input_w)} is `x+1` and '
                #     f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class SeTRPUPHead(nn.Module):
    """
    SETR-PUP with default parameters, for prompted Swin Encoder
    """

    def __init__(
        self,
        crop_size=224,
        patch_size=32,
        in_channels=1024,
        embed_dims=256,
        prompt_len=10,
        #  stages = 4,
        num_classes=20,
    ):
        super(SeTRPUPHead, self).__init__()
        self.in_channels = in_channels
        # self.stages = stages
        self.num_classes = num_classes
        self.n_patch = (crop_size // patch_size) ** 2  # 7*7
        self.feat_proj = nn.Identity()
        if prompt_len > 0:
            # B, HW/256+Prompt, C -> B, HW/256, C
            self.feat_proj = nn.Linear(self.n_patch + prompt_len, self.n_patch)
            nn.init.kaiming_uniform_(self.feat_proj.weight, a=0, mode="fan_in")
        self.act = nn.GELU()
        self.conv_0 = nn.Conv2d(
            in_channels, embed_dims, kernel_size=3, stride=1, padding=1
        )
        self.conv_1 = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1
        )
        self.conv_2 = nn.Conv2d(
            embed_dims, embed_dims // 2, kernel_size=3, stride=1, padding=1
        )
        self.conv_3 = nn.Conv2d(
            embed_dims // 2, embed_dims // 2, kernel_size=3, stride=1, padding=1
        )
        self.conv_4 = nn.Conv2d(
            embed_dims // 2, self.num_classes, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        """
        Args:
                x: B, C, HW/256
        Return
                pred, B, NUM_CLASSES, H, W
        """
        assert len(x.shape) == 3
        x = x.transpose(1, 2)
        x = self.feat_proj(x)
        x = x.transpose(1, 2)
        hw = x.shape[1]
        H, W = int(hw**0.5), int(hw**0.5)
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv_0(x)
        x = self.act(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv_1(x)
        x = self.act(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv_2(x)
        x = self.act(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv_3(x)
        x = self.act(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv_4(x)
        # x = self.act(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        return x


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(
        self,
        pool_scales,
        in_channels,
        channels,
        conv_cfg,
        norm_cfg,
        act_cfg,
        align_corners,
    ):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class SwinUperNetHead(nn.Module):
    """
    warpper class for transformer.UperNetHead used for swin transformer
    """

    def __init__(self, pool_scales=(1, 2, 3, 6)):
        super(SwinUperNetHead, self).__init__()
        self.config = UperNetConfig(num_labels=1)
        self.head = UperNetHead(config=self.config, in_channels=[256, 512, 1024, 1024])

    def forward(self, xs):
        """
        Args:
            xs: a list of features from swin transformer encoder,
                [B, H*W, C] for each element in the list

        """
        # convert into B C H W
        xs = [
            einops.rearrange(
                x,
                "b (h w) c -> b c h w",
                h=int(x.shape[1] ** 0.5),
                w=int(x.shape[1] ** 0.5),
            )
            for x in xs
        ]
        return self.head(xs)
