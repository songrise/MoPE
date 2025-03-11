#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
borrowed from the official swin implementation, with some modification.
search "prompt" for details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .decoder_heads import SwinUperNetHead
# from warnings import deprecated

USE_STATIC = 2


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        block_module=SwinTransformerBlock,
        # add two more parameters for prompt
        num_prompts=None,
        prompt_location=None,
        deep_prompt=None,
        use_instruct=True,
        d_cross=0,
        d_inter=0,
        moe_n_experts=0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.deep_prompt = deep_prompt
        self.use_instruct = use_instruct
        self.num_prompts = num_prompts * USE_STATIC if deep_prompt else 0
        self.prompt_location = prompt_location
        self.d_cross = d_cross
        self.d_inter = d_inter
        # build blocks
        if num_prompts is not None:
            self.deep_prompt = deep_prompt
            self.num_prompts = num_prompts
            self.num_prompts = self.num_prompts * USE_STATIC
            if use_instruct:
                self.num_prompts += 1
            self.prompt_location = prompt_location
            self.blocks = nn.ModuleList(
                [
                    block_module(
                        self.num_prompts,
                        prompt_location,
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=(
                            drop_path[i] if isinstance(drop_path, list) else drop_path
                        ),  # noqa
                        norm_layer=norm_layer,
                        d_cross=d_cross,
                        d_inter=d_inter,
                        moe_n_experts=moe_n_experts,
                        is_last=(i == depth - 1),
                    )
                    for i in range(depth)
                ]
            )
            # self.prompt_norm = nn.ModuleList(
            #     [norm_layer(dim) for i in range(depth)]
            # )

            if self.deep_prompt and self.prompt_location != "prepend":
                raise ValueError(
                    "deep prompt mode for swin is only applicable to prepend"
                )
        else:
            self.blocks = nn.ModuleList(
                [
                    block_module(
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=(
                            drop_path[i] if isinstance(drop_path, list) else drop_path
                        ),  # noqa
                        norm_layer=norm_layer,
                    )
                    for i in range(depth)
                ]
            )

        # patch merging layer
        if downsample is not None:
            if num_prompts is None:
                self.downsample = downsample(
                    input_resolution, dim=dim, norm_layer=norm_layer
                )
            else:
                if self.use_instruct:
                    self.downsample = downsample(
                        USE_STATIC * num_prompts + 1,
                        prompt_location,
                        deep_prompt,
                        input_resolution,
                        dim=dim,
                        norm_layer=norm_layer,
                    )
                else:
                    self.downsample = downsample(
                        USE_STATIC * num_prompts,
                        prompt_location,
                        deep_prompt,
                        input_resolution,
                        dim=dim,
                        norm_layer=norm_layer,
                    )
        else:
            self.downsample = None

    def forward(
        self,
        x,
        deep_prompt_embd=None,
        instruct_prompt_embd=None,
        use_attn_fuse: bool = False,
        use_mm_fuse: bool = False,
        cross_feature=None,
        all_prompt_experts=None,
        static_prompt=None,
        moe_scores=None,
    ):

        imp_losses = []
        if self.deep_prompt and deep_prompt_embd is None and not use_mm_fuse:
            raise ValueError("need deep_prompt embddings")
        if self.use_instruct:
            assert (
                instruct_prompt_embd is not None
            ), "instruction tuning is used but no instruction prompt embeddings are provided"
        if not self.deep_prompt:
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
        else:
            B = x.shape[0]  # batchsize
            num_blocks = len(self.blocks)
            if use_mm_fuse or deep_prompt_embd.shape[1] != num_blocks:
                # first layer
                for i in range(num_blocks):
                    if i == 0:
                        x = self.blocks[i](x)

                    elif use_attn_fuse:
                        prompt_emb = deep_prompt_embd[
                            :, :, i, ...
                        ]  # all experts at this block

                        assert self.use_instruct
                        x = self.blocks[i].forward_attn_moe(
                            x,
                            prompt_emb,
                            instruct_prompt_embd,
                            moe_scores[:, i, :],  # perlayer routing assumed
                        )  # assume promptedSwinTransformerBlock
                    elif use_mm_fuse:
                        x, imp_loss = self.blocks[i].forward_mm_moe(
                            x,
                            static_prompt[:, i, ...],
                            all_prompt_experts[:, :, i, :, :],
                            instruct_prompt_embd,
                            cross_feature,
                        )  # route inside the block
                        imp_losses.append(imp_loss)
                    else:
                        prompt_emb = deep_prompt_embd[:, i, ...]
                        if self.use_instruct:
                            prompt_emb = torch.cat(
                                (prompt_emb, instruct_prompt_embd), dim=1
                            )

                        x = torch.cat((prompt_emb, x[:, self.num_prompts :, :]), dim=1)
                        x = self.blocks[i](x)  # assume promptedSwinTransformerBlock

            else:
                # other layers
                for i in range(num_blocks):
                    prompt_emb = deep_prompt_embd[:, i, ...]
                    # prompt_emb = deep_prompt_embd[i].expand(B, -1, -1)
                    if instruct_prompt_embd is not None:
                        prompt_emb = torch.cat(
                            (prompt_emb, instruct_prompt_embd), dim=1
                        )
                    # prompt_emb = self.prompt_norm[i](prompt_emb)
                    x = torch.cat((prompt_emb, x[:, self.num_prompts :, :]), dim=1)
                    x = self.blocks[i](x)

        if self.downsample is not None:
            x = self.downsample(x)
        # if the imp_loss is not empty, then we need to return i
        if len(imp_losses) > 0:
            return x, imp_losses
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r"""Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                use_instruct=False,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        f = self.norm(x)  # B L C
        cls_ = self.avgpool(f.transpose(1, 2))  # B C 1
        cls_ = torch.flatten(cls_, 1)
        return cls_, f

    def forward(self, x):
        cls_, f = self.forward_features(x)
        logit = self.head(cls_)
        return logit

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


# ------------prompt version-----------------

import torchvision as tv
import math

from functools import reduce
from operator import mul
from torch.nn import Conv2d, Dropout

from timm.models.layers import to_2tuple


class PromptedSwinTransformer(SwinTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        prompt_length=10,
        prompt_type: str = "vpt",
        moe_n_experts: int = 8,
        use_static_prompt=False,
        use_instruct=True,
        prompt_init="uniform",
        d_cross=0,
        d_inter=0,
        **kwargs,
    ):
        super(PromptedSwinTransformer, self).__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            ape,
            patch_norm,
            use_checkpoint,
            **kwargs,
        )
        global USE_STATIC  # TODO Sep 30: refactor
        USE_STATIC = 2 if use_static_prompt else 1
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # todo add one for instruction.
        num_tokens = prompt_length

        self.prompt_dropout = Dropout(0.1)
        self.prompt_type = prompt_type
        self.depths = depths
        self.use_instruct = use_instruct
        # if project the prompt embeddings
        # if self.prompt_config.PROJECT > -1:
        self.prompt_proj = nn.Identity()

        # build layers
        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    self.patches_resolution[0] // (2**i_layer),
                    self.patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                block_module=PromptedSwinTransformerBlock,
                downsample=(
                    PromptedPatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
                num_prompts=num_tokens,
                prompt_location="prepend",
                deep_prompt=True,
                use_instruct=use_instruct,  #!HARDCODED: now, always use instruct
                d_cross=d_cross,
                d_inter=d_inter,
                moe_n_experts=moe_n_experts,
            )
            self.layers.append(layer)


        if True:
            # elif True: 
            self.moe_n_experts = moe_n_experts
            self.moe_top_k = 1
            val = math.sqrt(
                6.0 / float(3 * reduce(mul, patch_size, 1) + embed_dim)
            )  # noqa
            # for "prepend"
            self.extra_prompt_for_static = (
                1  # always add one for static prompt, even if not used
            )
            self.num_prompts = USE_STATIC * num_tokens
            if self.use_instruct:
                self.num_prompts += 1
            self.prompt_embeddings = nn.Parameter(
                torch.zeros(1, self.num_prompts, embed_dim)
            )  # +1 for the additional instruct prompt, +10 for the static prompt


            #! since now by-default inited with mope, the expert-0 is always the static prompt.
            #! so we always have an additional static prompt, and the total #prompt would be moe_n_expert + 1 per block
            self.deep_prompt_embeddings_0 = nn.Parameter(
                torch.zeros(
                    self.moe_n_experts + self.extra_prompt_for_static,
                    # depths[0] - 1,
                    depths[0],
                    num_tokens,
                    embed_dim,
                )
            )

            self.deep_prompt_embeddings_1 = nn.Parameter(
                torch.zeros(
                    self.moe_n_experts + self.extra_prompt_for_static,
                    depths[1],
                    num_tokens,
                    embed_dim * 2,
                )
            )

            self.deep_prompt_embeddings_2 = nn.Parameter(
                torch.zeros(
                    self.moe_n_experts + self.extra_prompt_for_static,
                    depths[2],
                    num_tokens,
                    embed_dim * 4,
                )
            )

            self.deep_prompt_embeddings_3 = nn.Parameter(
                torch.zeros(
                    self.moe_n_experts + self.extra_prompt_for_static,
                    depths[3],
                    num_tokens,
                    embed_dim * 8,
                )
            )
            self._init_prompt(method=prompt_init, val=val)
            # additional projection from text (instruction prompt)
            if True:  # set false to avoid extra param for ablation
                self.prompt_proj_act = nn.GELU()
                self.prompt_proj_0 = nn.Linear(384, embed_dim)
                self.prompt_proj_1 = nn.Linear(384, embed_dim * 2)
                self.prompt_proj_2 = nn.Linear(384, embed_dim * 4)
                self.prompt_proj_3 = nn.Linear(384, embed_dim * 8)

                #!HARDCODED Sep 21: these are the proj for multimodal moe
                #!HARDCODED Oct 31:  default to route per swin layer (4 layer)
                self.mm_prompt_instruction_proj = nn.Sequential(
                    nn.Linear(768, 384), nn.BatchNorm1d(384), nn.GELU()
                )


    def _init_prompt(self, method: str = "uniform", val=None):
        """
        How static prompt and prompt expert embeddings are initialized
        """
        if method == "uniform":
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            nn.init.uniform_(self.deep_prompt_embeddings_0.data, -val, val)
            nn.init.uniform_(self.deep_prompt_embeddings_1.data, -val, val)
            nn.init.uniform_(self.deep_prompt_embeddings_2.data, -val, val)
            nn.init.uniform_(self.deep_prompt_embeddings_3.data, -val, val)
        elif method == "normal":
            nn.init.trunc_normal_(self.prompt_embeddings.data, std=1)
            nn.init.trunc_normal_(self.deep_prompt_embeddings_0.data, std=1)
            nn.init.trunc_normal_(self.deep_prompt_embeddings_1.data, std=1)
            nn.init.trunc_normal_(self.deep_prompt_embeddings_2.data, std=1)
            nn.init.trunc_normal_(self.deep_prompt_embeddings_3.data, std=1)
        elif method == "othorgonal":
            nn.init.orthogonal_(self.prompt_embeddings.data)
            # othor for each block
            for i in range(self.depths[0] - 1):
                nn.init.orthogonal_(self.deep_prompt_embeddings_0[:, i, ...].data)
            for i in range(self.depths[1]):
                nn.init.orthogonal_(self.deep_prompt_embeddings_1[:, i, ...].data)
            for i in range(self.depths[2]):
                nn.init.orthogonal_(self.deep_prompt_embeddings_2[:, i, ...].data)
            for i in range(self.depths[3]):
                nn.init.orthogonal_(self.deep_prompt_embeddings_3[:, i, ...].data)

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.get_patch_embeddings(x)  # (batch_size, n_patches, hidden_dim)
        prompt_embd = self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1))
        x = torch.cat((prompt_embd, x), dim=1)
        return x

    def get_patch_embeddings(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            # first set all to eval and set the prompt to train later
            for module in self.children():
                module.train(False)
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.incorporate_prompt(x)


        deep_prompt_embds = [
            self.deep_prompt_embeddings_0[0].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_1[0].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_2[0].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_3[0].expand(B, -1, -1, -1),
        ]
        # if self.prompt_config.LOCATION == "prepend" and self.prompt_config.DEEP:
        #!HARDCODED Sep 06: only prepend and deep
        for layer, deep_prompt_embd in zip(self.layers, deep_prompt_embds):
            deep_prompt_embd = self.prompt_dropout(deep_prompt_embd)
            x = layer(x, deep_prompt_embd)

        f = self.norm(x)  # B L C
        cls_ = self.avgpool(f.transpose(1, 2))  # B C 1
        cls_ = torch.flatten(cls_, 1)
        return cls_, f

    def forward_features_instruct(self, x, y, return_internal=False):
        """
        x: image
        y: text features
        """
        B = x.shape[0]
        x = self.incorporate_prompt(x)

        # if self.prompt_config.LOCATION == "prepend" and self.prompt_config.DEEP:
        # forward instruct is forward with static prompt only
        y0 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_0(y))).unsqueeze(
            1
        )
        y1 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_1(y))).unsqueeze(
            1
        )
        y2 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_2(y))).unsqueeze(
            1
        )
        y3 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_3(y))).unsqueeze(
            1
        )

        #!HARDCODED Oct 13:  since now default inited with moe, use the first expert as the vanilla vpt, the dim 0 is the dim of experts
        deep_prompt_embds = [
            self.deep_prompt_embeddings_0[0].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_1[0].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_2[0].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_3[0].expand(B, -1, -1, -1),
        ]
        instruction_embds = [y0, y1, y2, y3]

        internal_features = []
        for i in range(len(self.layers)):
            deep_prompt_embd = deep_prompt_embds[i]
            deep_prompt_embd = self.prompt_dropout(deep_prompt_embd)
            x = self.layers[i](x, deep_prompt_embd, instruction_embds[i])
            internal_features.append(x[:, 1:, :])
        f = self.norm(x)  # B L C
        cls_ = self.avgpool(f.transpose(1, 2))  # B C 1
        cls_ = torch.flatten(cls_, 1)
        if return_internal:
            return cls_, f, internal_features
        return cls_, f

    def forward_features_instruct_moe(self, x, y, route_score, return_internal=False):
        """
        x: image
        y: text features
        route_score: [B, n_expert] determine which expert(s) to use. if more than one, interpolate between the prompts
                    [B, n_layer, n_expert] for per-layer routed moe
        return_internal: if true, return the internal features of each layer
        """
        if len(route_score.shape) == 2:
            route_per_layer = False
        elif len(route_score.shape) == 3:
            route_per_layer = True
            # assert route_score.shape[1] == len(self.layers)
        B = x.shape[0]
        x = self.incorporate_prompt(x)

        # if self.prompt_config.LOCATION == "prepend" and self.prompt_config.DEEP:

        y0 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_0(y))).unsqueeze(
            1
        )
        y1 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_1(y))).unsqueeze(
            1
        )
        y2 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_2(y))).unsqueeze(
            1
        )
        y3 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_3(y))).unsqueeze(
            1
        )
        instruction_embds = [y0, y1, y2, y3]

        all_prompt_experts = [
            self.deep_prompt_embeddings_0[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_1[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_2[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_3[1:, ...].expand(B, -1, -1, -1, -1),
        ]
        static_prompt = [
            self.deep_prompt_embeddings_0[0, ...].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_1[0, ...].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_2[0, ...].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_3[0, ...].expand(B, -1, -1, -1),
        ]
        # construct prompts
        moe_prompt_embds = []

        othro_loss = []
        for i in range(len(self.layers)):
            if route_per_layer:
                # prompt_embeds = torch.einsum(
                #     "bk, bklnh -> blnh", route_score[:, i, :], all_prompt_experts[i]
                # )  # dim 1 is dim of layers
                #!HARDCODED Nov 02: per block
                crt_block_depth = self.depths[i]
                crt_start = sum(self.depths[:i]) - 1
                if i == 0:
                    crt_block_depth -= 1
                    crt_start = 0

                dynamic_prompt = torch.einsum(
                    "blk, bklnh -> blnh",
                    route_score[:, crt_start : crt_start + crt_block_depth, :],
                    all_prompt_experts[i],
                )  # dim 1 is dim of layers
            else:
                # b: batch, k : num_experts, n: number of prompts l: number of blocks in this layer, h: prompt dim
                dynamic_prompt = torch.einsum(
                    "bk, bklnh -> blnh", route_score, all_prompt_experts[i]
                )
                # scale the prompt embeddings (linear interpolation)

            #!HARDCODED Sep 27: concat dynamic and static prompts
            if USE_STATIC == 2:  # 2 means use, 1 is not use
                prompt_embeds = torch.cat(
                    [dynamic_prompt, static_prompt[i]], dim=2
                )  # dim 2 is the number of prompt
            else:
                prompt_embeds = dynamic_prompt


            moe_prompt_embds.append(prompt_embeds)

        internal_features = []
        for i in range(len(self.layers)):
            deep_prompt_embd = moe_prompt_embds[i]
            deep_prompt_embd = self.prompt_dropout(deep_prompt_embd)
            x = self.layers[i](x, deep_prompt_embd, instruction_embds[i])
            # if return_internal:
            internal_features.append(x[:, self.num_prompts :, :])

        f = self.norm(x)  # B L C
        cls_ = self.avgpool(f.transpose(1, 2))  # B C 1
        cls_ = torch.flatten(cls_, 1)
        if return_internal:
            return cls_, internal_features
        # return cls_, f, torch.mean(torch.stack(othro_loss))
        return cls_, f, None

    def forward_features_attn_moe(self, x, y, route_score):
        """
        x: image
        y: text features
        route_score: [B, n_expert] determine which expert(s) to use. if more than one, interpolate between the prompts
        fuse attn values instead of learnable prompts
        """
        B = x.shape[0]
        x = self.incorporate_prompt(x)

        # if self.prompt_config.LOCATION == "prepend" and self.prompt_config.DEEP:

        y0 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_0(y))).unsqueeze(
            1
        )
        y1 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_1(y))).unsqueeze(
            1
        )
        y2 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_2(y))).unsqueeze(
            1
        )
        y3 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_3(y))).unsqueeze(
            1
        )
        instruction_embds = [y0, y1, y2, y3]

        all_prompt_experts = [
            self.deep_prompt_embeddings_0[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_1[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_2[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_3[1:, ...].expand(B, -1, -1, -1, -1),
        ]

        static_prompt = [
            self.deep_prompt_embeddings_0[0, ...].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_1[0, ...].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_2[0, ...].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_3[0, ...].expand(B, -1, -1, -1),
        ]

        # expand and concat all static prompts to prompt experts
        for i in range(len(all_prompt_experts)):
            # expand number_of_expert time and concat
            static_prompt_expanded = (
                static_prompt[i].unsqueeze(1).expand(-1, self.moe_n_experts, -1, -1, -1)
            )
            all_prompt_experts[i] = torch.cat(
                [all_prompt_experts[i], static_prompt_expanded], dim=3
            )  # dim 3 is the number of prompts

        for i in range(len(self.layers)):
            x = self.layers[i](
                x,
                all_prompt_experts[i],
                instruction_embds[i],
                use_attn_fuse=True,
                moe_scores=route_score,
            )

        f = self.norm(x)  # B L C
        cls_ = self.avgpool(f.transpose(1, 2))  # B C 1
        cls_ = torch.flatten(cls_, 1)
        return cls_, f

    def forward_features_instruct_multimodal_moe(self, x, y):
        """
        x: image
        y: text features
        """
        B = x.shape[0]
        x = self.incorporate_prompt(x)

        # if self.prompt_config.LOCATION == "prepend" and self.prompt_config.DEEP:
        y_ins = self.mm_prompt_instruction_proj(y)
        y0 = self.prompt_dropout(
            self.prompt_proj_act(self.prompt_proj_0(y_ins))
        ).unsqueeze(1)
        y1 = self.prompt_dropout(
            self.prompt_proj_act(self.prompt_proj_1(y_ins))
        ).unsqueeze(1)
        y2 = self.prompt_dropout(
            self.prompt_proj_act(self.prompt_proj_2(y_ins))
        ).unsqueeze(1)
        y3 = self.prompt_dropout(
            self.prompt_proj_act(self.prompt_proj_3(y_ins))
        ).unsqueeze(1)
        instruction_embds = [y0, y1, y2, y3]

        all_prompt_experts = [
            self.deep_prompt_embeddings_0[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_1[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_2[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_3[1:, ...].expand(B, -1, -1, -1, -1),
        ]

        static_prompt = [
            self.deep_prompt_embeddings_0[0, ...].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_1[0, ...].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_2[0, ...].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_3[0, ...].expand(B, -1, -1, -1),
        ]

        extra_out = {}
        all_layer_imp_loss = []
        # here, default to route per layer
        for i in range(len(self.layers)):

            x, imp_losses = self.layers[i](
                x,
                all_prompt_experts=all_prompt_experts[i],
                instruct_prompt_embd=instruction_embds[i],
                static_prompt=static_prompt[i],
                use_mm_fuse=True,
                cross_feature=y,
            )
            all_layer_imp_loss += imp_losses

        extra_out["importance_loss"] = torch.mean(torch.stack(all_layer_imp_loss))
        f = self.norm(x)  # B L C
        cls_ = self.avgpool(f.transpose(1, 2))  # B C 1
        cls_ = torch.flatten(cls_, 1)
        return cls_, f, extra_out

    def forward(self, x):
        cls_, f = self.forward_features(x)
        return f

    def load_state_dict(self, state_dict, strict):

        super(PromptedSwinTransformer, self).load_state_dict(state_dict, strict)



class PromptedPatchMerging(PatchMerging):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        num_prompts,
        prompt_location,
        deep_prompt,
        input_resolution,
        dim,
        norm_layer=nn.LayerNorm,
    ):
        super(PromptedPatchMerging, self).__init__(input_resolution, dim, norm_layer)
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if prompt_location == "prepend":
            if not deep_prompt:
                self.prompt_upsampling = None
                # self.prompt_upsampling = nn.Linear(dim, 4 * dim, bias=False)
            else:
                self.prompt_upsampling = None

    def upsample_prompt(self, prompt_emb):
        if self.prompt_upsampling is not None:
            prompt_emb = self.prompt_upsampling(prompt_emb)
        else:
            prompt_emb = torch.cat(
                (prompt_emb, prompt_emb, prompt_emb, prompt_emb), dim=-1
            )
        return prompt_emb

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        if self.prompt_location == "prepend":
            # change input size
            prompt_emb = x[:, : self.num_prompts, :]
            x = x[:, self.num_prompts :, :]
            L = L - self.num_prompts
            prompt_emb = self.upsample_prompt(prompt_emb)

        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(
            H * W, L
        )
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # add the prompt back:
        if self.prompt_location == "prepend":
            x = torch.cat((prompt_emb, x), dim=1)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PromptedSwinTransformerBlock(SwinTransformerBlock):
    def __init__(
        self,
        num_prompts,
        prompt_location,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        d_cross=0,
        d_inter=0,
        moe_n_experts=0,
        is_last=False,
    ):
        super(PromptedSwinTransformerBlock, self).__init__(
            dim,
            input_resolution,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop,
            attn_drop,
            drop_path,
            act_layer,
            norm_layer,
        )
        self.is_last = is_last
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if self.prompt_location == "prepend":
            self.attn = PromptedWindowAttention(
                num_prompts,
                prompt_location,
                dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        
        self.mm_prompt_global_pooler = nn.AdaptiveAvgPool1d(1)
        self.mm_prompt_self_proj = nn.Linear(dim, d_inter)
        self.mm_prompt_cross_proj = nn.Linear(768, d_cross)
        self.mm_frozen_expert_key_embed = nn.Parameter(
            torch.zeros(d_cross + d_inter, moe_n_experts)
        )
        # orthogonal initialize key_embed
        nn.init.orthogonal_(self.mm_frozen_expert_key_embed)
        self.mm_frozen_expert_key_embed.requires_grad = False

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        num_prompts = self.num_prompts
        if self.prompt_location == "prepend":
            # change input size
            prompt_emb = x[:, :num_prompts, :]
            x = x[:, num_prompts:, :]
            L = L - num_prompts

        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(
            H * W, L
        )

        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows --> nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C

        # add back the prompt for attn for parralel-based prompts
        # nW*B, num_prompts + window_size*window_size, C
        num_windows = int(x_windows.shape[0] / B)
        if self.prompt_location == "prepend":
            # expand prompts_embs
            # B, num_prompts, C --> nW*B, num_prompts, C
            prompt_emb = prompt_emb.unsqueeze(0)
            prompt_emb = prompt_emb.expand(num_windows, -1, -1, -1)
            prompt_emb = prompt_emb.reshape((-1, num_prompts, C))
            x_windows = torch.cat(
                (prompt_emb, x_windows), dim=1
            )  # append prompt to this window

        dummy_mask = torch.zeros(
            num_windows, self.window_size**2, self.window_size**2, device=x.device
        )
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask if self.attn_mask is not None else dummy_mask
        )
        # attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # seperate prompt embs --> nW*B, num_prompts, C
        if self.prompt_location == "prepend":
            # change input size
            prompt_emb = attn_windows[:, :num_prompts, :]
            attn_windows = attn_windows[:, num_prompts:, :]
            # change prompt_embs's shape:
            # nW*B, num_prompts, C - B, num_prompts, C
            prompt_emb = prompt_emb.view(-1, B, num_prompts, C)
            prompt_emb = prompt_emb.mean(0)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # add the prompt back:
        if self.prompt_location == "prepend":
            x = torch.cat((prompt_emb, x), dim=1)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def forward_attn_moe(
        self, x_input, all_prompt_experts, instruct_prompt_embd, moe_scores
    ):
     raise NotImplementedError

    def forward_mm_moe(
        self,
        x_input,
        static_prompt,
        all_prompt_experts,
        instruct_prompt_embd,
        cross_feature,
    ):
        """
        The main forward function for multimodal moe (MoPE)
        """

        # first synthesis the prompt embd
        x_pooled = self.mm_prompt_global_pooler(x_input.transpose(1, 2)).squeeze(
            -1
        )  # B, C

        moe_self_embd = self.mm_prompt_self_proj(x_pooled)  # B, D_moe_embd
        moe_cross_embd = self.mm_prompt_cross_proj(cross_feature)  # B,  D_moe_embd
        moe_joint_embd = torch.cat(
            [moe_self_embd, moe_cross_embd], dim=1
        )  # B,  D_moe_embd *2
        # moe_joint_embd = moe_cross_embd
        # B,  D_moe_embd *2
        # get the logit by dot product with expert key at this layer  B,  D_moe_embd *2 @ D_moe_embd *2 , k_expert -> B, k_expert
        moe_logits = moe_joint_embd @ self.mm_frozen_expert_key_embed  # B, k_expert # B, k_expert
        temperature = 0.1
        # add normal dis N(0, 1/n_experts^2)
        # add gumbel noise
        
        noise = torch.randn(moe_logits.shape).to(moe_logits.device) / (16**2)
        moe_logits = moe_logits / temperature

        moe_scores = F.softmax(moe_logits + noise, dim=-1)
        moe_mask = torch.zeros(moe_scores.shape).to(moe_scores.device)
        # if self.route_per_layer:
        moe_mask.scatter_(1, torch.topk(moe_scores, k=1, dim=-1)[1], 1)
        # uncomment to use top-1 sparse routing
        # moe_scores = moe_mask
        # get the dynamic prompt
        dynamic_prompt = torch.einsum("bk, bknh -> bnh", moe_scores, all_prompt_experts)
        # get the prompt embedding for this forward, which is concat of all
        prompt_emb = torch.cat(
            (static_prompt, dynamic_prompt, instruct_prompt_embd), dim=1
        )

        sum_scores = torch.sum(moe_scores, dim=0)  # N_expert
        std_scores = torch.std(sum_scores, dim=-1)  # 1
        mean_scores = torch.mean(sum_scores, dim=-1)  # 1
        threshold = 0.1
        importance_loss = (std_scores / mean_scores) ** 2 if mean_scores > 0 else 0
        # importance_loss = torch.where(importance_loss > threshold, importance_loss, torch.zeros_like(importance_loss))
        importance_loss = (
            importance_loss
            if importance_loss > threshold
            else torch.zeros_like(importance_loss)
        )
        # imp_loss = torch.mean(importance_loss)

        # the following is same as forward.
        x = torch.cat((prompt_emb, x_input[:, self.num_prompts :, :]), dim=1)
        return self.forward(x), importance_loss


class PromptedWindowAttention(WindowAttention):
    def __init__(
        self,
        num_prompts,
        prompt_location,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super(PromptedWindowAttention, self).__init__(
            dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop
        )
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        

        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww

        # account for prompt nums for relative_position_bias
        # attn: [1920, 6, 649, 649]
        # relative_position_bias: [6, 49, 49])

        if self.prompt_location == "prepend":
            # expand relative_position_bias
            _C, _H, _W = relative_position_bias.shape

            relative_position_bias = torch.cat(
                (
                    torch.zeros(_C, self.num_prompts, _W, device=attn.device),
                    relative_position_bias,
                ),
                dim=1,
            )
            relative_position_bias = torch.cat(
                (
                    torch.zeros(
                        _C, _H + self.num_prompts, self.num_prompts, device=attn.device
                    ),
                    relative_position_bias,
                ),
                dim=-1,
            )

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # incorporate prompt
            # mask: (nW, 49, 49) --> (nW, 49 + n_prompts, 49 + n_prompts)
            nW = mask.shape[0]
            if self.prompt_location == "prepend":
                # expand relative_position_bias
                exp_masks = torch.zeros(nW, self.num_prompts, _W).to(
                    attn.device
                )
                mask = torch.cat((exp_masks, mask), dim=1)
                mask = torch.cat(
                    (
                        torch.zeros(
                            nW,
                            _H + self.num_prompts,
                            self.num_prompts,
                            device=attn.device,
                        ),
                        mask,
                    ),
                    dim=-1,
                )
            # logger.info("before", attn.shape)
            # attn: B batch size for input image, nW: number of windows, nH: number of heads, N: window size (with prompt), N: window size (with prompt)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            # logger.info("after", attn.shape)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_moe(self, xs, mask=None, moe_scores=None):
        """
        Args:
            xs: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        all_x = []  # all x after self-attn
        for x in xs:
            B_, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)

            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww

            # account for prompt nums for relative_position_bias
            # attn: [1920, 6, 649, 649]
            # relative_position_bias: [6, 49, 49])

            if self.prompt_location == "prepend":
                # expand relative_position_bias
                _C, _H, _W = relative_position_bias.shape

                relative_position_bias = torch.cat(
                    (
                        torch.zeros(_C, self.num_prompts, _W, device=attn.device),
                        relative_position_bias,
                    ),
                    dim=1,
                )
                relative_position_bias = torch.cat(
                    (
                        torch.zeros(
                            _C,
                            _H + self.num_prompts,
                            self.num_prompts,
                            device=attn.device,
                        ),
                        relative_position_bias,
                    ),
                    dim=-1,
                )

            attn = attn + relative_position_bias.unsqueeze(0)
            # no mask here
            attn = self.softmax(attn)

            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            all_x.append(x)

        # blend by moe scores
        x = torch.stack(all_x, dim=1)  # B, n_experts, H*W, C
        # b: batch size, k: k experts in total, l: sequence length, h: hidden dim
        x = torch.einsum("bk, bklh -> blh", moe_scores, x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_swin_encoder(
    num_classes,
    crop_size,
    use_vpt=True,
    moe_n_experts=8,
    prompt_length=10,
    use_static_prompt=False,
    use_instruct=True,
    prompt_init="uniform",
    d_cross=0,
    d_inter=0,
):
    if use_vpt:
        if crop_size == 224:
            model = PromptedSwinTransformer(
                img_size=crop_size,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=7,
                drop_path_rate=0.5,
                num_classes=num_classes,
                moe_n_experts=moe_n_experts,
                prompt_length=prompt_length,
                use_static_prompt=use_static_prompt,
                use_instruct=use_instruct,
                prompt_init=prompt_init,
                d_cross=d_cross,
                d_inter=d_inter,
            )
        elif crop_size == 384:
            model = PromptedSwinTransformer(
                img_size=crop_size,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=12,
                drop_path_rate=0.2,
                num_classes=num_classes,
                moe_n_experts=moe_n_experts,
                prompt_length=prompt_length,
                use_static_prompt=use_static_prompt,
                use_instruct=use_instruct,
            )
        elif crop_size == 512:
            model = PromptedSwinTransformer(
                img_size=crop_size,
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                drop_path_rate=0.3,
                num_classes=num_classes,
                moe_n_experts=moe_n_experts,
                prompt_length=prompt_length,
                use_static_prompt=use_static_prompt,
                use_instruct=use_instruct,
            )

        # freeze all parameter except name with "prompt" or "moe"
        for name, param in model.named_parameters():
            if "prompt" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    else:
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=num_classes,
        )

    # load checkpoint
    if crop_size == 224:
        model_w = "pretrained/swin_base_patch4_window7_224_22k.pth"
    elif crop_size == 384:
        model_w = "pretrained/swin_base_patch4_window12_384_22k.pth"
    elif crop_size == 512:
        model_w = "/pretrained/upernet_swin_small_patch4_window7_512x512.pth"
    checkpoint = torch.load(model_w, map_location="cpu")
    state_dict = checkpoint["model"]
    # ignore head weight when loading
    state_dict.pop("head.weight")
    state_dict.pop("head.bias")
    model.load_state_dict(state_dict, strict=False)
    return model


def get_swin_classifier(
    num_classes,
    backbone,
    use_vpt,
    moe_n_experts,
    prompt_length=10,
    use_static_prompt=False,
    prompt_init="uniform",
    use_instruct=True,
    d_cross=0,
    d_inter=0,
):
    class SwinClassifier(nn.Module):
        def __init__(
            self,
            num_classes,
            use_vpt,
            moe_n_experts=8,
            prompt_length=10,
            use_static_prompt=False,
            prompt_init="uniform",
            use_instruct=True,
            d_cross=0,
            d_inter=0,
        ):
            super(SwinClassifier, self).__init__()
            crop_size = 224
            if "384" in backbone:
                crop_size = 384
            self.encoder = get_swin_encoder(
                num_classes,
                crop_size,
                use_vpt,
                moe_n_experts,
                prompt_length,
                use_static_prompt,
                prompt_init=prompt_init,
                use_instruct=use_instruct,
                d_cross=d_cross,
                d_inter=d_inter,
            )
            self.classifier = nn.Linear(1024, num_classes)

        def forward(self, x, return_features=False):
            cls_, _ = self.encoder.forward_features(x)
            x = self.classifier(cls_)
            if not return_features:
                return x
            else:
                return x, cls_

        def forward_instruct(
            self,
            x,
            text_feature=None,
            return_features=False,
        ):
            """
            project forzen text encoder cls as prompt
            """
            cls_, _ = self.encoder.forward_features_instruct(x, text_feature)
            x = self.classifier(cls_)
            if not return_features:
                return x
            else:
                return x, cls_

        def forward_instruct_moe(
            self, x, text_feature, route_score, return_features=False, attn_fuse=False
        ):
            if attn_fuse:
                cls_, _ = self.encoder.forward_features_attn_moe(
                    x, text_feature, route_score
                )
                extra_loss = 0.0
            else:
                cls_, _, extra_loss = self.encoder.forward_features_instruct_moe(
                    x, text_feature, route_score
                )
            x = self.classifier(cls_)
            if not return_features:
                return x, extra_loss
            else:
                return x, cls_, extra_loss

        def forward_instruct_multimodal_moe(
            self, x, text_feature, return_features=False
        ):
            """
            moe where the prompt vector is conditioned both on pooled img feature and the text feature.
            """
            cls_, _, extra_out = self.encoder.forward_features_instruct_multimodal_moe(
                x, text_feature
            )
            x = self.classifier(cls_)
            extra_out["cls"] = cls_
            if not return_features:
                return x, extra_out
            else:
                return x, cls_, extra_out



    return SwinClassifier(
        num_classes,
        use_vpt,
        moe_n_experts,
        prompt_length,
        use_static_prompt,
        prompt_init,
        use_instruct,
        d_cross=d_cross,
        d_inter=d_inter,
    )


def get_swin_segmentor(
    num_classes,
    crop_size,
    use_vpt,
    moe_n_experts,
    prompt_length=10,
    use_static_prompt=False,
):
    class SwinSegmentor(nn.Module):
        def __init__(
            self,
            num_classes,
            use_vpt,
            moe_n_experts=8,
            prompt_length=10,
            use_static_prompt=False,
        ):
            super(SwinSegmentor, self).__init__()
            self.encoder = get_swin_encoder(
                num_classes,
                crop_size,
                use_vpt,
                moe_n_experts,
                prompt_length,
                use_static_prompt,
            )
            # self.segmentor = SeTRPUPHead(crop_size, num_classes=num_classes, prompt_len= self.encoder.num_prompts)
            self.segmentor = SwinUperNetHead()
            # self.instruct_learned_down_sampler = nn.Linear(145, 144)

        def forward(self, x, return_features=False):
            cls_, f = self.encoder.forward_features(x)
            x = self.segmentor(f)
            if not return_features:
                return x
            else:
                return x, f

        def forward_instruct(self, x, text_feature, return_features=False, **kwargs):
            cls_, f, fs = self.encoder.forward_features_instruct(
                x, text_feature, return_internal=True
            )

            x = self.segmentor(fs)
            if not return_features:
                return x
            else:
                return x, f

        # TODO Oct 13:
        def forward_late_concat(self, x, text_feature, return_features=False, **kwargs):
            cls_, f = self.encoder.forward_features(x)
            # repeat text feature and add to f
            text_feature = text_feature.unsqueeze(1)
            text_feature = text_feature.expand(-1, -1, f.shape[1])
            f = f + text_feature

            x = self.segmentor(f)
            if not return_features:
                return x
            else:
                return x, f

        def forward_instruct_moe(
            self, x, text_feature, route_score, return_features=False, attn_fuse=False
        ):
            if attn_fuse:
                cls_, f = self.encoder.forward_features_attn_moe(
                    x, text_feature, route_score
                )
            else:
                cls_, fs = self.encoder.forward_features_instruct_moe(
                    x, text_feature, route_score, return_internal=True
                )
            x = self.segmentor(fs)
            if not return_features:
                return x
            else:
                return x, f

        def slide_inference(self, x, crop_size, stride):
            """
            x: B, C, H, W
            crop_size: [crop_h, crop_w]
            stride: [stride_h, stride_w]
            """
            B, C, H, W = x.shape
            crop_h, crop_w = crop_size
            stride_h, stride_w = stride
            assert (
                H % stride_h == 0 and W % stride_w == 0
            ), "input feature has wrong size, should be {}, got {}".format(H * W, L)
            assert (
                crop_h % stride_h == 0 and crop_w % stride_w == 0
            ), "crop size has wrong size, should be {}, got {}".format(H * W, L)
            assert (
                crop_h <= H and crop_w <= W
            ), "crop size should be smaller than input feature size"

            # crop
            ret = []
            for i in range(0, H - crop_h + 1, stride_h):
                for j in range(0, W - crop_w + 1, stride_w):
                    crop = x[:, :, i : i + crop_h, j : j + crop_w]
                    crop_ret = self.forward(crop)
                    ret.append(crop_ret)
            ret = torch.stack(ret, dim=1)
            ret = ret.view(B, -1, num_classes)
            return ret

    return SwinSegmentor(
        num_classes, use_vpt, moe_n_experts, prompt_length, use_static_prompt
    )


if __name__ == "__main__":
    crop_size = 224
    model = get_swin_encoder(crop_size, use_vpt=True)
    model.eval()

    dummy_input = torch.randn(1, 3, crop_size, crop_size)

    seg = get_swin_segmentor(20)
    ret = seg(dummy_input)
    dummy_input_2 = torch.randn(1, 3, crop_size, 600)
    ret = seg.slide_inference(dummy_input_2, [crop_size, crop_size], [128, 128])

    print(ret.shape)