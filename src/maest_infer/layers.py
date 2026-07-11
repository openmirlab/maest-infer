"""MAEST's Vision Transformer building blocks: Mlp, patch embeddings, attention, blocks.

Split out of the pre-split maest.py (lines ~38/175-401): `to_2tuple`, `Mlp`,
`PatchEmbed` (standard 2D conv patchify), `PatchEmbedFreq` (MAEST's
frequency-dependent patch embedding, one conv per frequency band) plus its
`replacePatchEmbedFreq` constructor, `Attention`, and `Block`. `first_RUN` is
a shared, mutable, process-wide debug-logging flag (never affects numerics --
only gates one-time `_logger.debug` calls) that model.MAEST also reads/writes
via `layers.first_RUN`, preserving the original single-module `global
first_RUN` semantics across the file split.

Reads: maest_infer.helpers.vit_helpers.DropPath; read by maest_infer.model,
maest_infer.loading
"""

import collections
import logging
import warnings

import torch
import torch.nn as nn

from .helpers.vit_helpers import DropPath

_logger = logging.getLogger("MAEST")


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

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


first_RUN = True


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.grid_size = (img_size[0] // stride[0], img_size[1] // stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if not (H == self.img_size[0] and W == self.img_size[1]):
            _logger.debug(
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if first_RUN:
            _logger.debug(f"self.norm(x) size: {x.size()}")
        return x


class PatchEmbedFreq(nn.Module):
    """2D Spectrogram to Patch Embedding with frequency-dependent projector."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        old_proj=None,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.grid_size = (img_size[0] // stride[0], img_size[1] // stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_f_patches = self.grid_size[0]
        self.flatten = flatten
        self.embed_dim = embed_dim
        self.projs = nn.ModuleList(
            [
                nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
                for _ in range(self.num_f_patches)
            ]
        )

        if old_proj:
            for proj in self.projs:
                proj.load_state_dict(old_proj.state_dict())

        self.lb = [i * self.stride[0] for i in range(self.num_f_patches)]
        self.hb = [
            i * self.stride[0] + self.patch_size[0] for i in range(self.num_f_patches)
        ]

        if first_RUN:
            _logger.debug(f"lb embeddings {self.lb}")
        if first_RUN:
            _logger.debug(f"hb embeddings {self.hb}")

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if not (H == self.img_size[0] and W == self.img_size[1]):
            warnings.warn(
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            )
        x = torch.stack(
            [
                conv(x[:, :, lb:hb, :]).squeeze()
                for conv, lb, hb in zip(self.projs, self.lb, self.hb)
            ]
        ).permute(1, 2, 0, 3)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if first_RUN:
            _logger.debug(f"self.norm(x): {x.size()}")
        return x


def replacePatchEmbedFreq(replace):
    return PatchEmbedFreq(
        img_size=replace.img_size,
        patch_size=replace.patch_size,
        stride=replace.stride,
        in_chans=1,
        embed_dim=replace.embed_dim,
        flatten=replace.flatten,
        old_proj=replace.proj,
    )


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
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

    def forward(self, x, return_self_attention=False):
        if return_self_attention:
            return self.attn(self.norm1(x))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
