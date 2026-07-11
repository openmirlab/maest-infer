"""The MAEST model: patch embed -> ViT encoder -> distilled classification head.

Split out of the pre-split maest.py (lines ~404-874 for the `MAEST` class,
~876-910 for its construction-time weight-init helpers). `forward()` accepts
raw waveform, a precomputed melspectrogram, or a batch of either and returns
`(logits, embeddings)`; `predict_labels()` wraps it with sigmoid + label
lookup. `_init_vit_weights`/`lecun_normal_` are kept here rather than in
loading.py (where the rest of the "weight-init/pos-embed adaptation" section
of the original file landed) because they run at MAEST construction time
(`self.apply(_init_vit_weights)` inside `__init__`/`init_weights`), and
loading.py needs to import `MAEST` -- keeping them here avoids a circular
import between the two modules.

Reads: maest_infer.layers (first_RUN shared state, PatchEmbed, Block),
maest_infer.helpers.vit_helpers (trunc_normal_, variance_scaling_),
maest_infer.helpers.melspectrogram, maest_infer.discogs_labels
"""

import logging
from collections import OrderedDict
from functools import partial
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from . import layers
from .discogs_labels import discogs_400labels, discogs_519labels
from .helpers.melspectrogram import MelSpectrogram
from .helpers.vit_helpers import trunc_normal_, variance_scaling_
from .layers import Block, PatchEmbed

_logger = logging.getLogger("MAEST")


class MAEST(nn.Module):
    """MAEST - Music Audio Efficient Spectrogram Transformer.

    Based on the implementation of Vision Transformer in timm library.
    """

    def __init__(
        self,
        u_patchout=0,
        s_patchout_t=0,
        s_patchout_f=0,
        s_patchout_f_indices=[],
        s_patchout_f_interleaved=0,
        s_patchout_t_indices=[],
        s_patchout_t_interleaved=0,
        img_size=(128, 998),
        patch_size=16,
        stride=16,
        in_chans=1,
        num_classes=527,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init="",
        distilled_type="mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.u_patchout = u_patchout
        self.img_size = img_size
        self.s_patchout_t = s_patchout_t
        self.s_patchout_f = s_patchout_f
        self.s_patchout_f_indices = s_patchout_f_indices
        self.s_patchout_f_interleaved = s_patchout_f_interleaved
        self.s_patchout_t_indices = s_patchout_t_indices
        self.s_patchout_t_interleaved = s_patchout_t_interleaved
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        self.distilled_type = distilled_type
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        if self.num_classes == 400:
            self.labels = discogs_400labels
        elif self.num_classes == 519:
            self.labels = discogs_519labels

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten=False,
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        )
        self.new_pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.freq_new_pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, self.patch_embed.grid_size[0], 1)
        )
        self.time_new_pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, 1, self.patch_embed.grid_size[1])
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity(),
        )
        self.head_dist = None
        if distilled:
            self.head_dist = (
                nn.Linear(self.embed_dim, self.num_classes)
                if num_classes > 0
                else nn.Identity()
            )

        self.init_weights(weight_init)
        self.melspectrogram = None

    def init_melspectrogram(self):
        """Initialize the melspectrogram extractor."""
        if self.melspectrogram is not None:
            _logger.warning(
                "Melspectrogram extractor already initialized, skipping re-initialization."
            )
            return
        self.melspectrogram = MelSpectrogram()

    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "nlhb", "")
        trunc_normal_(self.new_pos_embed, std=0.02)
        trunc_normal_(self.freq_new_pos_embed, std=0.02)
        trunc_normal_(self.time_new_pos_embed, std=0.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=0.02)
        if mode.startswith("jax"):
            raise RuntimeError("Not supported yet")
        else:
            trunc_normal_(self.cls_token, std=0.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "new_pos_embed",
            "freq_new_pos_embed",
            "time_new_pos_embed",
            "cls_token",
            "dist_token",
        }

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        if self.num_tokens == 2:
            self.head_dist = (
                nn.Linear(self.embed_dim, self.num_classes)
                if num_classes > 0
                else nn.Identity()
            )

    def forward_features(self, x, transformer_block=-1, return_self_attention=False):
        x = self.patch_embed(x)
        B_dim, E_dim, F_dim, T_dim = x.shape
        if layers.first_RUN:
            _logger.debug(f"patch_embed shape: {x.shape}")
        if layers.first_RUN:
            _logger.debug(
                f"self.time_new_pos_embed.shape: {self.time_new_pos_embed.shape}"
            )
        time_new_pos_embed = self.time_new_pos_embed
        if x.shape[-1] <= time_new_pos_embed.shape[-1]:
            if self.training:
                toffset = torch.randint(
                    1 + time_new_pos_embed.shape[-1] - x.shape[-1], (1,)
                ).item()
                if layers.first_RUN:
                    _logger.debug(
                        f"CUT with randomoffset={toffset} time_new_pos_embed.shape: {time_new_pos_embed.shape}",
                    )
                time_new_pos_embed = time_new_pos_embed[
                    :, :, :, toffset : toffset + x.shape[-1]
                ]
            else:
                time_new_pos_embed = time_new_pos_embed[:, :, :, : x.shape[-1]]
            if layers.first_RUN:
                _logger.debug(
                    f"CUT time_new_pos_embed.shape: {time_new_pos_embed.shape}"
                )
        else:
            raise Exception(
                f"the patches shape:{x.shape} are larger than the expected time encodings {time_new_pos_embed.shape},"
                " please reduce the input duration."
            )
        x = x + time_new_pos_embed
        if layers.first_RUN:
            _logger.debug(
                f"self.freq_new_pos_embed.shape: {self.freq_new_pos_embed.shape}"
            )
        x = x + self.freq_new_pos_embed

        # Structured Patchout (training only)
        if self.training and self.s_patchout_t:
            if layers.first_RUN:
                _logger.debug(
                    f"X Before time Patchout of {self.s_patchout_t}: {x.size()}"
                )
            random_indices = (
                torch.randperm(T_dim)[: T_dim - self.s_patchout_t].sort().values
            )
            x = x[:, :, :, random_indices]
            if layers.first_RUN:
                _logger.debug(f"X after time Patchout {x.size()}")
        if self.training and self.s_patchout_f:
            if layers.first_RUN:
                _logger.debug(
                    f"X Before Freq Patchout of {self.s_patchout_f}: {x.size()}"
                )
            random_indices = (
                torch.randperm(F_dim)[: F_dim - self.s_patchout_f].sort().values
            )
            x = x[:, :, random_indices, :]
            if layers.first_RUN:
                _logger.debug(f"X after freq Patchout: {x.size()}")

        if self.s_patchout_f_indices:
            if layers.first_RUN:
                _logger.debug(
                    "WARNING!! Applying freq patchout indices on feature extraction "
                )
            if layers.first_RUN:
                _logger.debug(
                    f"X Before Freq Patchout of bands {self.s_patchout_f_indices}: {x.size()}"
                )
            kept_indices = torch.arange(F_dim)
            for i in self.s_patchout_f_indices:
                kept_indices = kept_indices[kept_indices != int(i)]
            x = x[:, :, kept_indices, :]
            if layers.first_RUN:
                _logger.debug(f"X after freq Patchout: {x.size()}")

        if self.s_patchout_f_interleaved:
            if layers.first_RUN:
                _logger.debug(
                    "WARNING!! Applying freq patchout interleaved feature extraction"
                )
            if layers.first_RUN:
                _logger.debug(
                    f"X Before freq Patchout of {self.s_patchout_t_interleaved} bands: {x.size()}",
                )
            kept_indices = torch.arange(0, F_dim, self.s_patchout_f_interleaved)
            x = x[:, :, kept_indices, :]
            if layers.first_RUN:
                _logger.debug(f"X after temp Patchout: {x.size()}")

        if self.s_patchout_t_indices:
            if layers.first_RUN:
                _logger.debug(
                    "WARNING!! Applying temp patchout indices on feature extraction"
                )
            if layers.first_RUN:
                _logger.debug(
                    f"X Before temp Patchout of bands {self.s_patchout_t_indices}: {x.size()}"
                )
            kept_indices = torch.arange(T_dim)
            for i in self.s_patchout_t_indices:
                kept_indices = kept_indices[kept_indices != int(i)]
            x = x[:, :, :, kept_indices]
            if layers.first_RUN:
                _logger.debug(f"X after temp Patchout: {x.size()}")

        if self.s_patchout_t_interleaved:
            if layers.first_RUN:
                _logger.debug(
                    "WARNING!! Applying temp patchout interleaved on feature extraction"
                )
            if layers.first_RUN:
                _logger.debug(
                    f"X Before temp Patchout of {self.s_patchout_t_interleaved} bands",
                    x.size(),
                )
            kept_indices = torch.arange(0, T_dim, self.s_patchout_t_interleaved)
            x = x[:, :, :, kept_indices]
            if layers.first_RUN:
                _logger.debug(f"X after temp Patchout: {x.size()}")

        x = x.flatten(2).transpose(1, 2)
        if layers.first_RUN:
            _logger.debug(f"X flattened {x.size()}")
        if self.training and self.u_patchout:
            seq_len = x.shape[1]
            random_indices = (
                torch.randperm(seq_len)[: seq_len - self.u_patchout].sort().values
            )
            x = x[:, random_indices, :]
            if layers.first_RUN:
                _logger.debug(f"X After Unstructured Patchout: {x.size()}")

        if layers.first_RUN:
            _logger.debug(f"self.new_pos_embed.shape: {self.new_pos_embed.shape}")
        cls_tokens = self.cls_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, :1, :]
        if layers.first_RUN:
            _logger.debug("self.cls_tokens.shape: {cls_tokens.shape}")
        if self.dist_token is None:
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            dist_token = (
                self.dist_token.expand(B_dim, -1, -1) + self.new_pos_embed[:, 1:, :]
            )
            if layers.first_RUN:
                _logger.debug(f"self.dist_token.shape {dist_token.shape}")
            x = torch.cat((cls_tokens, dist_token, x), dim=1)

        if layers.first_RUN:
            _logger.debug(f"final sequence x: {x.shape}")
        x = self.pos_drop(x)

        layers.first_RUN = False

        if transformer_block == -1:
            x = self.blocks(x)
            x = self.norm(x)
            if self.dist_token is None:
                return self.pre_logits(x[:, 0])
            else:
                return x[:, 0], x[:, 1]
        else:
            for i, block in enumerate(self.blocks):
                if i == transformer_block:
                    if layers.first_RUN:
                        _logger.debug(f"returning self-attention from block {i}")
                    x = block(x, return_self_attention=return_self_attention)
                    break
                else:
                    x = block(x)

            if layers.first_RUN:
                _logger.debug(f"after {len(self.blocks)} atten blocks x {x.shape}")

            cls = x[:, 0, :]
            dist = x[:, 1, :]
            feats = torch.mean(x[:, 2:, :], dim=1)
            layers.first_RUN = False
            return torch.cat([cls, dist, feats], dim=1)

    def forward(
        self,
        x,
        transformer_block: int = -1,
        return_self_attention: bool = False,
        melspectrogram_input: bool = False,
    ) -> Tuple[torch.Tensor | None, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x: Input tensor, either raw audio or melspectrogram.
            transformer_block: If -1, returns the output of the last block.
                If >= 0, returns the output of the specified block.
            return_self_attention: If True, returns self-attention from the specified block.
            melspectrogram_input: If True, input is expected to be a melspectrogram.

        Returns:
            activations: Output tensor with the model's predictions. Requires transformer_block to be -1.
            embeddings: Output tensor with the model's embeddings of the selected transformer block.
        """
        if layers.first_RUN:
            _logger.debug(f"x size: {len(x)}")

        assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
        assert x.nelement() > 0, "Input tensor must not be empty"

        if len(x.shape) == 1:
            assert melspectrogram_input is False, (
                "Input is 1D, but melspectrogram_input is True. This is not supported."
            )
            _logger.debug("extracting melspec")
            if self.melspectrogram is None:
                self.init_melspectrogram()
            x = self.melspectrogram(x)
            if x.shape[1] >= self.img_size[1]:
                trim = x.shape[1] % self.img_size[1]
                if trim:
                    x = x[:, :-trim]
                x = x.reshape(self.img_size[0], 1, -1, self.img_size[1])
                x = torch.swapaxes(x, 0, 2)
            else:
                x = x.reshape(1, 1, x.shape[0], x.shape[1])

        elif len(x.shape) == 2 and melspectrogram_input:
            trim = x.shape[1] % self.img_size[1]
            if trim:
                x = x[:, :-trim]
            x = x.reshape(self.img_size[0], 1, -1, self.img_size[1])
            x = np.swapaxes(x, 0, 2)

        elif len(x.shape) == 2 and not melspectrogram_input:
            if self.melspectrogram is None:
                self.init_melspectrogram()
            x = self.melspectrogram(x)
            x.unsqueeze_(1)

        elif len(x.shape) == 3:
            x.unsqueeze_(1)

        x = self.forward_features(
            x,
            transformer_block=transformer_block,
            return_self_attention=return_self_attention,
        )
        if transformer_block != -1:
            return None, x

        if self.distilled_type == "mean":
            features = (x[0] + x[1]) / 2
            if layers.first_RUN:
                _logger.debug(f"features size: {features.size()}")
            x = self.head(features)
            if layers.first_RUN:
                _logger.debug(f"head size: {x.size()}")
            layers.first_RUN = False
            return x, features
        elif self.distilled_type == "separated":
            features = (x[0] + x[1]) / 2
            if layers.first_RUN:
                _logger.debug(f"features size: {features.size()}")
            x_cls = self.head(x[0])
            if layers.first_RUN:
                _logger.debug(f"head cls size: {x_cls.size()}")
            x_dist = self.head_dist(x[1])
            if layers.first_RUN:
                _logger.debug(f"head dist size: {x_dist.size()}")
            layers.first_RUN = False
            return x_cls, x_dist, features
        else:
            features = x
            if layers.first_RUN:
                _logger.debug(f"features size: {features.size()}")
            x = self.head(x)
        if layers.first_RUN:
            _logger.debug(f"head size: {x.size()}")
        return x, features

    def predict_labels(self, x):
        """Predict labels with sigmoid activations."""
        logits = self.forward(x)[0]
        activations = nn.functional.sigmoid(logits)
        activations = torch.mean(activations, dim=0)
        return activations.detach().cpu().numpy(), self.labels


def _init_vit_weights(
    module: nn.Module, name: str = "", head_bias: float = 0.0, jax_impl: bool = False
):
    """ViT weight initialization."""
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith("pre_logits"):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if "mlp" in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")
