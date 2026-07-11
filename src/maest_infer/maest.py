"""Backward-compatible re-export shim for the pre-split maest.py module.

Before the P4 restructuring (ADOPT campaign, feat/adopt-constitution),
everything in this package lived in one 1410-line maest.py. It's now split
into configs.py (model configs + variant registry), layers.py (ViT building
blocks), model.py (the MAEST class), and loading.py (checkpoint loading +
factories + get_maest). This module re-exports every name that used to be
importable from `maest_infer.maest` so `from maest_infer.maest import
get_maest`, `from maest_infer.maest import MAEST`, etc. keep working
unchanged -- pure rearrangement, verified bit-identical against the P0
baseline.

Reads: maest_infer.configs, maest_infer.layers, maest_infer.model,
maest_infer.loading
"""

import logging

from .configs import (  # noqa: F401
    DISCOGS_MEAN,
    DISCOGS_STD,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
    _cfg,
    default_cfgs,
)
from .layers import (  # noqa: F401
    Attention,
    Block,
    Mlp,
    PatchEmbed,
    PatchEmbedFreq,
    replacePatchEmbedFreq,
    to_2tuple,
)
from .loading import (  # noqa: F401
    _create_vision_transformer,
    adapt_image_pos_embed_to_passt,
    adapt_input_conv,
    adapt_passt_timefreq_embed,
    checkpoint_filter_fn,
    deit_base_distilled_patch16_384,
    discogs_maest_5s_pw_129e,
    discogs_maest_10s_dw_75e,
    discogs_maest_10s_fs_129e,
    discogs_maest_10s_pw_129e,
    discogs_maest_20s_pw_129e,
    discogs_maest_30s_pw_129e,
    discogs_maest_30s_pw_73e_ts,
    discogs_maest_30s_pw_129e_519l,
    fix_embedding_layer,
    get_maest,
    lighten_model,
    passt_s_swa_p16_128_ap476,
    resize_pos_embed,
)
from .model import MAEST, _init_vit_weights, lecun_normal_  # noqa: F401

_logger = logging.getLogger("MAEST")

# Historical module-level flag from the pre-split file (debug-log gating
# only, never affects numerics). The live, shared flag actually read/written
# by layers.PatchEmbed*/model.MAEST at runtime is `layers.first_RUN`; this
# name is kept here only so `from maest_infer.maest import first_RUN` still
# resolves, matching the old single-file layout.
from .layers import first_RUN  # noqa: F401,E402
