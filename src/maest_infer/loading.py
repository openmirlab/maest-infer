"""Checkpoint loading: pos-embed adaptation, the model factories, and get_maest().

Split out of the pre-split maest.py (lines ~913-1410): `resize_pos_embed` /
`adapt_image_pos_embed_to_passt` / `adapt_passt_timefreq_embed` adapt a
pretrained backbone's position embeddings to MAEST's time/frequency-split
scheme; `checkpoint_filter_fn` is the `pretrained_filter_fn` handed to
`vit_helpers.load_pretrained` (see P1: this is the callback that actually
runs during every pretrained checkpoint load). `adapt_input_conv` is carried
over unused/unreferenced, exactly as it was in the original file (dead code,
not part of this phase's scope to prune).

The 8 near-duplicate `discogs_maest_*` factory functions (identical body
shape: debug log -> default time-dim -> build kwargs -> stride-mismatch
warning -> optional forced num_classes -> `_create_vision_transformer`) are
now one registry-driven `_build_discogs_variant`, data-fed by
`configs.DISCOGS_VARIANTS`; the 8 original names are kept as thin aliases
(public API per README). `deit_base_distilled_patch16_384` and
`passt_s_swa_p16_128_ap476` are copied verbatim as standalone functions --
they are not in the README's public model table, and deit_base's body
differs in ways (no stride warning, different statement order) that made
folding it into the shared template not worth the risk. `get_maest`'s
architecture dispatch is now a dict lookup (`configs.ARCH_TO_FACTORY_NAME`)
instead of a 10-armed if/elif chain, same behavior for every input.

Reads: maest_infer.configs, maest_infer.model.MAEST, maest_infer.layers
(PatchEmbedFreq, replacePatchEmbedFreq), maest_infer.helpers.vit_helpers
(build_model_with_cfg)
"""

import logging
import math
import warnings

import torch
import torch.nn.functional as F

from .configs import ARCH_TO_FACTORY_NAME, DISCOGS_VARIANTS, default_cfgs
from .helpers.vit_helpers import build_model_with_cfg
from .layers import PatchEmbedFreq, replacePatchEmbedFreq  # noqa: F401 (PatchEmbedFreq kept for parity)
from .model import MAEST

_logger = logging.getLogger("MAEST")


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError("Weight format not supported by conversion.")
        else:
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= 3 / float(in_chans)
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=(), mode="bicubic"):
    """Rescale the grid of position embeddings when loading from state_dict."""
    _logger.info(
        "Resized position embedding: %s to %s with %s cls/dis tokens",
        posemb.shape,
        posemb_new.shape,
        num_tokens,
    )
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info("Position embedding grid-size from %s to %s", [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode=mode, align_corners=False
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def adapt_image_pos_embed_to_passt(posemb, num_tokens=1, gs_new=(), mode="bicubic"):
    """Adapt ImageNet position embeddings to PaSST format."""
    _logger.info(
        "Resized position embedding: %s to %s with %s cls/dis tokens",
        posemb.shape,
        gs_new,
        num_tokens,
    )
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))

    assert len(gs_new) >= 2
    _logger.info("Position embedding grid-size from %s to %s", [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode=mode, align_corners=False
    )
    freq_new_pos_embed = posemb_grid.mean(dim=3, keepdim=True)
    time_new_pos_embed = posemb_grid.mean(dim=2, keepdim=True)
    _logger.info("New Position cls/dstl embedding %s", posemb_tok.shape)
    _logger.info("New FREQ Position embedding %s", freq_new_pos_embed.shape)
    _logger.info("New TIME Position embedding %s", time_new_pos_embed.shape)
    return posemb_tok, freq_new_pos_embed, time_new_pos_embed


def adapt_passt_timefreq_embed(
    freqemb, timeemb, freq_gs_new=(), time_gs_new=(), mode="bicubic"
):
    """Adapt time/frequency embeddings to new dimensions."""
    freqemb_new = F.interpolate(
        freqemb, size=freq_gs_new, mode=mode, align_corners=False
    )
    timeemb_new = F.interpolate(
        timeemb, size=time_gs_new, mode=mode, align_corners=False
    )
    _logger.info("New FREQ Position embedding %s", freqemb_new.shape)
    _logger.info("New TIME Position embedding %s", timeemb_new.shape)
    return freqemb_new, timeemb_new


def checkpoint_filter_fn(state_dict, model):
    """Convert patch embedding weight from manual patchify + linear proj to conv."""
    out_dict = {}
    if "model" in state_dict:
        state_dict = state_dict["model"]
    state_dict = {k: v for k, v in state_dict.items()}
    if "time_new_pos_embed" not in state_dict:
        _logger.info("Adapting pos embedding from ImageNet pretrained model to MAEST.")
        v = state_dict.pop("pos_embed")
        (
            new_pos_embed,
            freq_new_pos_embed,
            time_new_pos_embed,
        ) = adapt_image_pos_embed_to_passt(
            v, getattr(model, "num_tokens", 1), model.patch_embed.grid_size
        )
        state_dict["new_pos_embed"] = new_pos_embed
        state_dict["freq_new_pos_embed"] = freq_new_pos_embed
        state_dict["time_new_pos_embed"] = time_new_pos_embed
    else:
        _logger.debug(f"new pos embed: {state_dict['new_pos_embed'].shape}")
        _logger.debug(f"new pos embed time: {state_dict['freq_new_pos_embed'].shape}")
        _logger.debug(f"new pos embed freq: {state_dict['time_new_pos_embed'].shape}")

        freq_old_dim = state_dict["freq_new_pos_embed"].numpy().shape[2]
        time_old_dim = state_dict["time_new_pos_embed"].numpy().shape[3]
        freq_new_dim = model.patch_embed.grid_size[0]
        time_new_dim = model.patch_embed.grid_size[1]

        if (freq_old_dim != freq_new_dim) or (time_old_dim != time_new_dim):
            _logger.info(
                "Adapting time/freq embedding from MAEST pre-trained model to a new configuration."
            )
            freq_old_pos_embed = state_dict.pop("freq_new_pos_embed")
            time_old_pos_embed = state_dict.pop("time_new_pos_embed")
            freq_new_gs = [freq_new_dim, 1]
            time_new_gs = [1, time_new_dim]
            freq_new_pos_embed, time_new_pos_embed = adapt_passt_timefreq_embed(
                freq_old_pos_embed,
                time_old_pos_embed,
                freq_new_gs,
                time_new_gs,
            )
            state_dict["freq_new_pos_embed"] = freq_new_pos_embed
            state_dict["time_new_pos_embed"] = time_new_pos_embed

    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k and len(v.shape) < 4:
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == "pos_embed" and v.shape != model.pos_embed.shape:
            v = resize_pos_embed(
                v,
                model.pos_embed,
                getattr(model, "num_tokens", 1),
                model.patch_embed.grid_size,
            )
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )

    default_num_classes = default_cfg["num_classes"]
    num_classes = kwargs.get("num_classes", default_num_classes)
    repr_size = kwargs.pop("representation_size", None)
    if repr_size is not None and num_classes != default_num_classes:
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        MAEST,
        variant,
        pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load="npz" in default_cfg["url"],
        **kwargs,
    )
    return model


def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """DeiT-base distilled model @ 384x384."""
    if not kwargs["img_size"][1]:
        kwargs["img_size"] = (kwargs["img_size"][0], 998)
    _logger.debug("Loading DEIT BASE 384")
    _logger.debug(f"Pretrained weights: {pretrained}")
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "deit_base_distilled_patch16_384",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def passt_s_swa_p16_128_ap476(pretrained=False, **kwargs):
    """PaSST pre-trained on AudioSet."""
    _logger.debug("Loading PaSST pre-trained on AudioSet")
    if not kwargs["img_size"][1]:
        kwargs["img_size"] = (kwargs["img_size"][0], 998)
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if model_kwargs.get("stride") != (10, 10):
        warnings.warn(
            f"This model was pre-trained with strides {(10, 10)}, but now you set (fstride,tstride) to {model_kwargs.get('stride')}."
        )
    model = _create_vision_transformer(
        "passt_s_swa_p16_128_ap476",
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def _build_discogs_variant(variant: str, pretrained: bool = False, **kwargs):
    """Registry-driven replacement for the 8 near-duplicate discogs_maest_* factories.

    Reproduces each original function's body exactly (see configs.DISCOGS_VARIANTS
    for the per-variant default_time_dim / log_message / force_num_classes),
    just data-driven instead of copy-pasted per variant.
    """
    spec = DISCOGS_VARIANTS[variant]
    _logger.debug(spec["log_message"])
    if not kwargs["img_size"][1]:
        kwargs["img_size"] = (kwargs["img_size"][0], spec["default_time_dim"])
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if model_kwargs.get("stride") != (10, 10):
        warnings.warn(
            f"This model was pre-trained with strides {(10, 10)}, but now you set (fstride,tstride) to {model_kwargs.get('stride')}."
        )
    force_num_classes = spec["force_num_classes"]
    if force_num_classes is not None and model_kwargs.get("num_classes") != force_num_classes:
        _logger.debug(f"Forcing `num_classes` to {force_num_classes}")
        model_kwargs["num_classes"] = force_num_classes
    model = _create_vision_transformer(
        variant,
        pretrained=pretrained,
        distilled=True,
        **model_kwargs,
    )
    return model


def discogs_maest_10s_fs_129e(pretrained=False, **kwargs):
    """MAEST pre-trained on Discogs data (10s, from scratch)."""
    return _build_discogs_variant("discogs_maest_10s_fs_129e", pretrained=pretrained, **kwargs)


def discogs_maest_10s_pw_129e(pretrained=False, **kwargs):
    """MAEST pre-trained on Discogs data (10s, PaSST weights)."""
    return _build_discogs_variant("discogs_maest_10s_pw_129e", pretrained=pretrained, **kwargs)


def discogs_maest_10s_dw_75e(pretrained=False, **kwargs):
    """MAEST pre-trained on Discogs data (10s, DeiT weights)."""
    return _build_discogs_variant("discogs_maest_10s_dw_75e", pretrained=pretrained, **kwargs)


def discogs_maest_5s_pw_129e(pretrained=False, **kwargs):
    """MAEST pre-trained on Discogs data (5s, PaSST weights)."""
    return _build_discogs_variant("discogs_maest_5s_pw_129e", pretrained=pretrained, **kwargs)


def discogs_maest_20s_pw_129e(pretrained=False, **kwargs):
    """MAEST pre-trained on Discogs data (20s, PaSST weights)."""
    return _build_discogs_variant("discogs_maest_20s_pw_129e", pretrained=pretrained, **kwargs)


def discogs_maest_30s_pw_129e(pretrained=False, **kwargs):
    """MAEST pre-trained on Discogs data (30s, PaSST weights)."""
    return _build_discogs_variant("discogs_maest_30s_pw_129e", pretrained=pretrained, **kwargs)


def discogs_maest_30s_pw_73e_ts(pretrained=False, **kwargs):
    """MAEST pre-trained on Discogs data (30s, teacher-student)."""
    return _build_discogs_variant("discogs_maest_30s_pw_73e_ts", pretrained=pretrained, **kwargs)


def discogs_maest_30s_pw_129e_519l(pretrained=False, **kwargs):
    """MAEST pre-trained on Discogs data (30s, 519 labels)."""
    return _build_discogs_variant("discogs_maest_30s_pw_129e_519l", pretrained=pretrained, **kwargs)


# arch string (as passed to get_maest) -> factory function, built after every
# factory above is defined so this can just look them up by name.
_FACTORY_FUNCTIONS = {
    "deit_base_distilled_patch16_384": deit_base_distilled_patch16_384,
    "passt_s_swa_p16_128_ap476": passt_s_swa_p16_128_ap476,
    "discogs_maest_10s_fs_129e": discogs_maest_10s_fs_129e,
    "discogs_maest_10s_pw_129e": discogs_maest_10s_pw_129e,
    "discogs_maest_10s_dw_75e": discogs_maest_10s_dw_75e,
    "discogs_maest_5s_pw_129e": discogs_maest_5s_pw_129e,
    "discogs_maest_20s_pw_129e": discogs_maest_20s_pw_129e,
    "discogs_maest_30s_pw_129e": discogs_maest_30s_pw_129e,
    "discogs_maest_30s_pw_73e_ts": discogs_maest_30s_pw_73e_ts,
    "discogs_maest_30s_pw_129e_519l": discogs_maest_30s_pw_129e_519l,
}


def fix_embedding_layer(model, embed="default"):
    if embed == "default":
        return model
    if embed == "overlap":
        model.patch_embed = PatchEmbedAdaptiveMean(replace=model.patch_embed)
    if embed == "am_keepconv":
        model.patch_embed = PatchEmbedAdaptiveMeanKeepConv(replace=model.patch_embed)
    if embed == "freq_embed":
        model.patch_embed = replacePatchEmbedFreq(model.patch_embed)
    return model


def lighten_model(model, cut_depth=0, remove_n_blocks=0):
    """Reduce model depth by removing layers."""
    if cut_depth:
        if cut_depth < 0:
            _logger.debug(f"Reducing model depth by removing every {-cut_depth} layer")
        else:
            _logger.debug(f"Reducing model depth by {cut_depth}")
            if len(model.blocks) < cut_depth + 2:
                raise ValueError(
                    f"Cut depth a VIT with {len(model.blocks)} "
                    f"layers should be between 1 and {len(model.blocks) - 2}"
                )
        _logger.debug(f"\n Before Cutting it was  {len(model.blocks)} \n\n")
        old_blocks = list(model.blocks.children())
        if cut_depth < 0:
            _logger.debug(f"cut_depth={cut_depth}")
            old_blocks = (
                [old_blocks[0]] + old_blocks[1:-1:-cut_depth] + [old_blocks[-1]]
            )
        else:
            old_blocks = [old_blocks[0]] + old_blocks[cut_depth + 1 :]
        model.blocks = torch.nn.Sequential(*old_blocks)
        _logger.debug(f"After Cutting it is {len(model.blocks)}")

    if remove_n_blocks:
        old_blocks = list(model.blocks.children())
        if remove_n_blocks:
            old_blocks = old_blocks[: len(old_blocks) - remove_n_blocks]
        model.blocks = torch.nn.Sequential(*old_blocks)
        _logger.debug(f"After Cutting it is {len(model.blocks)}")

    return model


def get_maest(
    arch: str,
    pretrained: bool = True,
    n_classes: int = 400,
    in_channels: int = 1,
    stride_f: int = 10,
    stride_t: int = 10,
    input_f: int = 96,
    input_t: int = None,
    u_patchout: int = 0,
    s_patchout_t: int = 0,
    s_patchout_f: int = 0,
    s_patchout_f_indices: tuple = (),
    s_patchout_f_interleaved: int = 0,
    s_patchout_t_indices: tuple = (),
    s_patchout_t_interleaved: int = 0,
    distilled_type: str = "mean",
    checkpoint: str = None,
    checkpoint_swa_weigts: bool = True,
    checkpoint_discard_head: bool = False,
):
    """Get a MAEST model for inference.

    Args:
        arch: Model architecture name (e.g., "discogs-maest-30s-pw-129e-519l")
        pretrained: Load pretrained weights
        n_classes: Number of output classes
        in_channels: Number of input channels (1 for mono)
        stride_f: Frequency stride for patches
        stride_t: Time stride for patches
        input_f: Input frequency bins
        input_t: Input time bins
        u_patchout: Unstructured patchout (training only)
        s_patchout_t: Structured time patchout (training only)
        s_patchout_f: Structured frequency patchout (training only)
        distilled_type: Distillation type ("mean" or "separated")
        checkpoint: Path to custom checkpoint
        checkpoint_swa_weigts: Use SWA weights from checkpoint
        checkpoint_discard_head: Discard classification head from checkpoint

    Returns:
        MAEST model ready for inference
    """
    input_size = (input_f, input_t)
    stride = (stride_f, stride_t)

    factory_name = ARCH_TO_FACTORY_NAME.get(arch)
    if factory_name is None:
        raise NotImplementedError(f"model {arch} not implemented")
    model_func = _FACTORY_FUNCTIONS[factory_name]

    model = model_func(
        pretrained=pretrained,
        num_classes=n_classes,
        in_chans=in_channels,
        img_size=input_size,
        stride=stride,
        u_patchout=u_patchout,
        s_patchout_t=s_patchout_t,
        s_patchout_f=s_patchout_f,
        s_patchout_f_indices=s_patchout_f_indices,
        s_patchout_f_interleaved=s_patchout_f_interleaved,
        s_patchout_t_indices=s_patchout_t_indices,
        s_patchout_t_interleaved=s_patchout_t_interleaved,
        distilled_type=distilled_type,
    )
    model = fix_embedding_layer(model)
    model = lighten_model(model)

    if checkpoint:
        state_dict = torch.load(checkpoint)["state_dict"]
        if checkpoint_swa_weigts:
            replace_str = "net_swa."
        else:
            replace_str = ""
        state_dict = {k.replace(replace_str, ""): v for k, v in state_dict.items()}
        if checkpoint_discard_head:
            state_dict = {k: v for k, v in state_dict.items() if "head" not in k}
        model.load_state_dict(state_dict, strict=False)

    return model
