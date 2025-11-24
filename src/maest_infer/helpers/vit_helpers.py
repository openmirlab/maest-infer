"""Vision Transformer helper utilities for MAEST.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import math
import warnings
from copy import deepcopy

import torch
from timm.models import load_pretrained
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out


def overlay_external_default_cfg(default_cfg, kwargs):
    """Overlay 'external_default_cfg' in kwargs on top of default_cfg arg."""
    external_default_cfg = kwargs.pop("external_default_cfg", None)
    if external_default_cfg:
        default_cfg.pop("url", None)
        default_cfg.pop("hf_hub", None)
        default_cfg.update(external_default_cfg)


def filter_kwargs(kwargs, names):
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)


def set_default_kwargs(kwargs, names, default_cfg):
    for n in names:
        if n == "img_size":
            input_size = default_cfg.get("input_size", None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == "in_chans":
            input_size = default_cfg.get("input_size", None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        else:
            default_val = default_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, default_cfg[n])


def update_default_cfg_and_kwargs(default_cfg, kwargs, kwargs_filter):
    """Update the default_cfg and kwargs before passing to model."""
    overlay_external_default_cfg(default_cfg, kwargs)
    default_kwarg_names = ("num_classes", "global_pool", "in_chans")
    if default_cfg.get("fixed_input_size", False):
        default_kwarg_names += ("img_size",)
    set_default_kwargs(kwargs, names=default_kwarg_names, default_cfg=default_cfg)
    filter_kwargs(kwargs, names=kwargs_filter)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Truncated normal initialization."""
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated normal distribution."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def build_model_with_cfg(
    model_cls,
    variant: str,
    pretrained: bool,
    default_cfg: dict,
    model_cfg=None,
    feature_cfg=None,
    pretrained_strict: bool = True,
    pretrained_filter_fn=None,
    pretrained_custom_load=False,
    kwargs_filter=None,
    **kwargs,
):
    """Build model with specified default_cfg and optional model_cfg."""
    pruned = kwargs.pop("pruned", False)
    features = False
    feature_cfg = feature_cfg or {}
    default_cfg = deepcopy(default_cfg) if default_cfg else {}
    update_default_cfg_and_kwargs(default_cfg, kwargs, kwargs_filter)
    default_cfg.setdefault("architecture", variant)

    if kwargs.pop("features_only", False):
        features = True
        feature_cfg.setdefault("out_indices", (0, 1, 2, 3, 4))
        if "out_indices" in kwargs:
            feature_cfg["out_indices"] = kwargs.pop("out_indices")

    model = (
        model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    )
    model.pretrained_cfg = default_cfg

    num_classes_pretrained = (
        0
        if features
        else getattr(model, "num_classes", kwargs.get("num_classes", 1000))
    )

    if pretrained:
        if pretrained_custom_load:
            load_custom_pretrained(model)
        else:
            load_pretrained(
                model,
                num_classes=num_classes_pretrained,
                in_chans=kwargs.get("in_chans", 3),
                filter_fn=pretrained_filter_fn,
                strict=pretrained_strict,
            )
    return model
