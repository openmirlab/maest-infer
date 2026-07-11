"""Vision Transformer helper utilities for MAEST.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

`load_pretrained` below is a vendored, trimmed copy of
`timm.models._builder.load_pretrained` (timm 0.9.16): MAEST's 10 default_cfgs
only ever populate a plain `url`, never `hf_hub_id`/`file`/`state_dict`/
`custom_load`/`label_offset`, so this covers exactly the one code path MAEST
actually exercises and raises loudly if a cfg ever needs more. See P1 of the
ADOPT campaign (feat/adopt-constitution) for the verification that timm's
`load_pretrained` was genuinely load-bearing (it performs the checkpoint
download + pos-embed adaptation + classifier-head resize), not dead code, and
that this vendored copy reproduces its output bit-for-bit.

Reads: maest_infer/data/checkpoints.json (checkpoint integrity registry);
read by maest_infer.loading (build_model_with_cfg), maest_infer.layers
(DropPath)
"""

import hashlib
import json
import logging
import math
import warnings
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torch.hub import get_dir as _torch_hub_get_dir
from torch.hub import load_state_dict_from_url
from torch.nn.init import _calculate_fan_in_and_fan_out

_logger = logging.getLogger("MAEST")

_CHECKPOINTS_JSON = Path(__file__).resolve().parent.parent / "data" / "checkpoints.json"
_checkpoint_registry_cache: dict | None = None


def _load_checkpoint_registry() -> dict:
    """Load data/checkpoints.json (url -> sha256/size/author), cached in-process."""
    global _checkpoint_registry_cache
    if _checkpoint_registry_cache is None:
        try:
            with open(_CHECKPOINTS_JSON, "r") as f:
                data = json.load(f)
            _checkpoint_registry_cache = {
                entry["url"]: entry for entry in data.get("checkpoints", {}).values()
            }
        except FileNotFoundError:
            _checkpoint_registry_cache = {}
    return _checkpoint_registry_cache


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checkpoint_integrity(url: str) -> None:
    """Verify the torch.hub-cached file for `url` against checkpoints.json, if known.

    Raises RuntimeError on a hash mismatch (corrupted download or a replaced
    third-party mirror). Logs and returns quietly if no hash is on record --
    this checks integrity, it does not require every URL to be pinned.
    """
    registry = _load_checkpoint_registry()
    entry = registry.get(url)
    if not entry or not entry.get("sha256"):
        return

    filename = Path(urlparse(url).path).name
    cached_path = Path(_torch_hub_get_dir()) / "checkpoints" / filename
    if not cached_path.is_file():
        _logger.debug(f"Checkpoint integrity check skipped: {cached_path} not found on disk.")
        return

    actual = _sha256_of_file(cached_path)
    expected = entry["sha256"]
    if actual != expected:
        raise RuntimeError(
            f"Checkpoint integrity check FAILED for {url}: "
            f"expected sha256={expected}, got sha256={actual}. "
            f"The downloaded file at {cached_path} does not match the recorded checksum -- "
            "it may be corrupted or the upstream mirror may have changed. Refusing to load it."
        )
    _logger.info(f"Checkpoint integrity verified for {filename} (sha256 matches checkpoints.json).")


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


def _adapt_input_conv(in_chans, conv_weight):
    """Adapt a pretrained conv's input channels (3 -> in_chans), e.g. RGB -> mono.

    Vendored from timm.models._manipulate.adapt_input_conv (0.9.16) -- identical
    algorithm to the (currently unused) copy at maest.py:adapt_input_conv.
    """
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # some weights are torch.half; ensure float for CPU sum
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


def load_pretrained(
    model,
    pretrained_cfg: dict | None = None,
    num_classes: int = 1000,
    in_chans: int = 3,
    filter_fn=None,
    strict: bool = True,
):
    """Load a pretrained checkpoint from a URL into `model` (vendored, URL-only).

    Trimmed from timm.models._builder.load_pretrained: MAEST's default_cfgs
    (see maest.py:default_cfgs) always set a plain `url` and never `hf_hub_id`,
    `file`, `state_dict`, `custom_load`, or `label_offset`, so only that one
    branch is implemented. Anything relying on the other branches raises
    RuntimeError instead of silently diverging from timm's behavior.
    """
    pretrained_cfg = pretrained_cfg or getattr(model, "pretrained_cfg", None)
    if not pretrained_cfg:
        raise RuntimeError(
            "Invalid pretrained config, cannot load weights. Use `pretrained=False` for random init."
        )

    if pretrained_cfg.get("hf_hub_id") or pretrained_cfg.get("file") or pretrained_cfg.get("state_dict"):
        raise RuntimeError(
            "This vendored load_pretrained only supports URL-sourced checkpoints "
            "(MAEST never uses hf_hub_id/file/state_dict overrides)."
        )
    pretrained_url = pretrained_cfg.get("url", None)
    if not pretrained_url:
        model_name = pretrained_cfg.get("architecture", "this model")
        raise RuntimeError(f"No pretrained weights exist for {model_name}. Use `pretrained=False` for random init.")
    if pretrained_cfg.get("custom_load", False):
        raise RuntimeError("This vendored load_pretrained does not support custom_load configs.")

    _logger.info(f"Loading pretrained weights from url ({pretrained_url})")
    state_dict = load_state_dict_from_url(
        pretrained_url,
        map_location="cpu",
        progress=False,
        check_hash=False,
    )
    _verify_checkpoint_integrity(pretrained_url)

    if filter_fn is not None:
        try:
            state_dict = filter_fn(state_dict, model)
        except TypeError:
            # backwards compat with filter fns that take one arg
            state_dict = filter_fn(state_dict)

    input_convs = pretrained_cfg.get("first_conv", None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + ".weight"
            try:
                state_dict[weight_name] = _adapt_input_conv(in_chans, state_dict[weight_name])
                _logger.info(
                    f"Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)"
                )
            except NotImplementedError:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f"Unable to convert pretrained {input_conv_name} weights, using random init for this layer."
                )

    classifiers = pretrained_cfg.get("classifier", None)
    label_offset = pretrained_cfg.get("label_offset", 0)
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != pretrained_cfg["num_classes"]:
            for classifier_name in classifiers:
                state_dict.pop(classifier_name + ".weight", None)
                state_dict.pop(classifier_name + ".bias", None)
            strict = False
        elif label_offset > 0:
            raise RuntimeError(
                "This vendored load_pretrained does not support label_offset configs "
                "(MAEST never sets label_offset)."
            )

    load_result = model.load_state_dict(state_dict, strict=strict)
    if load_result.missing_keys:
        _logger.info(
            f"Missing keys ({', '.join(load_result.missing_keys)}) discovered while loading pretrained weights."
            " This is expected if model is being adapted."
        )
    if load_result.unexpected_keys:
        _logger.warning(
            f"Unexpected keys ({', '.join(load_result.unexpected_keys)}) found while loading pretrained weights."
            " This may be expected if model is being adapted."
        )


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
