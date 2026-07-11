"""MAEST model configs: timm-style default_cfgs plus the variant registry.

Holds the per-checkpoint `default_cfg` dicts (url, mean/std, input_size,
classifier keys, ...) consumed by `loading.build_model_with_cfg`/
`load_pretrained`, and `DISCOGS_VARIANTS`: the data-driven registry that
replaced the 8 near-duplicate `discogs_maest_*` factory functions in the
pre-split maest.py (see loading.py). Pure data -- no torch imports needed.

Reads: (leaf module, no local imports); read by maest_infer.loading,
maest_infer.model
"""

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
DISCOGS_MEAN = 2.06755686098554
DISCOGS_STD = 1.268292820667291


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "passt_s_swa_p16_128_ap476": _cfg(
        url="https://github.com/kkoutini/PaSST/releases/download/v0.0.1-audioset/passt-s-f128-p16-s10-ap.476-swa.pt",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(1, 128, 998),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=527,
    ),
    "deit_base_distilled_patch16_384": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "discogs_maest_10s_fs_129e": _cfg(
        url="https://github.com/palonso/MAEST/releases/download/v0.0.0-beta/discogs-maest-10s-fs-129e-swa.ckpt",
        mean=DISCOGS_MEAN,
        std=DISCOGS_STD,
        input_size=(1, 96, 625),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=400,
    ),
    "discogs_maest_10s_dw_75e": _cfg(
        url="https://github.com/palonso/MAEST/releases/download/v0.0.0-beta/discogs-maest-10s-dw-75e-swa.ckpt",
        mean=DISCOGS_MEAN,
        std=DISCOGS_STD,
        input_size=(1, 96, 625),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=400,
    ),
    "discogs_maest_10s_pw_129e": _cfg(
        url="https://github.com/palonso/MAEST/releases/download/v0.0.0-beta/discogs-maest-10s-pw-129e-swa.ckpt",
        mean=DISCOGS_MEAN,
        std=DISCOGS_STD,
        input_size=(1, 96, 625),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=400,
    ),
    "discogs_maest_5s_pw_129e": _cfg(
        url="https://github.com/palonso/MAEST/releases/download/v0.0.0-beta/discogs-maest-5s-pw-129e-swa.ckpt",
        mean=DISCOGS_MEAN,
        std=DISCOGS_STD,
        input_size=(1, 96, 312),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=400,
    ),
    "discogs_maest_20s_pw_129e": _cfg(
        url="https://github.com/palonso/MAEST/releases/download/v0.0.0-beta/discogs-maest-20s-pw-129e-swa.ckpt",
        mean=DISCOGS_MEAN,
        std=DISCOGS_STD,
        input_size=(1, 128, 1250),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=400,
    ),
    "discogs_maest_30s_pw_129e": _cfg(
        url="https://github.com/palonso/MAEST/releases/download/v0.0.0-beta/discogs-maest-30s-pw-129e-swa.ckpt",
        mean=DISCOGS_MEAN,
        std=DISCOGS_STD,
        input_size=(1, 128, 1875),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=400,
    ),
    "discogs_maest_30s_pw_73e_ts": _cfg(
        url="https://github.com/palonso/MAEST/releases/download/v0.0.0-beta/discogs-maest-30s-pw-73e-ts-swa.ckpt",
        mean=DISCOGS_MEAN,
        std=DISCOGS_STD,
        input_size=(1, 128, 1875),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=400,
    ),
    "discogs_maest_30s_pw_129e_519l": _cfg(
        url="https://github.com/palonso/MAEST/releases/download/v0.0.0-beta/discogs-maest-30s-pw-129e-519l-swa.ckpt",
        mean=DISCOGS_MEAN,
        std=DISCOGS_STD,
        input_size=(1, 128, 1875),
        crop_pct=1.0,
        classifier=("head.1", "head_dist"),
        num_classes=519,
    ),
}


# Registry-driven replacement for the 8 near-duplicate `discogs_maest_*`
# factory functions (public API per README -- kept as thin aliases in
# loading.py). Every entry captures the one piece of per-variant state the
# original functions varied: the default time-bin count substituted when the
# caller doesn't set img_size[1], the debug log line, and (only for the
# 519-label variant) the forced num_classes. `passt_s_swa_p16_128_ap476` and
# `deit_base_distilled_patch16_384` are NOT in this registry: they are not
# part of the public README table, and deit_base_distilled_patch16_384's
# original body differs in ways (no stride-mismatch warning, different
# statement order) that make folding it into this template risky for no
# benefit -- both are kept as standalone functions in loading.py, copied
# verbatim from maest.py.
DISCOGS_VARIANTS = {
    "discogs_maest_10s_fs_129e": {
        "default_time_dim": 625,
        "log_message": "Loading MAEST 10s from scratch",
        "force_num_classes": None,
    },
    "discogs_maest_10s_pw_129e": {
        "default_time_dim": 625,
        "log_message": "Loading MAEST 10s PaSST weights",
        "force_num_classes": None,
    },
    "discogs_maest_10s_dw_75e": {
        "default_time_dim": 625,
        "log_message": "Loading MAEST 10s DeiT weights",
        "force_num_classes": None,
    },
    "discogs_maest_5s_pw_129e": {
        "default_time_dim": 312,
        "log_message": "Loading MAEST 5s PaSST weights",
        "force_num_classes": None,
    },
    "discogs_maest_20s_pw_129e": {
        "default_time_dim": 1250,
        "log_message": "Loading MAEST 20s PaSST weights",
        "force_num_classes": None,
    },
    "discogs_maest_30s_pw_129e": {
        "default_time_dim": 1875,
        "log_message": "Loading MAEST 30s PaSST weights",
        "force_num_classes": None,
    },
    "discogs_maest_30s_pw_73e_ts": {
        "default_time_dim": 1875,
        "log_message": "Loading MAEST 30s teacher-student",
        "force_num_classes": None,
    },
    "discogs_maest_30s_pw_129e_519l": {
        "default_time_dim": 1875,
        "log_message": "Loading MAEST 30s 519 labels",
        "force_num_classes": 519,
    },
}


# get_maest()'s public `arch` string -> underlying factory-function name.
# Preserves the exact original if/elif dispatch table as a dict lookup.
ARCH_TO_FACTORY_NAME = {
    "passt_deit_bd_p16_384": "deit_base_distilled_patch16_384",
    "passt_s_swa_p16_128_ap476": "passt_s_swa_p16_128_ap476",
    "discogs-maest-10s-fs-129e": "discogs_maest_10s_fs_129e",
    "discogs-maest-10s-pw-129e": "discogs_maest_10s_pw_129e",
    "discogs-maest-10s-dw-75e": "discogs_maest_10s_dw_75e",
    "discogs-maest-5s-pw-129e": "discogs_maest_5s_pw_129e",
    "discogs-maest-20s-pw-129e": "discogs_maest_20s_pw_129e",
    "discogs-maest-30s-pw-129e": "discogs_maest_30s_pw_129e",
    "discogs-maest-30s-pw-73e-ts": "discogs_maest_30s_pw_73e_ts",
    "discogs-maest-30s-pw-129e-519l": "discogs_maest_30s_pw_129e_519l",
}
