"""Smoke tests: the package imports cleanly and the model registry is intact.

Fast, no-network sanity checks -- catches an obviously broken package (a
missing module, a version string that didn't get wired up, a registry that
lost an entry during the P4 file split) before any model-loading test runs.

Reads: maest_infer, maest_infer.configs, maest_infer.maest (shim)
"""

import maest_infer


def test_package_imports():
    assert hasattr(maest_infer, "get_maest")
    assert hasattr(maest_infer, "MAEST")


def test_version_is_a_string():
    assert isinstance(maest_infer.__version__, str)
    assert maest_infer.__version__  # non-empty


def test_shim_module_preserves_import_paths():
    """`from maest_infer.maest import get_maest` etc. must keep working post-P4."""
    from maest_infer.maest import MAEST, get_maest

    assert get_maest is maest_infer.get_maest
    assert MAEST is maest_infer.MAEST


def test_model_registry_lists_ten_configs():
    from maest_infer.configs import ARCH_TO_FACTORY_NAME, default_cfgs

    assert len(default_cfgs) == 10
    assert len(ARCH_TO_FACTORY_NAME) == 10


def test_readme_documented_archs_are_registered():
    """The 8 discogs_maest_* variants documented in README.md's model table."""
    from maest_infer.configs import ARCH_TO_FACTORY_NAME

    documented_archs = [
        "discogs-maest-5s-pw-129e",
        "discogs-maest-10s-fs-129e",
        "discogs-maest-10s-pw-129e",
        "discogs-maest-10s-dw-75e",
        "discogs-maest-20s-pw-129e",
        "discogs-maest-30s-pw-129e",
        "discogs-maest-30s-pw-73e-ts",
        "discogs-maest-30s-pw-129e-519l",
    ]
    for arch in documented_archs:
        assert arch in ARCH_TO_FACTORY_NAME, f"{arch} missing from registry"
