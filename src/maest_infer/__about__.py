"""Single source of truth for the package version.

Read by hatchling's `[tool.hatch.version]` at build time and re-exported by
`maest_infer/__init__.py` at import time, so the version is never duplicated
between pyproject.toml and the package itself (it used to be hardcoded in
both).

Reads: (leaf file, no imports); read by: pyproject.toml [tool.hatch.version],
maest_infer/__init__.py
"""

__version__ = "0.1.0"
