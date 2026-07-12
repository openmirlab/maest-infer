# Changelog

All notable changes to maest-infer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## CI matrix + dependency floor refresh (2026-07-12, branch `fix/ci-matrix`)

Org audit found this repo tests only Python 3.10, and only inside
`publish.yml`'s release-gating job -- there was no dedicated `test.yml`
running on every push/PR, even though the classifiers claimed 3.10-3.12
support. This change builds that CI matrix and refreshes the stale
`torch`/`torchaudio`/`numpy` floors (org constitution article 3:
"floors, not ceilings"). Dependabot: 0 open alerts (reconfirmed), no
security fixes needed.

### Added
- **`.github/workflows/test.yml`**: a `test` job running the full `pytest`
  suite across a Python 3.10/3.11/3.12/3.13 matrix via `uv` (all four
  verified independently green with a genuinely fresh `uv sync` + `pytest -v`
  run before being added to the matrix -- torchaudio's newest release lags
  torch's by two minor versions, so 3.13 support hinged on whether a
  compatible wheel pair existed there too; it did), plus a `build` job
  (`needs: [test]`) doing the wheel-from-sdist build, a clean-venv install,
  an import smoke test (`from maest_infer import get_maest, MAEST,
  __version__`), and a check that the wheel contains `maest_infer/*.py`
  (not `src/maest_infer/*.py` -- hatchling strips the prefix correctly,
  confirmed by unzipping the built wheel) with no bundled checkpoint files
  (org constitution article 7).
- `Programming Language :: Python :: 3.13` classifier (3.13 verified green).

### Changed
- **`torch>=2.0` -> `>=2.11.0`, `torchaudio>=2.0` -> `>=2.11.0`** (not
  torch's absolute latest, `2.13.0`): torchaudio's newest PyPI release is
  `2.11.0`, two minor versions behind torch's `2.13.0`, and torchaudio's
  own metadata pins an exact matching torch version through `2.10.0`
  (`torchaudio==2.10.0` requires `torch==2.10.0`); `2.11.0` is the newest
  release where both packages exist as a matched pair, so the floor tracks
  that pairing rather than chasing torch's number alone. Note: `uv`'s
  resolver, given only these floors with no upper bound, actually resolves
  fresh installs to `torch==2.13.0` + `torchaudio==2.11.0` (torchaudio
  2.11.0 dropped the exact-pin metadata, so nothing stops the resolver
  reaching past it on the torch side) -- manually verified this specific
  mismatched-version pair still imports and runs `torchaudio.transforms`
  correctly, so it is not treated as a blocker, only documented here per
  the "floors not ceilings" policy (no upper bound added).
- **`numpy>=1.26` -> `>=2.2.6`**: the numpy/Python-version trap --
  `numpy>=2.3` requires Python>=3.11 and `numpy>=2.5` requires
  Python>=3.12, which would break this repo's Python>=3.10 floor if used
  as the pyproject floor. `2.2.6` is the newest 2.x release still
  compatible with Python 3.10; `uv`'s resolver still floats higher (up to
  `2.5.1`) on the 3.11/3.12/3.13 legs of the lockfile's per-Python split,
  same pattern as larsnet-infer's numpy floor fix.
- **Accuracy check before committing to the torch/torchaudio bump**: reran
  `tools/capture_baseline.py`'s exact clip synthesis + model forward pass
  under torch `2.13.0+cu130`/torchaudio `2.11.0+cu130` and diffed every
  array against the recorded fixture (captured on torch `2.9.1+cu128`).
  Mel-spectrogram output is bit-identical (`max_abs_diff=0.0`); embeddings,
  logits, and label probabilities differ only by ~1e-6 to ~6e-6 absolute
  (float-kernel ULP noise across torch builds, not a real regression);
  predicted labels match exactly. This is the expected, org-sanctioned
  outcome of article 2's environment guard: `tests/test_baseline_regression.py`'s
  bit-exact assertion now **skips** (rather than fails) once the installed
  torch no longer matches the fixture's recorded `torch_version` --
  visible as the test-count shift below.
- `uv.lock` regenerated (48 packages resolved); the file itself stays
  gitignored, as before, and is not part of this commit.

### Test counts
- Before (current committed floors, `torch>=2.0`/`torchaudio>=2.0`/`numpy>=1.26`,
  fresh `uv sync`): **7 passed, 10 deselected** on all of Python
  3.10/3.11/3.12/3.13 -- the resolver's chosen torch build (`2.9.1+cu128`)
  happened to exactly match the recorded baseline fixture's
  `torch_version`, so the bit-exact regression test ran (rather than
  skipped) and passed. This differs from the prior audit's noted baseline
  of "6 passed / 1 skipped / 10 deselected" purely because of which torch
  build the resolver picked up at verification time -- not a behavior
  change in this repo.
- After (new floors, fresh `uv sync`, torch `2.13.0`/torchaudio `2.11.0`):
  **6 passed, 1 skipped, 10 deselected** on all four Python versions -- the
  1 skip is `test_rerun_matches_baseline_bit_for_bit`, self-skipping per
  the environment guard above, not a failure.

## [0.2.0] - 2026-07-11

ADOPT campaign (openmirlab modernization, `feat/adopt-constitution`) --
accuracy-preserving refactor gated on a numerical baseline at every step.

### Added
- `tools/capture_baseline.py`: deterministic seeded-audio baseline fixture
  (`tests/fixtures/baseline_discogs-maest-5s-pw-129e.npz`) capturing
  mel-spectrogram, embeddings, logits, and top-20 label probabilities.
  Verified deterministic both within-process and across separate process
  invocations; used to prove every later change in this release is
  output-identical.
- `src/maest_infer/data/checkpoints.json`: url/sha256/size/original_author
  provenance for all 10 third-party checkpoints (MAEST, PaSST, DeiT
  releases). `helpers/vit_helpers.load_pretrained` now verifies a
  downloaded checkpoint's sha256 against this registry whenever one is on
  record, raising on a mismatch (corrupted download or a swapped mirror).
- `tools/check_weights_liveness.py` + `tests/test_weights_liveness.py`: a
  `pytest.mark.network`-gated liveness check (deselected by default; run
  with `pytest -m network`) that HEADs every checkpoint URL.
- `tests/test_import.py` and `tests/test_baseline_regression.py`: smoke
  tests (registry has all 10 configs, package imports) and a bit-for-bit
  regression test against the baseline fixture.
- `src/maest_infer/__about__.py`: single-sourced version, read by
  hatchling's `[tool.hatch.version]` and re-exported from `__init__.py`.
- `NOTICE` and this repo's `CLAUDE.md` (scope, module layout, file-top
  header convention, verification commands/status).
- A `test` job in `.github/workflows/publish.yml`, now required (`needs:
  [test]`) before `publish` runs.

### Changed
- **Removed the `timm` dependency entirely.** Traced its only usage
  (`timm.models.load_pretrained`) and found it genuinely load-bearing --
  contrary to earlier suspicion that it was dead code, it is the mechanism
  that downloads every pretrained checkpoint and adapts its position
  embeddings. Vendored the one code path MAEST actually exercises
  (URL-sourced checkpoints only) into `helpers/vit_helpers.py` instead of
  raising timm's version ceiling. Verified bit-identical against the
  baseline with timm and its transitive dependencies fully uninstalled.
- **Split the 1410-line `maest.py`** into `configs.py` (model configs +
  variant registry), `layers.py` (ViT building blocks), `model.py` (the
  `MAEST` class), and `loading.py` (checkpoint loading, factories,
  `get_maest`). `maest.py` is now a thin re-export shim -- every previously
  importable name still resolves. Collapsed the 8 near-duplicate
  `discogs_maest_*` factory functions into one registry-driven builder,
  keeping the 8 names as thin aliases (public API per README). Pure
  rearrangement, verified bit-identical.
- Replaced `helpers/melspectrogram.py`'s assumed mel-fidelity claim ("we
  assume minimal impact") with a measured one: final-embedding cosine
  similarity >=0.999 and 100% top-5 label agreement vs Essentia-derived
  features across 4 clips including real music, though only ~74-97% of
  individual mel bins meet the stricter per-bin tolerance claim. Docstring
  only, no numerics touched.
- `pyproject.toml`: version is now `dynamic`, sourced from `__about__.py`;
  added a `maintainers` entry (upstream author stays in `authors`); packaged
  `data/*.json`; added the `network` pytest marker (deselected by default).
- `LICENSE`: filled the previously unfilled AGPL-3.0 copyright header with
  dual attribution -- original MAEST work (Pablo Alonso-Jimenez / MTG-UPF)
  and this packaging (openmirlab).

### Not changed (deliberately, out of scope for this campaign)
- `helpers/melspectrogram.py`'s actual mel-spectrogram numerics.
- The pre-existing `passt_deit_bd_p16_384` arch construction failure
  (reproduced identically on the pre-refactor code -- an inherited upstream
  issue).
- `loading.fix_embedding_layer`'s unreachable `"overlap"`/`"am_keepconv"`
  branches, which reference classes that don't exist anywhere in this
  package (carried over verbatim from the original file).
