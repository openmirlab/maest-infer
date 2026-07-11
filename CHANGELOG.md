# Changelog

All notable changes to maest-infer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
