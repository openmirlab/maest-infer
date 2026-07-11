# maest-infer -- CLAUDE.md

## Scope

maest-infer is an inference-only repackaging of [MAEST](https://github.com/palonso/maest)
(Music Audio Efficient Spectrogram Transformer, Alonso-Jimenez et al., ISMIR
2023) -- no training code, no Essentia dependency at inference time. It wraps
10 pretrained checkpoint variants (5s/10s/20s/30s inputs, 400 or 519 Discogs
labels) behind a single `get_maest(arch=...)` entry point. See README.md for
the public API and the model table.

## Module layout (post-ADOPT split)

The original single 1410-line `maest.py` is now split by concern:

- `configs.py` -- timm-style `default_cfgs` per checkpoint + `DISCOGS_VARIANTS`
  / `ARCH_TO_FACTORY_NAME` registries (data only, no torch).
- `layers.py` -- ViT building blocks: `Mlp`, `PatchEmbed`, `PatchEmbedFreq`,
  `Attention`, `Block`, plus the shared `first_RUN` debug-log flag.
- `model.py` -- the `MAEST` nn.Module itself (patch embed -> encoder ->
  distilled head) and its construction-time weight-init helpers.
- `loading.py` -- checkpoint pos-embed adaptation, the registry-driven model
  factories (the 8 public `discogs_maest_*` names are thin aliases over one
  generic builder), and `get_maest()`.
- `maest.py` -- a thin re-export shim: `from maest_infer.maest import
  get_maest` (and every other name that used to live in the monolithic file)
  still works unchanged.
- `helpers/vit_helpers.py` -- vendored ViT-init math (`trunc_normal_`,
  `DropPath`, ...) plus a trimmed, URL-only vendored copy of
  `timm.models._builder.load_pretrained` (see "timm dependency" below).
- `helpers/melspectrogram.py` -- the torchaudio mel-spectrogram front end.
- `data/checkpoints.json` -- provenance (url/sha256/size/author) for all 10
  third-party checkpoints, consulted by `vit_helpers.load_pretrained` for
  integrity checking whenever a hash is on record.

## File-top header convention

Every module starts with a header of this shape (as the module docstring):

```python
"""One-line title.

2-3 sentences: what this file is for and *why* it exists this way -- the
design constraint or decision it embodies, not just a restatement of the
code.

Reads: <files/modules this one depends on>; read by: <files that depend on
this one>, where useful
"""
```

Keep headers in sync as files change -- this is what lets a session (or the
`/nav:sync` skill) grasp any file from its first ~12 lines without reading
the whole thing.

## timm dependency: removed, not just version-bumped

Earlier evidence suggested `timm.models.load_pretrained` (the package's only
timm usage) was "possibly never called." That was wrong: `get_maest(...,
pretrained=True)` (the default) calls it on every load to download the
checkpoint and adapt its position embeddings. Rather than keep timm pinned or
raise its ceiling, `vit_helpers.load_pretrained` now vendors the one code
path MAEST actually exercises (URL-sourced checkpoints only -- MAEST's 10
`default_cfgs` never use `hf_hub_id`/`file`/`state_dict`/`custom_load`), so
timm is no longer a dependency at all. Unsupported config shapes raise
`RuntimeError` instead of silently diverging from timm's original behavior.

## Verification status

- **Bit-identical to upstream MAEST** (verified 2026-07, matched torch
  2.9.1): mel/embeddings/logits max|delta|=0.0 against the original
  palonso/MAEST implementation.
- **Refactor baseline**: `tests/fixtures/baseline_discogs-maest-5s-pw-129e.npz`
  (captured by `tools/capture_baseline.py`, deterministic seed, verified
  reproducible both within-process and across separate process
  invocations) gates every structural change in this repo -- P1 (timm
  removal), P3 (checkpoint-integrity wiring), and P4 (the file split above)
  all re-ran it and diffed bit-identical.
- **torchaudio-vs-Essentia mel deviation** (see `helpers/melspectrogram.py`):
  this is upstream MAEST's own shipped inference design (Essentia was only
  used to build the *training* dataset), not a deviation introduced by this
  package. Empirically bounded (2026-07, 4 clips incl. real music,
  discogs-maest-10s-pw-129e): final-embedding cosine similarity >=0.999 and
  100% top-5 label agreement vs Essentia-derived features, despite only
  ~74-97% of individual mel bins meeting the docstring's rtol/atol=1e-3
  claim (mismatch concentrates in the lowest mel bands and edge frames).

## Verification commands

```bash
uv sync                                             # install/update the environment
uv run --with pytest python -m pytest -q            # unit tests (network tests deselected)
uv run --with pytest python -m pytest -m network -q # + live checkpoint-URL liveness check
uv run python tools/capture_baseline.py --verify-run   # re-capture + determinism check
uv run python tools/check_weights_liveness.py          # HEAD every checkpoint URL
```

## What was deliberately not touched

- `helpers/melspectrogram.py`'s numerics (the torchaudio mel implementation
  itself) -- a separate investigation quantified its deviation from
  Essentia; this campaign only updated the docstring's claim to match the
  measured result, per that investigation's findings.
- The pre-existing `passt_deit_bd_p16_384` arch string in `get_maest`'s
  dispatch table raises `RuntimeError` on construction (a patch-embed
  reshape shape mismatch) -- reproduced identically on the pre-refactor
  code, so this is an inherited upstream issue, not a regression.
- `loading.fix_embedding_layer`'s `"overlap"`/`"am_keepconv"` branches
  reference `PatchEmbedAdaptiveMean`/`PatchEmbedAdaptiveMeanKeepConv`, which
  are not defined anywhere in this package (or were in the original
  monolithic file). Unreachable in practice -- `get_maest` never passes
  `embed=` to `fix_embedding_layer`, so it always takes the `"default"`
  branch -- carried over verbatim rather than fixed, since this campaign is
  scoped to structure, not behavior changes.
