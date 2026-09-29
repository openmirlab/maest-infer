# Genre analysis API and runtime probe

> Generated: 2026-09-29 · Source: Paul's branch/probe request · Grounding: fresh
> Convention seeded from Phonon's docs/blueprints/plans/2026-09-15-configurable-input-duration-ceiling.md.

## Context

The existing MAESTSession owns loading, inference and release. Preserve all existing defaults,
imports and raw inference numerics. Add task-level genre analysis on a local experimental branch;
no Phonon/Conductor edits, deployment, release, or public skill rollout.
The baseline suite is 16 passed, 1 skipped (environment-bound fixture), 10 network deselected.

## Approach

Current public mode names are `segment`, `full_track` (default), and `time_curve`.
This plan was executed before that rename; historical probe criteria below keep
their original excerpt / aggregate / timeline labels.

1. Add session.classify with segment, full_track (default), time_curve; add classify_genre one-shot.
   Use 519-label/30s architecture for the new task API. Existing session default stays 5s/400.
2. Normalize explicit-rate mono float arrays/tensors at one boundary to CPU float32/16kHz.
   File decoding belongs to the caller in this experimental surface. No new dependencies.
3. Window 5..30 integer seconds; hop 1..window (omitted equals window); top_n 1..50.
   Segment start defaults zero, rejects hop; whole-track modes reject start. Curve labels only
   in time_curve. Use end-aligned final full windows, pad only shorter-than-window audio.
4. Batch bounded windows on the session's actual device, inference_mode, native shorter-window
   forward (2D input prevents legacy 1D truncation). Keep all logits/sigmoids until aggregation.
   Coverage-weighted mean with real-time weights; padding contributes no weight.
5. JSON-compatible output: taxonomy, score_type, effective analysis parameters, sorted rankings;
   time_curve returns a timeline payload with windows and dense label series,
   default candidates = union of per-window top_n,
   selected by peak score with stable label tie break. Explicit labels select exact series.
6. Add offline contract tests, paired README/CLAUDE documentation and CHANGELOG. Inspect public
   skills but leave unreleased branch-only examples out of the installed/public guidance.
7. Probe real local music plus controlled concatenation and a tiled 600s stress input. Do not
   redistribute audio. Store provenance hashes/environment and raw measurement JSON locally.

## Critical files

| File | Responsibility |
|---|---|
| src/maest_infer/genre*.py | normalization, windows, full-score aggregation and ranked output |
| src/maest_infer/clean_api.py | additive ready-only classify entry and actual device bridge |
| src/maest_infer/__init__.py | additive export |
| tests/test_genre*.py | contract, invariants, lifecycle and failure verification |
| /tmp/maest-genre-probe-20260929/probe.py | disposable local measurement harness, outside the package |
| README.md, CLAUDE.md, CHANGELOG.md | accurate experimental API and evidence |

## Preflight architecture gate

Scoped nav-audit of the touched domains: (1) checkpoint/model complexity already hidden behind
session; (2) package exports exist; (3) model injection supplies a real test seam; (4) preserve
existing model modules, add focused postprocessing rather than a second model; (5) plain torch/
numpy, no service runtime dependency; (6) additive methods, legacy regression required;
(7) shorter-window accuracy remains unproven; (8) new load-bearing modules need purpose headers.
Warnings: session.infer is deliberately raw, not an audio normalization API; model mel frontend
is lazily created on CPU, so new GPU task path must initialize and move it explicitly. Existing
5s session default conflicts with new task's 519-label assumption: reject incompatible models
clearly, document explicit arch for reusable sessions. No structural rewrite needed.
Inference-only grep found no Trainer/DataLoader/Dataset/training-loop/evaluation-framework surface.
The old model's training-mode internals are shared architecture, not a shipped training workflow.

## Verification / pre-registered probe verdicts

- Existing suite stays green; capture current-environment raw logits before implementation and
  require bit-identical legacy output afterward. Older environment-bound fixture skip is explicit.
- Excerpt/direct sigmoid parity <=1e-6; sorted finite scores, valid coverage and ready lifecycle.
- Aggregate vs timeline same parameters: identical global rankings.
- Batch 1 vs 2: max absolute score delta <=1e-5 and identical Top-10 on the same audio.
- 5/10/20/30s native windows: successful finite outputs. Compare ranking overlap and score deltas
  descriptively; this does not certify human genre accuracy or justify claiming equivalence.
- 60 vs tiled 600s at same window/hop/batch: peak torch allocated VRAM growth <=20%.
- Real excerpt beginning vs later vs aggregate: report differences even if minimal; no assumed win.
- GPU release returns allocated memory within 16 MiB of pre-load baseline across three cycles.
- Build wheel, inspect package contents and imports. Do not run training/evaluation tooling or
  publish data, weights, branches or results externally.

## Baseline evidence

- `PYTHONPATH=src .venv/bin/python -m pytest -q`: 16 passed, 1 skipped, 10 deselected.
- `rg -n 'Trainer|DataLoader|Dataset|wandb|tensorboard|sacred|def train|def evaluate' src pyproject.toml`:
  no matches (exit 1).
- `uv build --out-dir /tmp/maest-genre-probe-20260929/dist`: wheel-from-sdist succeeds;
  13 Python modules, checkpoint TOML present, no tools/tests/probe/training modules.
- CPU first-30-second logits and embeddings saved before task changes as
  `/tmp/maest-genre-probe-20260929/legacy-before.npz`; input hash in `inputs.json`.
- Execution delegated to strong executor for the API and disposable harness. Mechanical
  executor was unavailable on this account; no work was performed by that failed attempt.

## Out of scope

Hosted HTTP schema, artifact size policy, fleet admission certification, paid calls, model
retraining, dependency upgrades, upstream numerics changes, and license adjudication.

## Outcome

Implemented and probed locally; see [findings](../thoughts/2026-09-29-genre-analysis-probe.md).
All ten runtime checks and boundary probes passed; offline suite 39 passed, 1
environment-bound skip, 10 network deselected. No hosted integration or release.
