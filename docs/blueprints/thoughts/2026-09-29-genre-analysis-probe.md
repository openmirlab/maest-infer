# Genre analysis API — probe findings

Status: experimental branch; evidence feeds the API decision, not a hosted release approval.

## Verdict

The excerpt / aggregate / timeline API is mechanically viable with one 519-label
checkpoint. Stable sorted results, complete temporal coverage, dense selected-label
curves, bounded GPU batches, and legacy compatibility passed the checks below.
Keep 30 seconds as the default context. Shorter windows run successfully but
materially change rankings; these observations do not establish their accuracy.
No Phonon provider or Conductor registration has been implemented in this branch.

Current public mode names are `segment`, `full_track` (default), and `time_curve`.
The historical probe labels in this note use the original measured names:
excerpt = segment, aggregate = full_track, timeline = time_curve. The
`time_curve` mode still returns its dense curve data under the `timeline` key.

## Method and environment

Pre-registered rules: [plan](../plans/2026-09-29-genre-analysis-probe.md).
Implementation base: `562ad3177723580401fd90030b61687eee4e233e`.
Local branch: `feat/genre-analysis-probe`; experiment date: 2026-09-29.
RTX 4090; Python process used torch `2.13.0+cu130`, torchaudio `2.11.0+cu130`,
CUDA 13.0, four CPU torch threads. Checkpoint:
`discogs-maest-30s-pw-129e-519l`, already cached locally. Batch defaults to one.

Local demo inputs (decoded with ffmpeg to mono float32 at 16 kHz):

| Source | Duration | SHA-256 of original file |
|---|---:|---|
| Nujabes — Luv(sic) Part 2 feat. Shing02 | 270.199 s | `5cccd326c73b4e0f27b47e9071641984432647f3a2c5462780fecefa3cea844c` |
| NewJeans — Super Shy, official MV audio | 200.829375 s | `84ab2c7636999805431205641e9a013c2ca4f8ff68654636d3cdc3a6e4a400d7` |

The controlled 60-second signal concatenates each file's first 30 seconds. Its
10x repetition is a synthetic 600-second memory stress input, not a third song
or a classification-quality benchmark. Audio has not been copied into this repo.

Disposable scripts and raw result JSON are under
`/tmp/maest-genre-probe-20260929/` (not durable artifacts or wheel contents):
`baseline.py`, `probe.py`, `edges.py`, `summary.json`, and per-trial JSON.
`PYTHONPATH=src .venv/bin/python /tmp/maest-genre-probe-20260929/probe.py`
ran all main trials; the analogous `edges.py` command ran the boundary checks.

## Observations

### Excerpts and whole-track analysis are meaningfully different

| Input / scope | Top result | Score |
|---|---|---:|
| Super Shy, 0–30 s | Electronic — UK Garage | 0.28217 |
| Super Shy, 30–60 s | Pop — K-pop | 0.98693 |
| Super Shy, complete input, W30/H30 | Pop — K-pop | 0.77565 |
| Luv(sic), 0–30 s | Hip Hop — Jazzy Hip-Hop | 0.56665 |
| Luv(sic), 30–60 s | Hip Hop — Jazzy Hip-Hop | 0.44801 |
| Luv(sic), complete input, W30/H30 | Hip Hop — Jazzy Hip-Hop | 0.56748 |

These are sigmoid activations, not calibrated probabilities. The first track
shows that opening-only analysis can produce a different leading label; the
second shows that a change is not guaranteed. There is no human ground truth
in this probe and neither pattern alone proves classification accuracy.

### Window size is a semantic parameter

Controlled 60-second signal; hop equals window, batch one. Top-10 overlap is
against W30/H30 and counts shared labels, irrespective of order.

| Window | Windows | Top-10 overlap | Elapsed | Peak torch GPU allocation |
|---|---:|---:|---:|---:|
| 5 s | 12 | 5/10 | 0.0503 s | 350.02 MiB |
| 10 s | 6 | 6/10 | 0.0313 s | 378.52 MiB |
| 20 s | 3 | 9/10 | 0.0365 s | 474.59 MiB |
| 30 s | 2 | 10/10 | 0.0442 s | 630.60 MiB |

The intervention changes context and its resulting tiling; it does not isolate
context length from boundary placement. Treat this as evidence that the API
parameter changes observable behavior, not a ranking of model quality.

### Bounded batches control GPU allocation

| Input | W/H/batch | Elapsed | Peak torch GPU allocation |
|---|---|---:|---:|
| Controlled 60 s | 30/30/1 | 0.0461 s | 630.60 MiB |
| Repeated 600 s | 30/30/1 | 0.4835 s | 630.60 MiB |
| Luv(sic), full 270.199 s | 30/30/1 | 0.2164 s | 630.60 MiB |
| Super Shy, full 200.829 s | 30/30/1 | 0.1684 s | 630.60 MiB |

60-to-600-second peak ratio: **1.0**, below the pre-registered 1.2 limit.
Cached-checkpoint session load was approximately **0.947 s**. Timings are one
measurement per case after model loading, on decoded input, with CUDA sync;
they exclude network, ffmpeg, scheduling, provider overhead and download.
PyTorch allocated memory is not total GPU residency or a certified fleet
footprint. Process high-water host RSS reached roughly 1.28 GiB around loading;
this is not a measured host-RAM release guarantee.

## Contract and compatibility checks

- Main harness: all ten mandatory checks passed, no recorded errors.
- Raw CPU legacy logits and embeddings are bit-identical to the same-environment
  pre-change baseline. Model/loading numerical code was not changed.
- Direct GPU sigmoid vs excerpt: max delta **0** across 50 returned labels.
- Aggregate vs timeline: identical global rankings on both full real tracks.
- Repeated controlled60 aggregate: identical rankings and scores.
- Batch 1 vs 2: identical global Top-10 ordering; max delta **3.5762787e-7**
  across 50 fixed-label, two-window curves (limit 1e-5). This is not a full
  519-label bit-exact GPU claim across batch shapes.
- Overlap edge: W30/H5 over 65 seconds yields eight windows, full coverage,
  weights summing to 65 s; reconstructing selected global scores from returned
  curve scores/weights differs by at most **2.78e-17**.
- Short-input edge: one second padded to a five-second excerpt agrees with
  manually padded direct inference, max delta **0** across 50 labels.
- Tail excerpt starting at 64 s on a 65 s input retains that start and reports
  exactly one second of analyzed audio; no backward shift.
- Three load/classify/release cycles: no increase in live torch GPU allocation
  over the post-warm baseline; each returns to **8.125 MiB**. This residual is
  below the 16 MiB tolerance. It is not zero total process GPU memory, proof of
  host-RAM reclamation, or a substitute for a future worker-process lifecycle.
- Offline suite: **39 passed, 1 skipped, 10 network tests deselected**. The skip
  is the existing fixture recorded with a different torch build. The new local
  before/after comparison above passed; the old fixture was not overwritten.
- Rename verification after the public mode-name change: **42 passed, 1 skipped,
  10 network tests deselected**. A real GPU smoke compared `segment`,
  `full_track` (including omitted default mode), and `time_curve` against the
  earlier saved result JSON; scores, rankings, analysis data, windows, and curve
  series stayed identical except for the echoed `mode` string. Raw smoke output:
  `/tmp/maest-genre-probe-20260929/rename_smoke.json`.
- Wheel built from sdist successfully; installed-package layout contains the
  new API and checkpoint TOML, excludes probes/tests/audio, and exports
  `MAESTSession.classify` and `classify_genre` from the extracted wheel.

## API decisions retained and limitations

Three modes remain sufficient: opening = segment starting at zero. Every mode
returns sorted rankings; time_curve also returns dense selected-label curves.
All 519 scores are aggregated before Top-N. Default curve selection uses the
union of per-window Top-N, prioritizing peak scores; it is distinct from the
whole-track mean ranking and may highlight transient false positives.

This is a package prototype, not the hosted API: inputs are explicit-rate mono
float arrays/tensors, errors are Python exceptions, and results are in-memory
JSON-compatible dictionaries. Phonon still needs discovery/validation schemas,
size-bounded artifact delivery, admission and process lifecycle/OOM handling,
license/source-offer decisions, measured fleet resources, and end-to-end agent
acceptance. Conductor integration remains downstream of that provider work.

The public openmirlab-skills capability row was inspected; its existing package
routing remains valid. Unreleased branch-only API examples were not added there.
Two strong executor workers implemented API/tests and the disposable harness;
the main session reviewed both, corrected padding and measurement issues, and
independently reran the suite, built the wheel, and executed/interpreted probes.
