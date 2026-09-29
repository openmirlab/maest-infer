"""Task-level genre classification for the 519-label MAEST checkpoint.

This module owns waveform normalization, analysis windows, full-label score
aggregation, and JSON-compatible result shaping for the experimental genre API.
The legacy raw model forward path remains in model.py; this layer calls it only
with bounded 2D batches so long-audio behavior is explicit here.

Reads: torch, torchaudio.functional, maest_infer.discogs_labels; read by:
maest_infer.clean_api
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .discogs_labels import discogs_519labels

TARGET_SAMPLE_RATE = 16000
GENRE_ARCH = "discogs-maest-30s-pw-129e-519l"
TAXONOMY = {
    "id": "discogs-519",
    "revision": 1,
    "label_count": 519,
}
SCORE_TYPE = "sigmoid_activation"
MAX_WINDOWS = 4096
MAX_BATCH_SIZE = 32

_LABEL_TO_INDEX = {label: index for index, label in enumerate(discogs_519labels)}


@dataclass(frozen=True)
class _Window:
    index: int
    start_sample: int
    end_sample: int
    real_start_sample: int
    real_end_sample: int
    padding_samples: int


@dataclass(frozen=True)
class _ClassificationPlan:
    parameters: dict[str, Any]
    analysis: dict[str, Any] | None
    windows: list[_Window] | None
    weights: torch.Tensor | None
    sample_count: int | None


def genre_metadata() -> dict[str, Any]:
    """Return static genre API metadata without constructing a model."""
    return {
        "model_arch": GENRE_ARCH,
        "taxonomy": {
            **TAXONOMY,
            "labels": list(discogs_519labels),
        },
        "score_type": SCORE_TYPE,
        "sample_rate": TARGET_SAMPLE_RATE,
        "modes": {
            "default": "full_track",
            "supported": ["segment", "full_track", "time_curve"],
        },
        "parameters": {
            "window_seconds": {"default": 30, "minimum": 5, "maximum": 30},
            "hop_seconds": {"default": None, "minimum": 1, "maximum": "window_seconds"},
            "start_seconds": {"default": None, "minimum": 0, "modes": ["segment"]},
            "top_n": {"default": 10, "minimum": 1, "maximum": 50},
            "curve_labels": {"default": None, "maximum_count": 50, "modes": ["time_curve"]},
            "batch_size": {"default": 1, "minimum": 1, "maximum": MAX_BATCH_SIZE},
            "window_count": {"maximum": MAX_WINDOWS},
        },
    }


def preview_classification(
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    sample_count: int | None = None,
    mode: str = "full_track",
    window_seconds: int = 30,
    hop_seconds: int | None = None,
    start_seconds: float | int | None = None,
    top_n: int = 10,
    curve_labels: Sequence[str] | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Validate and preview a genre classification request without loading a model."""
    plan = _plan_classification(
        sample_rate=sample_rate,
        sample_count=sample_count,
        mode=mode,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        start_seconds=start_seconds,
        top_n=top_n,
        curve_labels=curve_labels,
        batch_size=batch_size,
    )
    return {
        "parameters": dict(plan.parameters),
        "analysis": dict(plan.analysis) if plan.analysis is not None else None,
    }


def classify_genre(audio: Any, *, sample_rate: int, device: str | None = "auto", **kwargs: Any) -> dict[str, Any]:
    """Load the 519-label MAEST checkpoint once and classify a mono waveform."""
    from .clean_api import MAESTSession

    with MAESTSession(arch=GENRE_ARCH, device=device) as session:
        return session.classify(audio, sample_rate=sample_rate, **kwargs)


def classify_with_session(
    session: Any,
    audio: Any,
    *,
    sample_rate: int,
    mode: str = "full_track",
    window_seconds: int = 30,
    hop_seconds: int | None = None,
    start_seconds: float | int | None = None,
    top_n: int = 10,
    curve_labels: Sequence[str] | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Run genre classification through an already-ready MAESTSession."""
    if getattr(session, "status", None) != "ready" or getattr(session, "_model", None) is None:
        raise RuntimeError("MAESTSession must be ready; call load() before classify()")
    _validate_arch(getattr(session, "arch", None), session._model)
    waveform = normalize_waveform(audio, sample_rate=sample_rate)
    plan = _plan_classification(
        sample_rate=TARGET_SAMPLE_RATE,
        sample_count=waveform.shape[0],
        mode=mode,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        start_seconds=start_seconds,
        top_n=top_n,
        curve_labels=curve_labels,
        batch_size=batch_size,
    )
    options = plan.parameters
    assert plan.windows is not None
    assert plan.weights is not None
    scores = _infer_window_scores(session._model, waveform, plan.windows, batch_size=options["batch_size"])
    weights = plan.weights
    aggregate_scores = _aggregate_scores(scores, weights)
    rankings = _rank_scores(aggregate_scores, options["top_n"])
    result: dict[str, Any] = {
        "mode": options["mode"],
        "parameters": dict(options),
        "taxonomy": dict(TAXONOMY),
        "score_type": SCORE_TYPE,
        "analysis": dict(plan.analysis or {}),
        "rankings": rankings,
    }
    if options["mode"] == "time_curve":
        curve_indices = _select_curve_indices(scores, options["top_n"], options["curve_labels"])
        result["timeline"] = {
            "windows": [_window_payload(window, weights[window.index], TARGET_SAMPLE_RATE) for window in plan.windows],
            "series": [_series_payload(index, scores[:, index]) for index in curve_indices],
        }
    return result


def normalize_waveform(audio: Any, *, sample_rate: int) -> torch.Tensor:
    """Return a finite CPU float32 mono waveform at MAEST's 16 kHz rate."""
    _validate_sample_rate(sample_rate)
    if isinstance(audio, torch.Tensor):
        if audio.dtype == torch.bool or not torch.is_floating_point(audio):
            raise ValueError("audio must be a floating-point mono waveform")
        waveform = audio.detach().to(device="cpu", dtype=torch.float32).contiguous()
    elif isinstance(audio, np.ndarray):
        if audio.dtype == np.bool_ or not np.issubdtype(audio.dtype, np.floating):
            raise ValueError("audio must be a floating-point mono waveform")
        waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    else:
        raise TypeError("audio must be a numpy.ndarray or torch.Tensor")
    if waveform.ndim != 1:
        raise ValueError("audio must be a mono 1D waveform")
    if waveform.numel() == 0:
        raise ValueError("audio must not be empty")
    if not torch.isfinite(waveform).all():
        raise ValueError("audio must contain only finite values")
    if sample_rate != TARGET_SAMPLE_RATE:
        import torchaudio.functional as F

        waveform = F.resample(waveform, orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError("normalized audio must be finite and nonempty")
    return waveform.to(device="cpu", dtype=torch.float32).contiguous()


def _validate_arch(arch: Any, model: Any) -> None:
    labels = getattr(model, "labels", discogs_519labels)
    label_count = len(labels)
    num_classes = getattr(model, "num_classes", label_count)
    if arch != GENRE_ARCH or num_classes != 519 or label_count != 519 or list(labels) != discogs_519labels:
        raise RuntimeError("classify() requires arch='discogs-maest-30s-pw-129e-519l' with 519 labels")


def _validate_options(**kwargs: Any) -> dict[str, Any]:
    mode = kwargs["mode"]
    if not isinstance(mode, str) or mode not in {"segment", "full_track", "time_curve"}:
        raise ValueError("mode must be 'segment', 'full_track', or 'time_curve'")
    window_seconds = _bounded_int("window_seconds", kwargs["window_seconds"], 5, 30)
    top_n = _bounded_int("top_n", kwargs["top_n"], 1, 50)
    batch_size = _bounded_int("batch_size", kwargs["batch_size"], 1, MAX_BATCH_SIZE)
    hop_seconds = kwargs["hop_seconds"]
    if mode == "segment":
        if hop_seconds is not None:
            raise ValueError("hop_seconds is only valid for full_track and time_curve modes")
        hop_value = None
    else:
        hop_value = window_seconds if hop_seconds is None else _bounded_int("hop_seconds", hop_seconds, 1, window_seconds)
    start_seconds = kwargs["start_seconds"]
    if mode == "segment":
        if start_seconds is None:
            start_value = 0.0
        else:
            start_value = _nonnegative_seconds("start_seconds", start_seconds)
    elif start_seconds is not None:
        raise ValueError("start_seconds is only valid for segment mode")
    else:
        start_value = None
    curve_labels = kwargs["curve_labels"]
    if curve_labels is not None:
        if mode != "time_curve":
            raise ValueError("curve_labels is only valid for time_curve mode")
        curve_labels = _validate_curve_labels(curve_labels)
    return {
        "mode": mode,
        "window_seconds": window_seconds,
        "hop_seconds": hop_value,
        "start_seconds": start_value,
        "top_n": top_n,
        "curve_labels": curve_labels,
        "batch_size": batch_size,
    }


def _plan_classification(
    *,
    sample_rate: int,
    sample_count: int | None,
    mode: str,
    window_seconds: int,
    hop_seconds: int | None,
    start_seconds: float | int | None,
    top_n: int,
    curve_labels: Sequence[str] | None,
    batch_size: int,
) -> _ClassificationPlan:
    _validate_sample_rate(sample_rate)
    options = _validate_options(
        mode=mode,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        start_seconds=start_seconds,
        top_n=top_n,
        curve_labels=curve_labels,
        batch_size=batch_size,
    )
    parameters = {
        "sample_rate": sample_rate,
        "sample_count": sample_count,
        **options,
        "curve_labels": list(options["curve_labels"]) if options["curve_labels"] is not None else None,
    }
    if sample_count is None:
        return _ClassificationPlan(
            parameters=parameters,
            analysis=None,
            windows=None,
            weights=None,
            sample_count=None,
        )
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer or None")
    target_sample_count = sample_count
    if sample_rate != TARGET_SAMPLE_RATE:
        target_sample_count = int(round(sample_count * TARGET_SAMPLE_RATE / sample_rate))
        if target_sample_count <= 0:
            raise ValueError("sample_count must resolve to at least one 16 kHz sample")
    windows = _build_windows(
        target_sample_count,
        mode=options["mode"],
        window_seconds=options["window_seconds"],
        hop_seconds=options["hop_seconds"],
        start_seconds=options["start_seconds"],
    )
    weights = _coverage_weights(windows, target_sample_count)
    analysis = _analysis_payload(
        target_sample_count,
        windows,
        mode=options["mode"],
        window_seconds=options["window_seconds"],
        hop_seconds=options["hop_seconds"],
        top_n=options["top_n"],
    )
    return _ClassificationPlan(
        parameters=parameters,
        analysis=analysis,
        windows=windows,
        weights=weights,
        sample_count=target_sample_count,
    )


def _validate_sample_rate(sample_rate: Any) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    return sample_rate


def _bounded_int(name: str, value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be from {minimum} to {maximum}")
    return value


def _nonnegative_seconds(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative number")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return value


def _validate_curve_labels(labels: Any) -> list[str]:
    if isinstance(labels, (str, bytes)) or not isinstance(labels, list):
        raise ValueError("curve_labels must be a list of label_id strings")
    if len(labels) > 50:
        raise ValueError("curve_labels may contain at most 50 labels")
    seen: set[str] = set()
    normalized: list[str] = []
    unknown: list[str] = []
    for label in labels:
        if not isinstance(label, str):
            raise ValueError("curve_labels must be a list of label_id strings")
        if label in seen:
            raise ValueError("curve_labels must be unique")
        seen.add(label)
        if label not in _LABEL_TO_INDEX:
            unknown.append(label)
        normalized.append(label)
    if unknown:
        raise ValueError(f"unknown curve_labels: {', '.join(unknown[:3])}")
    return normalized


def _build_windows(
    sample_count: int,
    *,
    mode: str,
    window_seconds: int,
    hop_seconds: int | None,
    start_seconds: float | None,
) -> list[_Window]:
    window_samples = window_seconds * TARGET_SAMPLE_RATE
    duration = sample_count / TARGET_SAMPLE_RATE
    windows: list[_Window] = []
    if mode == "segment":
        start_sample = int(round((start_seconds or 0.0) * TARGET_SAMPLE_RATE))
        if start_sample >= sample_count:
            raise ValueError("start_seconds must be within the audio duration")
        end_sample = start_sample + window_samples
        windows.append(_make_window(0, start_sample, end_sample, sample_count, window_samples))
        return windows

    hop_samples = (hop_seconds or window_seconds) * TARGET_SAMPLE_RATE
    if sample_count <= window_samples:
        windows.append(_make_window(0, 0, window_samples, sample_count, window_samples))
        return windows

    starts = list(range(0, sample_count - window_samples + 1, hop_samples))
    final_start = sample_count - window_samples
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    if len(starts) > MAX_WINDOWS:
        raise ValueError(f"analysis would create {len(starts)} windows; maximum is {MAX_WINDOWS}")
    for index, start_sample in enumerate(starts):
        end_sample = start_sample + window_samples
        windows.append(_make_window(index, start_sample, end_sample, sample_count, window_samples))
    if duration <= 0:
        raise ValueError("audio must not be empty")
    return windows


def _make_window(index: int, start_sample: int, end_sample: int, sample_count: int, window_samples: int) -> _Window:
    real_end_sample = min(end_sample, sample_count)
    padding_samples = max(0, window_samples - (real_end_sample - start_sample))
    return _Window(
        index=index,
        start_sample=start_sample,
        end_sample=end_sample,
        real_start_sample=start_sample,
        real_end_sample=real_end_sample,
        padding_samples=padding_samples,
    )


def _infer_window_scores(model: Any, waveform: torch.Tensor, windows: Sequence[_Window], *, batch_size: int) -> torch.Tensor:
    _prepare_model_for_genre(model)
    device = _model_device(model)
    batches: list[torch.Tensor] = []
    window_samples = max(window.end_sample - window.start_sample for window in windows)
    for offset in range(0, len(windows), batch_size):
        slices = []
        for window in windows[offset : offset + batch_size]:
            chunk = waveform[window.start_sample : window.real_end_sample]
            if chunk.numel() < window_samples:
                chunk = torch.nn.functional.pad(chunk, (0, window_samples - chunk.numel()))
            slices.append(chunk)
        batch = torch.stack(slices).to(device=device)
        with torch.inference_mode():
            output = model(batch)
        logits = output[0] if isinstance(output, tuple) else output
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("model must return logits as a torch.Tensor")
        if logits.shape != (len(slices), 519):
            raise RuntimeError(f"model returned logits with shape {tuple(logits.shape)}, expected {(len(slices), 519)}")
        if not torch.isfinite(logits).all():
            raise RuntimeError("model returned non-finite logits")
        scores = torch.sigmoid(logits).detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(scores).all():
            raise RuntimeError("model returned non-finite scores")
        batches.append(scores)
    return torch.cat(batches, dim=0)


def _prepare_model_for_genre(model: Any) -> None:
    if hasattr(model, "eval"):
        model.eval()
    if getattr(model, "melspectrogram", None) is None and hasattr(model, "init_melspectrogram"):
        model.init_melspectrogram()
    melspectrogram = getattr(model, "melspectrogram", None)
    if melspectrogram is not None and hasattr(melspectrogram, "to"):
        melspectrogram.to(_model_device(model))


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _coverage_weights(windows: Sequence[_Window], sample_count: int) -> torch.Tensor:
    boundaries = sorted({0, sample_count, *(w.real_start_sample for w in windows), *(w.real_end_sample for w in windows)})
    weights = [0.0 for _ in windows]
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            continue
        covering = [w.index for w in windows if w.real_start_sample < right and w.real_end_sample > left]
        if not covering:
            continue
        share = (right - left) / len(covering)
        for index in covering:
            weights[index] += share
    return torch.tensor(weights, dtype=torch.float64) / TARGET_SAMPLE_RATE


def _aggregate_scores(scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    total = weights.sum()
    if total <= 0:
        raise RuntimeError("window coverage must be positive")
    return (scores * weights[:, None]).sum(dim=0) / total


def _rank_scores(scores: torch.Tensor, top_n: int) -> list[dict[str, Any]]:
    ordered = sorted(range(len(discogs_519labels)), key=lambda index: (-float(scores[index]), discogs_519labels[index]))
    return [_ranking_payload(rank, index, float(scores[index])) for rank, index in enumerate(ordered[:top_n], start=1)]


def _ranking_payload(rank: int, index: int, score: float) -> dict[str, Any]:
    genre, subgenre = _split_label(discogs_519labels[index])
    return {
        "rank": rank,
        "label_id": discogs_519labels[index],
        "genre": genre,
        "subgenre": subgenre,
        "score": score,
    }


def _split_label(label: str) -> tuple[str, str]:
    genre, separator, subgenre = label.partition("---")
    if not separator:
        return label, ""
    return genre, subgenre


def _analysis_payload(
    sample_count: int,
    windows: Sequence[_Window],
    *,
    mode: str,
    window_seconds: int,
    hop_seconds: int | None,
    top_n: int,
) -> dict[str, Any]:
    start = windows[0].real_start_sample / TARGET_SAMPLE_RATE
    end = max(window.real_end_sample for window in windows) / TARGET_SAMPLE_RATE
    duration = sample_count / TARGET_SAMPLE_RATE
    return {
        "duration_seconds": duration,
        "start_seconds": start,
        "end_seconds": end,
        # Windows cover a contiguous interval (hop <= window). Count real samples
        # instead of summing fractional overlap weights, which can exceed duration
        # by a floating-point epsilon on dense curves.
        "analyzed_duration_seconds": (
            max(window.real_end_sample for window in windows) - windows[0].real_start_sample
        ) / TARGET_SAMPLE_RATE,
        "window_seconds": window_seconds,
        "hop_seconds": hop_seconds,
        "window_count": len(windows),
        "tail_policy": "excerpt_truncated_padded" if mode == "segment" else "end_aligned",
        "aggregation": "coverage_weighted_mean_v1",
        "top_n": top_n,
    }


def _window_payload(window: _Window, weight: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    start = window.real_start_sample / sample_rate
    end = window.real_end_sample / sample_rate
    return {
        "index": window.index,
        "start_seconds": start,
        "end_seconds": end,
        "center_seconds": (start + end) / 2,
        "padding_seconds": window.padding_samples / sample_rate,
        "aggregation_weight_seconds": float(weight),
    }


def _select_curve_indices(scores: torch.Tensor, top_n: int, curve_labels: Sequence[str] | None) -> list[int]:
    if curve_labels is not None:
        return [_LABEL_TO_INDEX[label] for label in curve_labels]
    candidates: set[int] = set()
    for row in scores:
        ordered = sorted(range(len(discogs_519labels)), key=lambda index: (-float(row[index]), discogs_519labels[index]))
        candidates.update(ordered[:top_n])
    return sorted(candidates, key=lambda index: (-float(scores[:, index].max()), discogs_519labels[index]))[:top_n]


def _series_payload(index: int, values: Iterable[torch.Tensor]) -> dict[str, Any]:
    genre, subgenre = _split_label(discogs_519labels[index])
    return {
        "label_id": discogs_519labels[index],
        "genre": genre,
        "subgenre": subgenre,
        "scores": [float(value) for value in values],
    }
