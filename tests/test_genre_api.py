"""Offline contract tests for the experimental genre classification API."""

import numpy as np
import pytest
import torch

from maest_infer.clean_api import MAESTSession, classify_genre
from maest_infer.discogs_labels import discogs_519labels
from maest_infer import genre, genre_metadata, preview_classification
from maest_infer.genre import GENRE_ARCH, _build_windows, _coverage_weights, normalize_waveform


class _GenreModel(torch.nn.Module):
    def __init__(self, logits_by_mean=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.labels = discogs_519labels
        self.num_classes = 519
        self.melspectrogram = None
        self.init_count = 0
        self.eval_count = 0
        self.calls = []
        self.logits_by_mean = logits_by_mean

    def init_melspectrogram(self):
        self.init_count += 1
        self.melspectrogram = torch.nn.Identity()

    def eval(self):
        self.eval_count += 1
        return super().eval()

    def forward(self, batch):
        self.calls.append(batch.detach().cpu().clone())
        logits = torch.full((batch.shape[0], 519), -12.0, device=batch.device)
        if self.logits_by_mean is None:
            logits[:, 0] = batch[:, 0]
            logits[:, 1] = -batch[:, 0]
            return logits
        means = batch.mean(dim=1)
        for row, mean in enumerate(means):
            logits[row] = self.logits_by_mean(float(mean))
        return logits


def _session(model=None, arch=GENRE_ARCH):
    return MAESTSession(arch=arch, model=model or _GenreModel())


def test_metadata_and_preview_do_not_load_model(monkeypatch):
    import maest_infer.loading as loading

    monkeypatch.setattr(loading, "get_maest", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("loaded")))

    metadata = genre_metadata()
    assert metadata["model_arch"] == GENRE_ARCH
    assert metadata["taxonomy"]["label_count"] == 519
    assert metadata["taxonomy"]["labels"] == discogs_519labels
    assert metadata["modes"]["default"] == "full_track"
    assert metadata["parameters"]["window_seconds"] == {"default": 30, "minimum": 5, "maximum": 30}

    preview = preview_classification(sample_count=65 * 16000, window_seconds=30, hop_seconds=30)
    assert preview["parameters"]["mode"] == "full_track"
    assert preview["parameters"]["hop_seconds"] == 30
    assert preview["analysis"]["window_count"] == 3


def test_preview_without_duration_validates_options_but_analysis_is_unknown():
    preview = preview_classification(mode="time_curve", curve_labels=[discogs_519labels[0]], batch_size=2)

    assert preview["analysis"] is None
    assert preview["parameters"]["curve_labels"] == [discogs_519labels[0]]
    assert preview["parameters"]["batch_size"] == 2


def test_preview_and_classify_share_window_parameters_and_analysis():
    audio = torch.zeros(65 * 16000)
    preview = preview_classification(
        sample_count=audio.numel(),
        mode="full_track",
        window_seconds=30,
        hop_seconds=30,
        top_n=3,
        batch_size=2,
    )
    result = _session().classify(
        audio,
        sample_rate=16000,
        mode="full_track",
        window_seconds=30,
        hop_seconds=30,
        top_n=3,
        batch_size=2,
    )

    assert result["parameters"] == preview["parameters"]
    assert result["analysis"] == preview["analysis"]


def test_full_track_ranks_after_full_label_weighted_mean():
    def logits_by_mean(mean):
        logits = torch.full((519,), -12.0)
        if mean < 0.5:
            logits[0] = 10.0
            logits[1] = -10.0
        else:
            logits[0] = -10.0
            logits[1] = 10.0
        logits[2] = 0.1
        return logits

    audio = np.concatenate(
        [
            np.zeros(5 * 16000, dtype=np.float32),
            np.ones(5 * 16000, dtype=np.float32),
        ]
    )
    result = _session(_GenreModel(logits_by_mean)).classify(
        audio,
        sample_rate=16000,
        window_seconds=5,
        hop_seconds=5,
        top_n=1,
        batch_size=2,
    )

    assert result["rankings"][0]["label_id"] == discogs_519labels[2]
    assert result["mode"] == "full_track"
    assert result["analysis"]["window_count"] == 2
    assert result["analysis"]["aggregation"] == "coverage_weighted_mean_v1"
    assert result["analysis"]["tail_policy"] == "end_aligned"
    assert result["analysis"]["analyzed_duration_seconds"] == pytest.approx(10.0)


def test_score_ties_sort_by_label_id(monkeypatch):
    monkeypatch.setattr(genre, "discogs_519labels", ["Z---Later", "A---Earlier"])
    rankings = genre._rank_scores(torch.tensor([0.5, 0.5]), top_n=2)

    assert [ranking["label_id"] for ranking in rankings] == ["A---Earlier", "Z---Later"]


def test_window_sequence_uses_end_aligned_final_window_and_overlap_weights():
    windows = _build_windows(
        65 * 16000,
        mode="full_track",
        window_seconds=30,
        hop_seconds=30,
        start_seconds=None,
    )
    assert [(w.real_start_sample / 16000, w.real_end_sample / 16000) for w in windows] == [
        (0.0, 30.0),
        (30.0, 60.0),
        (35.0, 65.0),
    ]

    weights = _coverage_weights(windows, 65 * 16000)
    assert weights.tolist() == pytest.approx([30.0, 17.5, 17.5])
    assert float(weights.sum()) == pytest.approx(65.0)


def test_segment_start_truncates_at_end_and_pads_requested_window():
    model = _GenreModel()
    audio = torch.ones(8 * 16000)
    result = _session(model).classify(
        audio,
        sample_rate=16000,
        mode="segment",
        window_seconds=5,
        start_seconds=6.0,
    )

    assert result["analysis"]["start_seconds"] == 6.0
    assert result["mode"] == "segment"
    assert result["analysis"]["end_seconds"] == 8.0
    assert result["analysis"]["analyzed_duration_seconds"] == pytest.approx(2.0)
    assert model.calls[0].shape == (1, 5 * 16000)
    assert torch.count_nonzero(model.calls[0][0, 2 * 16000 :]) == 0
    assert result["analysis"]["tail_policy"] == "excerpt_truncated_padded"


def test_time_curve_has_dense_scores_and_explicit_curves_preserve_order():
    model = _GenreModel()
    audio = torch.linspace(-1.0, 1.0, 10 * 16000)
    labels = [discogs_519labels[3], discogs_519labels[1]]
    result = _session(model).classify(
        audio,
        sample_rate=16000,
        mode="time_curve",
        window_seconds=5,
        hop_seconds=5,
        top_n=3,
        curve_labels=labels,
    )

    assert [series["label_id"] for series in result["timeline"]["series"]] == labels
    assert result["mode"] == "time_curve"
    assert len(result["timeline"]["windows"]) == 2
    assert "aggregation_weight_seconds" in result["timeline"]["windows"][0]
    assert all(len(series["scores"]) == 2 for series in result["timeline"]["series"])
    assert result["rankings"] == _session(model).classify(
        audio,
        sample_rate=16000,
        mode="full_track",
        window_seconds=5,
        hop_seconds=5,
        top_n=3,
    )["rankings"]


def test_time_curve_auto_curves_use_per_window_union_capped_to_top_n():
    def logits_by_mean(mean):
        logits = torch.full((519,), -12.0)
        logits[0 if mean < 0.5 else 1] = 10.0
        logits[2] = 0.1
        return logits

    audio = torch.cat([torch.zeros(5 * 16000), torch.ones(5 * 16000)])
    result = _session(_GenreModel(logits_by_mean)).classify(
        audio,
        sample_rate=16000,
        mode="time_curve",
        window_seconds=5,
        hop_seconds=5,
        top_n=2,
    )

    assert [series["label_id"] for series in result["timeline"]["series"]] == [
        discogs_519labels[0],
        discogs_519labels[1],
    ]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"mode": "bad"}, "mode"),
        ({"mode": "excerpt"}, "mode"),
        ({"mode": "aggregate"}, "mode"),
        ({"mode": "timeline"}, "mode"),
        ({"window_seconds": True}, "window_seconds"),
        ({"window_seconds": 31}, "window_seconds"),
        ({"hop_seconds": 1, "mode": "segment"}, "hop_seconds"),
        ({"start_seconds": 0, "mode": "full_track"}, "start_seconds"),
        ({"top_n": 0}, "top_n"),
        ({"batch_size": 33}, "batch_size"),
        ({"mode": "time_curve", "curve_labels": [discogs_519labels[0], discogs_519labels[0]]}, "unique"),
        ({"mode": "time_curve", "curve_labels": ["missing"]}, "unknown"),
        ({"mode": "full_track", "curve_labels": [discogs_519labels[0]]}, "curve_labels"),
    ],
)
def test_option_validation_is_strict(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _session().classify(torch.zeros(5 * 16000), sample_rate=16000, **kwargs)
    with pytest.raises(ValueError, match=match):
        preview_classification(sample_count=5 * 16000, **kwargs)


def test_preview_rejects_bad_sample_count_and_window_count_before_model_load(monkeypatch):
    import maest_infer.loading as loading

    monkeypatch.setattr(loading, "get_maest", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("loaded")))

    with pytest.raises(ValueError, match="sample_count"):
        preview_classification(sample_count=0)
    with pytest.raises(ValueError, match="sample_rate"):
        preview_classification(sample_rate=True, sample_count=16000)
    with pytest.raises(ValueError, match="start_seconds"):
        preview_classification(sample_count=16000, mode="segment", start_seconds=2.0)
    with pytest.raises(ValueError, match="maximum"):
        preview_classification(sample_count=(4097 + 4) * 16000, window_seconds=5, hop_seconds=1)


def test_audio_validation_rejects_bad_inputs_and_resamples():
    with pytest.raises(ValueError, match="positive integer"):
        normalize_waveform(torch.zeros(4), sample_rate=True)
    with pytest.raises(ValueError, match="floating-point"):
        normalize_waveform(np.zeros(4, dtype=np.int16), sample_rate=16000)
    with pytest.raises(ValueError, match="1D"):
        normalize_waveform(torch.zeros(1, 4), sample_rate=16000)
    with pytest.raises(ValueError, match="finite"):
        normalize_waveform(torch.tensor([0.0, float("nan")]), sample_rate=16000)

    resampled = normalize_waveform(torch.ones(8000), sample_rate=8000)
    assert resampled.dtype == torch.float32
    assert resampled.device.type == "cpu"
    assert resampled.numel() == 16000


def test_lifecycle_reuses_loaded_model_and_rejects_released_or_wrong_arch():
    model = _GenreModel()
    session = _session(model)
    session.classify(torch.zeros(5 * 16000), sample_rate=16000, window_seconds=5)
    session.classify(torch.zeros(5 * 16000), sample_rate=16000, window_seconds=5)
    assert model.init_count == 1
    assert len(model.calls) == 2

    session.release()
    with pytest.raises(RuntimeError, match="ready"):
        session.classify(torch.zeros(5 * 16000), sample_rate=16000, window_seconds=5)

    session.close()
    with pytest.raises(RuntimeError, match="ready"):
        session.classify(torch.zeros(5 * 16000), sample_rate=16000, window_seconds=5)

    with pytest.raises(RuntimeError, match="519 labels"):
        _session(_GenreModel(), arch="discogs-maest-5s-pw-129e").classify(
            torch.zeros(5 * 16000),
            sample_rate=16000,
            window_seconds=5,
        )

    mislabeled = _GenreModel()
    mislabeled.labels = list(reversed(discogs_519labels))
    with pytest.raises(RuntimeError, match="519 labels"):
        _session(mislabeled).classify(torch.zeros(5 * 16000), sample_rate=16000, window_seconds=5)


def test_short_full_track_pads_to_requested_window_but_weights_real_audio_only():
    model = _GenreModel()
    result = _session(model).classify(
        torch.ones(2 * 16000),
        sample_rate=16000,
        mode="full_track",
        window_seconds=5,
    )

    assert model.calls[0].shape == (1, 5 * 16000)
    assert torch.count_nonzero(model.calls[0][0, 2 * 16000 :]) == 0
    assert result["analysis"]["analyzed_duration_seconds"] == pytest.approx(2.0)


def test_one_shot_uses_genre_arch_and_closes_on_success_and_error(monkeypatch):
    events = []

    class FakeSession:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))
            self.fail = kwargs.get("device") == "fail"

        def __enter__(self):
            events.append(("enter", None))
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append(("exit", exc_type))
            return False

        def classify(self, audio, *, sample_rate, **kwargs):
            events.append(("classify", sample_rate, kwargs))
            if self.fail:
                raise RuntimeError("boom")
            return {"ok": True}

    import maest_infer.clean_api as clean_api

    monkeypatch.setattr(clean_api, "MAESTSession", FakeSession)
    assert classify_genre(torch.zeros(5), sample_rate=16000, top_n=3) == {"ok": True}
    assert events[0] == ("init", {"arch": GENRE_ARCH, "device": "auto"})
    assert events[2] == ("classify", 16000, {"top_n": 3})
    assert events[3] == ("exit", None)

    with pytest.raises(RuntimeError, match="boom"):
        classify_genre(torch.zeros(5), sample_rate=16000, device="fail")
    assert events[-1][0] == "exit"
    assert events[-1][1] is RuntimeError


def test_unknown_classify_keyword_raises_type_error():
    with pytest.raises(TypeError):
        _session().classify(torch.zeros(5 * 16000), sample_rate=16000, unknown=True)


def test_nonfinite_logits_and_bad_shape_fail_without_partial_result():
    class BadModel(_GenreModel):
        def __init__(self, logits):
            super().__init__()
            self.logits = logits

        def forward(self, batch):
            return self.logits.repeat(batch.shape[0], 1)

    with pytest.raises(RuntimeError, match="shape"):
        _session(BadModel(torch.zeros(1, 518))).classify(
            torch.zeros(5 * 16000),
            sample_rate=16000,
            window_seconds=5,
        )
    logits = torch.zeros(1, 519)
    logits[0, 0] = float("inf")
    with pytest.raises(RuntimeError, match="non-finite logits"):
        _session(BadModel(logits)).classify(
            torch.zeros(5 * 16000),
            sample_rate=16000,
            window_seconds=5,
        )


def test_window_count_guard_and_batch_bound_are_explicit():
    with pytest.raises(ValueError, match="maximum"):
        _build_windows(
            (4097 + 4) * 16000,
            mode="full_track",
            window_seconds=5,
            hop_seconds=1,
            start_seconds=None,
        )

    model = _GenreModel()
    _session(model).classify(
        torch.zeros(15 * 16000),
        sample_rate=16000,
        mode="full_track",
        window_seconds=5,
        hop_seconds=5,
        batch_size=2,
    )
    assert [call.shape[0] for call in model.calls] == [2, 1]


def test_dense_preview_reports_exact_analyzed_duration():
    from maest_infer import preview_classification
    result = preview_classification(sample_count=600 * 16000, mode="time_curve", hop_seconds=1)
    assert result["analysis"]["window_count"] == 571
    assert result["analysis"]["analyzed_duration_seconds"] == 600.0


@pytest.mark.parametrize("mode", ["segment", "time_curve"])
def test_explicit_range_matches_crop_and_excludes_outside_audio(mode):
    audio = torch.full((100 * 16000,), 9.0)
    audio[42 * 16000:81 * 16000] = torch.linspace(-0.5, 0.5, 39 * 16000)
    model = _GenreModel()
    options = dict(mode=mode, start_seconds=42, end_seconds=81, window_seconds=30, hop_seconds=15)
    result = _session(model).classify(audio, sample_rate=16000, **options)
    cropped = _session().classify(audio[42 * 16000:81 * 16000], sample_rate=16000, hop_seconds=15)
    preview = preview_classification(sample_count=audio.numel(), **options)
    assert result["rankings"] == cropped["rankings"]
    assert result["analysis"] == preview["analysis"]
    assert result["analysis"]["duration_seconds"] == 100
    assert result["analysis"]["analyzed_duration_seconds"] == 39
    assert result["analysis"]["window_count"] == 2
    assert result["analysis"]["tail_policy"] == "end_aligned"
    assert all(batch.abs().max() <= 0.5 for batch in model.calls)
    if mode == "time_curve":
        windows = result["timeline"]["windows"]
        assert [(w["start_seconds"], w["end_seconds"]) for w in windows] == [(42, 72), (51, 81)]
        assert sum(w["aggregation_weight_seconds"] for w in windows) == 39


def test_short_explicit_range_pads_without_reading_past_requested_end():
    audio = torch.ones(10 * 16000)
    audio[4 * 16000:] = 9
    model = _GenreModel()
    result = _session(model).classify(audio, sample_rate=16000, mode="segment", start_seconds=2, end_seconds=4, window_seconds=5)
    assert result["analysis"]["end_seconds"] == 4
    assert model.calls[0][0, :2 * 16000].eq(1).all()
    assert model.calls[0][0, 2 * 16000:].eq(0).all()


@pytest.mark.parametrize("mode", ["segment", "time_curve"])
def test_range_end_clips_at_eof_and_defaults_preserve_single_window(mode):
    preview = preview_classification(sample_count=50 * 16000, mode=mode, start_seconds=20, end_seconds=90)
    assert preview["analysis"]["end_seconds"] == 50
    assert preview["analysis"]["analyzed_duration_seconds"] == 30
    curve = preview_classification(sample_count=70 * 16000, mode="time_curve", start_seconds=20)
    assert curve["analysis"]["analyzed_duration_seconds"] == 50
    single = preview_classification(sample_count=70 * 16000, mode="segment", start_seconds=20)
    assert single["analysis"]["end_seconds"] == 50
    assert single["analysis"]["hop_seconds"] is None


@pytest.mark.parametrize("options", [
    {"end_seconds": True}, {"end_seconds": float("nan")}, {"end_seconds": float("inf")},
    {"end_seconds": -1}, {"start_seconds": 4, "end_seconds": 4},
    {"start_seconds": 4, "end_seconds": 3}, {"start_seconds": 60, "end_seconds": 80},
    {"start_seconds": 4, "end_seconds": 4.000001},
    {"mode": "full_track", "end_seconds": 30},
])
def test_invalid_range_rejected_by_preview_and_inference(options):
    kwargs = {"mode": "segment", **options}
    with pytest.raises(ValueError):
        preview_classification(sample_count=50 * 16000, **kwargs)
    model = _GenreModel()
    with pytest.raises(ValueError):
        _session(model).classify(torch.zeros(50 * 16000), sample_rate=16000, **kwargs)
    assert model.calls == []
