"""Explicit lifecycle facade for MAEST inference and checkpoint cache status.

Keeps model construction, readiness, release, and cache inspection in one thin
object while task-level postprocessing lives in focused modules. The genre task
API is additive here so the legacy raw infer path keeps its original behavior.

Reads: maest_infer.checkpoints, maest_infer.loading; read by: package users
"""
import hashlib
from pathlib import Path

from .checkpoints import checkpoint_artifact, checkpoint_for_url, torch_hub_checkpoint_path


def _sha256_of_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _device_label(device):
    if getattr(device, "type", None) == "cuda" and getattr(device, "index", None) is None:
        return "cuda:0"
    return str(device)


def _resolve_device(device):
    """Resolve MAEST's legacy automatic device choice and validate explicit requests.

    ``mps`` is rejected unconditionally, regardless of actual MPS
    availability: Apple MLX/MPS backends are permanently out of scope for
    this org's projects (org canon openmirlab-dev 5e588e6, art. 4b).
    ``"auto"`` never selects mps -- it is cuda-else-cpu only.
    """
    import torch

    if device is None or device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    if not isinstance(device, str):
        raise ValueError("device must be None, 'auto', 'cpu', 'cuda', or 'cuda:N'")
    if device == "mps":
        raise ValueError(
            "device 'mps' is not supported. Apple MLX/MPS backends are "
            "permanently out of scope for this project (org canon "
            "openmirlab-dev 5e588e6, art. 4b). Supported devices: 'auto', "
            "'cpu', 'cuda', or 'cuda:N'."
        )
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is not available")
        return torch.device("cuda:0")
    if not device.startswith("cuda:"):
        raise ValueError("device must be None, 'auto', 'cpu', 'cuda', or 'cuda:N'")
    index_text = device[5:]
    if not index_text.isdigit():
        raise ValueError("CUDA device index must be a non-negative integer")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available")
    index = int(index_text)
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {index} is not available")
    return torch.device(device)


class MAESTSession:
    def __init__(self, *, arch="discogs-maest-5s-pw-129e", model=None, device=None,
                 checkpoint=None, checkpoint_url=None, checkpoint_sha256=None, **kwargs):
        self.arch, self.device = arch, device
        self._device_request = device
        self._model, self._status = model, ("ready" if model is not None else "new")
        self._loaded_checkpoint_info = None
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.checkpoint_url, self.checkpoint_sha256 = checkpoint_url, checkpoint_sha256
        self._kwargs = kwargs

    @property
    def status(self):
        return self._status

    def load(self):
        if self._status == "closed":
            raise RuntimeError("cannot load a closed MAESTSession")
        if self._status == "ready":
            return self
        self._status = "loading"
        try:
            import torch
            from .loading import get_maest
            target = _resolve_device(self._device_request)
            options = dict(self._kwargs)
            options["pretrained"] = self.checkpoint is None
            if self.checkpoint is not None:
                options["checkpoint"] = str(self.checkpoint)
            elif self.checkpoint_url:
                options["external_default_cfg"] = {
                    "url": self.checkpoint_url,
                    "checkpoint_sha256": self.checkpoint_sha256,
                }
            self._model = get_maest(self.arch, **options)
            self._model.to(target).eval()
            self.device = target
            self._loaded_checkpoint_info = self._capture_loaded_checkpoint_info()
            self._status = "ready"
            return self
        except Exception:
            if self._model is not None and hasattr(self._model, "cpu"):
                try:
                    self._model.cpu()
                except Exception:
                    pass
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            self._loaded_checkpoint_info = None
            self._status = "failed"
            raise

    def infer(self, audio, **kwargs):
        if self._status != "ready" or self._model is None:
            raise RuntimeError("MAESTSession must be ready; call load() before infer()")
        return self._model(audio, **kwargs)

    def classify(self, audio, *, sample_rate, mode="full_track", window_seconds=30,
                 hop_seconds=None, start_seconds=None, end_seconds=None, top_n=10,
                 curve_labels=None, batch_size=1):
        """Classify a mono waveform with the experimental 519-label genre API."""
        from .genre import classify_with_session

        return classify_with_session(
            self,
            audio,
            sample_rate=sample_rate,
            mode=mode,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            top_n=top_n,
            curve_labels=curve_labels,
            batch_size=batch_size,
        )

    def release(self):
        if self._status == "closed":
            return self
        if self._model is not None and hasattr(self._model, "cpu"):
            self._model.cpu()
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self._status = "released"
        return self

    def close(self):
        if self._status == "closed":
            return self
        self.release()
        self._status = "closed"
        return self

    def cache_info(self):
        artifact = checkpoint_artifact(self.arch.replace("-", "_"))
        url = None if self.checkpoint is not None else self.checkpoint_url or artifact.get("url")
        resolved = self.checkpoint or torch_hub_checkpoint_path(url)
        declared = checkpoint_for_url(url) if url else None
        expected_sha = self.checkpoint_sha256 or (declared or {}).get("sha256")
        return {"model": self.arch, "status": self._status, "model_loaded": self._model is not None,
                "checkpoint_path": str(resolved), "cached": resolved.is_file(),
                "checkpoint_url": url,
                "checkpoint_sha256": expected_sha}

    def loaded_checkpoint_info(self):
        """Return provenance for the actual checkpoint bytes loaded into a ready session."""
        if self._status != "ready" or self._model is None:
            raise RuntimeError("MAESTSession must be ready; call load() before loaded_checkpoint_info()")
        if self._loaded_checkpoint_info is None:
            raise RuntimeError("Loaded checkpoint provenance is unavailable for this session")
        return dict(self._loaded_checkpoint_info)

    def _capture_loaded_checkpoint_info(self):
        loaded = getattr(self._model, "_maest_loaded_checkpoint", {}) or {}
        path = loaded.get("path")
        url = loaded.get("url")
        checkpoint_id = loaded.get("checkpoint_id")
        if path is None and url:
            path = torch_hub_checkpoint_path(url)
        path_obj = Path(path) if path is not None else None
        actual_sha = _sha256_of_file(path_obj) if path_obj is not None and path_obj.is_file() else None
        artifact = checkpoint_for_url(url) if url else None
        if checkpoint_id is None:
            checkpoint_id = (artifact or {}).get("name")
        if path_obj is None and url is None and checkpoint_id is None:
            return None
        return {
            "model": self.arch,
            "model_arch": self.arch,
            "device": _device_label(self.device),
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": str(path_obj) if path_obj is not None else None,
            "checkpoint_url": url,
            "checkpoint_sha256": actual_sha,
        }

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def get_maest_session(**kwargs):
    """Lazy compatibility facade: first inference loads the selected model."""
    session = MAESTSession(**kwargs)
    return session


def classify_genre(audio, *, sample_rate, **kwargs):
    """One-shot genre classification using the 519-label MAEST checkpoint."""
    from .genre import classify_genre as _classify_genre

    return _classify_genre(audio, sample_rate=sample_rate, **kwargs)


def genre_metadata():
    """Return static genre API metadata without loading MAEST weights."""
    from .genre import genre_metadata as _genre_metadata

    return _genre_metadata()


def preview_classification(**kwargs):
    """Validate and preview a genre classification request without loading weights."""
    from .genre import preview_classification as _preview_classification

    return _preview_classification(**kwargs)


__all__ = [
    "MAESTSession",
    "get_maest_session",
    "classify_genre",
    "genre_metadata",
    "preview_classification",
]
