"""Explicit lifecycle facade for MAEST inference and checkpoint cache status."""
from pathlib import Path

from .checkpoints import checkpoint_artifact, torch_hub_checkpoint_path


def _resolve_device(device):
    """Resolve MAEST's legacy automatic device choice and validate explicit requests."""
    import torch

    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    if not isinstance(device, str):
        raise ValueError("device must be None, 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'")
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was explicitly requested but is not available")
        return torch.device("mps")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is not available")
        return torch.device("cuda")
    if not device.startswith("cuda:"):
        raise ValueError("device must be None, 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'")
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
            target = _resolve_device(self._device_request)
            self._model.to(target).eval()
            self.device = target
            self._status = "ready"
            return self
        except Exception:
            self._status = "failed"
            raise

    def infer(self, audio, **kwargs):
        if self._status != "ready" or self._model is None:
            raise RuntimeError("MAESTSession must be ready; call load() before infer()")
        return self._model(audio, **kwargs)

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
        url = self.checkpoint_url or artifact.get("url")
        resolved = self.checkpoint or torch_hub_checkpoint_path(url)
        return {"model": self.arch, "status": self._status, "model_loaded": self._model is not None,
                "checkpoint_path": str(resolved), "cached": resolved.is_file(),
                "checkpoint_url": url,
                "checkpoint_sha256": self.checkpoint_sha256 or artifact.get("sha256")}

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def get_maest_session(**kwargs):
    """Lazy compatibility facade: first inference loads the selected model."""
    session = MAESTSession(**kwargs)
    return session


__all__ = ["MAESTSession", "get_maest_session"]
