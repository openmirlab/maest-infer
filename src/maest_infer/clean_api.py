"""Explicit lifecycle facade for MAEST inference."""
from pathlib import Path

from .checkpoints import checkpoint_metadata


class MAESTSession:
    def __init__(self, *, arch="discogs-maest-5s-pw-129e", model=None, device=None,
                 checkpoint=None, checkpoint_url=None, checkpoint_sha256=None, **kwargs):
        self.arch, self.device = arch, device
        self._model, self._status = model, ("ready" if model is not None else "new")
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.checkpoint_url, self.checkpoint_sha256 = checkpoint_url, checkpoint_sha256
        self._kwargs = kwargs

    @property
    def status(self):
        return self._status

    def load(self):
        if self._status == "ready":
            return self
        if self._status == "released":
            raise RuntimeError("cannot load a released MAESTSession")
        self._status = "loading"
        try:
            import torch
            from .loading import get_maest
            options = dict(self._kwargs)
            options["pretrained"] = self.checkpoint is None
            if self.checkpoint is not None:
                options["checkpoint"] = str(self.checkpoint)
            self._model = get_maest(self.arch, **options)
            target = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
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
        return self.release()

    def cache_info(self):
        metadata = checkpoint_metadata(self.arch.replace("-", "_"))
        artifact = next((a for a in metadata["artifacts"] if a.get("kind") == "checkpoint"), {})
        return {"model": self.arch, "status": self._status, "model_loaded": self._model is not None,
                "checkpoint_path": str(self.checkpoint) if self.checkpoint else None,
                "checkpoint_url": self.checkpoint_url or artifact.get("url"),
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
