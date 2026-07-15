"""Package-owned checkpoint metadata for MAEST."""
from pathlib import Path
import tomllib


def checkpoint_config_path() -> Path:
    return Path(__file__).with_name("config") / "checkpoints.toml"


def load_checkpoints(path=None) -> dict:
    path = Path(path) if path else checkpoint_config_path()
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid checkpoint config: {path}") from exc
    if data.get("schema", {}).get("version") != 1 or not isinstance(data.get("models"), dict):
        raise ValueError("checkpoint config must define schema.version=1 and models")
    for key, model in data["models"].items():
        artifacts = model.get("artifacts") if isinstance(model, dict) else None
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"invalid checkpoint metadata for {key}")
        for artifact in artifacts:
            digest = artifact.get("sha256") if isinstance(artifact, dict) else None
            if not str(artifact.get("url", "")).startswith("https://") or not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"invalid artifact metadata for {key}")
    return data


def checkpoint_metadata(model: str) -> dict:
    try:
        return dict(load_checkpoints()["models"][model])
    except KeyError as exc:
        raise KeyError(f"unknown checkpoint model: {model}") from exc


__all__ = ["checkpoint_config_path", "load_checkpoints", "checkpoint_metadata"]
