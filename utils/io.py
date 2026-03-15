import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_dirs(output_root: str | Path = "outputs") -> Dict[str, Path]:
    output_root = ensure_dir(output_root)

    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    logs_dir = ensure_dir(output_root / "logs")
    metrics_dir = ensure_dir(output_root / "metrics")

    return {
        "root": output_root,
        "checkpoints": checkpoints_dir,
        "logs": logs_dir,
        "metrics": metrics_dir,
    }


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)