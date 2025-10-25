"""JSON based persistence for viewer parameters."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


CONFIG_FILENAME = "config.json"
ALIGNMENT_FILENAME = "alignment.json"


@dataclass
class Config:
    preprocessing: Dict[str, Any]
    alignment: Dict[str, Any]
    segmentation: Dict[str, Any]


def load_config(slide_path: Path) -> Config:
    config_path = slide_path / CONFIG_FILENAME
    alignment_path = slide_path / ALIGNMENT_FILENAME

    def _read(path: Path) -> Dict[str, Any]:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    data = _read(config_path)
    alignment = _read(alignment_path)
    return Config(
        preprocessing=data.get("preprocessing", {}),
        alignment=alignment,
        segmentation=data.get("segmentation", {}),
    )


def save_config(slide_path: Path, config: Config) -> None:
    config_path = slide_path / CONFIG_FILENAME
    alignment_path = slide_path / ALIGNMENT_FILENAME
    config_path.write_text(json.dumps({
        "preprocessing": config.preprocessing,
        "segmentation": config.segmentation,
    }, indent=2))
    alignment_path.write_text(json.dumps(config.alignment, indent=2))
