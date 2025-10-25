"""Utility helpers for discovering cytogenetic image datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from skimage import io


CHANNEL_ORDER = ("dapi", "fitc", "cy3", "cy5")
IMAGE_EXTENSIONS = {".tif", ".tiff"}


@dataclass
class Metaphase:
    """Container describing a metaphase and its channel image files."""

    name: str
    path: Path
    channels: Dict[str, List[Path]] = field(default_factory=dict)

    def available_channels(self) -> List[str]:
        """Return the list of available channel names ordered for display."""
        ordered = [c for c in CHANNEL_ORDER if c in self.channels]
        tail = sorted(set(self.channels) - set(ordered))
        return ordered + tail


@dataclass
class Slide:
    """Collection of metaphases belonging to a slide."""

    name: str
    path: Path
    metaphases: Dict[str, Metaphase] = field(default_factory=dict)

    def available_metaphases(self) -> List[str]:
        return sorted(self.metaphases)


@dataclass
class Dataset:
    """Description of all slides discovered below a root folder."""

    root: Path
    slides: Dict[str, Slide] = field(default_factory=dict)

    def available_slides(self) -> List[str]:
        return sorted(self.slides)


def _collect_channel_files(channel_dir: Path) -> List[Path]:
    files = [p for p in channel_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort()
    return files


def discover_dataset(root: Path) -> Optional[Dataset]:
    """Return a :class:`Dataset` from *root* or ``None`` if nothing is found."""
    root = root.expanduser().resolve()
    if not root.exists():
        return None

    slides: Dict[str, Slide] = {}
    for slide_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        metaphases: Dict[str, Metaphase] = {}
        for metaphase_dir in sorted(p for p in slide_dir.iterdir() if p.is_dir()):
            channels: Dict[str, List[Path]] = {}
            for channel_dir in sorted(p for p in metaphase_dir.iterdir() if p.is_dir()):
                files = _collect_channel_files(channel_dir)
                if files:
                    channels[channel_dir.name.lower()] = files
            if channels:
                metaphases[metaphase_dir.name] = Metaphase(
                    name=metaphase_dir.name,
                    path=metaphase_dir,
                    channels=channels,
                )
        if metaphases:
            slides[slide_dir.name] = Slide(
                name=slide_dir.name,
                path=slide_dir,
                metaphases=metaphases,
            )
    if not slides:
        return None
    return Dataset(root=root, slides=slides)


def load_image(path: Path) -> np.ndarray:
    """Load an image as ``float32`` for further processing."""
    arr = io.imread(str(path)).astype(np.float32)
    return arr


def ensure_dataset(paths: Iterable[Path]) -> Dataset:
    """Return the first dataset found in *paths* or create a synthetic one."""
    for candidate in paths:
        dataset = discover_dataset(candidate)
        if dataset is not None:
            return dataset

    from .fake_data import create_synthetic_dataset

    synthetic_root = create_synthetic_dataset()
    dataset = discover_dataset(synthetic_root)
    if dataset is None:
        raise RuntimeError("Failed to create synthetic dataset")
    return dataset
