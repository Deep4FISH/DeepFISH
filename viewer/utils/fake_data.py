"""Synthetic dataset generation for development environments."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
from skimage import draw, io, filters


def _make_channel(seed: int, shape: Tuple[int, int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=2000, scale=300, size=shape).astype(np.float32)
    # Add a few Gaussian blobs to mimic chromosomes.
    for _ in range(8):
        rr, cc = draw.disk(
            center=(rng.integers(20, shape[0] - 20), rng.integers(20, shape[1] - 20)),
            radius=rng.integers(8, 18),
            shape=shape,
        )
        base[rr, cc] += rng.uniform(1200, 2500)
    base = filters.gaussian(base, sigma=1.5)
    base -= base.min()
    base /= base.max()
    return (base * 4095).astype(np.uint16)


def create_synthetic_dataset() -> Path:
    """Create a temporary dataset compatible with the viewer layout."""
    root = Path(tempfile.mkdtemp(prefix="cyto-viewer-demo-"))
    slide_dir = root / "demo-slide"
    metaphase_dir = slide_dir / "1"
    metaphase_dir.mkdir(parents=True, exist_ok=True)

    channels = {
        "dapi": _make_channel(1, (256, 256)),
        "cy3": _make_channel(2, (256, 256)),
        "cy5": _make_channel(3, (256, 256)),
    }

    for name, arr in channels.items():
        channel_dir = metaphase_dir / name
        channel_dir.mkdir()
        io.imsave(str(channel_dir / "1.tif"), arr)

    # Default config file showcasing the JSON format.
    config_path = slide_dir / "config.json"
    config_path.write_text(json.dumps({
        "preprocessing": {},
        "alignment": {},
        "segmentation": {},
    }, indent=2))
    return root
