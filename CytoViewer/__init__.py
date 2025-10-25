"""Compatibility wrapper to allow ``python -m CytoViewer``.

The original notebooks referenced a capitalised package name. Keep the
modern implementation in :mod:`cyto_viewer` as the single source of
truth and re-export its public entry points here.
"""
from cyto_viewer.viewer import main, run_viewer  # noqa: F401

__all__ = ["main", "run_viewer"]
