"""Segmentation controls and utilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from skimage import measure, morphology


@dataclass
class SegmentationSettings:
    block_size: int = 35
    offset: int = -10
    display_mode: str = "overlay"  # overlay, contour, filled
    brush_radius: int = 8
    min_size: int = 50
    max_size: int = 10000
    borderkill: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "SegmentationSettings":
        if not data:
            return cls()
        return cls(**data)


class SegmentationPanel(QtWidgets.QWidget):
    generateRequested = QtCore.pyqtSignal(str, object)
    saveRequested = QtCore.pyqtSignal(str)
    displayModeChanged = QtCore.pyqtSignal(str)
    brushChanged = QtCore.pyqtSignal(int, str)
    undoRequested = QtCore.pyqtSignal()
    redoRequested = QtCore.pyqtSignal()
    filterRequested = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._channel = "dapi"
        self._per_channel: Dict[str, SegmentationSettings] = {
            "dapi": SegmentationSettings(),
        }
        self._settings = self._per_channel[self._channel]
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.channel_label = QtWidgets.QLabel("Channel: dapi")
        layout.addWidget(self.channel_label)

        self.block_spin = QtWidgets.QSpinBox()
        self.block_spin.setRange(3, 255)
        self.block_spin.setSingleStep(2)
        self.block_spin.setValue(self._settings.block_size)
        layout.addWidget(QtWidgets.QLabel("Adaptive block size"))
        layout.addWidget(self.block_spin)

        self.offset_spin = QtWidgets.QSpinBox()
        self.offset_spin.setRange(-50, 50)
        self.offset_spin.setValue(self._settings.offset)
        layout.addWidget(QtWidgets.QLabel("Threshold offset"))
        layout.addWidget(self.offset_spin)

        self.display_combo = QtWidgets.QComboBox()
        self.display_combo.addItems(["overlay", "contour", "filled"])
        layout.addWidget(QtWidgets.QLabel("Display mode"))
        layout.addWidget(self.display_combo)

        self.brush_combo = QtWidgets.QComboBox()
        self.brush_combo.addItems(["brush", "eraser"])
        self.brush_radius = QtWidgets.QSpinBox()
        self.brush_radius.setRange(1, 50)
        self.brush_radius.setValue(self._settings.brush_radius)
        brush_layout = QtWidgets.QFormLayout()
        brush_layout.addRow("Tool", self.brush_combo)
        brush_layout.addRow("Radius", self.brush_radius)
        layout.addLayout(brush_layout)

        self.min_size_spin = QtWidgets.QSpinBox()
        self.min_size_spin.setRange(0, 50000)
        self.min_size_spin.setValue(self._settings.min_size)
        self.max_size_spin = QtWidgets.QSpinBox()
        self.max_size_spin.setRange(0, 100000)
        self.max_size_spin.setValue(self._settings.max_size)
        self.borderkill_check = QtWidgets.QCheckBox("Remove border-touching objects")
        self.borderkill_check.setChecked(self._settings.borderkill)
        size_layout = QtWidgets.QFormLayout()
        size_layout.addRow("Min size", self.min_size_spin)
        size_layout.addRow("Max size", self.max_size_spin)
        layout.addLayout(size_layout)
        layout.addWidget(self.borderkill_check)

        button_layout = QtWidgets.QHBoxLayout()
        self.generate_button = QtWidgets.QPushButton("Preview")
        self.filter_button = QtWidgets.QPushButton("Filter")
        self.save_button = QtWidgets.QPushButton("Save mask")
        button_layout.addWidget(self.generate_button)
        button_layout.addWidget(self.filter_button)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)

        undo_layout = QtWidgets.QHBoxLayout()
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.redo_button = QtWidgets.QPushButton("Redo")
        undo_layout.addWidget(self.undo_button)
        undo_layout.addWidget(self.redo_button)
        layout.addLayout(undo_layout)

        layout.addStretch()

        # Connections
        self.generate_button.clicked.connect(self._emit_generate)
        self.filter_button.clicked.connect(self._emit_filter)
        self.save_button.clicked.connect(lambda: self.saveRequested.emit(self._channel))
        self.display_combo.currentTextChanged.connect(self.displayModeChanged)
        self.brush_radius.valueChanged.connect(self._emit_brush)
        self.brush_combo.currentTextChanged.connect(self._emit_brush)
        self.undo_button.clicked.connect(self.undoRequested)
        self.redo_button.clicked.connect(self.redoRequested)

    def _emit_generate(self) -> None:
        self._sync_settings()
        self.generateRequested.emit(self._channel, self._settings)

    def _emit_brush(self) -> None:
        self._sync_settings()
        self.brushChanged.emit(self._settings.brush_radius, self.brush_combo.currentText())

    def _emit_filter(self) -> None:
        self._sync_settings()
        self.filterRequested.emit()

    def _sync_settings(self) -> None:
        self._settings.block_size = int(self.block_spin.value()) | 1
        self._settings.offset = int(self.offset_spin.value())
        self._settings.display_mode = self.display_combo.currentText()
        self._settings.brush_radius = int(self.brush_radius.value())
        self._settings.min_size = int(self.min_size_spin.value())
        self._settings.max_size = int(self.max_size_spin.value())
        self._settings.borderkill = self.borderkill_check.isChecked()
        self._per_channel[self._channel] = self._settings

    def set_channel(self, name: str, data: Optional[Dict]) -> None:
        self._channel = name
        self.channel_label.setText(f"Channel: {name}")
        if name not in self._per_channel:
            self._per_channel[name] = SegmentationSettings()
        if data and name in data:
            self._per_channel[name] = SegmentationSettings.from_dict(data[name])
        self._settings = self._per_channel[name]
        self.block_spin.setValue(self._settings.block_size)
        self.offset_spin.setValue(self._settings.offset)
        self.display_combo.setCurrentText(self._settings.display_mode)
        self.brush_radius.setValue(self._settings.brush_radius)
        self.min_size_spin.setValue(self._settings.min_size)
        self.max_size_spin.setValue(self._settings.max_size)
        self.borderkill_check.setChecked(self._settings.borderkill)

    def export_settings(self) -> Dict[str, Dict]:
        self._sync_settings()
        return {name: settings.to_dict() for name, settings in self._per_channel.items()}

    def settings_for(self, channel: str) -> SegmentationSettings:
        if channel not in self._per_channel:
            self._per_channel[channel] = SegmentationSettings()
        if channel == self._channel:
            self._sync_settings()
        return self._per_channel[channel]


# ---------------------------------------------------------------------------
# Algorithms


def adaptive_threshold(image: np.ndarray, settings: SegmentationSettings) -> np.ndarray:
    arr = image.astype(np.uint8)
    if arr.max() > 255:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    block_size = max(3, settings.block_size | 1)
    thresh = cv2.adaptiveThreshold(
        arr,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        settings.offset,
    )
    mask = (thresh > 0).astype(np.uint8)
    return mask


def apply_filters(mask: np.ndarray, settings: SegmentationSettings) -> np.ndarray:
    filtered = mask.astype(bool)
    if settings.min_size > 0:
        filtered = morphology.remove_small_objects(filtered, min_size=settings.min_size)
    if settings.max_size > 0:
        labeled = measure.label(filtered)
        props = measure.regionprops(labeled)
        allowed = np.zeros_like(filtered)
        for prop in props:
            if prop.area <= settings.max_size:
                allowed[labeled == prop.label] = True
        filtered = allowed
    if settings.borderkill:
        filtered &= ~_touches_border(filtered)
    return filtered.astype(np.uint8)


def _touches_border(mask: np.ndarray) -> np.ndarray:
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_pixels = mask & border
    labeled = measure.label(mask)
    kill = np.zeros_like(mask, dtype=bool)
    for label in np.unique(labeled[border_pixels]):
        if label == 0:
            continue
        kill[labeled == label] = True
    return kill
