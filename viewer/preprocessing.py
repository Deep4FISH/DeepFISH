"""Preprocessing controls and algorithms."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import colors as mpl_colors
from skimage import exposure, morphology


@dataclass
class PreprocessingSettings:
    """Configuration for a single channel."""

    radius: int = 10
    background_method: str = "median"
    display_mode: str = "grayscale"  # grayscale, inverted, false_color
    false_color_min: Tuple[float, float, float] = (0.6, 0.4, 0.9)  # HSV
    false_color_max: Tuple[float, float, float] = (0.0, 1.0, 1.0)
    clip_low: float = 2.0
    clip_high: float = 98.0

    def to_dict(self) -> Dict[str, float]:
        data = asdict(self)
        data["false_color_min"] = list(self.false_color_min)
        data["false_color_max"] = list(self.false_color_max)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "PreprocessingSettings":
        if not data:
            return cls()
        values = dict(data)
        if "false_color_min" in values:
            values["false_color_min"] = tuple(values["false_color_min"])
        if "false_color_max" in values:
            values["false_color_max"] = tuple(values["false_color_max"])
        return cls(**values)


class PreprocessingPanel(QtWidgets.QWidget):
    """Qt widget exposing preprocessing controls."""

    settingsChanged = QtCore.pyqtSignal(str, object)
    applyRequested = QtCore.pyqtSignal(str)
    batchRequested = QtCore.pyqtSignal(str)
    saveRequested = QtCore.pyqtSignal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._channel = "dapi"
        self._settings: Dict[str, PreprocessingSettings] = {
            "dapi": PreprocessingSettings(),
            "cy3": PreprocessingSettings(false_color_min=(0.15, 0.9, 0.95), false_color_max=(0.07, 1.0, 1.0)),
            "cy5": PreprocessingSettings(false_color_min=(0.0, 1.0, 0.6), false_color_max=(0.0, 1.0, 1.0)),
            "fitc": PreprocessingSettings(false_color_min=(0.33, 0.8, 0.7), false_color_max=(0.25, 1.0, 1.0)),
        }
        self._build_ui()
        self._update_ui()

    # ------------------------------------------------------------------
    # UI helpers
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.channel_label = QtWidgets.QLabel("Channel: dapi")
        layout.addWidget(self.channel_label)

        self.radius_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.radius_slider.setRange(0, 50)
        self.radius_slider.setValue(10)
        layout.addWidget(QtWidgets.QLabel("White top-hat radius"))
        layout.addWidget(self.radius_slider)

        self.background_combo = QtWidgets.QComboBox()
        self.background_combo.addItems(["median", "mode"])
        layout.addWidget(QtWidgets.QLabel("Background subtraction"))
        layout.addWidget(self.background_combo)

        self.display_combo = QtWidgets.QComboBox()
        self.display_combo.addItems(["grayscale", "inverted", "false_color"])
        layout.addWidget(QtWidgets.QLabel("Display mode"))
        layout.addWidget(self.display_combo)

        self.clip_low_spin = QtWidgets.QDoubleSpinBox()
        self.clip_low_spin.setRange(0.0, 50.0)
        self.clip_low_spin.setSuffix(" %")
        self.clip_low_spin.setValue(2.0)
        self.clip_high_spin = QtWidgets.QDoubleSpinBox()
        self.clip_high_spin.setRange(50.0, 100.0)
        self.clip_high_spin.setSuffix(" %")
        self.clip_high_spin.setValue(98.0)
        clip_layout = QtWidgets.QFormLayout()
        clip_layout.addRow("Clip low", self.clip_low_spin)
        clip_layout.addRow("Clip high", self.clip_high_spin)
        layout.addLayout(clip_layout)

        color_layout = QtWidgets.QHBoxLayout()
        self.min_color_button = QtWidgets.QPushButton("Min colour")
        self.max_color_button = QtWidgets.QPushButton("Max colour")
        color_layout.addWidget(self.min_color_button)
        color_layout.addWidget(self.max_color_button)
        layout.addLayout(color_layout)

        button_layout = QtWidgets.QHBoxLayout()
        self.apply_button = QtWidgets.QPushButton("Apply")
        self.batch_button = QtWidgets.QPushButton("Batch")
        self.save_button = QtWidgets.QPushButton("Save result")
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.batch_button)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)

        layout.addStretch()

        # Signals
        self.radius_slider.valueChanged.connect(self._emit_change)
        self.background_combo.currentTextChanged.connect(self._emit_change)
        self.display_combo.currentTextChanged.connect(self._emit_change)
        self.clip_low_spin.valueChanged.connect(self._emit_change)
        self.clip_high_spin.valueChanged.connect(self._emit_change)
        self.min_color_button.clicked.connect(lambda: self._choose_color(True))
        self.max_color_button.clicked.connect(lambda: self._choose_color(False))
        self.apply_button.clicked.connect(lambda: self.applyRequested.emit(self._channel))
        self.batch_button.clicked.connect(lambda: self.batchRequested.emit(self._channel))
        self.save_button.clicked.connect(lambda: self.saveRequested.emit(self._channel))

    def _choose_color(self, is_min: bool) -> None:
        settings = self._settings[self._channel]
        hsv = settings.false_color_min if is_min else settings.false_color_max
        qcolor = QtGui.QColor()
        qcolor.setHsvF(*hsv)
        chosen = QtWidgets.QColorDialog.getColor(qcolor, self)
        if chosen.isValid():
            hsv = (chosen.hueF(), chosen.saturationF(), chosen.valueF())
            if is_min:
                settings.false_color_min = hsv
            else:
                settings.false_color_max = hsv
            self._emit_change()

    def _update_ui(self) -> None:
        settings = self._settings[self._channel]
        self.channel_label.setText(f"Channel: {self._channel}")
        self.radius_slider.blockSignals(True)
        self.background_combo.blockSignals(True)
        self.display_combo.blockSignals(True)
        self.clip_low_spin.blockSignals(True)
        self.clip_high_spin.blockSignals(True)
        self.radius_slider.setValue(settings.radius)
        self.background_combo.setCurrentText(settings.background_method)
        self.display_combo.setCurrentText(settings.display_mode)
        self.clip_low_spin.setValue(settings.clip_low)
        self.clip_high_spin.setValue(settings.clip_high)
        self.radius_slider.blockSignals(False)
        self.background_combo.blockSignals(False)
        self.display_combo.blockSignals(False)
        self.clip_low_spin.blockSignals(False)
        self.clip_high_spin.blockSignals(False)

    def set_channel(self, name: str, data: Dict[str, Dict]) -> None:
        self._channel = name
        if name not in self._settings:
            self._settings[name] = PreprocessingSettings()
        if data and name in data:
            self._settings[name] = PreprocessingSettings.from_dict(data[name])
        self._update_ui()

    def export_settings(self) -> Dict[str, Dict]:
        self._emit_change()
        return {ch: cfg.to_dict() for ch, cfg in self._settings.items()}

    def settings_for(self, channel: str) -> PreprocessingSettings:
        return self._settings.setdefault(channel, PreprocessingSettings())

    def _emit_change(self) -> None:
        settings = self._settings[self._channel]
        settings.radius = self.radius_slider.value()
        settings.background_method = self.background_combo.currentText()
        settings.display_mode = self.display_combo.currentText()
        settings.clip_low = self.clip_low_spin.value()
        settings.clip_high = self.clip_high_spin.value()
        if settings.clip_low >= settings.clip_high:
            settings.clip_high = min(100.0, settings.clip_low + 1.0)
            self.clip_high_spin.blockSignals(True)
            self.clip_high_spin.setValue(settings.clip_high)
            self.clip_high_spin.blockSignals(False)
        self.settingsChanged.emit(self._channel, settings)


# ---------------------------------------------------------------------------
# Processing functions


def apply_white_tophat(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image
    selem = morphology.disk(radius)
    return morphology.white_tophat(image, selem)


def subtract_background(image: np.ndarray, method: str) -> np.ndarray:
    if method == "mode":
        hist, bin_edges = np.histogram(image, bins=512)
        mode_index = int(np.argmax(hist))
        background = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2
    else:
        background = float(np.median(image))
    corrected = image - background
    corrected[corrected < 0] = 0
    return corrected


def stretch_contrast(image: np.ndarray, low: float, high: float) -> np.ndarray:
    low_val, high_val = np.percentile(image, (low, high))
    if np.isclose(low_val, high_val):
        return np.zeros_like(image)
    return exposure.rescale_intensity(image, in_range=(low_val, high_val))


def preprocess_image(image: np.ndarray, settings: PreprocessingSettings) -> np.ndarray:
    arr = image.astype(np.float32)
    arr = apply_white_tophat(arr, settings.radius)
    arr = subtract_background(arr, settings.background_method)
    arr = stretch_contrast(arr, settings.clip_low, settings.clip_high)
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val > 0:
        arr = arr / max_val
    return arr


def map_false_colour(image: np.ndarray, settings: PreprocessingSettings) -> np.ndarray:
    arr = np.clip(image, 0, 1)
    h_low, s_low, v_low = settings.false_color_min
    h_high, s_high, v_high = settings.false_color_max
    h = h_low + (h_high - h_low) * arr
    s = s_low + (s_high - s_low) * arr
    v = v_low + (v_high - v_low) * arr
    hsv = np.stack([h, s, v], axis=-1)
    rgb = mpl_colors.hsv_to_rgb(hsv)
    return rgb
