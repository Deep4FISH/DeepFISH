"""Manual alignment controls."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets


@dataclass
class AlignmentState:
    offsets: Dict[str, Tuple[int, int]]

    def to_dict(self) -> Dict[str, Tuple[int, int]]:
        return {k: list(v) for k, v in self.offsets.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Tuple[int, int]]) -> "AlignmentState":
        return cls(offsets={k: tuple(v) for k, v in data.items()})


class AlignmentPanel(QtWidgets.QWidget):
    translationChanged = QtCore.pyqtSignal(str, int, int)
    applyAllRequested = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._channel = "cy3"
        self._offsets: Dict[str, Tuple[int, int]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel("Channel: cy3 relative to DAPI")
        layout.addWidget(self.label)

        grid = QtWidgets.QGridLayout()
        self.dy_spin = QtWidgets.QSpinBox()
        self.dy_spin.setRange(-200, 200)
        self.dx_spin = QtWidgets.QSpinBox()
        self.dx_spin.setRange(-200, 200)
        grid.addWidget(QtWidgets.QLabel("dy"), 0, 0)
        grid.addWidget(self.dy_spin, 0, 1)
        grid.addWidget(QtWidgets.QLabel("dx"), 1, 0)
        grid.addWidget(self.dx_spin, 1, 1)
        layout.addLayout(grid)

        arrows = QtWidgets.QGridLayout()
        self.btn_up = QtWidgets.QPushButton("↑")
        self.btn_down = QtWidgets.QPushButton("↓")
        self.btn_left = QtWidgets.QPushButton("←")
        self.btn_right = QtWidgets.QPushButton("→")
        arrows.addWidget(self.btn_up, 0, 1)
        arrows.addWidget(self.btn_left, 1, 0)
        arrows.addWidget(self.btn_right, 1, 2)
        arrows.addWidget(self.btn_down, 2, 1)
        layout.addLayout(arrows)

        self.apply_all_button = QtWidgets.QPushButton("Apply to all metaphases")
        layout.addWidget(self.apply_all_button)
        layout.addStretch()

        self.dy_spin.valueChanged.connect(self._emit_change)
        self.dx_spin.valueChanged.connect(self._emit_change)
        self.btn_up.clicked.connect(lambda: self.dy_spin.setValue(self.dy_spin.value() - 1))
        self.btn_down.clicked.connect(lambda: self.dy_spin.setValue(self.dy_spin.value() + 1))
        self.btn_left.clicked.connect(lambda: self.dx_spin.setValue(self.dx_spin.value() - 1))
        self.btn_right.clicked.connect(lambda: self.dx_spin.setValue(self.dx_spin.value() + 1))
        self.apply_all_button.clicked.connect(self.applyAllRequested)

    def set_channel(self, channel: str, data: Dict[str, Tuple[int, int]]) -> None:
        self._channel = channel
        self.label.setText(f"Channel: {channel} relative to DAPI")
        if channel not in self._offsets:
            self._offsets[channel] = (0, 0)
        if data and channel in data:
            self._offsets[channel] = tuple(data[channel])
        self.dy_spin.blockSignals(True)
        self.dx_spin.blockSignals(True)
        dy, dx = self._offsets[channel]
        self.dy_spin.setValue(int(dy))
        self.dx_spin.setValue(int(dx))
        self.dy_spin.blockSignals(False)
        self.dx_spin.blockSignals(False)

    def export_offsets(self) -> Dict[str, Tuple[int, int]]:
        return {ch: (int(dy), int(dx)) for ch, (dy, dx) in self._offsets.items()}

    def _emit_change(self) -> None:
        offsets = (self.dy_spin.value(), self.dx_spin.value())
        self._offsets[self._channel] = offsets
        self.translationChanged.emit(self._channel, *offsets)


def apply_translation(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(image, int(dy), axis=0), int(dx), axis=1)
