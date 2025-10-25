"""Batch processing helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from PyQt5 import QtCore, QtWidgets


@dataclass
class BatchRequest:
    root: str
    steps: Dict[str, bool]


class BatchPanel(QtWidgets.QWidget):
    runRequested = QtCore.pyqtSignal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.path_edit = QtWidgets.QLineEdit()
        self.choose_button = QtWidgets.QPushButton("Browse…")
        path_layout = QtWidgets.QHBoxLayout()
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.choose_button)
        layout.addLayout(path_layout)

        self.preprocess_check = QtWidgets.QCheckBox("Preprocessing")
        self.align_check = QtWidgets.QCheckBox("Alignment")
        self.segment_check = QtWidgets.QCheckBox("Segmentation")
        for chk in (self.preprocess_check, self.align_check, self.segment_check):
            chk.setChecked(True)
            layout.addWidget(chk)

        self.run_button = QtWidgets.QPushButton("Run batch")
        layout.addWidget(self.run_button)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        layout.addStretch()

        self.choose_button.clicked.connect(self._choose_directory)
        self.run_button.clicked.connect(self._emit_request)

    def _choose_directory(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select dataset root")
        if path:
            self.path_edit.setText(path)

    def _emit_request(self) -> None:
        request = BatchRequest(
            root=self.path_edit.text(),
            steps={
                "preprocessing": self.preprocess_check.isChecked(),
                "alignment": self.align_check.isChecked(),
                "segmentation": self.segment_check.isChecked(),
            },
        )
        self.runRequested.emit(request)

    def log_message(self, message: str) -> None:
        self.log.appendPlainText(message)

    def update_progress(self, percent: int) -> None:
        self.progress.setValue(percent)
