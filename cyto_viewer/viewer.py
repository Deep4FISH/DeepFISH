"""PyQt-based viewer for cytogenetic metaphase images.

This module implements a navigation interface similar to a light table
that lets users browse 12-bit spectral components (DAPI, Cy3, Cy5) and a
pseudo-colour composite. The implementation focusses on providing
adjustable linear histogram stretching per channel and specialised
colour assignments in HSV space instead of simply stacking channels in
RGB.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from matplotlib.colors import hsv_to_rgb
from skimage import exposure, io

from PyQt5 import QtCore, QtGui, QtWidgets

CHANNEL_ORDER = ["dapi", "cy3", "cy5"]
CHANNEL_NAMES = {"dapi": "DAPI", "cy3": "Cy3", "cy5": "Cy5"}

# HSV hues assigned to each spectral component (degrees / 360).
CHANNEL_HUES = {
    "dapi": 210.0 / 360.0,  # teal leaning blue
    "cy3": 40.0 / 360.0,    # orange-yellow
    "cy5": 350.0 / 360.0,   # cherry red
}


@dataclass
class HistogramSettings:
    """Configuration for linear histogram stretching."""

    low_percent: float = 1.0
    high_percent: float = 99.0

    def clamp(self) -> None:
        self.low_percent = max(0.0, min(self.low_percent, 100.0))
        self.high_percent = max(0.0, min(self.high_percent, 100.0))
        if self.high_percent <= self.low_percent:
            # Maintain at least 1 % separation to avoid degenerate scaling.
            if self.low_percent >= 99.0:
                self.low_percent = 98.0
                self.high_percent = 99.0
            else:
                self.high_percent = min(100.0, self.low_percent + 1.0)


@dataclass
class Metaphase:
    """Represents one metaphase folder and its spectral components."""

    name: str
    channels: Dict[str, List[Path]]

    def available_channels(self) -> Iterable[str]:
        return (ch for ch in CHANNEL_ORDER if ch in self.channels)


class ImageRepository:
    """Loads and caches 12-bit TIFF images from a dataset hierarchy."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def metaphases(self) -> List[Metaphase]:
        entries: List[Metaphase] = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir():
                continue
            channels: Dict[str, List[Path]] = {}
            for channel in CHANNEL_ORDER:
                chan_dir = path / channel
                if not chan_dir.exists() or not chan_dir.is_dir():
                    continue
                files = sorted(
                    p for p in chan_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff"}
                )
                if files:
                    channels[channel] = files
            if channels:
                entries.append(Metaphase(path.name, channels))
        return entries

    @lru_cache(maxsize=256)
    def load_image(self, path: Path) -> np.ndarray:
        data = io.imread(str(path))
        if data.ndim != 2:
            raise ValueError(f"Expected 2D image for {path}, got shape {data.shape}")
        return data.astype(np.float32)


class ThumbnailCache:
    """Caches generated thumbnails to avoid recomputing on every refresh."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str, int, float, float], QtGui.QPixmap] = {}

    def get(
        self,
        key: Tuple[str, str, int, float, float],
    ) -> Optional[QtGui.QPixmap]:
        return self._cache.get(key)

    def store(
        self,
        key: Tuple[str, str, int, float, float],
        pixmap: QtGui.QPixmap,
    ) -> None:
        self._cache[key] = pixmap

    def clear(self) -> None:
        self._cache.clear()


def apply_histogram_stretch(
    image: np.ndarray, settings: HistogramSettings
) -> np.ndarray:
    """Apply a linear histogram stretch based on percentile cut-offs."""

    # Ensure we always operate on a float array without mutating the cached source.
    working = image.astype(np.float32, copy=False)

    settings.clamp()
    with np.errstate(invalid="ignore"):
        low, high = np.nanpercentile(
            working, [settings.low_percent, settings.high_percent]
        )

    if not np.isfinite(low) or not np.isfinite(high):
        return np.zeros_like(working)

    if high < low:
        low, high = high, low

    if np.isclose(high, low):
        clipped = np.clip(working, low, high)
        if high <= 0:
            return np.zeros_like(working)
        return (clipped - low) / max(high - low, 1e-6)

    try:
        stretched = exposure.rescale_intensity(
            working, in_range=(low, high), out_range=(0.0, 1.0)
        )
    except Exception:
        clipped = np.clip(working, low, high)
        return (clipped - low) / max(high - low, 1e-6)

    # ``rescale_intensity`` may return a view with the original dtype. Guarantee float32.
    return stretched.astype(np.float32, copy=False)


def channel_to_qimage(image: np.ndarray) -> QtGui.QImage:
    """Convert a normalised single-channel image into a grayscale QImage."""

    arr = np.clip(image, 0.0, 1.0)
    data = (arr * 255).astype(np.uint8)
    height, width = data.shape
    bytes_per_line = width
    return QtGui.QImage(
        data.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8
    ).copy()


def rgb_array_to_qimage(image: np.ndarray) -> QtGui.QImage:
    """Convert a normalised RGB array into a QImage."""

    arr = np.clip(image, 0.0, 1.0)
    data = (arr * 255).astype(np.uint8)
    height, width, _ = data.shape
    bytes_per_line = 3 * width
    return QtGui.QImage(
        data.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
    ).copy()


def make_pseudo_colour(
    stretched_images: Dict[str, np.ndarray],
    hues: Dict[str, float] = CHANNEL_HUES,
) -> np.ndarray:
    """Generate a pseudo-colour composite by tinting each component in HSV."""

    rgb_layers: List[np.ndarray] = []
    for channel in CHANNEL_ORDER:
        image = stretched_images.get(channel)
        if image is None:
            continue
        hue = hues.get(channel, 0.0)
        hsv = np.stack(
            [np.full_like(image, hue), np.ones_like(image), image], axis=-1
        )
        rgb = hsv_to_rgb(hsv)
        rgb_layers.append(rgb)
    if not rgb_layers:
        raise ValueError("No channels provided for pseudo-colour composite")
    composite = np.zeros_like(rgb_layers[0])
    for layer in rgb_layers:
        composite += layer
    return np.clip(composite, 0.0, 1.0)


class ChannelControl(QtWidgets.QGroupBox):
    """Widget with percentile controls for a single channel."""

    settings_changed = QtCore.pyqtSignal(str, HistogramSettings)

    def __init__(self, channel: str, settings: HistogramSettings, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(CHANNEL_NAMES.get(channel, channel), parent)
        self.channel = channel
        self.settings = settings
        self.low_spin = QtWidgets.QDoubleSpinBox()
        self.high_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.low_spin, self.high_spin):
            spin.setRange(0.0, 100.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.25)
        self.low_spin.setValue(settings.low_percent)
        self.high_spin.setValue(settings.high_percent)
        layout = QtWidgets.QFormLayout(self)
        layout.addRow(self.tr("Seuil bas (%)"), self.low_spin)
        layout.addRow(self.tr("Seuil haut (%)"), self.high_spin)
        self.low_spin.valueChanged.connect(self._emit_changed)
        self.high_spin.valueChanged.connect(self._emit_changed)

    def _emit_changed(self) -> None:
        self.settings.low_percent = self.low_spin.value()
        self.settings.high_percent = self.high_spin.value()
        self.settings.clamp()
        # Update spin boxes in case clamp adjusted the values.
        self.low_spin.blockSignals(True)
        self.high_spin.blockSignals(True)
        self.low_spin.setValue(self.settings.low_percent)
        self.high_spin.setValue(self.settings.high_percent)
        self.low_spin.blockSignals(False)
        self.high_spin.blockSignals(False)
        self.settings_changed.emit(self.channel, self.settings)


class HistogramPanel(QtWidgets.QWidget):
    """Panel hosting controls for all channel histogram settings."""

    settings_changed = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.controls: Dict[str, ChannelControl] = {}
        layout = QtWidgets.QHBoxLayout(self)
        for channel in CHANNEL_ORDER:
            control = ChannelControl(channel, HistogramSettings())
            self.controls[channel] = control
            control.settings_changed.connect(self._relay)
            layout.addWidget(control)
        layout.addStretch(1)

    def get_settings(self) -> Dict[str, HistogramSettings]:
        return {ch: control.settings for ch, control in self.controls.items()}

    def _relay(self, channel: str, settings: HistogramSettings) -> None:
        del channel, settings
        self.settings_changed.emit()


class LightTable(QtWidgets.QMainWindow):
    """Main application window implementing the light-table viewer."""

    def __init__(self, repo: ImageRepository, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cytogenetic Light Table Viewer")
        self.repo = repo
        self.metaphases = repo.metaphases()
        self.thumbnail_cache = ThumbnailCache()
        self.hist_panel = HistogramPanel()
        self.hist_panel.settings_changed.connect(self.refresh_previews)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(len(CHANNEL_ORDER) + 1)
        headers = [CHANNEL_NAMES[ch] for ch in CHANNEL_ORDER] + [self.tr("Pseudo-couleur")]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.table.setIconSize(QtCore.QSize(160, 160))
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._update_preview_from_selection)

        self.preview_label = QtWidgets.QLabel(self.tr("Sélectionnez une image."))
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumSize(320, 320)
        self.preview_label.setBackgroundRole(QtGui.QPalette.Base)
        self.preview_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.file_list = QtWidgets.QListWidget()
        self.file_list.itemSelectionChanged.connect(self._file_index_changed)

        right_pane = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_pane)
        right_layout.addWidget(self.preview_label, stretch=3)
        right_layout.addWidget(QtWidgets.QLabel(self.tr("Fichiers disponibles")))
        right_layout.addWidget(self.file_list, stretch=1)
        right_layout.addWidget(self.hist_panel, stretch=0)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.table)
        splitter.addWidget(right_pane)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        self._metaphase_indices: Dict[str, Dict[str, int]] = {
            meta.name: {channel: 0 for channel in meta.channels}
            for meta in self.metaphases
        }

        self.populate_table()

    def populate_table(self) -> None:
        self.table.setRowCount(len(self.metaphases))
        for row, meta in enumerate(self.metaphases):
            name_item = QtWidgets.QTableWidgetItem(meta.name)
            name_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.table.setVerticalHeaderItem(row, name_item)
            for col, channel in enumerate(CHANNEL_ORDER):
                pixmap = self._make_channel_thumbnail(meta, channel)
                item = QtWidgets.QTableWidgetItem()
                if pixmap is not None:
                    item.setIcon(QtGui.QIcon(pixmap))
                    item.setToolTip(f"{meta.name} — {CHANNEL_NAMES[channel]}")
                else:
                    item.setText(self.tr("(absent)"))
                    item.setFlags(QtCore.Qt.NoItemFlags)
                self.table.setItem(row, col, item)
            pseudo_pixmap = self._make_pseudo_thumbnail(meta)
            pseudo_item = QtWidgets.QTableWidgetItem()
            if pseudo_pixmap is not None:
                pseudo_item.setIcon(QtGui.QIcon(pseudo_pixmap))
                pseudo_item.setToolTip(f"{meta.name} — {self.tr('Pseudo-couleur')}")
            else:
                pseudo_item.setText(self.tr("(indisponible)"))
                pseudo_item.setFlags(QtCore.Qt.NoItemFlags)
            self.table.setItem(row, len(CHANNEL_ORDER), pseudo_item)
        self.table.resizeRowsToContents()

    def _make_channel_thumbnail(self, meta: Metaphase, channel: str) -> Optional[QtGui.QPixmap]:
        files = meta.channels.get(channel)
        if not files:
            return None
        key = (meta.name, channel, 0, self.hist_panel.controls[channel].settings.low_percent,
               self.hist_panel.controls[channel].settings.high_percent)
        cached = self.thumbnail_cache.get(key)
        if cached is not None:
            return cached
        image = self.repo.load_image(files[0])
        stretched = apply_histogram_stretch(image, self.hist_panel.controls[channel].settings)
        pixmap = self._array_to_pixmap(stretched)
        self.thumbnail_cache.store(key, pixmap)
        return pixmap

    def _make_pseudo_thumbnail(self, meta: Metaphase) -> Optional[QtGui.QPixmap]:
        stretched: Dict[str, np.ndarray] = {}
        for channel in CHANNEL_ORDER:
            files = meta.channels.get(channel)
            if not files:
                continue
            image = self.repo.load_image(files[0])
            stretched[channel] = apply_histogram_stretch(
                image, self.hist_panel.controls[channel].settings
            )
        if not stretched:
            return None
        composite = make_pseudo_colour(stretched)
        pixmap = self._array_to_pixmap(composite)
        return pixmap

    def _array_to_pixmap(self, array: np.ndarray) -> QtGui.QPixmap:
        if array.ndim == 2:
            qimage = channel_to_qimage(array)
        else:
            qimage = rgb_array_to_qimage(array)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap.scaled(
            self.table.iconSize(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )

    def refresh_previews(self) -> None:
        self.thumbnail_cache.clear()
        self.populate_table()
        self._update_preview_from_selection()

    def _selected_cell(self) -> Optional[Tuple[int, int]]:
        items = self.table.selectedIndexes()
        if not items:
            return None
        index = items[0]
        return index.row(), index.column()

    def _update_preview_from_selection(self) -> None:
        selection = self._selected_cell()
        if selection is None:
            return
        row, column = selection
        if row >= len(self.metaphases):
            return
        meta = self.metaphases[row]
        if column < len(CHANNEL_ORDER):
            channel = CHANNEL_ORDER[column]
            self._show_channel(meta, channel)
        else:
            self._show_pseudo(meta)

    def _show_channel(self, meta: Metaphase, channel: str) -> None:
        files = meta.channels.get(channel)
        if not files:
            self.preview_label.setText(self.tr("Canal indisponible"))
            self.file_list.clear()
            return
        index = self._metaphase_indices[meta.name].get(channel, 0)
        index = max(0, min(index, len(files) - 1))
        self._metaphase_indices[meta.name][channel] = index
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for i, file in enumerate(files):
            item = QtWidgets.QListWidgetItem(file.name)
            item.setData(QtCore.Qt.UserRole, i)
            self.file_list.addItem(item)
            if i == index:
                item.setSelected(True)
        self.file_list.blockSignals(False)
        image = self.repo.load_image(files[index])
        stretched = apply_histogram_stretch(image, self.hist_panel.controls[channel].settings)
        pixmap = QtGui.QPixmap.fromImage(channel_to_qimage(stretched))
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )
        self.preview_label.setToolTip(f"{meta.name} — {CHANNEL_NAMES[channel]} — {files[index].name}")

    def _show_pseudo(self, meta: Metaphase) -> None:
        self.file_list.blockSignals(True)
        self.file_list.clear()
        self.file_list.blockSignals(False)
        stretched: Dict[str, np.ndarray] = {}
        for channel in CHANNEL_ORDER:
            files = meta.channels.get(channel)
            if not files:
                continue
            index = self._metaphase_indices[meta.name].get(channel, 0)
            index = max(0, min(index, len(files) - 1))
            image = self.repo.load_image(files[index])
            stretched[channel] = apply_histogram_stretch(
                image, self.hist_panel.controls[channel].settings
            )
        if not stretched:
            self.preview_label.setText(self.tr("Pas de données pour la pseudo-couleur"))
            return
        composite = make_pseudo_colour(stretched)
        pixmap = QtGui.QPixmap.fromImage(rgb_array_to_qimage(composite))
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )
        self.preview_label.setToolTip(f"{meta.name} — {self.tr('Pseudo-couleur')}")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_preview_from_selection()

    def _file_index_changed(self) -> None:
        selection = self._selected_cell()
        if selection is None:
            return
        row, column = selection
        meta = self.metaphases[row]
        if column >= len(CHANNEL_ORDER):
            return
        channel = CHANNEL_ORDER[column]
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        index = selected_items[0].data(QtCore.Qt.UserRole)
        self._metaphase_indices[meta.name][channel] = index
        self._show_channel(meta, channel)


def run_viewer(root: Path) -> None:
    """Start the Qt application for the provided dataset root."""

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    repo = ImageRepository(root)
    window = LightTable(repo)
    if not window.metaphases:
        QtWidgets.QMessageBox.warning(
            window,
            window.tr("Données manquantes"),
            window.tr("Aucune métaphase n'a été trouvée dans le dossier fourni."),
        )
    window.resize(1200, 700)
    window.show()
    app.exec_()


def _candidate_roots(start: Path) -> Iterable[Path]:
    """Yield plausible dataset folders relative to ``start`` and its parents."""

    dataset_names = [
        Path("Raw images") / "jpp21",
        Path("dataset"),
        Path("jpp21_downloaded_data"),
        Path("jpp21"),
    ]
    for base in [start, *start.parents]:
        for name in dataset_names:
            candidate = (base / name).resolve()
            yield candidate
        yield base.resolve()


def find_default_root() -> Path:
    """Try to guess a default dataset location."""

    for candidate in _candidate_roots(Path.cwd()):
        if candidate.exists() and candidate.is_dir():
            return candidate
    return Path.cwd()


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Viewer for cytogenetic spectral components")
    parser.add_argument(
        "root",
        nargs="?",
        default=str(find_default_root()),
        help="Folder containing metaphase sub-directories",
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unrecognised Qt arguments: {unknown}")
    run_viewer(Path(args.root))


if __name__ == "__main__":
    main()
