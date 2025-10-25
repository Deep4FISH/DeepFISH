"""Entry point for the Cytogenetic Image Multi-Channel Viewer."""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from . import alignment, batch, config, preprocessing, segmentation
from .utils import data
from skimage import morphology


def numpy_to_qimage(arr: np.ndarray) -> QtGui.QImage:
    """Convert a float image (0-1) or uint8/uint16 to a QImage."""
    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            norm = np.clip(arr, 0, 1)
            arr8 = (norm * 255).astype(np.uint8)
        else:
            arr8 = arr
        arr8 = np.ascontiguousarray(arr8)
        h, w = arr8.shape
        return QtGui.QImage(arr8.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()
    if arr.ndim == 3 and arr.shape[2] == 3:
        if arr.dtype != np.uint8:
            norm = np.clip(arr, 0, 1)
            arr8 = (norm * 255).astype(np.uint8)
        else:
            arr8 = arr
        arr8 = np.ascontiguousarray(arr8)
        h, w, _ = arr8.shape
        return QtGui.QImage(arr8.data, w, h, w * 3, QtGui.QImage.Format_RGB888).copy()
    raise ValueError("Unsupported array shape for QImage")


class ImageCanvas(QtWidgets.QGraphicsView):
    maskEdited = QtCore.pyqtSignal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)
        self._mask_item = QtWidgets.QGraphicsPixmapItem()
        self._mask_item.setOpacity(0.5)
        self.scene().addItem(self._mask_item)
        self._image_array: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._display_mode = "overlay"
        self._brush_radius = 8
        self._tool = "brush"
        self._editing = False
        self._undo_stack: List[np.ndarray] = []
        self._redo_stack: List[np.ndarray] = []
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

    # ------------------------------------------------------------------
    def set_image(self, qimage: QtGui.QImage, array: np.ndarray) -> None:
        self._image_array = array
        self._pixmap_item.setPixmap(QtGui.QPixmap.fromImage(qimage))
        self.scene().setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, QtCore.Qt.KeepAspectRatio)

    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        if mask is None:
            self._mask = None
            self._mask_item.setPixmap(QtGui.QPixmap())
            self._undo_stack.clear()
            self._redo_stack.clear()
            return
        self._mask = mask.astype(bool)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_mask_pixmap()

    def set_display_mode(self, mode: str) -> None:
        self._display_mode = mode
        self._update_mask_pixmap()

    def set_brush(self, radius: int, tool: str) -> None:
        self._brush_radius = max(1, int(radius))
        self._tool = tool

    def set_editing_enabled(self, enabled: bool) -> None:
        self._editing = enabled

    def undo(self) -> None:
        if not self._undo_stack:
            return
        current = self._mask.copy()
        self._redo_stack.append(current)
        prev = self._undo_stack.pop()
        self._mask = prev
        self._update_mask_pixmap()
        self.maskEdited.emit(self._mask.copy())

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self._mask.copy())
        next_mask = self._redo_stack.pop()
        self._mask = next_mask
        self._update_mask_pixmap()
        self.maskEdited.emit(self._mask.copy())

    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._editing and self._mask is not None and event.buttons() & QtCore.Qt.LeftButton:
            self._push_undo()
            self._apply_brush(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._editing and self._mask is not None and event.buttons() & QtCore.Qt.LeftButton:
            self._apply_brush(event.pos())
        super().mouseMoveEvent(event)

    def _push_undo(self) -> None:
        if self._mask is not None:
            self._undo_stack.append(self._mask.copy())
            self._redo_stack.clear()

    def _apply_brush(self, pos: QtCore.QPoint) -> None:
        if self._mask is None:
            return
        scene_pos = self.mapToScene(pos)
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        if not (0 <= x < self._mask.shape[1] and 0 <= y < self._mask.shape[0]):
            return
        yy, xx = np.ogrid[: self._mask.shape[0], : self._mask.shape[1]]
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= self._brush_radius ** 2
        if self._tool == "brush":
            self._mask[circle] = True
        else:
            self._mask[circle] = False
        self._update_mask_pixmap()
        self.maskEdited.emit(self._mask.copy())

    def _update_mask_pixmap(self) -> None:
        if self._mask is None:
            self._mask_item.setPixmap(QtGui.QPixmap())
            return
        mask_bool = self._mask.astype(bool)
        mask = mask_bool.astype(np.uint8)
        if self._display_mode == "contour":
            eroded = morphology.binary_erosion(mask_bool)
            edges = mask_bool & ~eroded
            rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            rgb[..., 0] = edges * 255
            rgb[..., 1] = edges * 255
            rgb[..., 2] = 0
            qimage = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], QtGui.QImage.Format_RGB888)
            self._mask_item.setPixmap(QtGui.QPixmap.fromImage(qimage))
            return
        color = np.zeros((*mask.shape, 4), dtype=np.uint8)
        if self._display_mode == "overlay":
            color[..., 0] = 255
            color[..., 1] = 128
            color[..., 3] = mask * 120
        else:  # filled
            color[..., 0] = mask * 255
            color[..., 3] = mask * 255
        qimage = QtGui.QImage(color.data, color.shape[1], color.shape[0], QtGui.QImage.Format_RGBA8888)
        self._mask_item.setPixmap(QtGui.QPixmap.fromImage(qimage))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, dataset: data.Dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.current_slide: Optional[data.Slide] = None
        self.current_metaphase: Optional[data.Metaphase] = None
        self.current_channel: str = "dapi"
        self.config: Optional[config.Config] = None
        self.raw_images: Dict[str, np.ndarray] = {}
        self.preprocessed: Dict[str, np.ndarray] = {}
        self.alignment_offsets: Dict[str, tuple] = {}
        self.segmentation_masks: Dict[str, np.ndarray] = {}
        self.segmentation_mask: Optional[np.ndarray] = None
        self._build_ui()
        self._populate_slides()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setWindowTitle("Cytogenetic Image Viewer")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)

        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)

        navigation_layout = QtWidgets.QHBoxLayout()
        self.slide_combo = QtWidgets.QComboBox()
        self.metaphase_combo = QtWidgets.QComboBox()
        navigation_layout.addWidget(QtWidgets.QLabel("Slide"))
        navigation_layout.addWidget(self.slide_combo)
        navigation_layout.addWidget(QtWidgets.QLabel("Metaphase"))
        navigation_layout.addWidget(self.metaphase_combo)
        left_layout.addLayout(navigation_layout)

        self.channel_buttons_layout = QtWidgets.QHBoxLayout()
        left_layout.addLayout(self.channel_buttons_layout)

        self.canvas = ImageCanvas()
        self.canvas.set_editing_enabled(False)
        left_layout.addWidget(self.canvas, stretch=1)

        layout.addWidget(left_container, stretch=3)

        self.tabs = QtWidgets.QTabWidget()
        self.preprocessing_panel = preprocessing.PreprocessingPanel()
        self.alignment_panel = alignment.AlignmentPanel()
        self.segmentation_panel = segmentation.SegmentationPanel()
        self.batch_panel = batch.BatchPanel()
        self.tabs.addTab(self.preprocessing_panel, "Preprocessing")
        self.tabs.addTab(self.alignment_panel, "Alignment")
        self.tabs.addTab(self.segmentation_panel, "Segmentation")
        self.tabs.addTab(self.batch_panel, "Batch")
        layout.addWidget(self.tabs, stretch=1)

        # Signals
        self.slide_combo.currentTextChanged.connect(self._on_slide_changed)
        self.metaphase_combo.currentTextChanged.connect(self._on_metaphase_changed)
        self.preprocessing_panel.settingsChanged.connect(self._on_preprocessing_settings)
        self.preprocessing_panel.applyRequested.connect(self._apply_preprocessing)
        self.preprocessing_panel.batchRequested.connect(self._run_preprocessing_batch)
        self.preprocessing_panel.saveRequested.connect(self._save_preprocessed)
        self.alignment_panel.translationChanged.connect(self._on_alignment_changed)
        self.alignment_panel.applyAllRequested.connect(self._apply_alignment_all)
        self.segmentation_panel.generateRequested.connect(self._on_generate_segmentation)
        self.segmentation_panel.saveRequested.connect(self._save_segmentation)
        self.segmentation_panel.displayModeChanged.connect(self.canvas.set_display_mode)
        self.segmentation_panel.brushChanged.connect(self._on_brush_changed)
        self.segmentation_panel.undoRequested.connect(self.canvas.undo)
        self.segmentation_panel.redoRequested.connect(self.canvas.redo)
        self.segmentation_panel.filterRequested.connect(self._apply_segmentation_filters)
        self.canvas.maskEdited.connect(self._on_mask_edited)
        self.batch_panel.runRequested.connect(self._on_batch_requested)

    # ------------------------------------------------------------------
    def _populate_slides(self) -> None:
        self.slide_combo.blockSignals(True)
        self.slide_combo.clear()
        for slide_name in self.dataset.available_slides():
            self.slide_combo.addItem(slide_name)
        self.slide_combo.blockSignals(False)
        if self.slide_combo.count() > 0:
            self.slide_combo.setCurrentIndex(0)
            self._on_slide_changed(self.slide_combo.currentText())

    def _on_slide_changed(self, name: str) -> None:
        if not name:
            return
        if self.current_slide is not None:
            self._persist_current_config()
        self.current_slide = self.dataset.slides[name]
        self.config = config.load_config(self.current_slide.path)
        self.alignment_offsets = {}
        for k, v in self.config.alignment.items():
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                self.alignment_offsets[k] = (int(v[0]), int(v[1]))
        self.preprocessing_panel.set_channel(self.current_channel, self.config.preprocessing)
        self.alignment_panel.set_channel(self.current_channel, self.alignment_offsets)
        self.segmentation_panel.set_channel(self.current_channel, self.config.segmentation)
        self._populate_metaphases()

    def _populate_metaphases(self) -> None:
        self.metaphase_combo.blockSignals(True)
        self.metaphase_combo.clear()
        if not self.current_slide:
            return
        for meta in self.current_slide.available_metaphases():
            self.metaphase_combo.addItem(meta)
        self.metaphase_combo.blockSignals(False)
        if self.metaphase_combo.count() > 0:
            self.metaphase_combo.setCurrentIndex(0)
            self._on_metaphase_changed(self.metaphase_combo.currentText())

    def _on_metaphase_changed(self, name: str) -> None:
        if not name or not self.current_slide:
            return
        self.current_metaphase = self.current_slide.metaphases[name]
        self.raw_images = {}
        self.preprocessed = {}
        self.segmentation_masks = {}
        self.segmentation_mask = None
        for channel, files in self.current_metaphase.channels.items():
            image = data.load_image(files[0])
            self.raw_images[channel] = image
        self._setup_channel_buttons()
        if self.current_channel not in self.raw_images:
            self.current_channel = self.current_metaphase.available_channels()[0]
        self.preprocessing_panel.set_channel(self.current_channel, self.config.preprocessing)
        self.alignment_panel.set_channel(self.current_channel, self.alignment_offsets)
        self.segmentation_panel.set_channel(self.current_channel, self.config.segmentation)
        self._refresh_view()

    def _setup_channel_buttons(self) -> None:
        # Clear layout
        while self.channel_buttons_layout.count():
            item = self.channel_buttons_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not self.current_metaphase:
            return
        for channel in self.current_metaphase.available_channels():
            btn = QtWidgets.QToolButton()
            btn.setText(channel)
            btn.setCheckable(True)
            btn.setChecked(channel == self.current_channel)
            btn.clicked.connect(functools.partial(self._on_channel_selected, channel))
            self.channel_buttons_layout.addWidget(btn)

    def _on_channel_selected(self, channel: str) -> None:
        self.current_channel = channel
        self.preprocessing_panel.set_channel(channel, self.config.preprocessing)
        self.alignment_panel.set_channel(channel, self.alignment_offsets)
        self.segmentation_panel.set_channel(channel, self.config.segmentation)
        self.segmentation_mask = self.segmentation_masks.get(channel)
        self._refresh_view()

    def _refresh_view(self) -> None:
        if not self.current_metaphase:
            return
        processed = self._get_processed_image(self.current_channel)
        settings = self.preprocessing_panel.settings_for(self.current_channel)
        norm = processed
        if norm.max() > 0:
            norm = processed / processed.max()
        if settings.display_mode == "inverted":
            norm = 1.0 - norm
            qimage = numpy_to_qimage(norm)
        elif settings.display_mode == "false_color":
            rgb = preprocessing.map_false_colour(norm, settings)
            qimage = numpy_to_qimage(rgb)
        else:
            qimage = numpy_to_qimage(norm)
        self.canvas.set_image(qimage, processed)
        mask = self.segmentation_masks.get(self.current_channel)
        self.segmentation_mask = mask
        self.canvas.set_mask(mask)
        self.canvas.set_editing_enabled(mask is not None)

    def _get_processed_image(self, channel: str) -> np.ndarray:
        if channel in self.preprocessed:
            return self.preprocessed[channel]
        raw = self.raw_images[channel]
        settings = self.preprocessing_panel.settings_for(channel)
        processed = preprocessing.preprocess_image(raw, settings)
        offsets = self.alignment_offsets.get(channel, (0, 0))
        processed = alignment.apply_translation(processed, offsets[0], offsets[1])
        self.preprocessed[channel] = processed
        return processed

    def _on_preprocessing_settings(self, channel: str, settings: preprocessing.PreprocessingSettings) -> None:
        if channel in self.preprocessed:
            del self.preprocessed[channel]
        self.config.preprocessing[channel] = settings.to_dict()
        if channel == self.current_channel:
            self._refresh_view()

    def _apply_preprocessing(self, channel: str) -> None:
        self.preprocessed.pop(channel, None)
        self._refresh_view()

    def _run_preprocessing_batch(self, channel: str) -> None:
        if not self.current_slide:
            return
        metaphases = list(self.current_slide.metaphases.values())
        targets = [(meta, meta.channels.get(channel)) for meta in metaphases]
        targets = [(meta, files) for meta, files in targets if files]
        if not targets:
            QtWidgets.QMessageBox.information(self, "Preprocessing", "No images found for this channel on the current slide.")
            return
        progress = QtWidgets.QProgressDialog("Preprocessing", "Cancel", 0, len(targets), self)
        progress.setWindowTitle("Preprocessing batch")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        settings = self.preprocessing_panel.settings_for(channel)
        from skimage import io

        for idx, (metaphase, files) in enumerate(targets, start=1):
            if progress.wasCanceled():
                break
            image = data.load_image(files[0])
            result = preprocessing.preprocess_image(image, settings)
            out_path = metaphase.path / f"{channel}_preprocessed.tif"
            io.imsave(str(out_path), (np.clip(result, 0, 1) * 65535).astype(np.uint16))
            progress.setValue(idx)
            progress.setLabelText(f"Saved {out_path.name}")
        progress.close()

    def _save_preprocessed(self, channel: str) -> None:
        if channel not in self.preprocessed:
            self._refresh_view()
        arr = self.preprocessed[channel]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save preprocessed image", f"{channel}.tif")
        if path:
            from skimage import io

            max_val = float(arr.max()) if arr.size else 1.0
            norm = arr / max_val if max_val > 0 else arr
            io.imsave(path, np.clip(norm, 0, 1))

    def _on_alignment_changed(self, channel: str, dy: int, dx: int) -> None:
        self.alignment_offsets[channel] = (dy, dx)
        self.config.alignment[channel] = [dy, dx]
        if channel in self.preprocessed:
            del self.preprocessed[channel]
        if channel == self.current_channel:
            self._refresh_view()

    def _apply_alignment_all(self) -> None:
        if not self.current_metaphase:
            return
        for channel in self.current_metaphase.available_channels():
            if channel == "dapi":
                continue
            offset = self.alignment_offsets.get(channel, (0, 0))
            self.config.alignment[channel] = [offset[0], offset[1]]
        QtWidgets.QMessageBox.information(self, "Alignment", "Offsets saved for all channels in the slide.")

    def _on_generate_segmentation(self, channel: str, settings: segmentation.SegmentationSettings) -> None:
        image = self._get_processed_image(channel)
        mask = segmentation.adaptive_threshold(image, settings)
        self.segmentation_mask = segmentation.apply_filters(mask, settings)
        self.config.segmentation[channel] = settings.to_dict()
        self.segmentation_masks[channel] = self.segmentation_mask
        self.canvas.set_mask(self.segmentation_mask)
        self.canvas.set_editing_enabled(True)

    def _apply_segmentation_filters(self) -> None:
        if self.segmentation_mask is None:
            return
        settings = self.segmentation_panel.settings_for(self.current_channel)
        self.segmentation_mask = segmentation.apply_filters(self.segmentation_mask, settings)
        self.segmentation_masks[self.current_channel] = self.segmentation_mask
        self.canvas.set_mask(self.segmentation_mask)

    def _save_segmentation(self, channel: str) -> None:
        mask = self.segmentation_masks.get(channel)
        if mask is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save mask", f"{channel}_mask.tif")
        if path:
            from skimage import io

            io.imsave(path, (mask > 0).astype(np.uint8) * 255)

    def _on_brush_changed(self, radius: int, tool: str) -> None:
        self.canvas.set_brush(radius, tool)

    def _on_mask_edited(self, mask: np.ndarray) -> None:
        self.segmentation_mask = mask
        self.segmentation_masks[self.current_channel] = mask

    def _persist_current_config(self) -> None:
        if not self.current_slide:
            return
        self.alignment_offsets = self.alignment_panel.export_offsets()
        cfg = config.Config(
            preprocessing=self.preprocessing_panel.export_settings(),
            alignment=self.alignment_offsets,
            segmentation=self.segmentation_panel.export_settings(),
        )
        config.save_config(self.current_slide.path, cfg)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._persist_current_config()
        super().closeEvent(event)

    def _on_batch_requested(self, request: batch.BatchRequest) -> None:
        root = Path(request.root) if request.root else self.current_slide.path
        dataset = data.discover_dataset(root)
        if dataset is None:
            QtWidgets.QMessageBox.warning(self, "Batch", "No dataset found at the selected path.")
            return
        slides = dataset.available_slides()
        if not slides:
            QtWidgets.QMessageBox.warning(self, "Batch", "Dataset does not contain any slides.")
            return
        total = len(slides)
        for idx, slide_name in enumerate(slides, start=1):
            slide = dataset.slides[slide_name]
            slide_cfg = config.load_config(slide.path)
            for metaphase in slide.metaphases.values():
                for channel, files in metaphase.channels.items():
                    image = data.load_image(files[0])
                    if request.steps["preprocessing"]:
                        settings = preprocessing.PreprocessingSettings.from_dict(
                            slide_cfg.preprocessing.get(channel, {})
                        )
                        image = preprocessing.preprocess_image(image, settings)
                    if request.steps["alignment"]:
                        dy, dx = slide_cfg.alignment.get(channel, (0, 0))
                        image = alignment.apply_translation(image, dy, dx)
                    if request.steps["segmentation"]:
                        seg_settings = segmentation.SegmentationSettings.from_dict(
                            slide_cfg.segmentation.get(channel, {})
                        )
                        mask = segmentation.adaptive_threshold(image, seg_settings)
                        mask = segmentation.apply_filters(mask, seg_settings)
                        out_path = metaphase.path / f"{channel}_mask.png"
                        from skimage import io

                        io.imsave(str(out_path), mask.astype(np.uint8) * 255)
            percent = int(idx / total * 100)
            self.batch_panel.update_progress(percent)
            self.batch_panel.log_message(f"Processed slide {slide_name}")
        QtWidgets.QMessageBox.information(self, "Batch", "Batch processing finished.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Cytogenetic image viewer")
    parser.add_argument("root", nargs="?", default=None, help="Dataset root directory")
    args = parser.parse_args()

    default_paths = [
        Path.cwd() / "dataset",
        Path.cwd() / "Raw images" / "jpp21",
        Path.cwd() / "jpp21_downloaded_data",
        Path.cwd(),
    ]
    if args.root:
        default_paths.insert(0, Path(args.root))

    dataset = data.ensure_dataset(default_paths)

    app = QtWidgets.QApplication([])
    window = MainWindow(dataset)
    window.resize(1400, 900)
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
