import os
import sys
import cv2
import torch
import yaml
import numpy as np
from PIL import Image
from threading import Lock, Thread
from collections import defaultdict
from torchvision import transforms
from ultralytics import YOLO
import queue
import time

try:
    from vehicle_reid.load_model import load_model_from_opts
except ImportError:
    class DummyReIDModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_feature_dim = 2048
            self.linear = torch.nn.Linear(
                224 * 224 * 3, self.dummy_feature_dim)

        def forward(self, x):
            return torch.rand(x.shape[0], self.dummy_feature_dim)

    def load_model_from_opts(opts_path, ckpt, remove_classifier):
        return DummyReIDModel()

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QSplitter, QScrollArea, QSizePolicy, QStackedWidget,
                             QComboBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# Configuration
SOURCES_FOLDER = './camera_input'
SOURCES = ['c1.mp4', 'c2.mp4', 'c3.mp4']
VIDEO_SOURCES = [os.path.join(SOURCES_FOLDER, src) for src in SOURCES]
YOLO_MODEL_PATH = 'yolo11s.pt'
REID_MODEL_PATH = './result/net_19.pth'
REID_OPTS_PATH = './result/opts.yaml'
SIMILARITY_THRESHOLD = 0.7
MAX_COMPARISONS = 5
YOLO_CLASSES = [2, 5, 7]

# Initialize models
reid_model = None
reid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    reid_model = load_model_from_opts(
        REID_OPTS_PATH, ckpt=REID_MODEL_PATH, remove_classifier=True)
    reid_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    reid_model = DummyReIDModel()
    reid_model.eval().to("cpu")


def extract_reid_feature(image):
    """Extracts a re-identification feature vector from an image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            return torch.rand(reid_model.dummy_feature_dim if hasattr(reid_model, 'dummy_feature_dim') else 2048)

        pil_image = Image.fromarray(image)
        tensor = reid_transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = reid_model(tensor).squeeze(0)
            if tensor.dim() == 4 and tensor.shape[3] > 1:
                feature += reid_model(torch.flip(tensor, dims=[3])).squeeze(0)
            feature = feature / torch.norm(feature, p=2)
        return feature.cpu()
    except Exception as e:
        return torch.rand(reid_model.dummy_feature_dim if hasattr(reid_model, 'dummy_feature_dim') else 2048)


yolo_model = None
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    pass


class VideoProcessor(QThread):
    """Optimized video processing thread with batch operations."""
    update_frame = pyqtSignal(list, dict)
    update_similarities = pyqtSignal(dict)
    update_query = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.caps = [cv2.VideoCapture(src) for src in VIDEO_SOURCES]
        self.running = True
        self.query_feature = None
        self.query_image = None
        self.query_lock = Lock()
        self.similar_vehicles = {}
        self.similar_vehicles_lock = Lock()
        self.frame_buffer = [None] * len(VIDEO_SOURCES)

    def set_query_feature(self, feature, image):
        """Sets the feature and image of the query vehicle."""
        with self.query_lock:
            if self.query_feature is not None:
                del self.query_feature
            self.query_feature = feature
            self.query_image = image.copy()
            self.update_query.emit(image)

    def run(self):
        """Main optimized processing loop with batch operations."""
        while self.running:
            frames_np = [None] * len(self.caps)
            all_detections = defaultdict(list)
            frames_to_process = []
            indices_to_process = []

            # Read frames and prepare for processing
            for i, cap in enumerate(self.caps):
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frames_np[i] = frame
                frames_to_process.append(frame)
                indices_to_process.append(i)

            # Batch processing if we have frames and models
            if yolo_model and frames_to_process:
                # Batch process frames with YOLO
                results = yolo_model(frames_to_process, classes=YOLO_CLASSES,
                                     verbose=False, conf=0.4, batch=len(frames_to_process))

                # Collect crops for batch ReID processing
                all_crops = []
                crop_info = []

                for j, res in enumerate(results):
                    i_cam = indices_to_process[j]
                    frame = frames_to_process[j]
                    boxes = res.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        all_crops.append(crop)
                        crop_info.append((i_cam, x1, y1, x2, y2))
                        all_detections[i_cam].append((x1, y1, x2, y2))

                # Batch process crops with ReID if we have a query
                with self.query_lock:
                    current_query_feature = self.query_feature

                if all_crops and current_query_feature is not None:
                    features = []
                    for crop in all_crops:
                        features.append(extract_reid_feature(crop))

                    # Convert to tensor for efficient computation
                    features_tensor = torch.stack(features)
                    similarities = torch.mm(
                        features_tensor,
                        current_query_feature.unsqueeze(1)
                    ).squeeze(1).cpu().numpy()

                    for idx, sim in enumerate(similarities):
                        if sim > SIMILARITY_THRESHOLD:
                            i_cam, x1, y1, x2, y2 = crop_info[idx]
                            key = (i_cam, x1, y1, x2, y2)
                            with self.similar_vehicles_lock:
                                self.similar_vehicles[key] = {
                                    'frame': all_crops[idx].copy(),
                                    'similarity': sim,
                                    'source': i_cam,
                                    'timestamp': time.time(),
                                    'feature': features_tensor[idx].cpu()
                                }

            self.update_frame.emit(frames_np, all_detections)

            # Clean up old similar vehicles
            with self.similar_vehicles_lock:
                current_time = time.time()
                self.similar_vehicles = {k: v for k, v in self.similar_vehicles.items()
                                         if current_time - v['timestamp'] <= 5}
                self.update_similarities.emit(self.similar_vehicles)

            time.sleep(0.03)

    def stop(self):
        """Stops the video processing thread and cleans up resources."""
        self.running = False
        for cap in self.caps:
            cap.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.wait()


class CameraView(QLabel):
    """Optimized camera view with selective rendering."""

    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 150)
        self.detections = []
        self.selected_box = None
        self.click_callback = None
        self.current_frame = None
        self.setStyleSheet("border: 1px solid #ddd; background-color: black;")
        self.setText(f"Cargando Cámara {self.camera_id+1}...")

    def update_frame(self, frame, detections, selected_box=None):
        """Updates the displayed frame and detections."""
        if frame is None:
            self.setText(f"Cámara {self.camera_id+1} Desconectada")
            self.clear()
            return

        self.current_frame = frame
        self.detections = detections
        self.selected_box = selected_box
        self._draw_and_display()

    def _draw_and_display(self):
        """Optimized drawing and display with selective rendering."""
        if self.current_frame is None or not self.isVisible() or self.size().isEmpty():
            return

        # Downsample for display
        display_frame = cv2.resize(self.current_frame,
                                   (self.width(), self.height()),
                                   interpolation=cv2.INTER_AREA)

        # Draw detections with scaled coordinates
        for (x1, y1, x2, y2) in self.detections:
            # Scale coordinates to display size
            sf_x = self.width() / self.current_frame.shape[1]
            sf_y = self.height() / self.current_frame.shape[0]
            dx1, dy1 = int(x1 * sf_x), int(y1 * sf_y)
            dx2, dy2 = int(x2 * sf_x), int(y2 * sf_y)

            # Draw rectangle
            color = (0, 0, 255) if self.selected_box and \
                self.selected_box[0] == self.camera_id and \
                self.selected_box[1:] == (x1, y1, x2, y2) else (255, 0, 0)
            cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), color, 2)
            cv2.putText(display_frame, f"Cámara {self.camera_id+1}", (dx1 + 5, dy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Convert to QImage
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, w * ch, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(q_img))

    def resizeEvent(self, event):
        """Handles widget resize events to redraw the frame."""
        super().resizeEvent(event)
        self._draw_and_display()

    def mousePressEvent(self, event):
        """Handles mouse clicks to select a vehicle."""
        if (event.button() == Qt.LeftButton and self.click_callback and
                self.pixmap() and self.current_frame is not None):

            # Calculate click position in original frame coordinates
            pixmap_size = self.pixmap().size()
            label_size = self.size()

            offset_x = (label_size.width() - pixmap_size.width()) / 2
            offset_y = (label_size.height() - pixmap_size.height()) / 2

            x_rel_pixmap = event.pos().x() - offset_x
            y_rel_pixmap = event.pos().y() - offset_y

            # Scale to original frame coordinates
            x_on_frame = int(
                x_rel_pixmap * self.current_frame.shape[1] / pixmap_size.width())
            y_on_frame = int(
                y_rel_pixmap * self.current_frame.shape[0] / pixmap_size.height())

            # Check if click is within any detection
            for (x1, y1, x2, y2) in self.detections:
                if x1 <= x_on_frame <= x2 and y1 <= y_on_frame <= y2:
                    self.click_callback(self.camera_id, x1, y1, x2, y2)
                    break


class SimilarVehicleWidget(QWidget):
    """Optimized widget for displaying similar vehicle matches."""

    def __init__(self, match_data, query_img_data=None):
        super().__init__()
        self.match_data = match_data
        self.query_img_data = query_img_data
        self.init_ui()

    def init_ui(self):
        """Initialize UI components."""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # Query image display
        self.query_label = QLabel()
        self.query_label.setFixedSize(100, 100)
        self.query_label.setAlignment(Qt.AlignCenter)
        self.query_label.setStyleSheet("border: 1px dashed gray;")
        if self.query_img_data and self.query_img_data[0] is not None:
            self.update_image_label(self.query_label, self.query_img_data[0])
        else:
            self.query_label.setText("Consulta")
        main_layout.addWidget(self.query_label)

        # Arrow indicator
        arrow_label = QLabel("->")
        arrow_label.setFont(arrow_label.font())
        arrow_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(arrow_label)

        # Match image display
        self.match_label = QLabel()
        self.match_label.setFixedSize(100, 100)
        self.match_label.setAlignment(Qt.AlignCenter)
        self.update_image_label(self.match_label, self.match_data['frame'])
        main_layout.addWidget(self.match_label)

        # Information display
        info_label = QLabel(
            f"Fuente: Cámara {self.match_data['source']+1}\n"
            f"Similitud: {self.match_data['similarity']:.3f}\n"
            f"Hora: {time.strftime('%H:%M:%S', time.localtime(self.match_data['timestamp']))}"
        )
        info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        main_layout.addWidget(info_label)

        main_layout.addStretch()
        self.setLayout(main_layout)
        self.setStyleSheet("""
            border: 1px solid #ccc; 
            border-radius: 5px; 
            background-color: #f0f0f0; 
            margin-bottom: 5px;
        """)

    def update_image_label(self, label_widget, img_np):
        """Optimized image display update."""
        if img_np is None or img_np.size == 0:
            label_widget.setText("Sin Imagen")
            label_widget.clear()
            return

        # Convert to RGB if needed
        img_rgb = cv2.cvtColor(
            img_np, cv2.COLOR_BGR2RGB) if img_np.shape[2] == 3 else img_np

        # Resize for display
        img_rgb = cv2.resize(
            img_rgb, (label_widget.width(), label_widget.height()))

        # Create QImage
        h, w, ch = img_rgb.shape
        q_img = QImage(img_rgb.data, w, h, w * ch, QImage.Format_RGB888)
        label_widget.setPixmap(QPixmap.fromImage(q_img))


class MainWindow(QMainWindow):
    """Optimized main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Re-Identificación de Vehículos")
        self.setGeometry(100, 100, 1400, 800)
        self.init_ui()
        self.init_video_processor()

    def init_ui(self):
        """Initialize the main UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Control bar
        control_bar_layout = QHBoxLayout()
        self.control_button = QPushButton("Iniciar Procesamiento")
        self.control_button.clicked.connect(self.toggle_processing)
        control_bar_layout.addWidget(self.control_button)

        control_bar_layout.addStretch()

        view_label = QLabel("Modo de Vista:")
        control_bar_layout.addWidget(view_label)
        self.view_selector = QComboBox()
        self.view_selector.addItems(["Vista de Cuadrícula", "Vista de Lista"])
        self.view_selector.currentIndexChanged.connect(self.switch_view)
        control_bar_layout.addWidget(self.view_selector)
        main_layout.addLayout(control_bar_layout)

        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Camera view area
        camera_view_wrapper = QWidget()
        camera_view_layout = QVBoxLayout(camera_view_wrapper)
        camera_view_layout.setContentsMargins(0, 0, 0, 0)
        camera_view_layout.setSpacing(0)

        self.camera_stacked_widget = QStackedWidget()
        camera_view_layout.addWidget(self.camera_stacked_widget)

        self.init_grid_view()
        self.init_list_view()

        main_splitter.addWidget(camera_view_wrapper)

        # Details panel
        self.details_panel = QWidget()
        details_layout = QVBoxLayout(self.details_panel)
        self.details_panel.setMinimumWidth(380)
        self.details_panel.setMaximumWidth(450)
        self.details_panel.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.details_panel.setStyleSheet(
            "background-color: #e8e8e8; border-left: 1px solid #aaa;")

        # Query section
        query_section_label = QLabel("--- Vehículo de Consulta ---")
        query_section_label.setAlignment(Qt.AlignCenter)
        query_section_label.setStyleSheet(
            "font-weight: bold; margin-top: 10px; margin-bottom: 5px;")
        details_layout.addWidget(query_section_label)

        self.query_display_label = QLabel(
            "Seleccione un vehículo de cualquier cámara para consultar.")
        self.query_display_label.setAlignment(Qt.AlignCenter)
        self.query_display_label.setFixedSize(300, 300)
        self.query_display_label.setStyleSheet(
            "border: 2px dashed #aaa; background-color: #f5f5f5;")
        query_label_container = QHBoxLayout()
        query_label_container.addStretch()
        query_label_container.addWidget(self.query_display_label)
        query_label_container.addStretch()
        details_layout.addLayout(query_label_container)
        details_layout.addSpacing(15)

        # Similar vehicles section
        similar_section_label = QLabel("--- Vehículos Similares ---")
        similar_section_label.setAlignment(Qt.AlignCenter)
        similar_section_label.setStyleSheet(
            "font-weight: bold; margin-bottom: 5px;")
        details_layout.addWidget(similar_section_label)

        self.similar_scroll_area = QScrollArea()
        self.similar_scroll_area.setWidgetResizable(True)
        self.similar_vehicles_widget = QWidget()
        self.similar_vehicles_layout = QVBoxLayout(
            self.similar_vehicles_widget)
        self.similar_vehicles_layout.setAlignment(Qt.AlignTop)
        self.similar_scroll_area.setWidget(self.similar_vehicles_widget)
        details_layout.addWidget(self.similar_scroll_area)

        details_layout.addStretch()
        main_splitter.addWidget(self.details_panel)
        main_splitter.setSizes(
            [int(self.width() * 0.7), int(self.width() * 0.3)])

        self.statusBar().showMessage("Listo")

    def init_grid_view(self):
        """Initialize grid view layout."""
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setContentsMargins(5, 5, 5, 5)
        grid_layout.setSpacing(10)

        self.camera_views_grid = []
        for i in range(len(VIDEO_SOURCES)):
            camera_view = CameraView(i)
            camera_view.click_callback = self.on_vehicle_selected
            self.camera_views_grid.append(camera_view)
            row = i // 2
            col = i % 2
            grid_layout.addWidget(camera_view, row, col)

        self.camera_stacked_widget.addWidget(grid_widget)
        self.grid_view_index = self.camera_stacked_widget.indexOf(grid_widget)

    def init_list_view(self):
        """Initialize list view layout."""
        list_scroll_area = QScrollArea()
        list_scroll_area.setWidgetResizable(True)
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(5, 5, 5, 5)
        list_layout.setSpacing(10)
        list_layout.setAlignment(Qt.AlignTop)

        self.camera_views_list = []
        for i in range(len(VIDEO_SOURCES)):
            camera_view = CameraView(i)
            camera_view.click_callback = self.on_vehicle_selected
            self.camera_views_list.append(camera_view)
            list_layout.addWidget(camera_view)
            camera_view.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding)

        list_scroll_area.setWidget(list_widget)
        self.camera_stacked_widget.addWidget(list_scroll_area)
        self.list_view_index = self.camera_stacked_widget.indexOf(
            list_scroll_area)

    def init_video_processor(self):
        """Initialize video processing thread."""
        self.processing = False
        self.query_feature = None
        self.query_image = None
        self.detections = defaultdict(list)
        self.selected_box = None

        self.video_processor = VideoProcessor()
        self.video_processor.update_frame.connect(self.update_camera_views)
        self.video_processor.update_similarities.connect(
            self.update_similar_vehicles)
        self.video_processor.update_query.connect(self.update_query_view)

    def switch_view(self, index):
        """Switch between grid and list views."""
        if index == 0:
            self.camera_stacked_widget.setCurrentIndex(self.grid_view_index)
            current_views = self.camera_views_grid
        elif index == 1:
            self.camera_stacked_widget.setCurrentIndex(self.list_view_index)
            current_views = self.camera_views_list
        else:
            return

        self.statusBar().showMessage(
            f"Cambiado a {self.view_selector.currentText()}")
        for cam_view in current_views:
            cam_view._draw_and_display()

    def toggle_processing(self):
        """Toggle video processing on/off."""
        if self.processing:
            self.video_processor.stop()
            self.control_button.setText("Iniciar Procesamiento")
            self.statusBar().showMessage("Procesamiento detenido")
            for cam_view in self.camera_views_grid + self.camera_views_list:
                cam_view.clear()
                cam_view.setText(f"Cámara {cam_view.camera_id+1} Detenida")
        else:
            self.video_processor.start()
            self.control_button.setText("Detener Procesamiento")
            self.statusBar().showMessage("Procesamiento iniciado")
        self.processing = not self.processing

    def update_camera_views(self, frames, detections):
        """Update all camera views with new frames and detections."""
        self.detections = detections

        for i, frame in enumerate(frames):
            if i < len(self.camera_views_grid):
                self.camera_views_grid[i].update_frame(
                    frame, detections[i], self.selected_box)
            if i < len(self.camera_views_list):
                self.camera_views_list[i].update_frame(
                    frame, detections[i], self.selected_box)

    def on_vehicle_selected(self, camera_id, x1, y1, x2, y2):
        """Handle vehicle selection from camera view."""
        self.selected_box = (camera_id, x1, y1, x2, y2)
        self.statusBar().showMessage(
            f"Vehículo seleccionado de la Cámara {camera_id+1}.")

        current_active_views = None
        if self.camera_stacked_widget.currentIndex() == self.grid_view_index:
            current_active_views = self.camera_views_grid
        elif self.camera_stacked_widget.currentIndex() == self.list_view_index:
            current_active_views = self.camera_views_list

        if current_active_views and camera_id < len(current_active_views):
            full_frame = current_active_views[camera_id].current_frame
            if full_frame is None or not (0 <= y1 < y2 <= full_frame.shape[0] and 0 <= x1 < x2 <= full_frame.shape[1]):
                self.statusBar().showMessage(
                    "Error: No se pudo recuperar el fotograma completo para recortar el vehículo seleccionado.")
                return

            crop = full_frame[y1:y2, x1:x2]

            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                self.statusBar().showMessage(
                    "Error: El recorte seleccionado está vacío o no es válido.")
                return

            for cam_view in self.camera_views_grid + self.camera_views_list:
                cam_view._draw_and_display()

            Thread(target=self.process_query, args=(crop,)).start()
        else:
            self.statusBar().showMessage(
                "Error: No se encontró la vista de la cámara para el vehículo seleccionado.")

    def process_query(self, crop_image):
        """Process selected vehicle image to set as query."""
        try:
            feature = extract_reid_feature(crop_image)
            self.query_feature = feature
            self.query_image = crop_image.copy()
            self.video_processor.set_query_feature(feature, crop_image)

            with self.video_processor.similar_vehicles_lock:
                self.video_processor.similar_vehicles.clear()
            self.update_similar_vehicles({})
            self.statusBar().showMessage(
                "Vehículo de consulta establecido. Buscando vehículos similares...")
        except Exception as e:
            self.statusBar().showMessage(f"Error al procesar la consulta: {e}")

    def update_query_view(self, image):
        """Update the query vehicle display."""
        if image is None or image.size == 0:
            self.query_display_label.setText(
                "No se ha seleccionado ningún vehículo de consulta.")
            self.query_display_label.clear()
            return

        # Convert to RGB if needed
        image_rgb = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image

        # Resize for display
        image_rgb = cv2.resize(
            image_rgb, (self.query_display_label.width(), self.query_display_label.height()))

        # Create QImage
        h, w, ch = image_rgb.shape
        q_img = QImage(image_rgb.data, w, h, w * ch, QImage.Format_RGB888)

        self.query_display_label.clear()
        self.query_display_label.setPixmap(QPixmap.fromImage(q_img))
        self.query_display_label.setText("")

    def update_similar_vehicles(self, similar_vehicles_dict):
        """Update the list of similar vehicles."""
        # Clear existing widgets
        for i in reversed(range(self.similar_vehicles_layout.count())):
            widget = self.similar_vehicles_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        if not similar_vehicles_dict and self.query_feature is None:
            no_matches = QLabel(
                "Seleccione un vehículo para comenzar a buscar similitudes.")
            no_matches.setAlignment(Qt.AlignCenter)
            self.similar_vehicles_layout.addWidget(no_matches)
            return
        elif not similar_vehicles_dict and self.query_feature is not None:
            no_matches = QLabel(
                "No se encontraron vehículos similares todavía.")
            no_matches.setAlignment(Qt.AlignCenter)
            self.similar_vehicles_layout.addWidget(no_matches)
            return

        # Add new matches
        sorted_vehicles = sorted(similar_vehicles_dict.items(),
                                 key=lambda x: x[1]['similarity'],
                                 reverse=True)[:MAX_COMPARISONS]

        query_data_for_display = (
            self.query_image, self.query_feature) if self.query_image is not None else None

        for (key, vehicle) in sorted_vehicles:
            widget = SimilarVehicleWidget(vehicle, query_data_for_display)
            self.similar_vehicles_layout.addWidget(widget)

        self.similar_vehicles_layout.addStretch()

    def closeEvent(self, event):
        """Handle window close event."""
        if self.processing:
            self.video_processor.stop()
        event.accept()


if __name__ == '__main__':
    os.makedirs(SOURCES_FOLDER, exist_ok=True)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
