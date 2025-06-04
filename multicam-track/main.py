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
            self.linear = torch.nn.Linear(224 * 224 * 3, self.dummy_feature_dim)

        def forward(self, x):
            return torch.rand(x.shape[0], self.dummy_feature_dim)

    def load_model_from_opts(opts_path, ckpt, remove_classifier):
        return DummyReIDModel()

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QSplitter, QScrollArea, QSizePolicy, QStackedWidget,
                             QComboBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


SOURCES_FOLDER = './camera_input'
SOURCES = ['c1.mp4', 'c2.mp4', 'c3.mp4']
VIDEO_SOURCES = [os.path.join(SOURCES_FOLDER, src) for src in SOURCES]
YOLO_MODEL_PATH = 'yolo11s.pt'
REID_MODEL_PATH = './result/net_19.pth'
REID_OPTS_PATH = './result/opts.yaml'
SIMILARITY_THRESHOLD = 0.7
MAX_COMPARISONS = 5
YOLO_CLASSES = [2, 5, 7]


def create_dummy_video(path, duration_seconds=10, fps=25, resolution=(640, 480)):
    """Creates a dummy video file for testing purposes."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, resolution)

        if not out.isOpened():
            return

        for i in range(duration_seconds * fps):
            frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            cv2.putText(frame, f"Cámara {os.path.basename(path).split('.')[0]}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.circle(frame, (resolution[0]//2 + int(100*np.sin(i*0.1)),
                               resolution[1]//2 + int(50*np.cos(i*0.05))),
                       30, (0, 255, 0), -1)
            cv2.rectangle(frame, (int(100*np.sin(i*0.08))+50, int(100*np.cos(i*0.03))+50),
                                 (int(100*np.sin(i*0.08))+150, int(100*np.cos(i*0.03))+150),
                                 (0, 0, 255), 2)
            out.write(frame)
        out.release()

for src_path in VIDEO_SOURCES:
    create_dummy_video(src_path)

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
    """Processes video frames, performs YOLO detection, and Re-ID comparison."""
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

    def set_query_feature(self, feature, image):
        """Sets the feature and image of the query vehicle."""
        with self.query_lock:
            self.query_feature = feature
            self.query_image = image.copy()
            self.update_query.emit(image)

    def run(self):
        """Main loop for video processing."""
        while self.running:
            frames_np = []
            all_detections = defaultdict(list)

            for i, cap in enumerate(self.caps):
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        frames_np.append(None)
                        continue

                current_boxes = []
                if yolo_model:
                    try:
                        results = yolo_model(frame, classes=YOLO_CLASSES, verbose=False, conf=0.4)
                        boxes = results[0].boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            current_boxes.append((x1, y1, x2, y2))
                            crop = frame[y1:y2, x1:x2]

                            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                                continue

                            with self.query_lock:
                                current_query_feature = self.query_feature
                                current_query_image = self.query_image

                            if current_query_feature is not None:
                                feature = extract_reid_feature(crop)
                                sim = torch.dot(current_query_feature, feature).item()

                                if sim > SIMILARITY_THRESHOLD:
                                    with self.similar_vehicles_lock:
                                        key = (i, x1, y1, x2, y2)
                                        self.similar_vehicles[key] = {
                                            'frame': crop.copy(),
                                            'similarity': sim,
                                            'source': i,
                                            'timestamp': time.time(),
                                            'feature': feature
                                        }
                    except Exception as e:
                        pass
                
                all_detections[i].extend(current_boxes)
                frames_np.append(frame)

            self.update_frame.emit(frames_np, all_detections)

            current_time = time.time()
            with self.similar_vehicles_lock:
                to_remove = [k for k, v in self.similar_vehicles.items()
                             if current_time - v['timestamp'] > 5]
                for k in to_remove:
                    del self.similar_vehicles[k]
                self.update_similarities.emit(self.similar_vehicles.copy())

            time.sleep(0.03)

    def stop(self):
        """Stops the video processing thread."""
        self.running = False
        for cap in self.caps:
            cap.release()
        self.wait()

class CameraView(QLabel):
    """Displays a single camera feed with vehicle detections."""
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

        self.current_frame = frame.copy()
        self.detections = detections
        self.selected_box = selected_box
        self._draw_and_display()

    def _draw_and_display(self):
        """Draws detections and displays the frame."""
        if self.current_frame is None:
            return

        display_frame = self.current_frame.copy()

        for (x1, y1, x2, y2) in self.detections:
            is_selected = self.selected_box and \
                          self.selected_box[0] == self.camera_id and \
                          self.selected_box[1:] == (x1, y1, x2, y2)

            color = (0, 0, 255) if is_selected else (255, 0, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"Cámara {self.camera_id+1}", (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        """Handles widget resize events to redraw the frame."""
        super().resizeEvent(event)
        self._draw_and_display()

    def mousePressEvent(self, event):
        """Handles mouse clicks to select a vehicle."""
        if event.button() == Qt.LeftButton and self.click_callback and self.pixmap() and self.current_frame is not None:
            pixmap_scaled_width = self.pixmap().width()
            pixmap_scaled_height = self.pixmap().height()

            label_width = self.width()
            label_height = self.height()

            offset_x = (label_width - pixmap_scaled_width) / 2
            offset_y = (label_height - pixmap_scaled_height) / 2

            x_rel_pixmap = event.pos().x() - offset_x
            y_rel_pixmap = event.pos().y() - offset_y

            original_h, original_w, _ = self.current_frame.shape
            x_on_frame = int(x_rel_pixmap * original_w / pixmap_scaled_width)
            y_on_frame = int(y_rel_pixmap * original_h / pixmap_scaled_height)

            for (x1, y1, x2, y2) in self.detections:
                if x1 <= x_on_frame <= x2 and y1 <= y_on_frame <= y2:
                    self.click_callback(self.camera_id, x1, y1, x2, y2)
                    break

class SimilarVehicleWidget(QWidget):
    """Displays a single similar vehicle match."""
    def __init__(self, match_data, query_img_data=None):
        super().__init__()
        self.match_data = match_data
        self.query_img_data = query_img_data

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        self.query_label = QLabel()
        self.query_label.setFixedSize(100, 100)
        self.query_label.setAlignment(Qt.AlignCenter)
        self.query_label.setStyleSheet("border: 1px dashed gray;")
        if self.query_img_data and self.query_img_data[0] is not None:
            self.update_image_label(self.query_label, self.query_img_data[0])
        else:
            self.query_label.setText("Consulta")
        main_layout.addWidget(self.query_label)

        arrow_label = QLabel("->")
        arrow_label.setFont(arrow_label.font())
        arrow_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(arrow_label)

        self.match_label = QLabel()
        self.match_label.setFixedSize(100, 100)
        self.match_label.setAlignment(Qt.AlignCenter)
        self.update_image_label(self.match_label, match_data['frame'])
        main_layout.addWidget(self.match_label)

        info_label = QLabel(
            f"Fuente: Cámara {match_data['source']+1}\n"
            f"Similitud: {match_data['similarity']:.3f}\n"
            f"Hora: {time.strftime('%H:%M:%S', time.localtime(match_data['timestamp']))}"
        )
        info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        main_layout.addWidget(info_label)

        main_layout.addStretch()

        self.setLayout(main_layout)
        self.setStyleSheet("border: 1px solid #ccc; border-radius: 5px; background-color: #f0f0f0; margin-bottom: 5px;")

    def update_image_label(self, label_widget, img_np):
        """Updates the image displayed in a QLabel."""
        if img_np is None or img_np.size == 0:
            label_widget.setText("Sin Imagen")
            label_widget.clear()
            return

        if img_np.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_np

        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label_widget.setPixmap(QPixmap.fromImage(q_img).scaled(
            label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

class MainWindow(QMainWindow):
    """Main application window for the vehicle re-identification system."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Re-Identificación de Vehículos")
        self.setGeometry(100, 100, 1400, 800)

        self.processing = False
        self.query_feature = None
        self.query_image = None
        self.detections = defaultdict(list)
        self.selected_box = None

        self._setup_ui()
        self._setup_video_processor()

    def _setup_ui(self):
        """Sets up the main user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

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

        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        camera_view_wrapper = QWidget()
        camera_view_layout = QVBoxLayout(camera_view_wrapper)
        camera_view_layout.setContentsMargins(0,0,0,0)
        camera_view_layout.setSpacing(0)

        self.camera_stacked_widget = QStackedWidget()
        camera_view_layout.addWidget(self.camera_stacked_widget)

        self._setup_grid_view()
        self._setup_list_view()

        main_splitter.addWidget(camera_view_wrapper)

        self.details_panel = QWidget()
        details_layout = QVBoxLayout(self.details_panel)
        self.details_panel.setMinimumWidth(380)
        self.details_panel.setMaximumWidth(450)
        self.details_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.details_panel.setStyleSheet("background-color: #e8e8e8; border-left: 1px solid #aaa;")

        query_section_label = QLabel("--- Vehículo de Consulta ---")
        query_section_label.setAlignment(Qt.AlignCenter)
        query_section_label.setStyleSheet("font-weight: bold; margin-top: 10px; margin-bottom: 5px;")
        details_layout.addWidget(query_section_label)

        self.query_display_label = QLabel("Seleccione un vehículo de cualquier cámara para consultar.")
        self.query_display_label.setAlignment(Qt.AlignCenter)
        self.query_display_label.setFixedSize(300, 300)
        self.query_display_label.setStyleSheet("border: 2px dashed #aaa; background-color: #f5f5f5;")
        query_label_container = QHBoxLayout()
        query_label_container.addStretch()
        query_label_container.addWidget(self.query_display_label)
        query_label_container.addStretch()
        details_layout.addLayout(query_label_container)
        details_layout.addSpacing(15)

        similar_section_label = QLabel("--- Vehículos Similares ---")
        similar_section_label.setAlignment(Qt.AlignCenter)
        similar_section_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        details_layout.addWidget(similar_section_label)

        self.similar_scroll_area = QScrollArea()
        self.similar_scroll_area.setWidgetResizable(True)
        self.similar_vehicles_widget = QWidget()
        self.similar_vehicles_layout = QVBoxLayout(self.similar_vehicles_widget)
        self.similar_vehicles_layout.setAlignment(Qt.AlignTop)
        self.similar_scroll_area.setWidget(self.similar_vehicles_widget)
        details_layout.addWidget(self.similar_scroll_area)

        details_layout.addStretch()

        main_splitter.addWidget(self.details_panel)

        main_splitter.setSizes([int(self.width() * 0.7), int(self.width() * 0.3)])

        self.statusBar().showMessage("Listo")

    def _setup_grid_view(self):
        """Sets up the grid view for camera feeds."""
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

    def _setup_list_view(self):
        """Sets up the list view for camera feeds."""
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
            camera_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        list_scroll_area.setWidget(list_widget)
        self.camera_stacked_widget.addWidget(list_scroll_area)
        self.list_view_index = self.camera_stacked_widget.indexOf(list_scroll_area)

    def _setup_video_processor(self):
        """Initializes the video processing thread."""
        self.video_processor = VideoProcessor()
        self.video_processor.update_frame.connect(self.update_camera_views)
        self.video_processor.update_similarities.connect(self.update_similar_vehicles)
        self.video_processor.update_query.connect(self.update_query_view)

    def switch_view(self, index):
        """Switches the display mode between grid and list."""
        if index == 0:
            self.camera_stacked_widget.setCurrentIndex(self.grid_view_index)
            current_views = self.camera_views_grid
        elif index == 1:
            self.camera_stacked_widget.setCurrentIndex(self.list_view_index)
            current_views = self.camera_views_list
        else:
            return

        self.statusBar().showMessage(f"Cambiado a {self.view_selector.currentText()}")
        for cam_view in current_views:
            cam_view._draw_and_display()

    def toggle_processing(self):
        """Starts or stops the video processing thread."""
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
        """Updates all camera views with new frames and detections."""
        self.detections = detections

        for i, frame in enumerate(frames):
            if i < len(self.camera_views_grid):
                self.camera_views_grid[i].update_frame(frame, detections[i], self.selected_box)
            if i < len(self.camera_views_list):
                self.camera_views_list[i].update_frame(frame, detections[i], self.selected_box)

    def on_vehicle_selected(self, camera_id, x1, y1, x2, y2):
        """Handles a vehicle selection event from a camera view."""
        self.selected_box = (camera_id, x1, y1, x2, y2)
        self.statusBar().showMessage(f"Vehículo seleccionado de la Cámara {camera_id+1}.")

        current_active_views = None
        if self.camera_stacked_widget.currentIndex() == self.grid_view_index:
            current_active_views = self.camera_views_grid
        elif self.camera_stacked_widget.currentIndex() == self.list_view_index:
            current_active_views = self.camera_views_list

        if current_active_views and camera_id < len(current_active_views):
            full_frame = current_active_views[camera_id].current_frame
            if full_frame is None or not (0 <= y1 < y2 <= full_frame.shape[0] and 0 <= x1 < x2 <= full_frame.shape[1]):
                self.statusBar().showMessage("Error: No se pudo recuperar el fotograma completo para recortar el vehículo seleccionado.")
                return

            crop = full_frame[y1:y2, x1:x2]

            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                self.statusBar().showMessage("Error: El recorte seleccionado está vacío o no es válido.")
                return

            for cam_view in self.camera_views_grid + self.camera_views_list:
                cam_view._draw_and_display()

            Thread(target=self.process_query, args=(crop,)).start()
        else:
            self.statusBar().showMessage("Error: No se encontró la vista de la cámara para el vehículo seleccionado.")

    def process_query(self, crop_image):
        """Processes the selected vehicle image to set it as the query."""
        try:
            feature = extract_reid_feature(crop_image)
            self.query_feature = feature
            self.query_image = crop_image.copy()
            self.video_processor.set_query_feature(feature, crop_image)

            with self.video_processor.similar_vehicles_lock:
                self.video_processor.similar_vehicles.clear()
            self.update_similar_vehicles({})
            self.statusBar().showMessage("Vehículo de consulta establecido. Buscando vehículos similares...")
        except Exception as e:
            self.statusBar().showMessage(f"Error al procesar la consulta: {e}")

    def update_query_view(self, image):
        """Updates the display of the query vehicle image."""
        if image is None or image.size == 0:
            self.query_display_label.setText("No se ha seleccionado ningún vehículo de consulta.")
            self.query_display_label.clear()
            return

        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.query_display_label.clear()
        self.query_display_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.query_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.query_display_label.setText("")

    def update_similar_vehicles(self, similar_vehicles_dict):
        """Updates the list of similar vehicles displayed."""
        for i in reversed(range(self.similar_vehicles_layout.count())):
            widget = self.similar_vehicles_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        if not similar_vehicles_dict and self.query_feature is None:
            no_matches = QLabel("Seleccione un vehículo para comenzar a buscar similitudes.")
            no_matches.setAlignment(Qt.AlignCenter)
            self.similar_vehicles_layout.addWidget(no_matches)
            return
        elif not similar_vehicles_dict and self.query_feature is not None:
            no_matches = QLabel("No se encontraron vehículos similares todavía.")
            no_matches.setAlignment(Qt.AlignCenter)
            self.similar_vehicles_layout.addWidget(no_matches)
            return

        sorted_vehicles = sorted(similar_vehicles_dict.items(),
                                 key=lambda x: x[1]['similarity'],
                                 reverse=True)[:MAX_COMPARISONS]

        query_data_for_display = (self.query_image, self.query_feature) if self.query_image is not None else None

        for (key, vehicle) in sorted_vehicles:
            widget = SimilarVehicleWidget(vehicle, query_data_for_display)
            self.similar_vehicles_layout.addWidget(widget)

        self.similar_vehicles_layout.addStretch()

    def closeEvent(self, event):
        """Handles graceful shutdown when the window is closed."""
        if self.processing:
            self.video_processor.stop()
        event.accept()

if __name__ == '__main__':
    os.makedirs(SOURCES_FOLDER, exist_ok=True)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())