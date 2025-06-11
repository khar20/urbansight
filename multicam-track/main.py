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

# Attempt to import vehicle_reid model, fall back to dummy if not available
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
            # Simulate a feature vector
            return torch.rand(x.shape[0], self.dummy_feature_dim)

    def load_model_from_opts(opts_path, ckpt, remove_classifier):
        return DummyReIDModel()

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QSplitter, QScrollArea, QSizePolicy, QStackedWidget,
                             QComboBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# --- NEW IMPORT FOR MAP ---
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
except ImportError:
    print("PyQtWebEngine not found. Map embedding will be disabled. "
          "Please install it with 'pip install PyQtWebEngine'")
    QWebEngineView = None
    QWebEngineSettings = None
# --- END NEW IMPORT ---

# Configuration
SOURCES_FOLDER = './camera_input'
SOURCES = ['c1.mp4', 'c2.mp4', 'c3.mp4']
VIDEO_SOURCES = [os.path.join(SOURCES_FOLDER, src) for src in SOURCES]
YOLO_MODEL_PATH = 'yolo11s.pt'
REID_MODEL_PATH = './result/net_19.pth'
REID_OPTS_PATH = './result/opts.yaml'
SIMILARITY_THRESHOLD = 0.7
MAX_COMPARISONS = 5
YOLO_CLASSES = [2, 5, 7]  # Classes for 'car', 'bus', 'truck'

# --- NEW: Camera Locations (Dummy data for demonstration) ---
# Format: {'source_filename': (latitude, longitude)}
# These are approximate locations in Downtown Los Angeles
CAMERA_LOCATIONS = {
    'c1.mp4': (-8.109105, -79.043799),  # Near Pershing Square
    'c2.mp4': (-8.109135, -79.043676),  # Near Grand Park
    # Near Staples Center / Crypto.com Arena
    'c3.mp4': (-8.115759, -79.043962)
}
# Map source names to their configured lat/lon, maintaining order for camera IDs
CONFIGURED_CAMERA_LOCATIONS = [
    CAMERA_LOCATIONS.get(src, (0, 0)) for src in SOURCES
]
# If a source is not found in CAMERA_LOCATIONS, it will default to (0,0) which will be skipped by the map.
# --- END NEW NEW ---

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
    print(f"Error loading ReID model: {e}. Using DummyReIDModel.")
    reid_model = DummyReIDModel()
    reid_model.eval().to("cpu")


def extract_reid_feature(image):
    """Extracts a re-identification feature vector from an image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            print(
                "Warning: Attempted to extract feature from empty image. Returning dummy feature.")
            return torch.rand(reid_model.dummy_feature_dim if hasattr(reid_model, 'dummy_feature_dim') else 2048)

        pil_image = Image.fromarray(image)
        tensor = reid_transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = reid_model(tensor).squeeze(0)
            # Add horizontal flip augmentation during feature extraction for robustness
            if tensor.dim() == 4 and tensor.shape[3] > 1:
                feature += reid_model(torch.flip(tensor, dims=[3])).squeeze(0)
            feature = feature / torch.norm(feature, p=2)  # L2 normalization
        return feature.cpu()
    except Exception as e:
        print(f"Error extracting ReID feature: {e}. Returning dummy feature.")
        return torch.rand(reid_model.dummy_feature_dim if hasattr(reid_model, 'dummy_feature_dim') else 2048)


yolo_model = None
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    # Move YOLO model to GPU if available
    if torch.cuda.is_available():
        yolo_model.to("cuda")
except Exception as e:
    print(f"Error loading YOLO model: {e}. Detection will be disabled.")
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
        self.query_lock = Lock()  # Protects query_feature and query_image
        self.similar_vehicles = {}
        self.similar_vehicles_lock = Lock()  # Protects similar_vehicles dictionary
        # Not used, but kept from original structure
        self.frame_buffer = [None] * len(VIDEO_SOURCES)

    def set_query_feature(self, feature, image):
        """Sets the feature and image of the query vehicle."""
        with self.query_lock:
            # Explicitly delete old feature to free memory if it was a tensor
            if self.query_feature is not None:
                del self.query_feature
            self.query_feature = feature
            self.query_image = image.copy()  # Make a copy to prevent modification issues
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
                    # Loop video if it ends
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()  # Try reading again
                    if not ret:  # If still no frame, skip this camera
                        continue
                frames_np[i] = frame
                # Only add frames that successfully read for batch processing
                frames_to_process.append(frame)
                indices_to_process.append(i)

            # Batch processing if we have frames and YOLO model is loaded
            if yolo_model and frames_to_process:
                try:
                    # Batch process frames with YOLO
                    results = yolo_model(frames_to_process, classes=YOLO_CLASSES,
                                         verbose=False, conf=0.4, batch=len(frames_to_process))

                    # Collect crops for batch ReID processing
                    all_crops_for_reid = []
                    crop_info_for_reid = []

                    for j, res in enumerate(results):
                        i_cam = indices_to_process[j]
                        frame = frames_to_process[j]
                        boxes = res.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            crop = frame[y1:y2, x1:x2]
                            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                                continue  # Skip empty crops
                            all_crops_for_reid.append(crop)
                            crop_info_for_reid.append((i_cam, x1, y1, x2, y2))
                            all_detections[i_cam].append((x1, y1, x2, y2))

                    # Batch process crops with ReID if we have a query and crops
                    with self.query_lock:
                        current_query_feature = self.query_feature

                    if all_crops_for_reid and current_query_feature is not None:
                        # Prepare list of features for batch extraction
                        features_list = []
                        for crop in all_crops_for_reid:
                            features_list.append(extract_reid_feature(crop))

                        # Convert to tensor for efficient cosine similarity computation
                        features_tensor = torch.stack(features_list)

                        # Calculate similarities
                        # Ensure current_query_feature is 1D for unsqueeze(1)
                        if current_query_feature.dim() == 1:
                            similarities = torch.mm(
                                features_tensor,
                                current_query_feature.unsqueeze(1)
                            ).squeeze(1).cpu().numpy()
                        # Handle potential case where current_query_feature is already 2D (e.g., [1, dim])
                        else:
                            similarities = torch.mm(
                                features_tensor,
                                current_query_feature.T
                            ).squeeze(1).cpu().numpy()

                        for idx, sim in enumerate(similarities):
                            if sim > SIMILARITY_THRESHOLD:
                                i_cam, x1, y1, x2, y2 = crop_info_for_reid[idx]
                                # Unique key for each detection instance
                                key = (i_cam, x1, y1, x2, y2)
                                with self.similar_vehicles_lock:
                                    self.similar_vehicles[key] = {
                                        # Store a copy of the crop image
                                        'frame': all_crops_for_reid[idx].copy(),
                                        'similarity': sim,
                                        'source': i_cam,
                                        'timestamp': time.time(),
                                        # Store the feature if needed later
                                        'feature': features_tensor[idx].cpu()
                                    }
                except Exception as e:
                    print(f"Error during YOLO or ReID batch processing: {e}")
                    # Continue loop even if an error occurs to prevent crash

            self.update_frame.emit(frames_np, all_detections)

            # Clean up old similar vehicles to keep the display fresh
            with self.similar_vehicles_lock:
                current_time = time.time()
                # Remove matches older than 5 seconds
                self.similar_vehicles = {k: v for k, v in self.similar_vehicles.items()
                                         if current_time - v['timestamp'] <= 5}
                self.update_similarities.emit(self.similar_vehicles)

            # Control loop speed
            time.sleep(0.03)  # Approx 30 FPS if processing is fast enough

    def stop(self):
        """Stops the video processing thread and cleans up resources."""
        self.running = False
        for cap in self.caps:
            cap.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory
        self.wait()  # Wait for the thread to finish execution


class CameraView(QLabel):
    """Optimized camera view with selective rendering."""

    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 150)
        self.detections = []
        # (camera_id, x1, y1, x2, y2) of the selected box
        self.selected_box = None
        self.click_callback = None
        self.current_frame = None  # Store the original frame for cropping
        self.setStyleSheet("border: 1px solid #ddd; background-color: black;")
        self.setText(f"Cargando Cámara {self.camera_id+1}...")

    def update_frame(self, frame, detections, selected_box=None):
        """Updates the displayed frame and detections."""
        if frame is None:
            self.setText(f"Cámara {self.camera_id+1} Desconectada")
            self.clear()  # Clear any previous pixmap
            return

        # Only update if the frame or detections have changed significantly
        # or if the selected_box state changes.
        # This basic check avoids redundant redraws for identical frames.
        if (self.current_frame is not None and
            np.array_equal(self.current_frame, frame) and
            self.detections == detections and
                self.selected_box == selected_box):
            return

        self.current_frame = frame
        self.detections = detections
        self.selected_box = selected_box
        self._draw_and_display()

    def _draw_and_display(self):
        """Optimized drawing and display with selective rendering."""
        if self.current_frame is None or not self.isVisible() or self.size().isEmpty():
            return

        # Create a copy to draw on, preventing modification of the original frame
        display_frame = self.current_frame.copy()

        # Downsample for display if the frame is much larger than the widget
        if display_frame.shape[0] > self.height() or display_frame.shape[1] > self.width():
            display_frame = cv2.resize(display_frame,
                                       (self.width(), self.height()),
                                       interpolation=cv2.INTER_AREA)

        # Calculate scaling factors for drawing detections
        sf_x = display_frame.shape[1] / self.current_frame.shape[1]
        sf_y = display_frame.shape[0] / self.current_frame.shape[0]

        # Draw detections with scaled coordinates
        for (x1, y1, x2, y2) in self.detections:
            # Scale coordinates to display size
            dx1, dy1 = int(x1 * sf_x), int(y1 * sf_y)
            dx2, dy2 = int(x2 * sf_x), int(y2 * sf_y)

            # Draw rectangle
            color = (0, 0, 255)  # Default: Blue
            # If this box is the selected one, highlight it in red
            if self.selected_box and \
               self.selected_box[0] == self.camera_id and \
               self.selected_box[1:] == (x1, y1, x2, y2):
                color = (255, 0, 0)  # Red for selected

            cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), color, 2)
            # Draw camera ID on the box
            cv2.putText(display_frame, f"Camara {self.camera_id+1}", (dx1 + 5, dy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Convert to QImage and display
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h,
                       bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(q_img))

    def resizeEvent(self, event):
        """Handles widget resize events to redraw the frame at the new size."""
        super().resizeEvent(event)
        self._draw_and_display()  # Redraw when resized

    def mousePressEvent(self, event):
        """Handles mouse clicks to select a vehicle."""
        if (event.button() == Qt.LeftButton and self.click_callback and
                self.pixmap() and self.current_frame is not None):

            # Calculate click position relative to the image within the label
            # This handles cases where the image might not fill the entire label due to aspect ratio
            pixmap_size = self.pixmap().size()
            label_size = self.size()

            # Calculate the effective area of the pixmap within the QLabel
            # Assuming Qt.AlignCenter for the pixmap
            scaled_width = pixmap_size.width()
            scaled_height = pixmap_size.height()

            # Adjust for aspect ratio difference if image was stretched or scaled
            # (In _draw_and_display, we resize to fit the label, so this simplifies)
            x_offset = (label_size.width() - scaled_width) / 2
            y_offset = (label_size.height() - scaled_height) / 2

            x_rel_pixmap = event.pos().x() - x_offset
            y_rel_pixmap = event.pos().y() - y_offset

            # Check if click is within the pixmap area
            if not (0 <= x_rel_pixmap <= scaled_width and 0 <= y_rel_pixmap <= scaled_height):
                return  # Click was outside the image area

            # Scale back to original frame coordinates
            # Current frame might be different dimensions from the displayed pixmap
            # So scale from displayed pixmap coordinates to original frame coordinates
            original_frame_width = self.current_frame.shape[1]
            original_frame_height = self.current_frame.shape[0]

            x_on_original_frame = int(
                x_rel_pixmap * original_frame_width / scaled_width)
            y_on_original_frame = int(
                y_rel_pixmap * original_frame_height / scaled_height)

            # Check if click is within any detection bounding box
            for (x1, y1, x2, y2) in self.detections:
                if x1 <= x_on_original_frame <= x2 and y1 <= y_on_original_frame <= y2:
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
        self.query_label.setStyleSheet(
            "border: 1px dashed gray; background-color: #e0e0e0;")
        if self.query_img_data is not None and self.query_img_data[0] is not None:
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
        if img_np is None or img_np.size == 0 or img_np.shape[0] == 0 or img_np.shape[1] == 0:
            label_widget.setText("Sin Imagen")
            label_widget.clear()
            return

        # Convert to RGB if needed (assuming BGR from OpenCV)
        img_rgb = cv2.cvtColor(
            img_np, cv2.COLOR_BGR2RGB) if img_np.ndim == 3 and img_np.shape[2] == 3 else img_np

        # Resize for display, maintaining aspect ratio while fitting the label
        h, w, ch = img_rgb.shape
        q_img = QImage(img_rgb.data, w, h, w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label_widget.setPixmap(pixmap.scaled(
            label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


# --- NEW: MapViewWidget Class ---
class MapViewWidget(QWidget):
    def __init__(self, camera_locations):
        super().__init__()
        self.camera_locations = camera_locations
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        # No extra margins inside the widget
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)  # Small space between label and map/message

        map_label = QLabel("--- Ubicaciones de Cámaras ---")
        map_label.setAlignment(Qt.AlignCenter)
        map_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        layout.addWidget(map_label)

        if QWebEngineView:
            self.web_view = QWebEngineView()
            # Enable local content to access remote URLs (for Leaflet CDN)
            self.web_view.settings().setAttribute(
                QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
            # Enable local storage for map tiles (optional but good practice)
            self.web_view.settings().setAttribute(
                QWebEngineSettings.LocalStorageEnabled, True)
            self.load_map()
            layout.addWidget(self.web_view)
        else:
            no_web_engine_label = QLabel(
                "PyQtWebEngine no está instalado. El mapa interactivo no se puede mostrar.\n"
                "Instálelo con 'pip install PyQtWebEngine'."
            )
            no_web_engine_label.setAlignment(Qt.AlignCenter)
            no_web_engine_label.setStyleSheet(
                "color: red; padding: 20px; border: 1px dashed red;")
            layout.addWidget(no_web_engine_label)
            # Add a stretch to make sure the label doesn't take minimal space
            layout.addStretch()

    def load_map(self):
        # Calculate a reasonable map center based on camera locations
        valid_lats = [loc[0] for loc in self.camera_locations if loc[0] != 0]
        valid_lons = [loc[1] for loc in self.camera_locations if loc[1] != 0]

        map_center_lat = np.mean(
            valid_lats) if valid_lats else 34.0522  # Default to LA
        map_center_lon = np.mean(
            valid_lons) if valid_lons else -118.2437  # Default to LA

        # Generate JavaScript for adding markers to the map
        markers_js = []
        for i, (lat, lon) in enumerate(self.camera_locations):
            if lat != 0 and lon != 0:  # Only add markers for valid locations
                markers_js.append(
                    f"L.marker([{lat}, {lon}]).addTo(map).bindPopup('<b>Cámara {i+1}</b><br>Lat: {lat:.4f}<br>Lon: {lon:.4f}');"
                )
        markers_js_str = "\n".join(markers_js)

        # HTML content with Leaflet.js
        # --- MODIFIED: Removed integrity and crossorigin attributes ---
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cámaras</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <!-- Leaflet CSS and JS from CDN -->
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <style>
                body {{ margin:0; padding:0; height:100%; width:100%; overflow:hidden; }}
                html {{ height:100%; width:100%; }}
                #mapid {{ width: 100%; height: 100%; min-height: 200px; }} /* Ensure minimum height */
            </style>
        </head>
        <body>
            <div id="mapid"></div>
            <script>
                var map = L.map('mapid').setView([{map_center_lat}, {map_center_lon}], 13); // Zoom level 13

                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                }}).addTo(map);

                {markers_js_str}
            </script>
        </body>
        </html>
        """
        self.web_view.setHtml(html_content)
# --- END NEW: MapViewWidget Class ---


class MainWindow(QMainWindow):
    """Optimized main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Re-Identificación de Vehículos")
        self.setGeometry(100, 100, 1400, 800)  # Initial window size
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

        control_bar_layout.addStretch()  # Pushes next widgets to the right

        view_label = QLabel("Modo de Vista:")
        control_bar_layout.addWidget(view_label)
        self.view_selector = QComboBox()
        self.view_selector.addItems(["Vista de Cuadrícula", "Vista de Lista"])
        self.view_selector.currentIndexChanged.connect(self.switch_view)
        control_bar_layout.addWidget(self.view_selector)
        main_layout.addLayout(control_bar_layout)

        # Main splitter (Left: Camera views | Right: Details panel)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Camera view area (Left side of main splitter)
        camera_view_wrapper = QWidget()
        camera_view_layout = QVBoxLayout(camera_view_wrapper)
        camera_view_layout.setContentsMargins(0, 0, 0, 0)
        camera_view_layout.setSpacing(0)  # No spacing around camera views

        self.camera_stacked_widget = QStackedWidget()
        camera_view_layout.addWidget(self.camera_stacked_widget)

        self.init_grid_view()
        self.init_list_view()

        # Add camera area to splitter
        main_splitter.addWidget(camera_view_wrapper)

        # Details panel (Right side of main splitter - will contain map, query, similar vehicles)
        self.details_panel = QWidget()
        details_layout = QVBoxLayout(self.details_panel)
        # Minimum width for details panel
        self.details_panel.setMinimumWidth(380)
        # Maximum width for details panel
        self.details_panel.setMaximumWidth(450)
        self.details_panel.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Expanding)  # Fixed width, expanding height
        self.details_panel.setStyleSheet(
            "background-color: #e8e8e8; border-left: 1px solid #aaa;")  # Styling for panel

        # --- NEW: Map View section (Top part of details panel) ---
        self.map_view_widget = MapViewWidget(CONFIGURED_CAMERA_LOCATIONS)
        # Give the map view a stretch factor so it expands vertically
        details_layout.addWidget(self.map_view_widget, 2)
        details_layout.addSpacing(10)  # Small space after map
        # --- END NEW ---

        # Query section
        query_section_label = QLabel("--- Vehículo de Consulta ---")
        query_section_label.setAlignment(Qt.AlignCenter)
        query_section_label.setStyleSheet(
            "font-weight: bold; margin-top: 10px; margin-bottom: 5px;")
        details_layout.addWidget(query_section_label)

        self.query_display_label = QLabel(
            "Seleccione un vehículo de cualquier cámara para consultar.")
        self.query_display_label.setAlignment(Qt.AlignCenter)
        # Adjusted fixed size to better fit a potentially smaller area due to map
        self.query_display_label.setFixedSize(250, 250)  # Reduced size
        self.query_display_label.setStyleSheet(
            "border: 2px dashed #aaa; background-color: #f5f5f5;")
        query_label_container = QHBoxLayout()
        query_label_container.addStretch()  # Pushes query label to center
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
        self.similar_vehicles_layout.setAlignment(
            Qt.AlignTop)  # Align content to top
        self.similar_scroll_area.setWidget(self.similar_vehicles_widget)
        # Give similar vehicles section more stretch factor for more space
        details_layout.addWidget(self.similar_scroll_area, 3)

        details_layout.addStretch()  # This will push all content up if there's extra space
        # Add details panel to splitter
        main_splitter.addWidget(self.details_panel)
        # Set initial sizes for the splitter parts (cameras 70%, details 30%)
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
            row = i // 2  # Arrange in 2 columns
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
        list_layout.setAlignment(Qt.AlignTop)  # Align content to top

        self.camera_views_list = []
        for i in range(len(VIDEO_SOURCES)):
            camera_view = CameraView(i)
            camera_view.click_callback = self.on_vehicle_selected
            self.camera_views_list.append(camera_view)
            list_layout.addWidget(camera_view)
            # Ensure camera views expand vertically in list mode
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
        if index == 0:  # Grid View
            self.camera_stacked_widget.setCurrentIndex(self.grid_view_index)
            current_views = self.camera_views_grid
        elif index == 1:  # List View
            self.camera_stacked_widget.setCurrentIndex(self.list_view_index)
            current_views = self.camera_views_list
        else:
            return

        self.statusBar().showMessage(
            f"Cambiado a {self.view_selector.currentText()}")
        # Force a redraw of all active camera views after switching
        for cam_view in current_views:
            cam_view._draw_and_display()

    def toggle_processing(self):
        """Toggle video processing on/off."""
        if self.processing:
            self.video_processor.stop()
            self.control_button.setText("Iniciar Procesamiento")
            self.statusBar().showMessage("Procesamiento detenido")
            # Clear and show stopped message on all camera views
            for cam_view in self.camera_views_grid + self.camera_views_list:
                cam_view.clear()
                cam_view.setText(f"Camara {cam_view.camera_id+1} Detenida")
        else:
            # Ensure cameras are available before starting
            if not all(os.path.exists(src) for src in VIDEO_SOURCES):
                self.statusBar().showMessage("Error: Algunas fuentes de video no existen.")
                return

            self.video_processor.start()
            self.control_button.setText("Detener Procesamiento")
            self.statusBar().showMessage("Procesamiento iniciado")
        self.processing = not self.processing

    def update_camera_views(self, frames, detections):
        """Update all camera views with new frames and detections."""
        self.detections = detections  # Store global detections for other parts if needed

        # Update both grid and list views for consistency,
        # but only the visible one will actually paint.
        for i, frame in enumerate(frames):
            # Check bounds to prevent index errors if frames list is shorter than views
            if i < len(self.camera_views_grid):
                self.camera_views_grid[i].update_frame(
                    frame, detections.get(i, []), self.selected_box)
            if i < len(self.camera_views_list):
                self.camera_views_list[i].update_frame(
                    frame, detections.get(i, []), self.selected_box)

    def on_vehicle_selected(self, camera_id, x1, y1, x2, y2):
        """Handle vehicle selection from camera view."""
        self.selected_box = (camera_id, x1, y1, x2, y2)
        self.statusBar().showMessage(
            f"Vehículo seleccionado de la Cámara {camera_id+1}.")

        # Get the currently active camera views to retrieve the frame
        current_active_views = None
        if self.camera_stacked_widget.currentIndex() == self.grid_view_index:
            current_active_views = self.camera_views_grid
        elif self.camera_stacked_widget.currentIndex() == self.list_view_index:
            current_active_views = self.camera_views_list

        if current_active_views and camera_id < len(current_active_views):
            full_frame = current_active_views[camera_id].current_frame
            # Validate coordinates and frame before cropping
            if full_frame is None or not (0 <= y1 < y2 <= full_frame.shape[0] and 0 <= x1 < x2 <= full_frame.shape[1]):
                self.statusBar().showMessage(
                    "Error: No se pudo recuperar el fotograma completo o coordenadas inválidas para recortar el vehículo seleccionado.")
                # Clear selected box if invalid
                self.selected_box = None
                # Redraw all views to remove highlight
                for cam_view in self.camera_views_grid + self.camera_views_list:
                    cam_view._draw_and_display()
                return

            crop = full_frame[y1:y2, x1:x2]

            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                self.statusBar().showMessage(
                    "Error: El recorte seleccionado está vacío o no es válido.")
                self.selected_box = None
                for cam_view in self.camera_views_grid + self.camera_views_list:
                    cam_view._draw_and_display()
                return

            # Force redraw of all camera views to update selection highlight
            for cam_view in self.camera_views_grid + self.camera_views_list:
                cam_view._draw_and_display()

            # Process query in a separate thread to avoid freezing UI
            Thread(target=self.process_query, args=(crop,)).start()
        else:
            self.statusBar().showMessage(
                "Error: No se encontró la vista de la cámara para el vehículo seleccionado.")

    def process_query(self, crop_image):
        """Process selected vehicle image to set as query."""
        try:
            feature = extract_reid_feature(crop_image)
            self.query_feature = feature
            self.query_image = crop_image.copy()  # Store a copy of the image
            self.video_processor.set_query_feature(feature, crop_image)

            # Clear existing similar vehicles immediately for a fresh search
            with self.video_processor.similar_vehicles_lock:
                self.video_processor.similar_vehicles.clear()
            self.update_similar_vehicles({})  # Update UI to show empty list
            self.statusBar().showMessage(
                "Vehículo de consulta establecido. Buscando vehículos similares...")
        except Exception as e:
            self.statusBar().showMessage(f"Error al procesar la consulta: {e}")

    def update_query_view(self, image):
        """Update the query vehicle display."""
        if image is None or image.size == 0:
            self.query_display_label.setText(
                "No se ha seleccionado ningún vehículo de consulta.")
            self.query_display_label.clear()  # Clear any pixmap
            return

        # Convert to RGB if needed
        image_rgb = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB) if image.ndim == 3 and image.shape[2] == 3 else image

        # Create QImage and scale to fit the label, maintaining aspect ratio
        h, w, ch = image_rgb.shape
        q_img = QImage(image_rgb.data, w, h, w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        self.query_display_label.clear()
        self.query_display_label.setPixmap(
            pixmap.scaled(self.query_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.query_display_label.setText("")  # Clear placeholder text

    def update_similar_vehicles(self, similar_vehicles_dict):
        """Update the list of similar vehicles."""
        # Clear existing widgets efficiently
        while self.similar_vehicles_layout.count():
            item = self.similar_vehicles_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()  # Schedule for deletion

        if not similar_vehicles_dict and self.query_feature is None:
            no_matches = QLabel(
                "Seleccione un vehículo para comenzar a buscar similitudes.")
            no_matches.setAlignment(Qt.AlignCenter)
            no_matches.setStyleSheet("color: #666; padding: 20px;")
            self.similar_vehicles_layout.addWidget(no_matches)
            return
        elif not similar_vehicles_dict and self.query_feature is not None:
            no_matches = QLabel(
                "No se encontraron vehículos similares todavía.")
            no_matches.setAlignment(Qt.AlignCenter)
            no_matches.setStyleSheet("color: #666; padding: 20px;")
            self.similar_vehicles_layout.addWidget(no_matches)
            return

        # Add new matches, sorted by similarity in descending order
        # Limit to MAX_COMPARISONS to prevent overcrowding
        sorted_vehicles = sorted(similar_vehicles_dict.items(),
                                 key=lambda x: x[1]['similarity'],
                                 reverse=True)[:MAX_COMPARISONS]

        # Pass query image data for display in each similar vehicle widget
        query_data_for_display = (
            self.query_image, self.query_feature) if self.query_image is not None else None

        for (key, vehicle) in sorted_vehicles:
            widget = SimilarVehicleWidget(vehicle, query_data_for_display)
            self.similar_vehicles_layout.addWidget(widget)

        self.similar_vehicles_layout.addStretch()  # Push content to the top

    def closeEvent(self, event):
        """Handle window close event."""
        if self.processing:
            self.video_processor.stop()  # Ensure processing thread is stopped
        event.accept()


if __name__ == '__main__':
    # Ensure camera input folder exists
    os.makedirs(SOURCES_FOLDER, exist_ok=True)

    # Dummy video files for testing if they don't exist
    # Create empty video files to prevent crashes if video files are missing
    for src_path in VIDEO_SOURCES:
        if not os.path.exists(src_path):
            print(f"Creating dummy video file: {src_path}")
            try:
                # Create a simple 1-second black video using OpenCV
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
                out = cv2.VideoWriter(src_path, fourcc, 20.0, (640, 480))
                if not out.isOpened():
                    raise IOError(
                        f"Could not open video writer for {src_path}")
                # Write 20 frames (1 second at 20 fps)
                for _ in range(20):
                    out.write(np.zeros((480, 640, 3), dtype=np.uint8))
                out.release()
            except Exception as e:
                print(f"Failed to create dummy video {src_path}: {e}")
                # If creating dummy fails, the app might still run but with warnings/errors for that camera

    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()  # Start with maximized window
    sys.exit(app.exec_())
