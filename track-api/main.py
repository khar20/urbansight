import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import time
import json
from collections import defaultdict
from threading import Lock, Thread
import uvicorn
import base64
from queue import Queue

# Configuration
TRACKER_CONFIG = "botsort.yaml"

SOURCES_FOLDER = './camera_input'
SOURCES = [
    'c1.mp4',
    'c2.mp4',
    'c3.mp4',
    #'c4.mp4',
    #'c5.mp4',
    #'c6.mp4',
    #'c7.mp4'
]

class AppConfig:
    MODEL_NAME = 'yolo11s.pt'
    VIDEO_SOURCES = [os.path.join(SOURCES_FOLDER, src) for src in SOURCES]
    HOMOGRAPHY_MATRIX_PATH = 'homography.json'
    #CONF_THRESHOLD = 0.5
    #IMGSZ = 640

# Utilities
class HomographyTransformer:
    def __init__(self, matrix_path):
        self.matrix = self._load_matrix(matrix_path)

    def _load_matrix(self, path):
        try:
            with open(path, 'r') as f:
                return np.array(json.load(f))
        except (OSError, ValueError):
            return np.eye(3)

    def project(self, pt):
        pt_array = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt_array, self.matrix)
        return tuple(map(int, projected[0][0]))

class VehicleTracker:
    def __init__(self, model_path, homography_path):
        self.model = YOLO(model_path)
        self.transformer = HomographyTransformer(homography_path)
        self.track_history = defaultdict(list)
        self.lock = Lock()

    def record_tracks(self, results):
        data = []
        with self.lock:
            for result in results:
                for box in result.boxes:
                    if box.id is None:
                        continue
                    tid = int(box.id.item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bottom_center = ((x1 + x2) // 2, y2)
                    projected = self.transformer.project(bottom_center)
                    self.track_history[tid].append(projected)
                    data.append({
                        "id": tid,
                        "bbox": [x1, y1, x2, y2],
                        "position": projected,
                        "confidence": float(box.conf[0])
                    })
        return data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

trackers = [VehicleTracker(AppConfig.MODEL_NAME, AppConfig.HOMOGRAPHY_MATRIX_PATH) for _ in AppConfig.VIDEO_SOURCES]
data_queue = Queue()


def track_video(source_index, video_source):
    model = trackers[source_index].model
    transformer = trackers[source_index].transformer

    stream = model.track(
        source=video_source,
        stream=True,
        verbose=False,
        persist=True,
        tracker=TRACKER_CONFIG,
        device='0',
    )

    for result in stream:
        start = time.time()
        tracking_info = trackers[source_index].record_tracks([result])
        frame = result.orig_img.copy()
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "source": video_source,
            "timestamp": time.time(),
            "fps": 1 / (time.time() - start),
            "tracks": tracking_info,
            "frame": frame_base64,
        }
        data_queue.put(payload)

@app.websocket("/ws/track")
async def ws_track(websocket: WebSocket):
    await websocket.accept()
    sent_dimensions = {}

    # Start threads for each video source
    threads = [Thread(target=track_video, args=(i, src), daemon=True) for i, src in enumerate(AppConfig.VIDEO_SOURCES)]
    for thread in threads:
        thread.start()

    try:
        while True:
            payload = data_queue.get()
            source = payload["source"]

            if source not in sent_dimensions:
                frame_data = base64.b64decode(payload["frame"])
                np_frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
                height, width = frame.shape[:2]
                payload["originalWidth"] = width
                payload["originalHeight"] = height
                sent_dimensions[source] = True

            await websocket.send_json(payload)
    except WebSocketDisconnect:
        print("Tracking data WebSocket disconnected")

# Entry Point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
