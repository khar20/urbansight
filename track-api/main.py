from contextlib import asynccontextmanager
import os
import json
import time
from av import VideoFrame
import cv2
from fastapi.websockets import WebSocketState
import numpy as np
import asyncio
import uvicorn
from collections import defaultdict
from threading import Lock
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcdatachannel import RTCDataChannel

# Configuration
TRACKER_CONFIG = "botsort.yaml"
SOURCES_FOLDER = './camera_input'
SOURCES = ['c1.mp4', 'c2.mp4', 'c3.mp4']


class AppConfig:
    MODEL_NAME = 'yolo11s.pt'
    VIDEO_SOURCES = [os.path.join(SOURCES_FOLDER, src) for src in SOURCES]
    HOMOGRAPHY_MATRIX_PATH = 'homography.json'
    MAX_TRACK_HISTORY = 100  # Limit track history to prevent memory leaks


class HomographyTransformer:
    """Simplified homography transformer for demonstration"""

    def __init__(self, homography_path):
        with open(homography_path) as f:
            data = json.load(f)
        self.matrix = np.array(data['matrix'])

    def project(self, point):
        """Project a point using the homography matrix"""
        px, py = point
        src_point = np.array([px, py, 1])
        dst_point = self.matrix @ src_point
        return (dst_point[0]/dst_point[2], dst_point[1]/dst_point[2])


class TrackingVideoStreamTrack(VideoStreamTrack):
    """Enhanced video track with integrated vehicle tracking"""

    def __init__(self, video_source, model, transformer, data_channel: RTCDataChannel = None):
        super().__init__()
        self.data_channel = data_channel
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source {video_source}")
        self.model = model
        self.transformer = transformer
        self.track_history = defaultdict(list)
        self.lock = Lock()
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.fps = 0

    async def recv(self):
        """Process each frame, perform tracking, and send results"""
        pts, time_base = await self.next_timestamp()

        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Video capture failed")

        # Calculate FPS
        self.frame_count += 1
        if time.time() - self.last_fps_update > 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_update = time.time()

        # Run tracking
        results = self.model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            verbose=False
        )

        # Process tracking data
        tracking_info = []
        for result in results:
            for box in result.boxes:
                if box.id is None:
                    continue

                tid = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bottom_center = ((x1 + x2) // 2, y2)
                projected = self.transformer.project(bottom_center)

                with self.lock:
                    # Maintain limited track history
                    self.track_history[tid].append(projected)
                    if len(self.track_history[tid]) > AppConfig.MAX_TRACK_HISTORY:
                        self.track_history[tid].pop(0)

                    tracking_info.append({
                        "id": tid,
                        "bbox": [x1, y1, x2, y2],
                        "position": projected,
                        "confidence": float(box.conf[0]),
                        "timestamp": time.time(),
                        "fps": self.fps
                    })

        # Send tracking data if data channel is available
        if self.data_channel and self.data_channel.readyState == "open":
            try:
                await self.data_channel.send(json.dumps({
                    "tracks": tracking_info,
                    "timestamp": time.time()
                }))
            except Exception as e:
                print(f"Data channel send error: {e}")

        # return frame
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame

    def release(self):
        """Clean up resources"""
        self.cap.release()
        super().release()


class VehicleTracker:
    """Optimized vehicle tracking class"""

    def __init__(self, model_path, homography_path):
        self.model = YOLO(model_path)
        # self.transformer = HomographyTransformer(homography_path)
        self.track_history = defaultdict(list)
        self.lock = Lock()

    def record_tracks(self, results):
        """Record tracking data with thread safety"""
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

                    # Maintain limited track history
                    self.track_history[tid].append(projected)
                    if len(self.track_history[tid]) > AppConfig.MAX_TRACK_HISTORY:
                        self.track_history[tid].pop(0)

                    data.append({
                        "id": tid,
                        "bbox": [x1, y1, x2, y2],
                        "position": projected,
                        "confidence": float(box.conf[0])
                    })
        return data


active_connections = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for pc in active_connections.values():
        await pc.close()
        active_connections.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/signaling")
async def websocket_signaling(websocket: WebSocket):
    """Handle WebRTC signaling and connection setup"""
    await websocket.accept()
    pc = None
    data_channel = None

    try:
        # Initial message must be the offer
        data = await websocket.receive_text()
        message = json.loads(data)

        if message["type"] != "offer":
            raise ValueError("First message must be an offer")

        # Create peer connection
        pc = RTCPeerConnection()
        active_connections[websocket] = pc

        # Setup error handlers
        @pc.on("iceconnectionstatechange")
        async def on_ice_state_change():
            print(f"ICE connection state: {pc.iceConnectionState}")
            if pc.iceConnectionState == "failed":
                await pc.close()
                active_connections.pop(websocket, None)

        # Create tracker
        tracker = VehicleTracker(
            AppConfig.MODEL_NAME,
            AppConfig.HOMOGRAPHY_MATRIX_PATH
        )

        # Create data channel
        data_channel = pc.createDataChannel("trackingData")

        @data_channel.on("open")
        def on_open():
            print("Data channel opened")

        # Create video track
        video_track = TrackingVideoStreamTrack(
            AppConfig.VIDEO_SOURCES[0],
            tracker.model,
            # tracker.transformer,
            data_channel
        )
        pc.addTrack(video_track)

        # Handle offer
        offer = RTCSessionDescription(
            sdp=message["sdp"],
            type=message["type"]
        )
        await pc.setRemoteDescription(offer)

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Send answer
        await websocket.send_text(json.dumps({
            "type": "answer",
            "sdp": pc.localDescription.sdp
        }))

        # Handle subsequent ICE candidates
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "ice":
                try:
                    candidate_dict = message["candidate"]
                    candidate = RTCIceCandidate(
                        foundation=candidate_dict.get("foundation", ""),
                        component=candidate_dict.get("component", 1),
                        protocol=candidate_dict.get("protocol", "udp"),
                        priority=candidate_dict.get("priority", 0),
                        ip=candidate_dict.get("ip", ""),
                        port=candidate_dict.get("port", 0),
                        type=candidate_dict.get("type", "host"),
                        relatedAddress=candidate_dict.get(
                            "relatedAddress", ""),
                        relatedPort=candidate_dict.get("relatedPort", 0),
                        sdpMid=candidate_dict.get("sdpMid", 0),
                        sdpMLineIndex=candidate_dict.get("sdpMLineIndex", 0),
                        # candidate=candidate_dict.get("candidate", "")
                    )
                    await pc.addIceCandidate(candidate)
                except Exception as e:
                    print(f"Error adding ICE candidate: {e}")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if pc:
            await pc.close()
        if websocket in active_connections:
            active_connections.pop(websocket, None)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=10,
        ws_ping_timeout=30
    )
