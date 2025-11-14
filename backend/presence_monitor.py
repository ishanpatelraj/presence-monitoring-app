import cv2
import numpy as np
import time
import argparse
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from collections import deque
import logging
from deepface import DeepFace
from mtcnn import MTCNN
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PresenceMonitor")


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
@dataclass
class MonitorConfig:
    face_match_threshold: float = 0.55
    yolo_confidence: float = 0.5
    yolo_iou: float = 0.45
    max_persons_allowed: int = 1
    blocked_object_classes: List[str] = None
    alert_frames_threshold: int = 3
    enable_liveness: bool = False
    resize_width: int = 640
    camera_id: int = 0

    def __post_init__(self):
        if self.blocked_object_classes is None:
            self.blocked_object_classes = [
                "cell phone", "phone", "book", "laptop", "handbag",
                "backpack", "suitcase", "bottle", "cup", "knife"
            ]


# ------------------------------------------------------------
# RESULT DATA CLASS
# ------------------------------------------------------------
@dataclass
class DetectionResult:
    timestamp: float
    is_authorized: bool
    alert_reasons: List[str]
    num_persons: int
    num_objects: int
    face_distance: Optional[float]
    detected_objects: List[str]


# ------------------------------------------------------------
# MAIN CLASS
# ------------------------------------------------------------
class PresenceMonitor:
    def __init__(self, config: MonitorConfig, authorized_path: str):
        self.config = config

        logger.info("Loading authorized embedding...")
        self.authorized_embedding = self._load_authorized_embedding(authorized_path)

        logger.info("Loading YOLOv8 model...")
        self.yolo = YOLO("yolov8n.pt")

        logger.info("Loading MTCNN face detector...")
        self.detector = MTCNN()

        self.frame_alert_buffer = deque(maxlen=config.alert_frames_threshold)

        logger.info("PresenceMonitor initialized.")

    # --------------------------------------------------------
    # LOAD DEEPFACE EMBEDDING
    # --------------------------------------------------------
    def _load_authorized_embedding(self, path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError("authorized.pkl not found")

        with open(p, "rb") as f:
            data = pickle.load(f)

        # DeepFace format stores embedding under "embedding"
        if "embedding" in data:
            logger.info(f"Loaded embedding for: {data.get('name', 'Unknown')}")
            return np.array(data["embedding"])

        raise ValueError("authorized.pkl is not in DeepFace format")

    # --------------------------------------------------------
    # EXTRACT FACE EMBEDDING FOR LIVE FRAME
    # --------------------------------------------------------
    def extract_embedding(self, frame):
        try:
            result = DeepFace.represent(
                img_path=frame,
                model_name="Facenet512",
                detector_backend="mtcnn",
                enforce_detection=True
            )
            return np.array(result[0]["embedding"])
        except Exception as e:
            logger.warning(f"DeepFace embedding error: {e}")
            return None

    # --------------------------------------------------------
    # PROCESS FRAME
    # --------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        ts = time.time()

        # Resize for performance
        h, w = frame.shape[:2]
        if w > self.config.resize_width:
            scale = self.config.resize_width / w
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        alerts = []
        detected_objects = []
        face_dist = None
        is_authorized = False

        # -----------------------------
        # YOLO OBJECT DETECTION
        # -----------------------------
        results = self.yolo(frame, verbose=False)[0]

        person_count = 0
        object_count = 0
        person_boxes = []

        for box in results.boxes:
            cls = int(box.cls[0])
            label = self.yolo.names[cls]
            conf = float(box.conf[0])

            if label == "person":
                person_count += 1
                person_boxes.append(box)
            else:
                object_count += 1
                detected_objects.append(label)

        # Person rules
        if person_count == 0:
            alerts.append("NO_PERSON_DETECTED")

        if person_count > self.config.max_persons_allowed:
            alerts.append(f"MULTIPLE_PERSONS ({person_count})")

        # Blocked object rules
        blocked = [o for o in detected_objects if o in self.config.blocked_object_classes]
        if blocked:
            alerts.append(f"BLOCKED_OBJECTS ({', '.join(blocked)})")

        # -----------------------------
        # FACE EMBEDDING MATCHING
        # -----------------------------
        if person_count == 1 and not alerts:
            embedding = self.extract_embedding(frame)

            if embedding is None:
                alerts.append("FACE_NOT_DETECTED")
            else:
                # Compute cosine distance
                face_dist = 1 - np.dot(embedding, self.authorized_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(self.authorized_embedding))

                if face_dist > self.config.face_match_threshold:
                    alerts.append(f"UNAUTHORIZED_FACE (dist={face_dist:.3f})")
                else:
                    is_authorized = True

        # Add sustained alert logic
        self.frame_alert_buffer.append(len(alerts) > 0)
        sustained_alert = sum(self.frame_alert_buffer) >= self.config.alert_frames_threshold

        # -----------------------------
        # DRAW BOXES + TEXT
        # -----------------------------
        annotated = frame.copy()

        color = (0, 255, 0) if is_authorized else ((0, 0, 255) if sustained_alert else (0, 165, 255))
        status = "AUTHORIZED" if is_authorized else ("ALERT" if sustained_alert else "CHECKING")

        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 60), color, -1)
        cv2.putText(annotated, status, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

        y_offset = 80
        for a in alerts[:3]:
            cv2.putText(annotated, f"! {a}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

        return annotated, DetectionResult(
            timestamp=ts,
            is_authorized=is_authorized,
            alert_reasons=alerts,
            num_persons=person_count,
            num_objects=object_count,
            face_distance=face_dist,
            detected_objects=detected_objects
        )


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", required=True)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    config = MonitorConfig(camera_id=args.camera)

    monitor = PresenceMonitor(config, args.encoding)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Camera not available")
        return

    logger.info("Starting monitoring... Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, result = monitor.process_frame(frame)
        cv2.imshow("Presence Monitor", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
