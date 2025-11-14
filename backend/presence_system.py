"""
Unified DeepFace + YOLO Presence Monitoring System
==================================================
Replaces all dlib/face_recognition dependencies.

Modes:
1. Enrollment:
    python presence_system.py enroll --image myphoto.jpg --name Ishan --output authorized.pkl

2. Monitoring:
    python presence_system.py monitor --encoding authorized.pkl
"""

import argparse
import pickle
import cv2
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
import time
import logging
from pathlib import Path
from collections import deque

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ================================================================
#                   ENROLLMENT (DeepFace)
# ================================================================

def extract_embedding(image_path):
    """Extract a 512-dimensional FaceNet embedding."""
    logger.info(f"Extracting embedding from: {image_path}")

    try:
        analysis = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet512",
            enforce_detection=True
        )
        emb = np.array(analysis[0]["embedding"])
        logger.info("Embedding extracted successfully.")
        return emb

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None


def enroll(image_path, name, output_path):
    """Save authorized user's embedding."""
    emb = extract_embedding(image_path)
    if emb is None:
        logger.error("Enrollment failed.")
        return

    data = {
        "name": name,
        "embedding": emb.tolist()
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    logger.info(f"Enrollment complete! Saved to {output_path}")


# ================================================================
#                   MONITORING (YOLO + DeepFace)
# ================================================================

class PresenceMonitor:
    def __init__(self, encoding_path):
        # Load authorized embedding
        with open(encoding_path, "rb") as f:
            data = pickle.load(f)

        self.auth_name = data["name"]
        self.auth_embedding = np.array(data["embedding"])

        logger.info(f"Loaded encoding for: {self.auth_name}")

        # Load YOLO
        self.yolo = YOLO("yolov8n.pt")
        self.class_names = self.yolo.names

        # Temporal smoothing
        self.alert_buffer = deque(maxlen=3)

    def verify_face(self, face_img, threshold=0.9):
        """Return True if face matches authorized person."""
        try:
            emb = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet512",
                enforce_detection=False
            )
            if not emb:
                return False

            emb = np.array(emb[0]["embedding"])
            dist = np.linalg.norm(emb - self.auth_embedding)

            logger.info(f"Face distance = {dist:.4f}")
            return dist < threshold
        except:
            return False

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open camera.")
            return

        logger.info("Monitoring started. Press Q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # YOLO detection
            results = self.yolo(frame, verbose=False)[0]

            persons = []
            objects = []

            for box in results.boxes:
                cls = int(box.cls[0])
                name = self.class_names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if name == "person":
                    persons.append((x1, y1, x2, y2))
                else:
                    objects.append(name)

            alert_reasons = []

            # RULE 1: Exactly one person must be visible
            if len(persons) == 0:
                alert_reasons.append("NO_PERSON")
            elif len(persons) > 1:
                alert_reasons.append("MULTIPLE_PERSONS")

            # RULE 2: No foreign objects allowed
            if len(objects) > 0:
                alert_reasons.append(f"OBJECTS: {','.join(objects)}")

            is_authorized = False

            if len(persons) == 1 and not objects:
                # Crop face
                x1, y1, x2, y2 = persons[0]
                face_img = frame[y1:y2, x1:x2]

                # Save temporary
                tmp = "temp_face.jpg"
                cv2.imwrite(tmp, face_img)

                if self.verify_face(tmp):
                    is_authorized = True
                else:
                    alert_reasons.append("UNAUTHORIZED_FACE")

            # Temporal smoothing
            self.alert_buffer.append(len(alert_reasons) > 0)
            sustained_alert = sum(self.alert_buffer) >= 2

            # Draw status
            if is_authorized:
                color = (0, 255, 0)
                status = "AUTHORIZED"
            elif sustained_alert:
                color = (0, 0, 255)
                status = "ALERT"
            else:
                color = (0, 165, 255)
                status = "CHECKING"

            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Draw boxes
            for (x1, y1, x2, y2) in persons:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            cv2.imshow("Presence Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ================================================================
#                   MAIN ENTRY POINT
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    # ENROLL
    e = sub.add_parser("enroll")
    e.add_argument("--image", required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--output", default="authorized.pkl")

    # MONITOR
    m = sub.add_parser("monitor")
    m.add_argument("--encoding", required=True)

    args = parser.parse_args()

    if args.mode == "enroll":
        enroll(args.image, args.name, args.output)

    elif args.mode == "monitor":
        monitor = PresenceMonitor(args.encoding)
        monitor.run()


if __name__ == "__main__":
    main()
