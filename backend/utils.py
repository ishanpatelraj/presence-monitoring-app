# utils.py
import base64
import cv2
import numpy as np
import time
import threading

# image utils

def b64_to_cv2_img(b64_string: str):
    """
    Accepts dataURL or plain base64 image string.
    Returns BGR cv2 image or None
    """
    if b64_string.startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]
    try:
        data = base64.b64decode(b64_string)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def cv2_to_b64_jpeg(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")

# simple in-memory OTP store with TTL
class OTPStore:
    def __init__(self):
        self._store = {}  # key -> (otp, expiry_ts)
        self._lock = threading.Lock()

    def set(self, key, otp, ttl=120):
        expiry = time.time() + ttl
        with self._lock:
            self._store[key] = (otp, expiry)

    def verify(self, key, otp):
        with self._lock:
            rec = self._store.get(key)
            if not rec:
                return False
            saved, expiry = rec
            if time.time() > expiry:
                del self._store[key]
                return False
            if saved == otp:
                del self._store[key]
                return True
            return False

    def cleanup(self):
        now = time.time()
        with self._lock:
            to_del = [k for k, (_, e) in self._store.items() if e < now]
            for k in to_del:
                del self._store[k]

otp_store = OTPStore()
