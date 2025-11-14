# server.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from utils import b64_to_cv2_img, otp_store
from presence_monitor import PresenceMonitor
from db_utils import init_db, get_all_users, log_event, get_logs, save_user
from presence_system import extract_embedding_from_image
import base64
import numpy as np
import os
import random
import string
import time

app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app)

# ensure db/schema exists
init_db()

# instantiate monitor
MON = PresenceMonitor(model_path="models/yolov8n.pt", face_threshold=0.45)

# simple OTP sender stub — replace by real email/SMS integration
def send_otp_stub(user_name, otp):
    # In production, send via SendGrid / Twilio / SMTP
    print(f"[OTP] for user {user_name}: {otp}")

def gen_otp():
    return "".join(random.choices(string.digits, k=6))

@app.route("/")
def index():
    # serve frontend index if present
    try:
        return send_from_directory(app.static_folder, "index.html")
    except Exception:
        return jsonify({"status": "backend running"})

@app.post("/register")  # register user via API (image upload)
def api_register():
    """
    Accepts multipart/form-data with:
      - 'name' (string)
      - 'image' (file)  OR JSON { image: dataurl }
      - optional 'export' boolean to write authorized.pkl
    """
    name = request.form.get("name") or request.json.get("name") if request.is_json else None
    export_flag = (request.form.get("export") == "1") or (request.json.get("export") is True if request.is_json else False)

    # get image (file or dataURL)
    if "image" in request.files:
        f = request.files["image"].read()
        img = None
        import numpy as np, cv2, io
        arr = np.frombuffer(f, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        # maybe JSON body with dataURL
        data = request.json.get("image")
        img = b64_to_cv2_img(data)

    if img is None:
        return jsonify({"ok": False, "error": "no_image"}), 400

    try:
        emb = extract_embedding_from_image_from_array(img)
    except Exception:
        # use DeepFace.represent directly (presence_system.extract can't accept cv2 array)
        from deepface import DeepFace
        rep = DeepFace.represent(img, model_name="Facenet512", enforce_detection=True)
        if not rep:
            return jsonify({"ok": False, "error": "could_not_extract_embedding"}), 500
        emb = np.array(rep[0]["embedding"], dtype=np.float32)

    # save to DB
    save_user(name, emb)
    MON.refresh_users()
    if export_flag:
        import pickle
        with open("authorized.pkl", "wb") as f:
            pickle.dump({"name": name, "embedding": emb}, f)
    return jsonify({"ok": True, "name": name})

# helper (local) to extract embedding from cv2 array (used above)
def extract_embedding_from_image_from_array(img):
    from deepface import DeepFace
    rep = DeepFace.represent(img, model_name="Facenet512", enforce_detection=True)
    if not rep:
        raise RuntimeError("embedding failed")
    return np.array(rep[0]["embedding"], dtype=np.float32)

@app.post("/detect")
def api_detect():
    """
    POST JSON: { frame: "<dataurl or base64>" }
    Returns JSON with status
    """
    data = request.json
    if not data or "frame" not in data:
        return jsonify({"ok": False, "error": "no_frame"}), 400

    img = b64_to_cv2_img(data["frame"])
    if img is None:
        return jsonify({"ok": False, "error": "invalid_image"}), 400

    result = MON.process_frame(img)
    return jsonify({"ok": True, "result": result})

@app.post("/request_otp")
def api_request_otp():
    """
    After face match & liveness success, frontend calls this to request OTP.
    POST JSON: {"user": "Alice"}
    """
    body = request.json
    user = body.get("user")
    if not user:
        return jsonify({"ok": False, "error": "no_user"}), 400

    otp = gen_otp()
    otp_store.set(user, otp, ttl=180)  # 3 minutes
    send_otp_stub(user, otp)
    return jsonify({"ok": True, "message": "OTP sent (stub)"})

@app.post("/verify_otp")
def api_verify_otp():
    body = request.json
    user = body.get("user")
    otp = body.get("otp")
    if not user or not otp:
        return jsonify({"ok": False, "error": "bad_request"}), 400
    ok = otp_store.verify(user, otp)
    if ok:
        log_event(user, "AUTHORIZED", "otp_verified")
        return jsonify({"ok": True, "authorized": True})
    else:
        log_event(user, "OTP_FAIL", f"otp={otp}")
        return jsonify({"ok": False, "authorized": False}), 403

@app.get("/users")
def api_users():
    users = get_all_users()
    # present names only
    return jsonify([{"id": u["id"], "name": u["name"]} for u in users])

@app.get("/logs")
def api_logs():
    rows = get_logs(limit=200)
    return jsonify([{"id": r[0], "user": r[1], "status": r[2], "info": r[3], "ts": r[4]} for r in rows])

if __name__ == "__main__":
    # create DB if missing
    init_db()
    # run dev server
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=True)
