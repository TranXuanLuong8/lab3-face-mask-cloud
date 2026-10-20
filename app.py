import base64
import io
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from inference import Predictor

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "mask_detector_3class_refined.onnx"
FACE_DIR = ROOT / "face_detector"
DATABASE_URL = os.getenv("DATABASE_URL", "")
SQLITE_PATH = ROOT / "detections.db"

CLASS_NAMES = ("mask", "no_mask", "incorrect_mask")
DISPLAY_LABELS = {
    "mask": "Đeo khẩu trang",
    "no_mask": "Không đeo khẩu trang",
    "incorrect_mask": "Đeo khẩu trang sai cách",
}
ALERT_ENABLED = {"mask": False, "no_mask": True, "incorrect_mask": True}
ALERT_THRESHOLD = 0.80
ALERT_COOLDOWN_S = 30

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TG_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"

app = Flask(__name__)
CORS(app)
predictor = Predictor(MODEL_PATH, FACE_DIR)
state_lock = threading.Lock()
last_snapshot = None
last_result = None
last_alert_at = {name: 0.0 for name in CLASS_NAMES}
telegram_enabled = True
last_update_id = 0


def db_connection():
    if DATABASE_URL:
        import psycopg
        from psycopg.rows import dict_row
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)
    connection = sqlite3.connect(str(SQLITE_PATH))
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with db_connection() as conn:
        if DATABASE_URL:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id BIGSERIAL PRIMARY KEY, timestamp TEXT NOT NULL,
                    label TEXT NOT NULL, confidence DOUBLE PRECISION,
                    class1_prob DOUBLE PRECISION, class2_prob DOUBLE PRECISION,
                    class3_prob DOUBLE PRECISION, face_count INTEGER NOT NULL,
                    inference_ms DOUBLE PRECISION, rtt_ms DOUBLE PRECISION,
                    total_ms DOUBLE PRECISION, telegram_sent BOOLEAN DEFAULT FALSE
                )
            """)
        else:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                    label TEXT NOT NULL, confidence REAL,
                    class1_prob REAL, class2_prob REAL, class3_prob REAL,
                    face_count INTEGER NOT NULL, inference_ms REAL,
                    rtt_ms REAL, total_ms REAL, telegram_sent INTEGER DEFAULT 0
                )
            """)
        conn.commit()


def telegram_configured():
    return bool(BOT_TOKEN and CHAT_ID)


def send_telegram_photo(image_bytes, caption):
    if not telegram_enabled or not telegram_configured() or not image_bytes:
        return False
    try:
        response = requests.post(
            f"{TG_BASE}/sendPhoto",
            data={"chat_id": CHAT_ID, "caption": caption},
            files={"photo": ("frame.jpg", io.BytesIO(image_bytes), "image/jpeg")},
            timeout=15,
        )
        return response.ok and response.json().get("ok", False)
    except requests.RequestException as exc:
        app.logger.error("Telegram sendPhoto failed: %s", exc)
        return False


def send_telegram_text(text):
    if not telegram_enabled or not telegram_configured():
        return False
    try:
        response = requests.post(
            f"{TG_BASE}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text},
            timeout=10,
        )
        return response.ok and response.json().get("ok", False)
    except requests.RequestException as exc:
        app.logger.error("Telegram sendMessage failed: %s", exc)
        return False


def result_caption(result, prefix="🚨 LAB 3-2"):
    probabilities = result["probabilities"]
    return (
        f"{prefix}\n"
        f"🏷 {DISPLAY_LABELS.get(result['label'], result['label'])}\n"
        f"🎯 Độ tin cậy: {result['confidence'] * 100:.1f}%\n"
        f"📊 3 nhãn: {probabilities['mask'] * 100:.1f}% | "
        f"{probabilities['no_mask'] * 100:.1f}% | "
        f"{probabilities['incorrect_mask'] * 100:.1f}%\n"
        f"⚡ Suy luận Cloud: {result['inference_ms']:.1f} ms\n"
        f"🔁 RTT local ↔ Cloud: {result.get('rtt_ms', 0):.1f} ms\n"
        f"⏱ Toàn trình: {result.get('total_ms', 0):.1f} ms"
    )


def maybe_alert(result, image_bytes):
    label = result["label"]
    now = time.time()
    with state_lock:
        allowed = (
            telegram_enabled
            and ALERT_ENABLED.get(label, False)
            and result["confidence"] >= ALERT_THRESHOLD
            and now - last_alert_at[label] >= ALERT_COOLDOWN_S
        )
        if allowed:
            last_alert_at[label] = now
    if not allowed:
        return False
    threading.Thread(
        target=send_telegram_photo,
        args=(image_bytes, result_caption(result)),
        daemon=True,
    ).start()
    return True


def save_detection(result, rtt_ms, total_ms, telegram_sent):
    probabilities = result["probabilities"]
    values = (
        datetime.now(timezone.utc).isoformat(), result["label"],
        result["confidence"], probabilities["mask"],
        probabilities["no_mask"], probabilities["incorrect_mask"],
        result["face_count"], result["inference_ms"], rtt_ms, total_ms,
        telegram_sent,
    )
    with db_connection() as conn:
        if DATABASE_URL:
            row = conn.execute(
                """INSERT INTO detections
                   (timestamp,label,confidence,class1_prob,class2_prob,class3_prob,
                    face_count,inference_ms,rtt_ms,total_ms,telegram_sent)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                values,
            ).fetchone()
            record_id = row[0]
        else:
            cursor = conn.execute(
                """INSERT INTO detections
                   (timestamp,label,confidence,class1_prob,class2_prob,class3_prob,
                    face_count,inference_ms,rtt_ms,total_ms,telegram_sent)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                values,
            )
            record_id = cursor.lastrowid
        conn.commit()
    return record_id


def parse_image():
    uploaded = request.files.get("image")
    if uploaded:
        image_bytes = uploaded.read()
    else:
        payload = request.get_json(silent=True) or {}
        encoded = payload.get("image_base64", "")
        image_bytes = base64.b64decode(encoded.split(",", 1)[-1]) if encoded else b""
    if not image_bytes or len(image_bytes) > 8 * 1024 * 1024:
        raise ValueError("Ảnh bắt buộc và phải nhỏ hơn 8 MB.")
    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Không đọc được ảnh JPEG/PNG.")
    return image_bytes, frame


@app.get("/")
def index():
    return render_template("index.html", model_sha256=predictor.model_sha256)


@app.get("/health")
def health():
    return jsonify({"ok": True, "inference": "cloud", "database": "postgres" if DATABASE_URL else "sqlite"})


@app.post("/predict")
def predict():
    started = time.perf_counter()
    try:
        image_bytes, frame = parse_image()
        faces, timings = predictor.predict_with_metrics(frame)
        inference_ms = timings["inference_ms"]
        if faces:
            winning = max(faces, key=lambda face: face["confidence"])
            result = {
                "label": winning["label"],
                "display_label": DISPLAY_LABELS[winning["label"]],
                "confidence": winning["confidence"],
                "probabilities": winning["probabilities"],
                "faces": faces,
                "face_count": len(faces),
                "inference_ms": inference_ms,
            }
        else:
            result = {
                "label": "no_face", "display_label": "Không tìm thấy khuôn mặt",
                "confidence": 0, "probabilities": {name: 0 for name in CLASS_NAMES},
                "faces": [], "face_count": 0, "inference_ms": inference_ms,
            }
        with state_lock:
            global last_snapshot, last_result
            last_snapshot = image_bytes
            last_result = result
        telegram_sent = maybe_alert(result, image_bytes) if result["face_count"] else False
        total_ms = (time.perf_counter() - started) * 1000
        record_id = save_detection(result, 0, total_ms, telegram_sent)
        result.update({
            "id": record_id, "rtt_ms": 0, "total_ms": total_ms,
            "telegram_sent": telegram_sent, "timings": timings,
        })
        return jsonify(result)
    except (ValueError, cv2.error) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Cloud inference failed."}), 500


@app.get("/history")
def history():
    limit = min(max(request.args.get("limit", 100, type=int), 1), 500)
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM detections ORDER BY id DESC LIMIT %s" % limit
        ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.get("/stats")
def stats():
    with db_connection() as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS total, AVG(inference_ms) AS avg_inference,
                   AVG(rtt_ms) AS avg_rtt, AVG(total_ms) AS avg_total,
                   SUM(CASE WHEN label='mask' THEN 1 ELSE 0 END) AS mask_count,
                   SUM(CASE WHEN label='no_mask' THEN 1 ELSE 0 END) AS no_mask_count,
                   SUM(CASE WHEN label='incorrect_mask' THEN 1 ELSE 0 END) AS incorrect_count
            FROM detections
        """).fetchone()
    values = dict(row)
    return jsonify({key: (round(value or 0, 2) if isinstance(value, (float, int)) else value)
                    for key, value in values.items()})


@app.post("/telegram-toggle")
def telegram_toggle():
    global telegram_enabled
    with state_lock:
        telegram_enabled = not telegram_enabled
    return jsonify({"enabled": telegram_enabled})


def handle_bot_updates(updates):
    global telegram_enabled
    for update in updates:
        message = update.get("message", {})
        if str(message.get("chat", {}).get("id", "")) != CHAT_ID:
            continue
        command = message.get("text", "").strip()
        if command == "/photo":
            with state_lock:
                image, result = last_snapshot, last_result
            if image and result:
                send_telegram_photo(image, result_caption(result, "📸 ẢNH HIỆN TẠI"))
            else:
                send_telegram_text("Chưa có ảnh nhận dạng.")
        elif command == "/on":
            telegram_enabled = True
            send_telegram_text("✅ Đã bật cảnh báo Telegram.")
        elif command == "/off":
            telegram_enabled = False
            send_telegram_text("⏸ Đã tắt cảnh báo Telegram.")
        elif command == "/status":
            send_telegram_text(f"Telegram: {'ON' if telegram_enabled else 'OFF'}\nNgưỡng: {ALERT_THRESHOLD:.0%}\nCooldown: {ALERT_COOLDOWN_S}s")
        elif command == "/stats":
            send_telegram_text(str(stats().json))


def bot_poll_loop():
    global last_update_id
    while True:
        try:
            response = requests.get(
                f"{TG_BASE}/getUpdates",
                params={"offset": last_update_id + 1, "timeout": 8},
                timeout=12,
            )
            if response.ok:
                updates = response.json().get("result", [])
                if updates:
                    last_update_id = updates[-1]["update_id"]
                    handle_bot_updates(updates)
        except requests.RequestException:
            app.logger.warning("Telegram polling temporarily unavailable.")
        time.sleep(2)


init_db()
if telegram_configured():
    threading.Thread(target=bot_poll_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5050")))
