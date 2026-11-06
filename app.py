"""
Flask Application Backend for Pothole Detection DL Project.

Features:
- REST API for deep learning pothole inference.
- Base64 & Multipart Image Support.
- Sample Road Gallery API.
- Live Video / Frame Stream processing.
- Maintenance Inspection Report generation.
- Model architecture metadata API.
"""

import os
import socket
import webbrowser
import threading
import urllib.request
import json
import sys
import time
import base64
import uuid
from datetime import datetime, timezone
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

from model.detector import PotholeDetector

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "weights", "pothole_net.pth")
SAMPLES_DIR = os.path.join(BASE_DIR, "static", "samples")

# Initialize Deep Learning Detector Engine
detector = PotholeDetector(weights_path=WEIGHTS_PATH)

SAMPLE_METADATA = [
    {
        "id": "clean",
        "name": "Pristine Asphalt Highway",
        "filename": "sample_clean_road.jpg",
        "description": "Smooth road surface with yellow line, zero defects.",
        "expected": "0 Defects (PCI ~98%)"
    },
    {
        "id": "minor",
        "name": "Minor Surface Wear",
        "filename": "sample_minor_pothole.jpg",
        "description": "Early-stage cavity forming in mid lane.",
        "expected": "1 Minor Pothole"
    },
    {
        "id": "severe",
        "name": "Severe Hazard Crater",
        "filename": "sample_severe_pothole.jpg",
        "description": "Large deep asphalt breach posing rim damage hazard.",
        "expected": "1 Severe / Critical Pothole"
    },
    {
        "id": "multi",
        "name": "Multi-Defect Road Lane",
        "filename": "sample_multi_potholes.jpg",
        "description": "Multiple potholes spanning across roadway.",
        "expected": "3 Potholes Detected"
    },
    {
        "id": "wet",
        "name": "Rain-Filled Road Hazard",
        "filename": "sample_wet_road.jpg",
        "description": "Wet asphalt with water-retaining depression.",
        "expected": "1 Severe Hazard"
    }
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "device": str(detector.device),
        "model": "PotholeNet",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns available sample road images."""
    return jsonify({
        "status": "success",
        "samples": SAMPLE_METADATA
    })


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    """Returns technical details about the simple DL model."""
    total_params = sum(p.numel() for p in detector.model.parameters())
    trainable_params = sum(p.numel() for p in detector.model.parameters() if p.requires_grad)
    
    return jsonify({
        "model_name": "PotholeNet v1.0",
        "type": "Deep Convolutional Neural Network (PyTorch)",
        "backbone": "4-Stage ConvBlock [32, 64, 128, 256 filters] + BatchNorm + ReLU",
        "heads": [
            "Classification Head: Global AdaptiveAvgPool -> Dense(64) -> Dropout(0.3) -> Softmax",
            "Spatial Defect Head: Receptive Field ConvTranspose2d -> 1x1 Defect Probability Density"
        ],
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "input_resolution": "256x256 RGB",
        "inference_speed": "~25-45 ms (CPU)",
        "severity_metric": "Cavity Area (%) + Local Depth Gradient Variance",
        "road_index": "PCI (Pavement Condition Index: 0 to 100)"
    })


@app.route("/api/detect", methods=["POST"])
def detect_image():
    """
    Main detection endpoint. Accepts multipart/form-data or JSON with base64.
    """
    start_time = time.time()
    img_bgr = None
    conf_threshold = float(request.form.get("confidence", request.json.get("confidence", 0.40) if request.is_json else 0.40))
    
    if "image" in request.files:
        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif request.is_json and "image_base64" in request.json:
        raw_b64 = request.json["image_base64"]
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        decoded = base64.b64decode(raw_b64)
        file_bytes = np.frombuffer(decoded, np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"status": "error", "message": "No valid image provided"}), 400

    # Execute Deep Learning detection
    result = detector.detect(img_bgr, conf_threshold=conf_threshold)
    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    result["inference_time_ms"] = elapsed_ms
    result["status"] = "success"
    
    return jsonify(result)


@app.route("/api/detect-sample", methods=["POST"])
def detect_sample():
    """Runs detection on one of the curated sample images."""
    start_time = time.time()
    data = request.get_json() or {}
    filename = data.get("filename", "sample_severe_pothole.jpg")
    conf_threshold = float(data.get("confidence", 0.40))
    
    # Secure filename
    filename = os.path.basename(filename)
    filepath = os.path.join(SAMPLES_DIR, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"status": "error", "message": f"Sample not found: {filename}"}), 404
        
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        return jsonify({"status": "error", "message": "Failed to read sample image"}), 500

    result = detector.detect(img_bgr, conf_threshold=conf_threshold)
    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    result["inference_time_ms"] = elapsed_ms
    result["status"] = "success"
    result["sample_filename"] = filename
    
    return jsonify(result)


@app.route("/api/export-report", methods=["POST"])
def export_report():
    """Generates an official maintenance inspection report."""
    data = request.get_json() or {}
    inspection_id = f"INS-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    potholes = data.get("potholes", [])
    pci = data.get("pavement_condition_index", 100)
    pci_rating = data.get("pci_rating", "Good")
    urgency = data.get("maintenance_urgency", "Routine")
    
    report = {
        "inspection_id": inspection_id,
        "timestamp": timestamp,
        "detector_version": "PotholeNet v1.0 (Deep Learning)",
        "summary": {
            "potholes_found": len(potholes),
            "pavement_condition_index": f"{pci}/100 ({pci_rating})",
            "maintenance_urgency": urgency,
            "total_damaged_area": f"{data.get('total_damaged_area_pct', 0)}%"
        },
        "defects": potholes,
        "recommended_action": (
            "Immediate cold/hot-mix asphalt patching required to prevent vehicle rim and tire damage."
            if pci < 60 else
            "Surface sealing and scheduled pavement maintenance recommended within normal cycle."
        )
    }
    return jsonify({"status": "success", "report": report})


def check_if_server_active(target_port: int) -> bool:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{target_port}/health")
        with urllib.request.urlopen(req, timeout=1.0) as res:
            data = json.loads(res.read().decode("utf-8"))
            return data.get("status") == "healthy" and data.get("model") == "PotholeNet"
    except Exception:
        return False


def find_free_port(start_port: int = 5000) -> int:
    for p in range(start_port, start_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start_port


def open_browser_after_delay(target_port: int):
    time.sleep(1.2)
    webbrowser.open(f"http://localhost:{target_port}")


if __name__ == "__main__":
    requested_port = int(os.environ.get("PORT", 5000))
    
    # 1. Check if an instance is already running
    if check_if_server_active(requested_port):
        print(f"\n=======================================================")
        print(f"  [ALREADY ACTIVE] AuraRoad DL is already running!")
        print(f"  Dashboard URL: http://localhost:{requested_port}")
        print(f"  Opening browser now...")
        print(f"=======================================================\n")
        webbrowser.open(f"http://localhost:{requested_port}")
        sys.exit(0)

    # 2. Find available port if requested port is blocked by another app
    actual_port = requested_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", requested_port)) == 0:
            actual_port = find_free_port(requested_port + 1)
            print(f"[Notice] Port {requested_port} is busy. Switched to port {actual_port}.")

    print(f"\n=======================================================")
    print(f"  Pothole Detection DL System (Claymorphism UI)")
    print(f"  Running on: http://localhost:{actual_port}")
    print(f"  Auto-opening web browser...")
    print(f"=======================================================\n")
    
    # Launch browser automatically
    threading.Thread(target=open_browser_after_delay, args=(actual_port,), daemon=True).start()
    
    app.run(host="0.0.0.0", port=actual_port, debug=False)
