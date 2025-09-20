import cv2
import numpy as np
import pandas as pd
import math
import time
import threading
import json
import serial
import sqlite3
import pickle
import warnings
import base64
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops
from collections import deque
from sklearn.preprocessing import StandardScaler

# Flask and WebSocket imports
from flask import Flask, render_template_string, jsonify, Response
from flask_socketio import SocketIO, emit
import eventlet

warnings.filterwarnings('ignore')

# ================== Web Application Setup ==================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'slope_monitoring_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global variables for video streaming and data
current_frame = None
processed_frame = None
frame_lock = threading.Lock()

# Global data storage
current_monitoring_data = {
    'timestamp': datetime.now().strftime('%H:%M:%S'),
    'slope_angle': 25.0,
    'displacement': 0.0,
    'crack_density': 0.0,
    'temperature': 22.0,
    'humidity': 65.0,
    'vibration_magnitude': 9.8,
    'vibration_x': 0.0,
    'vibration_y': 0.0,
    'vibration_z': 9.8,
    'ml_predicted_value': 0.1,
    'ml_stability_status': 'STABLE',
    'ml_risk_assessment': 'LOW',
    'rockfall_detected': False,
    'edge_density': 0.0,
    'moisture_index': 0.3,
    'texture_contrast': 150.0,
    'crack_width_max': 0.0,
    'crack_length_max': 0.0,
    'total_crack_length': 0.0,
    'rockfall_area': 0.0,
    'frames_analyzed': 0,
    'analysis_quality': 'GOOD',
    'slope_confidence': 100,
    'slope_status': 'STABLE'
}
data_lock = threading.Lock()

# ================== Arduino Serial Reader ==================
arduino_data = {
    "temp": 22.0,
    "humidity": 65.0,
    "ax": 0.0,
    "ay": 0.0,
    "az": 9.8
}

def read_arduino():
    global arduino_data
    try:
        ser = serial.Serial('COM3', 115200, timeout=1)
        print("Arduino connected on COM3")
        while True:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue
            try:
                arduino_data = json.loads(line)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"Arduino connection failed: {e}")
        print("Running with simulated Arduino data...")
        # Generate mock Arduino data
        while True:
            with data_lock:
                arduino_data = {
                    "temp": round(22 + np.random.normal(0, 2), 1),
                    "humidity": round(65 + np.random.normal(0, 5), 1),
                    "ax": round(np.random.normal(0, 0.1), 3),
                    "ay": round(np.random.normal(0, 0.1), 3),
                    "az": round(9.8 + np.random.normal(0, 0.1), 3)
                }
            time.sleep(1)

# ================== Data Broadcasting Function ==================
def broadcast_data():
    """Continuously broadcast current data to all connected clients"""
    global current_monitoring_data
    
    while True:
        try:
            with data_lock:
                data_to_send = current_monitoring_data.copy()
            
            # Add current timestamp
            data_to_send['timestamp'] = datetime.now().strftime('%H:%M:%S')
            
            # Broadcast to all connected clients
            socketio.emit('slope_data', data_to_send)
            print(f"Broadcasting data: Slope={data_to_send['slope_angle']:.1f}°, Status={data_to_send['ml_stability_status']}")
            
        except Exception as e:
            print(f"Error broadcasting data: {e}")
        
        time.sleep(2)  # Broadcast every 2 seconds

# ================== Machine Learning Predictor ==================
class MLPredictor:
    def __init__(self):
        self.is_model_loaded = False
        print("ML Predictor initialized (using mock predictions)")
    
    def make_prediction(self, slope_angle, crack_density, displacement, temperature, vibration):
        """Make mock prediction for demo"""
        # Generate realistic prediction based on input parameters
        risk_factor = (
            (displacement * 2) + 
            (crack_density * 0.5) + 
            (abs(slope_angle - 25) * 0.01) +
            (vibration * 0.1)
        )
        
        predicted_value = max(0.05, min(1.5, 0.1 + risk_factor + np.random.normal(0, 0.02)))
        
        # Assess stability
        if predicted_value < 0.1:
            stability = 'STABLE'
            risk = 'LOW'
        elif predicted_value < 0.3:
            stability = 'MODERATELY_STABLE'
            risk = 'MEDIUM'
        elif predicted_value < 0.7:
            stability = 'UNSTABLE'
            risk = 'HIGH'
        else:
            stability = 'CRITICALLY_UNSTABLE'
            risk = 'CRITICAL'
        
        return {
            'predicted_value': predicted_value,
            'stability_status': stability,
            'risk_assessment': risk
        }

ml_predictor = MLPredictor()

# ================== Video Processing Functions ==================
def calc_texture(gray):
    try:
        small_gray = cv2.resize(gray, (max(1, gray.shape[1]//4), max(1, gray.shape[0]//4)))
        small_gray = (small_gray // 32) * 32
        glcm = graycomatrix(small_gray, [1], [0], levels=256, symmetric=True, normed=True)
        return graycoprops(glcm, 'contrast')[0, 0]
    except:
        return 150.0

def detect_cracks(edges):
    try:
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        h_cracks = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        v_cracks = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        cracks = cv2.addWeighted(h_cracks, 0.5, v_cracks, 0.5, 0)
        contours, _ = cv2.findContours(cracks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_width_m = max_length_m = total_length_m = 0.0
        for cnt in contours:
            if cv2.contourArea(cnt) < 10:
                continue
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1] if rect[1][0] > 0 and rect[1][1] > 0 else (1, 1)
            crack_len = max(w, h) / 50  # PIXELS_PER_METER
            crack_wid = min(w, h) / 50
            max_length_m = max(max_length_m, crack_len)
            max_width_m = max(max_width_m, crack_wid)
            total_length_m += crack_len
        
        return max_width_m, max_length_m, total_length_m
    except:
        return 0.0, 0.0, 0.0

def calculate_slope_angle(gray_frame):
    try:
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=20)
        
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:
                    angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                    if angle > 90:
                        angle = 180 - angle
                    angles.append(angle)
            
            if angles:
                return np.median(angles)
        
        return 25.0  # Default angle
    except:
        return 25.0

# ================== Video Processing Thread ==================
def video_processing_thread():
    global current_monitoring_data, processed_frame, arduino_data
    
    print("Starting video processing...")
    
    # Try to open video file or webcam
    VIDEO_PATH = r"C:\Users\prasa\Desktop\open pit\Recording 2025-09-12 184424.mp4"
    cap = None
    
    if VIDEO_PATH and VIDEO_PATH.strip():
        cap = cv2.VideoCapture(VIDEO_PATH)
        if cap.isOpened():
            print(f"Using video file: {VIDEO_PATH}")
        else:
            cap = None
    
    if cap is None:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("Using webcam")
        else:
            print("No video source available, running in demo mode")
            demo_mode()
            return
    
    # Initialize background subtractor and variables
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    frame_count = 0
    start_time = time.time()
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Cannot read first frame, switching to demo mode")
        cap.release()
        demo_mode()
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    print("Video analysis started successfully")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop video if it ends
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Process frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Calculate optical flow for displacement
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            displacement = np.mean(mag[mag > 1.0]) / 50 if np.any(mag > 1.0) else 0.0  # Convert to meters
            
            # Background subtraction for rockfall detection
            fgmask = bg_sub.apply(frame)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rockfall_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100) / (50**2)
            
            # Crack detection
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 50, 150)
            crack_width, crack_length, total_crack_length = detect_cracks(edges)
            
            # Calculate other parameters
            slope_angle = calculate_slope_angle(gray)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Moisture index from HSV
            mean_s = np.mean(hsv[:,:,1])
            mean_v = np.mean(hsv[:,:,2])
            moisture_index = (mean_s * 0.7 + (255 - mean_v) * 0.3) / 255
            
            # Texture analysis
            texture_contrast = calc_texture(gray)
            
            # Get Arduino data safely
            with data_lock:
                current_arduino = arduino_data.copy()
            
            # Calculate vibration magnitude
            vibration_magnitude = np.sqrt(current_arduino['ax']**2 + current_arduino['ay']**2 + current_arduino['az']**2)
            
            # Make ML prediction
            ml_result = ml_predictor.make_prediction(
                slope_angle, edge_density, displacement, current_arduino['temp'], vibration_magnitude
            )
            
            # Update global data
            with data_lock:
                current_monitoring_data.update({
                    'slope_angle': float(slope_angle),
                    'displacement': float(displacement),
                    'crack_density': float(edge_density),
                    'temperature': float(current_arduino['temp']),
                    'humidity': float(current_arduino['humidity']),
                    'vibration_magnitude': float(vibration_magnitude),
                    'vibration_x': float(current_arduino['ax']),
                    'vibration_y': float(current_arduino['ay']),
                    'vibration_z': float(current_arduino['az']),
                    'ml_predicted_value': float(ml_result['predicted_value']),
                    'ml_stability_status': ml_result['stability_status'],
                    'ml_risk_assessment': ml_result['risk_assessment'],
                    'rockfall_detected': rockfall_area > 0.001,
                    'edge_density': float(edge_density),
                    'moisture_index': float(moisture_index),
                    'texture_contrast': float(texture_contrast),
                    'crack_width_max': float(crack_width),
                    'crack_length_max': float(crack_length),
                    'total_crack_length': float(total_crack_length),
                    'rockfall_area': float(rockfall_area),
                    'frames_analyzed': frame_count,
                    'analysis_quality': 'GOOD',
                    'slope_confidence': 95.0,
                    'slope_status': 'STABLE' if slope_angle < 30 else 'WARNING'
                })
            
            # Create display frame with overlays
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Slope: {slope_angle:.1f}deg", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Displacement: {displacement:.4f}m", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Status: {ml_result['stability_status']}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Risk: {ml_result['risk_assessment']}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
            cv2.putText(display_frame, f"Temp: {current_arduino['temp']:.1f}C", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            # Store processed frame
            with frame_lock:
                processed_frame = display_frame.copy()
            
            prev_gray = gray.copy()
            time.sleep(0.1)  # Control processing speed
            
    except Exception as e:
        print(f"Video processing error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()

def demo_mode():
    """Run in demo mode with simulated data"""
    global current_monitoring_data, processed_frame
    
    print("Running in DEMO mode with simulated data")
    
    # Create demo frame
    demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(demo_frame, "SLOPE MONITORING DEMO", (120, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(demo_frame, "Simulated Data Mode", (180, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    with frame_lock:
        processed_frame = demo_frame.copy()
    
    start_time = time.time()
    
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Generate realistic simulated data
        slope_angle = 25.0 + np.sin(elapsed_time / 10) * 5
        displacement = 0.01 + abs(np.sin(elapsed_time / 5)) * 0.05
        crack_density = 0.001 + abs(np.sin(elapsed_time / 8)) * 0.005
        
        # Simulate occasional events
        rockfall_detected = np.random.random() < 0.02
        rockfall_area = np.random.uniform(0.1, 2.0) if rockfall_detected else 0.0
        
        # Get simulated Arduino data
        with data_lock:
            current_arduino = arduino_data.copy()
        
        vibration_magnitude = np.sqrt(current_arduino['ax']**2 + current_arduino['ay']**2 + current_arduino['az']**2)
        
        # Make ML prediction
        ml_result = ml_predictor.make_prediction(
            slope_angle, crack_density, displacement, current_arduino['temp'], vibration_magnitude
        )
        
        # Update global data
        with data_lock:
            current_monitoring_data.update({
                'slope_angle': float(slope_angle),
                'displacement': float(displacement),
                'crack_density': float(crack_density),
                'temperature': float(current_arduino['temp']),
                'humidity': float(current_arduino['humidity']),
                'vibration_magnitude': float(vibration_magnitude),
                'vibration_x': float(current_arduino['ax']),
                'vibration_y': float(current_arduino['ay']),
                'vibration_z': float(current_arduino['az']),
                'ml_predicted_value': float(ml_result['predicted_value']),
                'ml_stability_status': ml_result['stability_status'],
                'ml_risk_assessment': ml_result['risk_assessment'],
                'rockfall_detected': rockfall_detected,
                'edge_density': float(crack_density),
                'moisture_index': 0.3 + np.sin(elapsed_time / 20) * 0.1,
                'texture_contrast': 150.0 + np.sin(elapsed_time / 15) * 20,
                'crack_width_max': 0.01,
                'crack_length_max': 0.5,
                'total_crack_length': crack_density * 100,
                'rockfall_area': float(rockfall_area),
                'frames_analyzed': int(elapsed_time * 10),
                'analysis_quality': 'GOOD',
                'slope_confidence': 95.0,
                'slope_status': 'STABLE' if slope_angle < 30 else 'WARNING'
            })
        
        # Update demo frame
        demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(demo_frame, "SLOPE MONITORING DEMO", (120, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(demo_frame, f"Slope: {slope_angle:.1f}deg", (50, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(demo_frame, f"Status: {ml_result['stability_status']}", (50, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(demo_frame, f"Risk: {ml_result['risk_assessment']}", (50, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        cv2.putText(demo_frame, f"Temp: {current_arduino['temp']:.1f}C", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        cv2.putText(demo_frame, f"Time: {elapsed_time:.0f}s", (50, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        with frame_lock:
            processed_frame = demo_frame.copy()
        
        time.sleep(2)

# ================== Video Streaming ==================
def generate_video_stream():
    while True:
        with frame_lock:
            if processed_frame is not None:
                ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ================== Flask Routes ==================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_data():
    with data_lock:
        return jsonify(current_monitoring_data)

# ================== WebSocket Events ==================
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send current data immediately
    with data_lock:
        data_to_send = current_monitoring_data.copy()
    data_to_send['timestamp'] = datetime.now().strftime('%H:%M:%S')
    emit('slope_data', data_to_send)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('request_data')
def handle_request_data():
    with data_lock:
        data_to_send = current_monitoring_data.copy()
    data_to_send['timestamp'] = datetime.now().strftime('%H:%M:%S')
    emit('slope_data', data_to_send)

# ================== HTML Template ==================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Slope Monitoring System</title>
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50, #4a69bd);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status-card {
            grid-column: 1 / -1;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        .status-dot {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .status-stable { background-color: #2ecc71; }
        .status-moderate { background-color: #f39c12; }
        .status-unstable { background-color: #e74c3c; }
        .status-critical { background-color: #c0392b; }
        .status-unknown { background-color: #7f8c8d; }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        .metric-value {
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .chart-container {
            height: 300px;
            margin-top: 20px;
        }
        
        .alert-banner {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: bold;
            display: none;
        }
        .alert-high { background-color: #ff6b6b; color: white; }
        .alert-critical { background-color: #c0392b; color: white; animation: pulse 1.5s infinite; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .last-update {
            text-align: right;
            font-size: 0.9em;
            color: #777;
            margin-top: 10px;
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 5px;
            color: white;
            font-weight: bold;
            z-index: 1000;
        }
        .connected { background-color: #2ecc71; }
        .disconnected { background-color: #e74c3c; }
        
        .system-info {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #4a69bd;
        }
    </style>
</head>
<body>
    <div id="connection-status" class="connection-status disconnected">Connecting...</div>
    
    <div class="container">
        <div class="header">
            <h1>Slope Stability Monitoring System</h1>
            <p>Real-time monitoring and analysis of slope stability using computer vision and machine learning</p>
        </div>
        
        <div id="alert-banner" class="alert-banner"></div>
        
        <div class="dashboard">
            <div class="card status-card">
                <div>
                    <h2>Slope Status</h2>
                    <div class="status-indicator">
                        <div id="status-dot" class="status-dot status-unknown"></div>
                        <div>
                            <h3 id="stability-status">Initializing...</h3>
                            <p id="risk-assessment">Risk assessment: Loading...</p>
                        </div>
                    </div>
                    
                    <div class="metric">
                        <span>Slope Angle:</span>
                        <span id="slope-angle" class="metric-value">-- °</span>
                    </div>
                    <div class="metric">
                        <span>Displacement:</span>
                        <span id="displacement" class="metric-value">-- m</span>
                    </div>
                    <div class="metric">
                        <span>Crack Density:</span>
                        <span id="crack-density" class="metric-value">--</span>
                    </div>
                    <div class="metric">
                        <span>Temperature:</span>
                        <span id="temperature" class="metric-value">-- °C</span>
                    </div>
                    <div class="metric">
                        <span>Humidity:</span>
                        <span id="humidity" class="metric-value">-- %</span>
                    </div>
                </div>
                
                <div>
                    <h2>Sensor Data</h2>
                    <div class="metric">
                        <span>Vibration Magnitude:</span>
                        <span id="vibration" class="metric-value">-- m/s²</span>
                    </div>
                    <div class="metric">
                        <span>Vibration X:</span>
                        <span id="vibration-x" class="metric-value">-- m/s²</span>
                    </div>
                    <div class="metric">
                        <span>Vibration Y:</span>
                        <span id="vibration-y" class="metric-value">-- m/s²</span>
                    </div>
                    <div class="metric">
                        <span>Vibration Z:</span>
                        <span id="vibration-z" class="metric-value">-- m/s²</span>
                    </div>
                    <div class="metric">
                        <span>Rockfall Detected:</span>
                        <span id="rockfall" class="metric-value">--</span>
                    </div>
                    <div class="metric">
                        <span>Rockfall Area:</span>
                        <span id="rockfall-area" class="metric-value">-- m²</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Displacement Trend</h2>
                <div class="chart-container">
                    <canvas id="displacement-chart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>Slope Angle Trend</h2>
                <div class="chart-container">
                    <canvas id="slope-chart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>Vibration Trend</h2>
                <div class="chart-container">
                    <canvas id="vibration-chart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>Crack Analysis</h2>
                <div class="metric">
                    <span>Max Crack Width:</span>
                    <span id="crack-width" class="metric-value">-- m</span>
                </div>
                <div class="metric">
                    <span>Max Crack Length:</span>
                    <span id="crack-length" class="metric-value">-- m</span>
                </div>
                <div class="metric">
                    <span>Total Crack Length:</span>
                    <span id="total-crack-length" class="metric-value">-- m</span>
                </div>
                <div class="metric">
                    <span>Texture Contrast:</span>
                    <span id="texture-contrast" class="metric-value">--</span>
                </div>
                <div class="metric">
                    <span>Moisture Index:</span>
                    <span id="moisture-index" class="metric-value">--</span>
                </div>
            </div>
            
            <div class="card">
                <h2>System Information</h2>
                <div class="metric">
                    <span>Frames Analyzed:</span>
                    <span id="frames-analyzed" class="metric-value">--</span>
                </div>
                <div class="metric">
                    <span>Analysis Quality:</span>
                    <span id="analysis-quality" class="metric-value">--</span>
                </div>
                <div class="metric">
                    <span>Slope Confidence:</span>
                    <span id="slope-confidence" class="metric-value">-- %</span>
                </div>
                <div class="metric">
                    <span>ML Predicted Value:</span>
                    <span id="ml-predicted" class="metric-value">--</span>
                </div>
                <div class="last-update">
                    Last updated: <span id="last-update">--</span>
                </div>
            </div>
        </div>
        
        <div class="system-info">
            <h3>System Status: <span id="system-status">Operational</span></h3>
            <p>This system monitors slope stability in real-time using computer vision and sensor data.</p>
        </div>
    </div>

    <script>
        console.log('Starting Slope Monitoring System...');
        
        // Initialize SocketIO connection
        const socket = io({
            transports: ['websocket', 'polling'],
            timeout: 20000,
            forceNew: true
        });
        
        // Connection status indicator
        const connectionStatus = document.getElementById('connection-status');
        
        // Initialize charts
        const displacementCtx = document.getElementById('displacement-chart').getContext('2d');
        const slopeCtx = document.getElementById('slope-chart').getContext('2d');
        const vibrationCtx = document.getElementById('vibration-chart').getContext('2d');
        
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0  // Disable animations for better performance
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            elements: {
                point: {
                    radius: 3
                }
            }
        };
        
        const displacementChart = new Chart(displacementCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Displacement (m)',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    tension: 0.1,
                    fill: false
                }]
            },
            options: chartOptions
        });
        
        const slopeChart = new Chart(slopeCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Slope Angle (°)',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.1,
                    fill: false
                }]
            },
            options: chartOptions
        });
        
        const vibrationChart = new Chart(vibrationCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Vibration (m/s²)',
                    data: [],
                    borderColor: 'rgb(153, 102, 255)',
                    backgroundColor: 'rgba(153, 102, 255, 0.1)',
                    tension: 0.1,
                    fill: false
                }]
            },
            options: chartOptions
        });
        
        // Data history for charts
        const maxDataPoints = 50;
        const timestamps = [];
        const displacementData = [];
        const slopeData = [];
        const vibrationData = [];
        
        // Socket event handlers
        socket.on('connect', function() {
            console.log('Connected to server');
            connectionStatus.textContent = 'Connected';
            connectionStatus.className = 'connection-status connected';
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from server');
            connectionStatus.textContent = 'Disconnected';
            connectionStatus.className = 'connection-status disconnected';
        });
        
        socket.on('slope_data', function(data) {
            console.log('Received data:', data);
            updateDashboard(data);
        });
        
        // Update dashboard with new data
        function updateDashboard(data) {
            // Update status indicators
            document.getElementById('stability-status').textContent = data.ml_stability_status;
            document.getElementById('risk-assessment').textContent = 'Risk assessment: ' + data.ml_risk_assessment;
            
            // Update status dot color based on stability
            const statusDot = document.getElementById('status-dot');
            statusDot.className = 'status-dot ';
            if (data.ml_stability_status === 'STABLE') {
                statusDot.classList.add('status-stable');
            } else if (data.ml_stability_status === 'MODERATELY_STABLE') {
                statusDot.classList.add('status-moderate');
            } else if (data.ml_stability_status === 'UNSTABLE') {
                statusDot.classList.add('status-unstable');
            } else if (data.ml_stability_status === 'CRITICALLY_UNSTABLE') {
                statusDot.classList.add('status-critical');
            } else {
                statusDot.classList.add('status-unknown');
            }
            
            // Update metrics
            document.getElementById('slope-angle').textContent = data.slope_angle.toFixed(1) + ' °';
            document.getElementById('displacement').textContent = data.displacement.toFixed(4) + ' m';
            document.getElementById('crack-density').textContent = data.crack_density.toFixed(4);
            document.getElementById('temperature').textContent = data.temperature.toFixed(1) + ' °C';
            document.getElementById('humidity').textContent = data.humidity.toFixed(1) + ' %';
            document.getElementById('vibration').textContent = data.vibration_magnitude.toFixed(2) + ' m/s²';
            document.getElementById('vibration-x').textContent = data.vibration_x.toFixed(2) + ' m/s²';
            document.getElementById('vibration-y').textContent = data.vibration_y.toFixed(2) + ' m/s²';
            document.getElementById('vibration-z').textContent = data.vibration_z.toFixed(2) + ' m/s²';
            document.getElementById('rockfall').textContent = data.rockfall_detected ? 'YES' : 'NO';
            document.getElementById('rockfall-area').textContent = data.rockfall_area.toFixed(2) + ' m²';
            document.getElementById('crack-width').textContent = data.crack_width_max.toFixed(3) + ' m';
            document.getElementById('crack-length').textContent = data.crack_length_max.toFixed(2) + ' m';
            document.getElementById('total-crack-length').textContent = data.total_crack_length.toFixed(2) + ' m';
            document.getElementById('texture-contrast').textContent = data.texture_contrast.toFixed(1);
            document.getElementById('moisture-index').textContent = data.moisture_index.toFixed(2);
            document.getElementById('frames-analyzed').textContent = data.frames_analyzed;
            document.getElementById('analysis-quality').textContent = data.analysis_quality;
            document.getElementById('slope-confidence').textContent = data.slope_confidence.toFixed(1) + ' %';
            document.getElementById('ml-predicted').textContent = data.ml_predicted_value.toFixed(4);
            document.getElementById('last-update').textContent = data.timestamp;
            
            // Update charts
            timestamps.push(data.timestamp);
            displacementData.push(data.displacement);
            slopeData.push(data.slope_angle);
            vibrationData.push(data.vibration_magnitude);
            
            // Limit data points
            if (timestamps.length > maxDataPoints) {
                timestamps.shift();
                displacementData.shift();
                slopeData.shift();
                vibrationData.shift();
            }
            
            // Update chart data
            displacementChart.data.labels = [...timestamps];
            displacementChart.data.datasets[0].data = [...displacementData];
            displacementChart.update('none');
            
            slopeChart.data.labels = [...timestamps];
            slopeChart.data.datasets[0].data = [...slopeData];
            slopeChart.update('none');
            
            vibrationChart.data.labels = [...timestamps];
            vibrationChart.data.datasets[0].data = [...vibrationData];
            vibrationChart.update('none');
            
            // Show alerts if needed
            const alertBanner = document.getElementById('alert-banner');
            if (data.ml_risk_assessment === 'CRITICAL') {
                alertBanner.textContent = 'CRITICAL ALERT: Slope instability detected! Immediate action required!';
                alertBanner.className = 'alert-banner alert-critical';
                alertBanner.style.display = 'block';
            } else if (data.ml_risk_assessment === 'HIGH') {
                alertBanner.textContent = 'WARNING: High risk of slope instability detected';
                alertBanner.className = 'alert-banner alert-high';
                alertBanner.style.display = 'block';
            } else {
                alertBanner.style.display = 'none';
            }
        }
        
        // Request initial data
        socket.emit('request_data');
        
        // Periodically request data as backup to the push mechanism
        setInterval(() => {
            if (socket.connected) {
                socket.emit('request_data');
            }
        }, 5000);
        
        console.log('Slope Monitoring System initialized');
    </script>
</body>
</html>
"""

# ================== Main Application ==================
if __name__ == '__main__':
    print("Starting Slope Monitoring System...")
    
    # Start Arduino reader thread
    arduino_thread = threading.Thread(target=read_arduino, daemon=True)
    arduino_thread.start()
    
    # Start video processing thread
    video_thread = threading.Thread(target=video_processing_thread, daemon=True)
    video_thread.start()
    
    # Start data broadcasting thread
    broadcast_thread = threading.Thread(target=broadcast_data, daemon=True)
    broadcast_thread.start()
    
    print("All threads started. Starting web server...")
    print("Open http://localhost:5000 in your browser to view the dashboard")
    
    # Start Flask-SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)