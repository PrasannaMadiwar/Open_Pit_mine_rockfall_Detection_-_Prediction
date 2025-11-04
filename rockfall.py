import cv2
import numpy as np
import pandas as pd
import math
import time
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops
from collections import deque
import json
import serial, threading 
import sqlite3
import pickle
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# ================== Arduino Serial Reader ==================
arduino_data = {}  

def read_arduino():
    global arduino_data
    try:
        ser = serial.Serial('COM3', 115200, timeout=1)
        while True:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue
            try:
                arduino_data = json.loads(line)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print("Arduino read error:", e)

threading.Thread(target=read_arduino, daemon=True).start()

# ================== Machine Learning Model Integration ==================
class MLPredictor:
    def __init__(self, model_path='model.pkl'):
        self.model = None
        self.scaler = None
        self.feature_names = ['slope_angle', 'crack_density', 'displacement', 'temperature', 'vibration']
        self.is_model_loaded = False
        
        # Load the trained regression model
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained regression model from pickle file"""
        try:
            with open(model_path, 'rb') as file:
                model_data = pickle.load(file)
            
            # Handle different pickle formats
            if isinstance(model_data, dict):
                # If pickle contains both model and scaler
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
            else:
                # If pickle contains only the model
                self.model = model_data
                self.scaler = StandardScaler()  # Create default scaler
            
            self.is_model_loaded = True
            print(f"Regression model loaded successfully from {model_path}")
            
            # Print model info if available
            if hasattr(self.model, '__class__'):
                print(f"Model type: {self.model.__class__.__name__}")
            
        except FileNotFoundError:
            print(f"Model file '{model_path}' not found. Predictions will be disabled.")
            self.is_model_loaded = False
        except Exception as e:
            print(f"Error loading model: {str(e)}. Predictions will be disabled.")
            self.is_model_loaded = False
    
    def prepare_features(self, prediction_df):
        """Prepare features for regression prediction"""
        if prediction_df.empty:
            return None
        
        # Get the latest row for prediction
        latest_data = prediction_df.iloc[-1:][self.feature_names].copy()
        
        # Handle missing values
        latest_data = latest_data.fillna(0)  # Replace NaN with 0
        
        # Check if we have valid data
        if latest_data.isnull().all().all():
            print("All features are null, skipping prediction")
            return None
        
        return latest_data.values
    
    def make_prediction(self, prediction_df):
        """Make regression prediction using the loaded model"""
        if not self.is_model_loaded:
            return {
                'predicted_value': None,
                'prediction_error': None,
                'stability_status': 'UNKNOWN',
                'risk_assessment': 'UNKNOWN',
                'status': 'Model not loaded'
            }
        
        try:
            # Prepare features
            features = self.prepare_features(prediction_df)
            if features is None:
                return {
                    'predicted_value': None,
                    'prediction_error': None,
                    'stability_status': 'UNKNOWN',
                    'risk_assessment': 'UNKNOWN',
                    'status': 'Invalid features'
                }
            
            # Scale features if scaler is available
            if self.scaler is not None:
                try:
                    features_scaled = self.scaler.transform(features)
                except:
                    # If scaler fails, use original features
                    features_scaled = features
            else:
                features_scaled = features
            
            # Make regression prediction
            predicted_value = self.model.predict(features_scaled)[0]
            
            # Calculate prediction confidence/error if available
            prediction_error = None
            if hasattr(self.model, 'predict') and hasattr(self.model, 'score'):
                try:
                    # For models that support prediction intervals or have built-in error estimation
                    if len(prediction_df) > 1:
                        # Use historical data to estimate prediction variance
                        recent_features = self.prepare_recent_features(prediction_df)
                        if recent_features is not None:
                            recent_predictions = self.model.predict(recent_features)
                            prediction_error = float(np.std(recent_predictions))
                except:
                    prediction_error = None
            
            # Assess stability based on predicted value
            stability_status = self.assess_stability(predicted_value)
            risk_assessment = self.assess_risk(predicted_value, prediction_error)
            
            return {
                'predicted_value': float(predicted_value),
                'prediction_error': float(prediction_error) if prediction_error is not None else None,
                'stability_status': stability_status,
                'risk_assessment': risk_assessment,
                'status': 'Success'
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return {
                'predicted_value': None,
                'prediction_error': None,
                'stability_status': 'ERROR',
                'risk_assessment': 'ERROR',
                'status': f'Error: {str(e)}'
            }
    
    def prepare_recent_features(self, prediction_df, n_recent=5):
        """Prepare recent features for error estimation"""
        if len(prediction_df) < n_recent:
            return None
        
        recent_data = prediction_df.tail(n_recent)[self.feature_names].copy()
        recent_data = recent_data.fillna(0)
        
        if self.scaler is not None:
            try:
                return self.scaler.transform(recent_data.values)
            except:
                return recent_data.values
        return recent_data.values
    
    def assess_stability(self, predicted_value):
        """Assess slope stability based on predicted value"""
        try:
            # Customize these thresholds based on what your model predicts
            # For example, if your model predicts displacement or stability index:
            
            if predicted_value < 0.1:  # Low displacement/high stability
                return 'STABLE'
            elif predicted_value < 0.5:  # Medium displacement/moderate stability
                return 'MODERATELY_STABLE'
            elif predicted_value < 1.0:  # High displacement/low stability
                return 'UNSTABLE'
            else:  # Very high displacement/very low stability
                return 'CRITICALLY_UNSTABLE'
                
        except:
            return 'UNKNOWN'
    
    def assess_risk(self, predicted_value, prediction_error=None):
        """Assess risk level based on predicted value and uncertainty"""
        try:
            base_risk = 'LOW'
            
            # Base risk assessment from predicted value
            if predicted_value < 0.1:
                base_risk = 'LOW'
            elif predicted_value < 0.5:
                base_risk = 'MEDIUM'
            elif predicted_value < 1.0:
                base_risk = 'HIGH'
            else:
                base_risk = 'CRITICAL'
            
            # Adjust risk based on prediction uncertainty
            if prediction_error is not None and prediction_error > 0.2:
                # High uncertainty increases risk level
                if base_risk == 'LOW':
                    base_risk = 'MEDIUM'
                elif base_risk == 'MEDIUM':
                    base_risk = 'HIGH'
            
            return base_risk
            
        except:
            return 'UNKNOWN'

# Initialize ML Predictor first
ml_predictor = MLPredictor('model.pkl')

# ================== DataFrame Management Class ==================
class SlopeDataManager:
    def __init__(self, prediction_window_size=50):
        self.prediction_window_size = prediction_window_size
        
        # Define columns for database DataFrame (all columns + prediction columns)
        self.db_columns = [
            'timestamp',
            'elapsed_seconds',
            'slope_angle',
            'crack_density',
            'displacement',
            'temperature',
            'humidity', 
            'vibration_x',
            'vibration_y', 
            'vibration_z',
            'vibration_magnitude',
            'crack_width_max',
            'crack_length_max',
            'total_crack_length',
            'edge_density',
            'moisture_index',
            'texture_contrast',
            'rockfall_area',
            'rockfall_detected',
            # ML Prediction columns for regression
            'ml_predicted_value',
            'ml_prediction_error', 
            'ml_stability_status',
            'ml_risk_assessment',
            'ml_status'
        ]
        
        # Define columns for prediction DataFrame (only selected columns)
        self.prediction_columns = [
            'slope_angle',
            'crack_density', 
            'displacement',
            'temperature',
            'vibration'  # This will be vibration_magnitude
        ]
        
        # Permanent DataFrame for database storage (keeps all data)
        self.db_dataframe = pd.DataFrame(columns=self.db_columns)
        
        # Temporary DataFrame for predictions (rolling window with selected columns only)
        self.prediction_dataframe = pd.DataFrame(columns=self.prediction_columns)
        
        print(f"Database DataFrame: {len(self.db_columns)} columns - Stores all historical data + ML regression predictions")
        print(f"Prediction DataFrame: {len(self.prediction_columns)} columns - Rolling window of {prediction_window_size} records")
        print(f"Prediction columns: {self.prediction_columns}")
        print(f"ML Regression model enabled: {ml_predictor.is_model_loaded}")
    
    def add_data_point(self, data_dict):
        """Add a new data point to both DataFrames"""
        # Create timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Extract Arduino data with defaults
        arduino = data_dict.get('arduino', {})
        temp = arduino.get('temp', np.nan)
        humidity = arduino.get('humidity', np.nan) 
        ax = arduino.get('ax', np.nan)
        ay = arduino.get('ay', np.nan)
        az = arduino.get('az', np.nan)
        
        # Calculate vibration magnitude
        if all(v is not None and not np.isnan(v) for v in [ax, ay, az]):
            vibration_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
        else:
            vibration_magnitude = np.nan
        
        # Create new row for database (all columns)
        db_row = {
            'timestamp': timestamp,
            'elapsed_seconds': data_dict.get('elapsed_seconds', 0),
            'slope_angle': data_dict.get('slope_angle_deg', np.nan),
            'crack_density': data_dict.get('edge_density', {}).get('avg', np.nan),
            'displacement': data_dict.get('displacement_m', {}).get('avg', np.nan),
            'temperature': temp,
            'humidity': humidity,
            'vibration_x': ax,
            'vibration_y': ay,
            'vibration_z': az,
            'vibration_magnitude': vibration_magnitude,
            'crack_width_max': data_dict.get('crack_width_m', {}).get('max', np.nan),
            'crack_length_max': data_dict.get('crack_length_m', {}).get('max', np.nan),
            'total_crack_length': data_dict.get('total_crack_length_m', {}).get('avg', np.nan),
            'edge_density': data_dict.get('edge_density', {}).get('avg', np.nan),
            'moisture_index': data_dict.get('moisture_index', {}).get('avg', np.nan),
            'texture_contrast': data_dict.get('texture_contrast', {}).get('avg', np.nan),
            'rockfall_area': data_dict.get('rockfall_area_m2', {}).get('avg', np.nan),
            'rockfall_detected': data_dict.get('rockfall_detected', False)
        }
        
        # Create new row for prediction (only selected columns)
        pred_row = {
            'slope_angle': data_dict.get('slope_angle_deg', np.nan),
            'crack_density': data_dict.get('edge_density', {}).get('avg', np.nan),
            'displacement': data_dict.get('displacement_m', {}).get('avg', np.nan),
            'temperature': temp,
            'vibration': vibration_magnitude  # Using vibration_magnitude for 'vibration' column
        }
        
        # Add to prediction DataFrame with rolling window (selected columns only)
        self.prediction_dataframe = pd.concat([self.prediction_dataframe, pd.DataFrame([pred_row])], ignore_index=True)
        
        # Keep only the last N records in prediction DataFrame
        if len(self.prediction_dataframe) > self.prediction_window_size:
            self.prediction_dataframe = self.prediction_dataframe.tail(self.prediction_window_size).reset_index(drop=True)
        
        # Make ML prediction using the prediction DataFrame
        ml_result = ml_predictor.make_prediction(self.prediction_dataframe)
        
        # Add ML regression prediction results to database row
        db_row.update({
            'ml_predicted_value': ml_result['predicted_value'],
            'ml_prediction_error': ml_result['prediction_error'],
            'ml_stability_status': ml_result['stability_status'],
            'ml_risk_assessment': ml_result['risk_assessment'],
            'ml_status': ml_result['status']
        })
        
        # Add to database DataFrame (permanent storage with all columns + predictions)
        self.db_dataframe = pd.concat([self.db_dataframe, pd.DataFrame([db_row])], ignore_index=True)
        
        print(f"Data added - DB records: {len(self.db_dataframe)}, Prediction records: {len(self.prediction_dataframe)}")
        
        # Print ML regression prediction results if available
        if ml_result['predicted_value'] is not None:
            print(f"ML Prediction: {ml_result['predicted_value']:.4f} | Status: {ml_result['stability_status']} | Risk: {ml_result['risk_assessment']}")
            if ml_result['prediction_error']:
                print(f"   Prediction Error: +/-{ml_result['prediction_error']:.4f}")
        
        return ml_result  # Return prediction results
    
    def get_db_dataframe(self):
        """Get the complete database DataFrame"""
        return self.db_dataframe.copy()
    
    def get_prediction_dataframe(self):
        """Get the prediction DataFrame (recent data only)"""
        return self.prediction_dataframe.copy()
    
    def get_latest_prediction(self):
        """Get the most recent ML regression prediction"""
        if len(self.db_dataframe) == 0:
            return None
        
        latest_row = self.db_dataframe.iloc[-1]
        return {
            'predicted_value': latest_row.get('ml_predicted_value'),
            'prediction_error': latest_row.get('ml_prediction_error'),
            'stability_status': latest_row.get('ml_stability_status'),
            'risk_assessment': latest_row.get('ml_risk_assessment'),
            'status': latest_row.get('ml_status'),
            'timestamp': latest_row.get('timestamp')
        }
    
    def get_prediction_history(self, last_n=10):
        """Get recent regression prediction history"""
        if len(self.db_dataframe) == 0:
            return pd.DataFrame()
        
        prediction_cols = ['timestamp', 'ml_predicted_value', 'ml_prediction_error', 'ml_stability_status', 'ml_risk_assessment', 'ml_status']
        return self.db_dataframe[prediction_cols].tail(last_n).reset_index(drop=True)
    
    def save_to_database(self, db_path='slope_monitoring.db'):
        """Save database DataFrame to SQLite database"""
        try:
            conn = sqlite3.connect(db_path)
            self.db_dataframe.to_sql('slope_monitoring_data', conn, if_exists='replace', index=False)
            conn.close()
            print(f"Database saved to {db_path} with {len(self.db_dataframe)} records")
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    def export_to_csv(self, db_filename='database_export.csv', pred_filename='prediction_data.csv'):
        """Export both DataFrames to CSV files"""
        try:
            self.db_dataframe.to_csv(db_filename, index=False)
            self.prediction_dataframe.to_csv(pred_filename, index=False)
            print(f"Exported - Database ({len(self.db_columns)} cols): {db_filename}")
            print(f"Exported - Prediction ({len(self.prediction_columns)} cols): {pred_filename}")
        except Exception as e:
            print(f"Error exporting CSV: {e}")
    
    def print_summary(self):
        """Print summary statistics of both DataFrames"""
        print("\n" + "="*60)
        print("DATABASE DATAFRAME SUMMARY")
        print("="*60)
        print(f"Total records: {len(self.db_dataframe)}")
        if len(self.db_dataframe) > 0:
            print("\nNumeric columns summary:")
            numeric_cols = self.db_dataframe.select_dtypes(include=[np.number]).columns
            print(self.db_dataframe[numeric_cols].describe())
        
        print("\n" + "="*60)
        print("PREDICTION DATAFRAME SUMMARY")
        print("="*60)
        print(f"Recent records: {len(self.prediction_dataframe)}")
        if len(self.prediction_dataframe) > 0:
            print("\nNumeric columns summary:")
            numeric_cols = self.prediction_dataframe.select_dtypes(include=[np.number]).columns
            print(self.prediction_dataframe[numeric_cols].describe())

# ================== Initialize Data Manager ==================
data_manager = SlopeDataManager(prediction_window_size=100)

# ================== Original Constants ==================
PIXELS_PER_METER = 50
MIN_CRACK_LENGTH = 10
MIN_ROCKFALL_AREA = 100

OUTPUT_INTERVAL = 1.0
SAVE_TO_FILE = True
OUTPUT_FILE = "slope_monitoring_data.json"

SLOPE_BUFFER_SIZE = 30
HOUGH_MIN_LINE_LENGTH = 100
HOUGH_MAX_LINE_GAP = 20

VIDEO_PATH = r"C:\Users\prasa\Desktop\open pit\Recording 2025-09-12 184424.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open the video file.")

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
slope_buffer = deque(maxlen=SLOPE_BUFFER_SIZE)
stable_slope_angle = 0.0
reference_slope_set = False

displacement_buffer = []
rockfall_buffer = []
crack_width_buffer = []
crack_length_buffer = []
total_crack_length_buffer = []
edge_density_buffer = []
moisture_buffer = []
texture_buffer = []

start_time = time.time()
last_output_time = start_time
all_data = []

ret, prev = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("Cannot read the first frame.")
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

def calc_texture(gray):
    small_gray = cv2.resize(gray, (gray.shape[1]//4, gray.shape[0]//4))
    small_gray = (small_gray // 32) * 32
    try:
        glcm = graycomatrix(small_gray, [1], [0], levels=256, symmetric=True, normed=True)
        return graycoprops(glcm, 'contrast')[0, 0]
    except Exception:
        return 0.0

def detect_cracks(edges):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    vertical_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    h_cracks = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    v_cracks = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
    cracks   = cv2.addWeighted(h_cracks, 0.5, v_cracks, 0.5, 0)
    contours, _ = cv2.findContours(cracks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_width_m = max_length_m = total_length_m = 0.0
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CRACK_LENGTH:
            continue
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        crack_len  = max(w, h) / PIXELS_PER_METER
        crack_wid  = min(w, h) / PIXELS_PER_METER
        max_length_m = max(max_length_m, crack_len)
        max_width_m  = max(max_width_m, crack_wid)
        total_length_m += crack_len
    return max_width_m, max_length_m, total_length_m

def calculate_stable_slope_angle(gray_frame):
    global slope_buffer, stable_slope_angle, reference_slope_set
    blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.dilate(edges, kernel, iterations=1)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                           minLineLength=HOUGH_MIN_LINE_LENGTH,
                           maxLineGap=HOUGH_MAX_LINE_GAP)
    if lines is not None and len(lines) > 0:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if angle > 90:
                angle = 180 - angle
            angles.append(angle)
        if angles:
            current_angle = np.median(angles)
            slope_buffer.append(current_angle)
            if len(slope_buffer) >= 5:
                q1, q3 = np.percentile(list(slope_buffer), [25, 75])
                iqr = q3 - q1
                lb = q1 - 1.5 * iqr
                ub = q3 + 1.5 * iqr
                filtered = [a for a in slope_buffer if lb <= a <= ub]
                if filtered:
                    stable_slope_angle = np.mean(filtered)
                    reference_slope_set = True
    return stable_slope_angle

def filter_displacement(flow_magnitude, threshold=1.0):
    sig = flow_magnitude[flow_magnitude > threshold]
    return 0.0 if len(sig) == 0 else np.mean(sig)

def calculate_averages(buffer):
    if not buffer:
        return {"avg": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
    arr = np.array(buffer)
    return {
        "avg": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "std": float(np.std(arr))
    }

def output_synchronized_data(timestamp, elapsed_seconds):
    global displacement_buffer, rockfall_buffer, crack_width_buffer
    global crack_length_buffer, total_crack_length_buffer, edge_density_buffer
    global moisture_buffer, texture_buffer, all_data
    
    data_point = {
        "timestamp": timestamp,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "displacement_m": calculate_averages(displacement_buffer),
        "rockfall_area_m2": calculate_averages(rockfall_buffer),
        "rockfall_detected": any(r > 0 for r in rockfall_buffer) if rockfall_buffer else False,
        "crack_width_m": calculate_averages(crack_width_buffer),
        "crack_length_m": calculate_averages(crack_length_buffer),
        "total_crack_length_m": calculate_averages(total_crack_length_buffer),
        "slope_angle_deg": round(stable_slope_angle, 2),
        "slope_status": "STABLE" if reference_slope_set and len(slope_buffer) >= SLOPE_BUFFER_SIZE else "CALIBRATING",
        "slope_confidence": min(100, (len(slope_buffer) / SLOPE_BUFFER_SIZE) * 100),
        "edge_density": calculate_averages(edge_density_buffer),
        "moisture_index": calculate_averages(moisture_buffer),
        "texture_contrast": calculate_averages(texture_buffer) if texture_buffer else {"avg": 0.0, "max": 0.0, "min": 0.0, "std": 0.0},
        "frames_analyzed": len(displacement_buffer),
        "analysis_quality": "GOOD" if len(displacement_buffer) >= fps * 0.8 else "LOW",
        "arduino": arduino_data.copy() if arduino_data else
                  {"temp": None, "humidity": None, "ax": None, "ay": None, "az": None}
    }
    
    # Add data to both DataFrames and get ML prediction
    ml_result = data_manager.add_data_point(data_point)
    
    print(f"\n=== DATA OUTPUT - {timestamp} (T+{elapsed_seconds:.1f}s) ===")
    print(f"Displacement (avg): {data_point['displacement_m']['avg']:.4f} m")
    print(f"Rockfall Area (avg): {data_point['rockfall_area_m2']['avg']:.4f} mÂ²")
    print(f"Rockfall Detected: {data_point['rockfall_detected']}")
    print(f"Crack Width (max): {data_point['crack_width_m']['max']:.4f} m")
    print(f"Crack Length (max): {data_point['crack_length_m']['max']:.4f} m")
    print(f"Total Crack Length: {data_point['total_crack_length_m']['avg']:.4f} m")
    print(f"Slope Angle: {data_point['slope_angle_deg']}Â° ({data_point['slope_status']}) "
          f"- Confidence: {data_point['slope_confidence']:.0f}%")
    print(f"Edge Density: {data_point['edge_density']['avg']:.4f}")
    print(f"Moisture Index: {data_point['moisture_index']['avg']:.3f}")
    print(f"Texture Contrast: {data_point['texture_contrast']['avg']:.2f}")
    a = data_point["arduino"]
    print(f"Arduino - Temp: {a['temp']}Â°C, Humidity: {a['humidity']}%, "
          f"Vibration: X={a['ax']}, Y={a['ay']}, Z={a['az']} m/sÂ²")
    
    # Display ML regression prediction results
    if ml_result and ml_result['predicted_value'] is not None:
        print(f"ML REGRESSION: {ml_result['predicted_value']:.4f} | Status: {ml_result['stability_status']} | Risk: {ml_result['risk_assessment']}")
        if ml_result['prediction_error']:
            print(f"   Prediction Error: +/-{ml_result['prediction_error']:.4f} | Model Status: {ml_result['status']}")
        else:
            print(f"   Model Status: {ml_result['status']}")
    else:
        print("ML REGRESSION: Not available")

    all_data.append(data_point)
    displacement_buffer.clear()
    rockfall_buffer.clear()
    crack_width_buffer.clear()
    crack_length_buffer.clear()
    total_crack_length_buffer.clear()
    edge_density_buffer.clear()
    moisture_buffer.clear()
    texture_buffer.clear()

frame_count = 0
print("Starting video analysis...")
print(f"Data will be output every {OUTPUT_INTERVAL} second(s)")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    avg_disp_m = filter_displacement(mag, threshold=1.0) / PIXELS_PER_METER
    displacement_buffer.append(avg_disp_m)

    fgmask = bg_sub.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area_m2 = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > MIN_ROCKFALL_AREA) / (PIXELS_PER_METER**2)
    rockfall_buffer.append(total_area_m2)

    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    max_w, max_l, total_l = detect_cracks(edges)
    crack_width_buffer.append(max_w)
    crack_length_buffer.append(max_l)
    total_crack_length_buffer.append(total_l)

    slope_angle = calculate_stable_slope_angle(gray)
    edge_density = np.sum(edges > 0) / edges.size
    edge_density_buffer.append(edge_density)

    mean_s = np.mean(hsv[:,:,1])
    mean_v = np.mean(hsv[:,:,2])
    moisture_index = (mean_s * 0.7 + (255 - mean_v) * 0.3) / 255
    moisture_buffer.append(moisture_index)

    texture_contrast = calc_texture(gray)
    if texture_contrast > 0:
        texture_buffer.append(texture_contrast)

    if current_time - last_output_time >= OUTPUT_INTERVAL:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
        output_synchronized_data(timestamp, elapsed_time)
        last_output_time = current_time

    status_color = (0, 255, 0) if reference_slope_set else (0, 255, 255)
    confidence = min(100, (len(slope_buffer) / SLOPE_BUFFER_SIZE) * 100)
    cv2.putText(frame, f"Slope: {slope_angle:.1f}Â° ({confidence:.0f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Real-time ML regression prediction display on video
    latest_prediction = data_manager.get_latest_prediction()
    if latest_prediction and latest_prediction['predicted_value'] is not None:
        pred_value = latest_prediction['predicted_value']
        pred_text = f"ML: {pred_value:.3f} ({latest_prediction['stability_status']})"
        
        # Color coding based on risk assessment
        risk = latest_prediction['risk_assessment']
        if risk == 'CRITICAL':
            pred_color = (0, 0, 255)  # Red
        elif risk == 'HIGH':
            pred_color = (0, 100, 255)  # Orange
        elif risk == 'MEDIUM':
            pred_color = (0, 255, 255)  # Yellow
        else:
            pred_color = (0, 255, 0)  # Green
        
        cv2.putText(frame, pred_text, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)
        
        if latest_prediction['prediction_error']:
            error_text = f"Error: +/-{latest_prediction['prediction_error']:.3f}"
            cv2.putText(frame, error_text, (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1)
    
    cv2.imshow("Video", frame)
    cv2.imshow("Edges", edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    prev_gray = gray.copy()

# Final data point if needed
current_time = time.time()
elapsed_time = current_time - start_time
if displacement_buffer:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
    output_synchronized_data(timestamp, elapsed_time)

cap.release()
cv2.destroyAllWindows()

# Save and export data
print("\n" + "="*60)
print("PROCESSING COMPLETED")
print("="*60)

# Print summary
data_manager.print_summary()

# Save to database and export CSVs
data_manager.save_to_database()
data_manager.export_to_csv()

# Access DataFrames for further processing
print("\n" + "="*60)
print("ACCESSING DATAFRAMES")
print("="*60)
db_df = data_manager.get_db_dataframe()
pred_df = data_manager.get_prediction_dataframe()

print(f"Database DataFrame shape: {db_df.shape}")
print(f"Prediction DataFrame shape: {pred_df.shape}")
print("\nPrediction DataFrame columns:")
print(list(pred_df.columns))
print("\nRequired prediction columns available:")
for col in data_manager.prediction_columns:
    print(f"  {col}: {'âœ“' if col in pred_df.columns else 'âœ—'}")

print(f"\nDatabase DataFrame columns: {len(db_df.columns)}")
print(f"Prediction DataFrame columns: {len(pred_df.columns)}")

# Display ML Regression Prediction Summary
print("\n" + "="*60)
print("ML REGRESSION PREDICTION SUMMARY")
print("="*60)
if ml_predictor.is_model_loaded:
    prediction_history = data_manager.get_prediction_history()
    if not prediction_history.empty:
        print("Recent Regression Predictions:")
        print(prediction_history.to_string(index=False))
        
        # Statistics on predicted values
        predicted_values = prediction_history['ml_predicted_value'].dropna()
        if len(predicted_values) > 0:
            print(f"\nPredicted Values Statistics:")
            print(f"  Mean: {predicted_values.mean():.4f}")
            print(f"  Std:  {predicted_values.std():.4f}")
            print(f"  Min:  {predicted_values.min():.4f}")
            print(f"  Max:  {predicted_values.max():.4f}")
        
        # Stability status distribution
        stability_counts = prediction_history['ml_stability_status'].value_counts()
        print(f"\nStability Status Distribution:")
        for status, count in stability_counts.items():
            print(f"  {status}: {count}")
            
        # Risk assessment distribution
        risk_counts = prediction_history['ml_risk_assessment'].value_counts()
        print(f"\nRisk Assessment Distribution:")
        for risk, count in risk_counts.items():
            print(f"  {risk}: {count}")
    else:
        print("No predictions available yet.")
else:
    print("ML Regression Model not loaded - predictions disabled.")

# Save original JSON format if enabled
if SAVE_TO_FILE and all_data:
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({
            "metadata": {
                "video_path": VIDEO_PATH,
                "total_duration_seconds": elapsed_time,
                "total_frames_processed": frame_count,
                "fps": fps,
                "output_interval_seconds": OUTPUT_INTERVAL,
                "total_data_points": len(all_data)
            },
            "data": all_data
        }, f, indent=2)
    print(f"\nOriginal JSON data saved to: {OUTPUT_FILE}")

print(f"\nFinal stable slope angle: {stable_slope_angle:.2f}Â°")
print(f"Total processing time: {elapsed_time:.1f} seconds")
print(f"Total frames processed: {frame_count}")
print(f"Total database records: {len(db_df)}")