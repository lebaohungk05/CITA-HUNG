import cv2
import torch
import numpy as np
import os
import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import time # Thêm time để đo FPS

# Add path to import models and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet18_pytorch import resnet18_pytorch
from screen_capture import ScreenCapturer
from face_detector import FaceDetector # Import class mới

# --- CONFIGURATION ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          '../trained_models/emotion_models/fer2013_resnet18_best_sgd.pth')
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
ENGAGEMENT_WEIGHTS = {
    'Happy': 1.0,    
    'Neutral': 0.8,  
    'Surprise': 0.6, 
    'Sad': 0.4,      
    'Fear': 0.3,     
    'Angry': 0.2,    
    'Disgust': 0.2   
}
INPUT_SIZE = 112

class TeacherToolApp:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.engagement_history = []
        self.full_history = [] 
        self.timestamps = []   
        self.max_history = 100 

        # 1. Load Emotion Model
        print("Loading Emotion Model...")
        self.model = resnet18_pytorch(num_classes=7)
        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        # 2. Initialize Face Detection (NEW: MediaPipe Wrapper)
        print("Initializing Face Detector (OpenCV Haar Cascade)...")
        self.detector = FaceDetector()

        # 3. Initialize Screen Capturer
        self.capturer = ScreenCapturer()

    def preprocess_face(self, face_img):
        """Preprocess for PyTorch ResNet"""
        try:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        except:
            return None # Handle case where crop failed
            
        face_img = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
        face_img = face_img.astype('float32') / 255.0
        face_img = (face_img - 0.5) / 0.5
        face_tensor = torch.from_numpy(face_img).unsqueeze(0).unsqueeze(0)
        return face_tensor.to(self.device)

    def run(self):
        # Auto-select primary monitor
        self.capturer.monitor = self.capturer.monitors[1] if len(self.capturer.monitors) > 1 else self.capturer.monitors[0]
        print(f">>> FULL SCREEN MODE ACTIVATED: {self.capturer.monitor}")

        self.capturer.start()
        print(">>> TEACHER TOOL RUNNING.")
        print(">>> Press 'q' to STOP.")
        print(">>> Press 'v' to TOGGLE VIDEO.")
        
        show_video = True 
        window_name = "Student Engagement Analysis (Teacher Tool)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480) 
        cv2.moveWindow(window_name, 0, 0)       

        prev_time = time.time() # For FPS calculation

        while True:
            frame = self.capturer.get_frame()
            if frame is None:
                continue

            h, w, _ = frame.shape
            
            # --- BLIND SPOT (Avoid Infinite Mirror) ---
            # Thu nhỏ vùng đen lại (chỉ che khu vực App hiển thị, mặc định App 640x480)
            process_frame = frame.copy()
            cv2.rectangle(process_frame, (0, 0), (650, 500), (0, 0, 0), -1)

            # --- VISUALIZATION FRAME ---
            if show_video:
                display_frame = frame.copy()
            else:
                display_frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(display_frame, "VIDEO HIDDEN", (50, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # --- DETECT FACES (Using new class) ---
            faces = self.detector.detect_faces(process_frame)

            face_count = 0
            current_total_score = 0
            
            for (x, y, bw, bh) in faces:
                # Sanity check for size
                if bw < 30 or bh < 30: continue
                
                # Check bounds again just to be safe
                if x+bw > w or y+bh > h: continue

                face_count += 1
                face_crop = frame[y:y+bh, x:x+bw]
                
                emotion_text = "Unknown"
                color = (200, 200, 200)

                # Predict Emotion
                with torch.no_grad():
                    face_tensor = self.preprocess_face(face_crop)
                    if face_tensor is not None:
                        outputs = self.model(face_tensor)
                        prob = torch.softmax(outputs, dim=1)
                        score, pred_idx = torch.max(prob, 1)
                        
                        emotion_name = EMOTIONS[pred_idx]
                        current_total_score += ENGAGEMENT_WEIGHTS[emotion_name]
                        emotion_text = f"{emotion_name} ({score.item():.2f})"

                        if emotion_name in ['Angry', 'Fear', 'Sad']:
                            color = (0, 0, 255) # Red for negative
                        elif emotion_name in ['Happy', 'Neutral', 'Surprise']:
                            color = (0, 255, 0) # Green for positive

                if show_video:
                    cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), color, 2)
                    cv2.putText(display_frame, emotion_text, (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- CALCULATE METRICS ---
            class_score = 0
            if face_count > 0:
                class_score = (current_total_score / face_count) * 100
            
            self.engagement_history.append(class_score)
            if len(self.engagement_history) > self.max_history:
                self.engagement_history.pop(0)
            
            # Save metrics (simple sampling)
            if len(self.full_history) == 0 or (datetime.now() - datetime.strptime(self.timestamps[-1], "%H:%M:%S.%f")).total_seconds() > 1.0:
                 self.full_history.append(class_score)
                 self.timestamps.append(datetime.now().strftime("%H:%M:%S.%f"))

            # --- DASHBOARD DRAWING ---
            # 1. Top Bar
            bar_width = 400
            bar_x, bar_y = 20, 80
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 30), (50, 50, 50), -1)
            fill_w = int((class_score / 100) * bar_width)
            bar_color = (0, 0, 255) if class_score < 40 else (0, 255, 255) if class_score < 70 else (0, 255, 0)
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 30), bar_color, -1)
            cv2.putText(display_frame, f"CLASS ENGAGEMENT: {class_score:.1f}%", (bar_x, bar_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 2. Graph
            graph_x, graph_y = 20, 150
            graph_h = 100
            cv2.rectangle(display_frame, (graph_x, graph_y), (graph_x + self.max_history * 2, graph_y + graph_h), (30, 30, 30), -1)
            for i in range(1, len(self.engagement_history)):
                pt1 = (graph_x + (i-1)*2, graph_y + graph_h - int(self.engagement_history[i-1]))
                pt2 = (graph_x + i*2, graph_y + graph_h - int(self.engagement_history[i]))
                cv2.line(display_frame, pt1, pt2, (0, 255, 255), 1)

            # 3. FPS & Info
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            cv2.putText(display_frame, f"FPS: {int(fps)}", (w - 150, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Students: {face_count}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(display_frame, "MediaPipe Enabled", (20, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                show_video = not show_video

        self.capturer.stop()
        cv2.destroyAllWindows()
        self.save_report()

    def save_report(self):
        print("\n>>> GENERATING SESSION REPORT...")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '../reports')
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
            
        csv_path = os.path.join(report_dir, f'session_log_{timestamp_str}.csv')
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Engagement Score'])
                for t, s in zip(self.timestamps, self.full_history):
                    writer.writerow([t, f"{s:.2f}"])
            print(f"   -> Saved Log: {csv_path}")
        except Exception as e:
            print(f"   -> Error saving CSV: {e}")

        chart_path = os.path.join(report_dir, f'session_chart_{timestamp_str}.png')
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.full_history, label='Class Engagement', color='green')
            plt.axhline(y=np.mean(self.full_history) if len(self.full_history) > 0 else 0, color='red', linestyle='--', label='Average')
            plt.title(f'Engagement History (Session: {timestamp_str})')
            plt.ylabel('Score (0-100)')
            plt.xlabel('Time')
            plt.ylim(0, 105)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(chart_path)
            plt.close()
            print(f"   -> Saved Chart: {chart_path}")
        except Exception as e:
            print(f"   -> Error saving Chart: {e}")
        
        print(">>> REPORT GENERATED SUCCESSFULLY.")

if __name__ == "__main__":
    app = TeacherToolApp()
    app.run()