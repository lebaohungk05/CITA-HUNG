import sys
import os
import customtkinter as ctk
from tkinter import messagebox, filedialog
import threading
import time
import matplotlib
matplotlib.use('Agg') 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageGrab
import csv
from datetime import datetime
import sqlite3
import torch

# Fix path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from models.resnet18_custom import resnet18_custom
    from teacher_tool.face_detector import FaceDetector
    from teacher_tool.smoothing import EMASmoother
except ImportError:
    try:
        from src.models.resnet18_custom import resnet18_custom
        from src.teacher_tool.face_detector import FaceDetector
        from src.teacher_tool.smoothing import EMASmoother
    except ImportError as e:
        print(f"Import Error: {e}")

# --- PALETTE (Modern Dark/Light Hybrid) ---
COLOR_BG = "#F8FAFC"       # Slate 50 (Main Background)
COLOR_SIDEBAR = "#FFFFFF"  # White
COLOR_CARD = "#FFFFFF"     # White
COLOR_PRIMARY = "#0EA5E9"  # Sky 500 (Light Sea Blue)
COLOR_PRIMARY_HOVER = "#38BDF8" # Sky 400
COLOR_TEXT_MAIN = "#1E293B" # Slate 800
COLOR_TEXT_SUB = "#64748B"  # Slate 500
COLOR_BORDER = "#E2E8F0"    # Slate 200
COLOR_SUCCESS = "#10B981"   # Emerald 500
COLOR_WARNING = "#F59E0B"   # Amber 500
COLOR_DANGER = "#EF4444"    # Red 500

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

# --- CONFIG ---
MODEL_EMOTION_PATH = os.path.join(src_dir, '../trained_models/emotion_models/fer2013_resnet18_mixup_best.pth')
MODEL_GENDER_PATH = os.path.join(src_dir, '../trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5') 
DB_PATH = os.path.join(src_dir, '../student_engagement.db')

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
GENDERS = ['Female', 'Male']
ENGAGEMENT_WEIGHTS = {
    'Happy': 1.0, 'Neutral': 0.8, 'Surprise': 0.6,
    'Sad': 0.4, 'Fear': 0.3, 'Angry': 0.2, 'Disgust': 0.2
}

# --- BACKEND ---
class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._migrate()

    def _migrate(self):
        self.cursor.execute('CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, start_time TEXT, end_time TEXT)')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, timestamp TEXT,
                engagement_score REAL, student_count INTEGER, emotions TEXT,
                male_count INTEGER DEFAULT 0, female_count INTEGER DEFAULT 0
            )
        ''')
        try: self.cursor.execute("ALTER TABLE logs ADD COLUMN male_count INTEGER DEFAULT 0")
        except: pass
        try: self.cursor.execute("ALTER TABLE logs ADD COLUMN female_count INTEGER DEFAULT 0")
        except: pass
        self.conn.commit()

    def start_session(self):
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("INSERT INTO sessions (start_time) VALUES (?)", (start_time,))
        self.conn.commit()
        self.session_id = self.cursor.lastrowid

    def log_data(self, score, count, emotions_str="", m_count=0, f_count=0):
        if not hasattr(self, 'session_id'): return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("""
            INSERT INTO logs (session_id, timestamp, engagement_score, student_count, emotions, male_count, female_count) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (self.session_id, timestamp, score, count, emotions_str, m_count, f_count))
        self.conn.commit()

    def end_session(self):
        if not hasattr(self, 'session_id'): return
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("UPDATE sessions SET end_time = ? WHERE id = ?", (end_time, self.session_id))
        self.conn.commit()

    def get_all_sessions(self):
        self.cursor.execute("""
            SELECT s.id, s.start_time, s.end_time, 
                   MAX(l.student_count) as max_students, 
                   AVG(l.engagement_score) as avg_score,
                   'Completed' as status
            FROM sessions s
            LEFT JOIN logs l ON s.id = l.session_id
            GROUP BY s.id
            ORDER BY s.id DESC
        """)
        return self.cursor.fetchall()

    def get_session_logs(self, session_id):
        self.cursor.execute("SELECT * FROM logs WHERE session_id = ? ORDER BY id ASC", (session_id,))
        return self.cursor.fetchall()

class TeacherMonitorApp:
    def __init__(self):
        self.running = False
        self.source_type = "Webcam"
        self.cap = None
        self.db = DatabaseManager(DB_PATH)
        self.frame_skip = 3
        self.frame_count = 0
        self.last_results = []
        
        # Performance Tracking
        self.fps = 0
        self.prev_time = 0
        self.history_focus = [0] * 50 # History for real-time chart
        
        # FORCE CPU to fix "Input type (cuda) and weight type (cpu)" mismatch
        self.device = torch.device("cpu") 
        self.face_detector = FaceDetector()
        self.emotion_model = None
        self.gender_model = None
        self.smoother = EMASmoother(alpha=0.15) # Smoothing Factor
        self._load_models()

    def _load_models(self):
        try:
            print(f"[AI] Loading Emotion Model to {self.device}...")
            # Use custom resnet model
            model = resnet18_custom(num_classes=7, pretrained=False)
            state_dict = torch.load(MODEL_EMOTION_PATH, map_location=self.device)
            # Handle potential key mismatch if model was saved weirdly
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            self.emotion_model = model
            print("[AI] Emotion Loaded Successfully.")
        except Exception as e:
            print(f"[Err] Emotion Model Load Failed: {e}")
            self.emotion_model = None

        try:
            from keras.models import load_model
            if os.path.exists(MODEL_GENDER_PATH):
                self.gender_model = load_model(MODEL_GENDER_PATH, compile=False)
                print("[AI] Gender Loaded.")
        except Exception as e: print(f"[Err] Gender: {e}")

    def start_monitoring(self, source_type="Webcam"):
        self.source_type = source_type
        self.running = True
        self.smoother.reset() # Reset smoothing on new session
        if self.source_type == "Webcam":
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened(): self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.db.start_session()
        self.last_log_time = time.time()

    def stop_monitoring(self):
        self.running = False
        if self.cap: self.cap.release()
        self.db.end_session()

    def process_frame(self):
        if not self.running: return None, None
        
        if self.source_type == "Webcam":
            if not self.cap: return None, None
            ret, frame = self.cap.read()
            if not ret: return None, None
            # MIRROR EFFECT (Flip Horizontal)
            frame = cv2.flip(frame, 1)
        else:
            img = ImageGrab.grab()
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            h, w = frame.shape[:2]
            if w > 1280: frame = cv2.resize(frame, (1280, int(h * 1280/w)))

        self.frame_count += 1
        
        if self.frame_count % self.frame_skip == 0 or not self.last_results:
            faces = self.face_detector.detect_faces(frame)
            new_results = []
            
            for (x, y, w, h) in faces:
                if w < 30 or h < 30: continue
                face_crop = frame[y:y+h, x:x+w]
                
                emotion = "Neutral"
                if self.emotion_model:
                    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    # Resize to 48x48 as per training
                    resized = cv2.resize(gray, (48, 48)).astype('float32') / 255.0
                    
                    # Tensor to self.device (CPU)
                    tensor = torch.from_numpy((resized - 0.5) / 0.5).unsqueeze(0).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = self.emotion_model(tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)[0]
                        # Debug: Print top 3 emotions
                        top3_prob, top3_idx = torch.topk(probs, 3)
                        # debug_str = " | ".join([f"{EMOTIONS[idx.item()]}: {prob.item():.2f}" for prob, idx in zip(top3_prob, top3_idx)])
                        # print(f"[DEBUG] {debug_str}") 
                        emotion = EMOTIONS[torch.argmax(output).item()]
                
                gender = "Unknown"
                if self.gender_model:
                    g_gray = cv2.resize(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY), (64, 64))
                    g_input = np.expand_dims(np.expand_dims(g_gray.astype('float32') / 255.0, -1), 0)
                    g_pred = self.gender_model.predict(g_input, verbose=0)
                    gender = GENDERS[np.argmax(g_pred)]
                
                new_results.append({'box': (x, y, w, h), 'emotion': emotion, 'gender': gender})
            
            self.last_results = new_results

        total_score = 0
        m_count, f_count = 0, 0
        emo_counts = {e: 0 for e in EMOTIONS}
        
        for res in self.last_results:
            (x, y, w, h) = res['box']
            emo, gen = res['emotion'], res['gender']
            
            emo_counts[emo] += 1
            if gen == 'Male': m_count += 1
            elif gen == 'Female': f_count += 1
            total_score += ENGAGEMENT_WEIGHTS.get(emo, 0.5)
            
            color = (0, 200, 83) if emo in ['Happy', 'Neutral', 'Surprise'] else (255, 23, 68)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Label background
            label = f"{gen} | {emo}"
            (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y-20), (x+t_w+10, y), color, -1)
            cv2.putText(frame, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        student_count = len(self.last_results)
        raw_focus = (total_score / student_count * 100) if student_count > 0 else 0
        avg_focus = self.smoother.update(raw_focus) # Apply Smoothing
        
        # Update History
        self.history_focus.pop(0)
        self.history_focus.append(avg_focus)
        
        # Calculate FPS
        curr_time = time.time()
        self.fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = curr_time

        stats = {
            'avg_focus': avg_focus,
            'total_students': student_count,
            'male_count': m_count,
            'female_count': f_count,
            'fps': self.fps
        }
        stats.update(emo_counts)
        
        if time.time() - self.last_log_time > 1.5:
            emo_str = ",".join([f"{k}:{v}" for k,v in emo_counts.items() if v > 0])
            self.db.log_data(avg_focus, student_count, emo_str, m_count, f_count)
            self.last_log_time = time.time()
            
        return frame, stats

# --- GUI ---
class StatCard(ctk.CTkFrame):
    def __init__(self, master, title, icon, color=COLOR_TEXT_MAIN):
        super().__init__(master, fg_color=COLOR_CARD, corner_radius=12, border_width=1, border_color=COLOR_BORDER)
        
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        
        icon_box = ctk.CTkFrame(self, fg_color="#F0F9FF", width=45, height=45, corner_radius=10)
        icon_box.grid(row=0, column=0, rowspan=2, padx=(15, 10), pady=15, sticky="w")
        icon_box.pack_propagate(False)
        ctk.CTkLabel(icon_box, text=icon, font=("Segoe UI Emoji", 20)).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(self, text=title, font=("Inter", 12, "bold"), text_color=COLOR_TEXT_SUB).grid(row=0, column=1, sticky="sw", pady=(15, 0), padx=(0, 15))
        self.value_lbl = ctk.CTkLabel(self, text="--", font=("Inter", 20, "bold"), text_color=color)
        self.value_lbl.grid(row=1, column=1, sticky="nw", pady=(0, 15), padx=(0, 15))

    def update_value(self, value):
        self.value_lbl.configure(text=str(value))

class ModernTeacherApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Teacher Station Pro 2.3")
        self.geometry("1400x850")
        self.configure(fg_color=COLOR_BG)
        
        self.is_monitoring = False
        self.app_core = None
        self.current_stats = None
        self.session_start_time = None
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.setup_sidebar()
        self.setup_main_area()
        
        self.active_frame = None
        self.frames = {}
        self.init_frames()
        self.show_frame("dashboard")
        
        self.after(200, self.lazy_load_core)

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color=COLOR_SIDEBAR)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)

        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.grid(row=0, column=0, padx=25, pady=30, sticky="w")
        ctk.CTkLabel(logo_frame, text="⚡", font=("Segoe UI Emoji", 26)).pack(side="left")
        ctk.CTkLabel(logo_frame, text=" Insight AI", font=("Inter", 20, "bold"), text_color=COLOR_TEXT_MAIN).pack(side="left")

        self.nav_btns = {}
        nav_items = [("dashboard", "Dashboard", "🏠"), ("analytics", "Analytics", "📊"), ("settings", "Settings", "⚙️")]
        for idx, (key, txt, icon) in enumerate(nav_items, start=1):
            btn = ctk.CTkButton(
                self.sidebar, text=f"  {icon}   {txt}", height=45, anchor="w",
                font=("Inter", 14), fg_color="transparent", text_color=COLOR_TEXT_SUB,
                hover_color=COLOR_BG, corner_radius=8,
                command=lambda k=key: self.show_frame(k)
            )
            btn.grid(row=idx, column=0, padx=15, pady=5, sticky="ew")
            self.nav_btns[key] = btn

        status_box = ctk.CTkFrame(self.sidebar, fg_color=COLOR_BG, corner_radius=10)
        status_box.grid(row=5, column=0, padx=15, pady=20, sticky="ew")
        self.status_dot = ctk.CTkLabel(status_box, text="●", font=("Arial", 18), text_color=COLOR_WARNING)
        self.status_dot.pack(side="left", padx=10, pady=10)
        self.status_txt = ctk.CTkLabel(status_box, text="Initializing...", font=("Inter", 12), text_color=COLOR_TEXT_MAIN)
        self.status_txt.pack(side="left")

    def setup_main_area(self):
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.grid(row=0, column=1, sticky="nsew", padx=30, pady=30)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

    def init_frames(self):
        self.frames["dashboard"] = self.create_dashboard()
        self.frames["analytics"] = self.create_analytics()
        self.frames["settings"] = self.create_settings()

    def create_dashboard(self):
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=7) # Video 70%
        frame.grid_columnconfigure(1, weight=3) # Controls 30%
        frame.grid_rowconfigure(0, weight=1)

        # Left: Video Container
        vid_panel = ctk.CTkFrame(frame, fg_color=COLOR_CARD, corner_radius=16, border_width=1, border_color=COLOR_BORDER)
        vid_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 20), pady=(0, 10))
        vid_panel.grid_propagate(False) 
        vid_panel.grid_columnconfigure(0, weight=1)
        vid_panel.grid_rowconfigure(1, weight=1)
        
        vid_header = ctk.CTkFrame(vid_panel, fg_color="transparent", height=40)
        vid_header.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 5))
        ctk.CTkLabel(vid_header, text="Live Feed", font=("Inter", 16, "bold"), text_color=COLOR_TEXT_MAIN).pack(side="left")
        self.live_badge = ctk.CTkLabel(vid_header, text="OFFLINE", font=("Inter", 11, "bold"), text_color=COLOR_TEXT_SUB, fg_color=COLOR_BG, corner_radius=6, width=60)
        self.live_badge.pack(side="left", padx=10)
        
        self.fps_lbl = ctk.CTkLabel(vid_header, text="FPS: 0", font=("Inter", 11), text_color=COLOR_TEXT_SUB)
        self.fps_lbl.pack(side="left", padx=5)

        self.timer_lbl = ctk.CTkLabel(vid_header, text="00:00:00", font=("Monospace", 14), text_color=COLOR_TEXT_SUB)
        self.timer_lbl.pack(side="right")

        self.video_lbl = ctk.CTkLabel(vid_panel, text="", corner_radius=8)
        self.video_lbl.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Mini Real-time Graph Container
        self.mini_fig = Figure(figsize=(4, 1.2), dpi=80, facecolor=COLOR_CARD)
        self.mini_ax = self.mini_fig.add_subplot(111)
        self.mini_ax.set_facecolor(COLOR_BG)
        self.mini_ax.set_ylim(0, 100)
        self.mini_ax.axis('off')
        self.mini_plot, = self.mini_ax.plot([0]*50, color=COLOR_PRIMARY, lw=2)
        self.mini_canvas = FigureCanvasTkAgg(self.mini_fig, master=vid_panel)
        self.mini_canvas.get_tk_widget().grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 15))
        
        # Right: Stats & Controls
        ctrl_panel = ctk.CTkFrame(frame, fg_color="transparent", width=320)
        ctrl_panel.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        ctrl_panel.grid_propagate(False)

        ctrl_box = ctk.CTkFrame(ctrl_panel, fg_color=COLOR_CARD, corner_radius=16, border_width=1, border_color=COLOR_BORDER)
        ctrl_box.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(ctrl_box, text="Controls", font=("Inter", 14, "bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", padx=20, pady=(15, 10))
        
        self.source_var = ctk.StringVar(value="Webcam")
        ctk.CTkOptionMenu(ctrl_box, variable=self.source_var, values=["Webcam", "Screen Capture"], fg_color=COLOR_BG, text_color=COLOR_TEXT_MAIN, button_color=COLOR_BORDER, button_hover_color=COLOR_BORDER).pack(fill="x", padx=20, pady=(0, 10))
        
        self.btn_start = ctk.CTkButton(ctrl_box, text="Start Session", height=40, font=("Inter", 14, "bold"), fg_color=COLOR_PRIMARY, hover_color=COLOR_PRIMARY_HOVER, command=self.toggle_session)
        self.btn_start.pack(fill="x", padx=20, pady=(0, 20))

        self.card_focus = StatCard(ctrl_panel, "Focus Score", "🎯", COLOR_PRIMARY)
        self.card_focus.pack(fill="x", pady=(0, 15))
        
        self.card_count = StatCard(ctrl_panel, "Students", "👥")
        self.card_count.pack(fill="x", pady=(0, 15))
        
        self.card_mood = StatCard(ctrl_panel, "Top Mood", "😊", COLOR_SUCCESS)
        self.card_mood.pack(fill="x", pady=(0, 15))

        self.card_gender = StatCard(ctrl_panel, "M/F Ratio", "⚤")
        self.card_gender.pack(fill="x")

        return frame

    def create_analytics(self):
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        
        head = ctk.CTkFrame(frame, fg_color="transparent")
        head.pack(fill="x", pady=(0, 20))
        ctk.CTkLabel(head, text="Session Analytics", font=("Inter", 24, "bold"), text_color=COLOR_TEXT_MAIN).pack(side="left")
        ctk.CTkButton(head, text="↻ Refresh", width=100, fg_color=COLOR_BG, text_color=COLOR_TEXT_MAIN, hover_color=COLOR_BORDER, command=self.load_analytics).pack(side="right")

        chart_box = ctk.CTkFrame(frame, fg_color=COLOR_CARD, corner_radius=16, border_width=1, border_color=COLOR_BORDER)
        chart_box.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor=COLOR_CARD)
        self.ax_focus = self.fig.add_subplot(211)
        self.ax_pie = self.fig.add_subplot(223)
        self.ax_bar = self.fig.add_subplot(224)
        self.fig.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.92, wspace=0.25, hspace=0.35)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_box)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)
        
        return frame

    def create_settings(self):
        frame = ctk.CTkFrame(self.container, fg_color="transparent")
        ctk.CTkLabel(frame, text="System Settings", font=("Inter", 24, "bold"), text_color=COLOR_TEXT_MAIN).pack(anchor="w", pady=(0, 20))
        
        box = ctk.CTkFrame(frame, fg_color=COLOR_CARD, corner_radius=16, border_width=1, border_color=COLOR_BORDER)
        box.pack(fill="x")
        
        row1 = ctk.CTkFrame(box, fg_color="transparent")
        row1.pack(fill="x", padx=20, pady=20)
        ctk.CTkLabel(row1, text="Audible Alert", font=("Inter", 14, "bold"), text_color=COLOR_TEXT_MAIN).pack(side="left")
        ctk.CTkLabel(row1, text="Beep when focus < 30%", font=("Inter", 12), text_color=COLOR_TEXT_SUB).pack(side="left", padx=10)
        self.sw_alert = ctk.CTkSwitch(row1, text="", progress_color=COLOR_PRIMARY)
        self.sw_alert.pack(side="right")
        
        ctk.CTkFrame(box, height=1, fg_color=COLOR_BORDER).pack(fill="x", padx=20)
        
        row2 = ctk.CTkFrame(box, fg_color="transparent")
        row2.pack(fill="x", padx=20, pady=20)
        ctk.CTkLabel(row2, text="Data Export", font=("Inter", 14, "bold"), text_color=COLOR_TEXT_MAIN).pack(side="left")
        ctk.CTkButton(row2, text="Download CSV Report", fg_color=COLOR_BG, text_color=COLOR_TEXT_MAIN, border_width=1, border_color=COLOR_BORDER, command=self.export_csv).pack(side="right")

        return frame

    def show_frame(self, name):
        for k, btn in self.nav_btns.items():
            is_active = (k == name)
            btn.configure(fg_color="#F0F9FF" if is_active else "transparent", text_color=COLOR_PRIMARY if is_active else COLOR_TEXT_SUB)
        
        if self.active_frame: self.active_frame.grid_forget()
        self.active_frame = self.frames[name]
        self.active_frame.grid(row=0, column=0, sticky="nsew")

    def lazy_load_core(self):
        try:
            self.app_core = TeacherMonitorApp()
            self.status_dot.configure(text_color=COLOR_SUCCESS)
            self.status_txt.configure(text="System Ready")
            self.load_analytics()
        except Exception as e:
            self.status_dot.configure(text_color=COLOR_DANGER)
            self.status_txt.configure(text="Core Error")
            print(e)

    def toggle_session(self):
        if not self.is_monitoring:
            self.is_monitoring = True
            self.btn_start.configure(text="Stop Session", fg_color=COLOR_DANGER, hover_color="#DC2626")
            self.live_badge.configure(text="LIVE", fg_color=COLOR_DANGER, text_color="white")
            self.session_start_time = time.time()
            self.app_core.start_monitoring(self.source_var.get())
            threading.Thread(target=self.loop, daemon=True).start()
            self.update_timer()
        else:
            self.is_monitoring = False
            self.btn_start.configure(text="Start Session", fg_color=COLOR_PRIMARY, hover_color=COLOR_PRIMARY_HOVER)
            self.live_badge.configure(text="OFFLINE", fg_color=COLOR_BG, text_color=COLOR_TEXT_SUB)
            self.app_core.stop_monitoring()
            self.video_lbl.configure(image=None, text="")
            messagebox.showinfo("Saved", "Session data logged.")
            self.load_analytics()

    def update_timer(self):
        if self.is_monitoring:
            elapsed = int(time.time() - self.session_start_time)
            h, m, s = elapsed//3600, (elapsed%3600)//60, elapsed%60
            self.timer_lbl.configure(text=f"{h:02}:{m:02}:{s:02}")
            self.after(1000, self.update_timer)

    def loop(self):
        while self.is_monitoring:
            try:
                frame, stats = self.app_core.process_frame()
                if frame is not None:
                    # Convert color space in thread to save main thread work
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Pass numpy array (thread-safe) to main thread
                    self.after(0, self.update_video_feed, img_rgb)
                
                if stats:
                    self.current_stats = stats
                    self.after(0, self.update_dashboard)
                
                time.sleep(0.015)
            except Exception as e: print(f"Loop error: {e}"); time.sleep(0.1)

    def update_video_feed(self, img_rgb):
        if not self.is_monitoring: return
        try:
            pil_img = Image.fromarray(img_rgb)
            
            # Robust size calculation
            w_panel = self.video_lbl.master.winfo_width() - 20
            h_panel = self.video_lbl.master.winfo_height() - 60 
            
            if w_panel > 10 and h_panel > 10:
                img_ratio = pil_img.width / pil_img.height
                panel_ratio = w_panel / h_panel
                
                if panel_ratio > img_ratio:
                    new_h = h_panel
                    new_w = int(new_h * img_ratio)
                else:
                    new_w = w_panel
                    new_h = int(new_w / img_ratio)
                    
                ctk_img = ctk.CTkImage(light_image=pil_img, size=(new_w, new_h))
                self.video_lbl.configure(image=ctk_img, text="")
                self.video_lbl.image = ctk_img # Keep reference
        except Exception as e:
            print(f"Update feed error: {e}")

    def update_dashboard(self):
        if not self.current_stats: return
        s = self.current_stats
        
        # Update Cards
        self.card_focus.update_value(f"{s['avg_focus']:.0f}%")
        self.card_count.update_value(str(s['total_students']))
        
        # UI/UX: Update FPS
        self.fps_lbl.configure(text=f"FPS: {s['fps']:.1f}")
        
        emos = {k:v for k,v in s.items() if k in EMOTIONS}
        dom = max(emos, key=emos.get) if any(emos.values()) else "-"
        self.card_mood.update_value(dom)
        self.card_gender.update_value(f"{s['male_count']}/{s['female_count']}")
        
        # Update Real-time Graph
        if self.is_monitoring:
            self.mini_plot.set_ydata(self.app_core.history_focus)
            self.mini_canvas.draw_idle()
        
        if self.sw_alert.get() == 1 and s['avg_focus'] < 30: print('\a')

    def load_analytics(self):
        if not self.app_core: return
        try:
            sessions = self.app_core.db.get_all_sessions()
            if not sessions: return
            logs = self.app_core.db.get_session_logs(sessions[0][0])
            self.ax_focus.clear(); self.ax_pie.clear(); self.ax_bar.clear()
            if logs:
                eng = [r[3] for r in logs]
                self.ax_focus.plot(eng, color=COLOR_PRIMARY, lw=2)
                self.ax_focus.set_title("Engagement Trend", fontsize=10, color=COLOR_TEXT_SUB)
                self.ax_focus.grid(True, linestyle='--', alpha=0.3)
                
                m = np.mean([r[6] for r in logs]); f = np.mean([r[7] for r in logs])
                if m+f > 0: self.ax_pie.pie([m, f], labels=['M','F'], autopct='%1.0f%%', colors=['#3B82F6','#EC4899'])
                
                emo_sum = {}
                for r in logs:
                    if r[5]:
                        for p in r[5].split(','):
                            k,v = p.split(':'); emo_sum[k] = emo_sum.get(k,0) + int(v)
                if emo_sum: self.ax_bar.bar(emo_sum.keys(), emo_sum.values(), color=COLOR_PRIMARY)
            
            self.fig.tight_layout(); self.canvas.draw()
        except Exception as e: print(e)

    def export_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv")
        if path:
            sessions = self.app_core.db.get_all_sessions()
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Start", "End", "Max Students", "Avg Score", "Status"])
                writer.writerows(sessions)
            messagebox.showinfo("Export", "Success")

if __name__ == "__main__":
    app = ModernTeacherApp()
    app.mainloop()