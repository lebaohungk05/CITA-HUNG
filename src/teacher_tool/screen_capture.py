import mss
import numpy as np
import cv2
from pynput import mouse
import threading
import time

class ScreenCapturer:
    def __init__(self):
        self.monitor = None  # Vùng chụp {top, left, width, height}
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        
        # Chỉ dùng mss tạm thời để lấy thông tin monitor lúc đầu
        with mss.mss() as sct:
            self.monitors = sct.monitors

        # Biến dùng cho việc chọn vùng
        self.start_x = 0
        self.start_y = 0
        self.current_x = 0
        self.current_y = 0
        self.selecting = False
        self.selected = False

    def select_region(self):
        """
        Cho phép người dùng kéo chuột để chọn vùng màn hình (ROI).
        Sử dụng overlay cửa sổ trong suốt của OpenCV (cách đơn giản).
        """
        print("\n" + "="*50)
        print("   CHUẨN BỊ CHUYỂN SANG CỬA SỔ ZOOM/MEET...")
        print("="*50)
        for i in range(3, 0, -1):
            print(f">>> Chụp màn hình sau {i} giây...")
            time.sleep(1)
        
        print(">>> [DEBUG] Taking full screenshot for selection...")
        try:
            # Chụp toàn màn hình để làm nền cho việc chọn
            with mss.mss() as sct:
                full_screen_monitor = sct.monitors[0]
                screenshot = sct.grab(full_screen_monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            print(f">>> [DEBUG] Screenshot taken. Size: {img.shape}")
        except Exception as e:
            print(f">>> [ERROR] Failed to grab screen: {e}")
            return False
        
        print(">>> [DEBUG] Creating OpenCV window...")
        # Tạo cửa sổ full màn hình
        window_name = "Select Region (Drag mouse & Press Enter)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print(">>> Vui lòng kéo chuột để chọn vùng cần giám sát (Zoom/Meet)...")
        print(">>> Nhấn 'ENTER' hoặc 'SPACE' để xác nhận vùng chọn.")
        print(">>> Nhấn 'q' để hủy.")
        
        # Callback chuột
        roi = [0, 0, 0, 0] # x, y, w, h
        
        def on_mouse(event, x, y, flags, param):
            nonlocal roi
            if event == cv2.EVENT_LBUTTONDOWN:
                self.start_x, self.start_y = x, y
                self.selecting = True
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.selecting:
                    self.current_x, self.current_y = x, y
                    
            elif event == cv2.EVENT_LBUTTONUP:
                self.selecting = False
                self.current_x, self.current_y = x, y
                # Tính toán ROI
                x_min = min(self.start_x, self.current_x)
                y_min = min(self.start_y, self.current_y)
                w = abs(self.current_x - self.start_x)
                h = abs(self.current_y - self.start_y)
                roi = [x_min, y_min, w, h]

        cv2.setMouseCallback(window_name, on_mouse)

        print(">>> [DEBUG] Entering selection loop. Check your taskbar if window is minimized.")
        while True:
            display_img = img.copy()
            
            # Vẽ hình chữ nhật đang chọn
            if self.selecting:
                cv2.rectangle(display_img, (self.start_x, self.start_y), (self.current_x, self.current_y), (0, 255, 0), 2)
            elif roi[2] > 0 and roi[3] > 0:
                cv2.rectangle(display_img, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
                # Làm tối vùng xung quanh để nổi bật ROI
                overlay = display_img.copy()
                cv2.rectangle(overlay, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (255, 255, 255), -1)
                cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img)

            cv2.imshow(window_name, display_img)
            # Quan trọng: waitKey cần thiết để OpenCV xử lý sự kiện vẽ
            key = cv2.waitKey(20) & 0xFF 
            
            if key == 13 or key == 32: # Enter or Space
                if roi[2] > 0 and roi[3] > 0:
                    self.monitor = {"top": roi[1], "left": roi[0], "width": roi[2], "height": roi[3]}
                    print(f"Selected Region: {self.monitor}")
                    break
            elif key == ord('q'):
                print("Cancelled selection.")
                cv2.destroyAllWindows()
                return False

        cv2.destroyAllWindows()
        return True

    def start(self):
        """Bắt đầu luồng chụp ảnh liên tục"""
        if not self.monitor:
            print("Error: Region not selected yet!")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        print("Capture started...")

    def _capture_loop(self):
        # Khởi tạo mss instance mới riêng cho thread này để tránh lỗi '_thread._local'
        with mss.mss() as sct:
            while self.running:
                try:
                    # Chụp ảnh vùng đã chọn
                    sct_img = sct.grab(self.monitor)
                    # Chuyển sang định dạng OpenCV (BGR)
                    frame = np.array(sct_img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    
                    with self.lock:
                        self.frame = frame
                    
                    # Giới hạn FPS (để không ngốn CPU quá mức)
                    time.sleep(0.03) # ~30 FPS
                except Exception as e:
                    print(f"Capture error: {e}")
                    self.running = False
                    break

    def get_frame(self):
        """Lấy frame mới nhất (thread-safe)"""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        print("Capture stopped.")

# --- Test Script ---
if __name__ == "__main__":
    capturer = ScreenCapturer()
    if capturer.select_region():
        capturer.start()
        
        print("Press 'q' to quit preview.")
        while True:
            frame = capturer.get_frame()
            if frame is not None:
                cv2.imshow("Real-time Monitor (Teacher Tool)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        capturer.stop()
        cv2.destroyAllWindows()
