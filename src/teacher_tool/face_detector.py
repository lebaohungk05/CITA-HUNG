import cv2
import os

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=1):
        """
        Face Detector Hybrid:
        1. Thử load YuNet (nếu file xịn).
        2. Nếu lỗi -> Dùng Haar Cascade Ultra-Sensitive (đã tối ưu cho mặt nghiêng).
        """
        self.detector_type = "haar" # Mặc định an toàn
        self.detector = None
        
        # --- THỬ LOAD YUNET ---
        model_path = 'D:/fer2013/face_classification/trained_models/detection_models/face_detection_yunet_2023mar.onnx'
        if os.path.exists(model_path) and os.path.getsize(model_path) > 10000:
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=model_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.5,
                    nms_threshold=0.3,
                    top_k=5000
                )
                self.detector_type = "yunet"
                print(f"[INFO] Initialized YuNet Face Detector.")
            except Exception as e:
                print(f"[WARNING] YuNet Error: {e}. Switching to Haar Cascade.")
        
        # --- LOAD HAAR CASCADE (FALLBACK) ---
        if self.detector_type == "haar":
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            if self.detector.empty():
                print("[ERROR] Failed to load Haar Cascade XML.")
            else:
                print("[INFO] Using Ultra-Sensitive Haar Cascade (Optimized for side faces).")

    def detect_faces(self, image):
        h, w, _ = image.shape
        bboxes = []
        
        if self.detector_type == "yunet":
            try:
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(image)
                if faces is not None:
                    for face in faces:
                        box = face[0:4].astype(int)
                        x, y, bw, bh = box[0], box[1], box[2], box[3]
                        x, y = max(0, x), max(0, y)
                        bboxes.append((x, y, bw, bh))
            except Exception:
                pass # Bỏ qua lỗi runtime nếu có
                    
        elif self.detector_type == "haar":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Tinh chỉnh cực đại cho mặt nghiêng:
            # scaleFactor=1.05: Quét rất kỹ (chậm hơn tí nhưng bắt dính)
            # minNeighbors=3: Chấp nhận độ tin cậy thấp hơn -> Bắt được mặt mờ/nghiêng
            faces = self.detector.detectMultiScale(
                gray, 
                scaleFactor=1.05, 
                minNeighbors=3, 
                minSize=(30, 30)
            )
            for (x, y, bw, bh) in faces:
                bboxes.append((x, y, bw, bh))
            
        return bboxes