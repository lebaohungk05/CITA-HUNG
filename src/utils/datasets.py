import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .data_augmentation import get_transforms

class FERDataset(Dataset):
    """
    PyTorch Dataset chuẩn cho FER2013 với cấu trúc thư mục.
    Tự động áp dụng các transform từ utils/data_augmentation.py
    """
    def __init__(self, directory, image_size=(112, 112), mode='train'):
        self.directory = directory
        self.mode = mode
        self.transform = get_transforms(image_size, mode)
        
        self.class_to_idx = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
        self.samples = []
        
        # Load danh sách ảnh
        if os.path.exists(directory):
            print(f"[{mode.upper()}] Loading data from {directory}...")
            for emotion_name, emotion_idx in self.class_to_idx.items():
                folder = os.path.join(directory, emotion_name)
                if not os.path.exists(folder): continue
                
                files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for filename in files:
                    self.samples.append((os.path.join(folder, filename), emotion_idx))
                print(f"  -> Class '{emotion_name}': {len(files)} images")
        else:
            print(f"Warning: Directory {directory} does not exist.")
            
        print(f"[{mode.upper()}] Total: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Đọc ảnh bằng OpenCV
        image = cv2.imread(img_path)
        
        # Xử lý lỗi ảnh hỏng
        if image is None:
            image = np.zeros((112, 112, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Áp dụng PyTorch Transforms (bao gồm cả ToTensor và Normalize)
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- Legacy Support (Giữ lại DataManager cũ nếu cần) ---
from scipy.io import loadmat
import pandas as pd

class DataManager(object):
    """Legacy DataManager for loading old .csv/.mat formats."""
    def __init__(self, dataset_name='imdb', dataset_path=None, image_size=(48, 48)):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size
        # ... (Legacy code omitted for brevity but structure kept)