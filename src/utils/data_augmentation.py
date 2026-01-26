from torchvision import transforms
import torch

def get_transforms(image_size=(112, 112), mode='train'):
    """
    Augmentation chiến lược "Heavy" để chống Overfitting.
    Thêm: RandomAffine, RandomErasing, Tăng góc xoay.
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            
            # Hình học: Xoay mạnh hơn, lật, co kéo nhẹ
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20), # Tăng từ 10 lên 20
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Co kéo
            
            # Màu sắc: Giữ nguyên
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            
            # Che khuất: Random Erasing (Quan trọng để chống học vẹt)
            # Buộc model phải nhận diện bằng các phần khác của khuôn mặt
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), value=0),
            
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])
    else:
        # Validation
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
