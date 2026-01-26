import torch
import time
import numpy as np
from models.resnet18_custom import resnet18_custom

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Lưu ý: Script tự fallback sang random weights nếu không tìm thấy file, nên path sai cũng không sao để đo tốc độ
MODEL_PATH = '../trained_models/emotion_models/fer2013_resnet18_best_optimized.pth' 
INPUT_SIZE = 48
NUM_CLASSES = 7
NUM_TESTS = 500 # Số lần test

def benchmark(batch_size):
    print(f"\n--- Benchmarking Batch Size: {batch_size} ---")
    
    # 1. Model (Random init is fine for speed test)
    model = resnet18_custom(num_classes=NUM_CLASSES).to(DEVICE)
    model.eval()

    # 2. Input
    dummy_input = torch.randn(batch_size, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)

    # 3. Warm-up
    print("Warming up...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)

    # 4. Measure
    print(f"Running {NUM_TESTS} iterations...")
    times = []
    with torch.no_grad():
        for _ in range(NUM_TESTS):
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _ = model(dummy_input)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time() # This line was missing in the original, added for correctness
            times.append(end - start)

    # 5. Stats
    avg_time_per_batch = np.mean(times)
    total_time = np.sum(times)
    total_images = NUM_TESTS * batch_size
    
    fps = total_images / total_time
    avg_time_per_image = (avg_time_per_batch / batch_size) * 1000 # ms

    print(f"-> Batch Processing Time: {avg_time_per_batch*1000:.2f} ms")
    print(f"-> Latency per Image: {avg_time_per_image:.4f} ms")
    print(f"-> Throughput: {fps:.2f} Images/sec")
    
    return fps, avg_time_per_image

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    
    # Case 1: Real-time (Batch=1)
    fps_1, lat_1 = benchmark(1)
    
    # Case 2: Batch Processing (Batch=128)
    fps_128, lat_128 = benchmark(128)
    
    print("\n================ FINAL REPORT ================")
    print(f"Real-time Latency (BS=1):   {lat_1:.2f} ms ({fps_1:.0f} FPS)")
    print(f"Batch Throughput (BS=128):  {fps_128:.0f} Images/sec")
    print("==============================================")