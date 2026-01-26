class EMASmoother:
    """
    Exponential Moving Average (EMA) Smoother.
    Giúp làm mượt dữ liệu thời gian thực, giảm nhiễu (jitter) nhưng vẫn bám sát xu hướng.
    
    Formula: S_t = alpha * x_t + (1 - alpha) * S_{t-1}
    - alpha (0.0 - 1.0): Hệ số làm mượt. 
      + alpha nhỏ (0.1): Rất mượt, nhưng trễ (lag) nhiều.
      + alpha lớn (0.9): Bám sát dữ liệu gốc, ít mượt.
    """
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None
