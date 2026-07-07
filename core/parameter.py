from core.backend import xp as np


class Parameter:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data)
