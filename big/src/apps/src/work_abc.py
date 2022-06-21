import numpy as np
from abc import ABC, abstractmethod


class WorkABC(ABC):
    @abstractmethod
    def load_image(self) -> np.ndarray:
        pass

    @abstractmethod
    def load_video(self) -> np.ndarray:
        pass
