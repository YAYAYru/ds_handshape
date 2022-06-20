import numpy as np

from abc import ABC, abstractmethod

class ModelInference(ABC):
    @abstractmethod
    def load_model(self, path: str):
        raise NotImplementedError
    
    @abstractmethod
    def load_class_list(self, path_json_for_model: str):
        raise NotImplementedError   

    @abstractmethod
    def predict_classes(self, x) -> np.ndarray:
        raise NotImplementedError      

    @abstractmethod
    def predict_proba(self, x) -> np.ndarray:
        raise NotImplementedError      