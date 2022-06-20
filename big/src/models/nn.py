import json

import numpy as np

from tensorflow.keras import models
from big.src.models.abc import ModelInference

class Predicter(ModelInference):
    def load_model(self, path):
        self.model = models.load_model(path)
        self.model.summary()

    def __inverse_transform(self, y_encoder, list_class):
        y = []
        for n in y_encoder:
            y.append(list_class[n])
        return np.array(y)
    
    def load_class_list(self, path_json_for_model):
        with open(path_json_for_model, "r") as f:
            self.data = json.load(f)

    def predict_classes(self, x):
        pred_y_encoder = np.argmax(self.model.predict(x), axis=-1)
        list_class = self.data["class"]  
        pred_y = self.__inverse_transform(pred_y_encoder, list_class)
        return pred_y

    def predict_proba(self, x):
        return self.model.predict(x)
    
    def evaluate(self, x, y_encoder, return_dict=True):
        return self.model.evaluate(x, y_encoder, return_dict=return_dict)